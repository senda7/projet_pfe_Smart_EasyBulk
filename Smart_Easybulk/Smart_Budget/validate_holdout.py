"""
╔══════════════════════════════════════════════════════════════════╗
║  validate_holdout.py                                             ║
║                                                                  ║
║  VALIDATION EXTERNE DU MODÈLE                                    ║
║                                                                  ║
║  Objectif : prouver la crédibilité du modèle en le testant sur   ║
║  une base qu'il n'a JAMAIS vue au training.                      ║
║                                                                  ║
║  Principe                                                        ║
║  ─────────                                                       ║
║   1. Modèles entraînés  → output/model_rf_{7,14,30}j.pkl         ║
║      (entraînés sur data/*.json, seed=42)                        ║
║                                                                  ║
║   2. Base holdout       → data_holdout/*.json                    ║
║      (même pattern métier, seed=999 → tirages indépendants)      ║
║                                                                  ║
║   3. Pipeline holdout   → output_holdout/features.csv            ║
║      (mêmes scripts, chemins surchargés via env vars)            ║
║                                                                  ║
║   4. Évaluation         → régression (MAE/RMSE/MAPE)             ║
║                           classification (accuracy/recall)       ║
║                                                                  ║
║  Si le modèle obtient des métriques comparables sur le holdout,  ║
║  il a appris le PATTERN (et non mémorisé les données).           ║
║                                                                  ║
║  Usage : python validate_holdout.py                              ║
╚══════════════════════════════════════════════════════════════════╝
"""
import os, sys, json, subprocess, joblib
from collections import Counter
import numpy  as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

BASE             = os.path.dirname(os.path.abspath(__file__))
DATA_HOLDOUT     = os.path.join(BASE, "data_holdout")
OUT_HOLDOUT      = os.path.join(BASE, "output_holdout")
OUT_TRAIN        = os.path.join(BASE, "output")
os.makedirs(OUT_HOLDOUT, exist_ok=True)

# B5 : constantes centralisées dans config.py
from config import SEUIL_SAFE, SEUIL_CRITIQUE, ALPHA_PRUDENT, CLASSES_RISQUE as CLASSES


def title(t):
    print(); print("┌" + "─"*64 + "┐"); print(f"│  {t:<62}│"); print("└" + "─"*64 + "┘")
def ok(m):   print(f"  ✓  {m}")
def info(m): print(f"  ·  {m}")
def warn(m): print(f"  ⚠  {m}")


def classifier_risque(restant, total):
    if total <= 0 or restant <= 0: return "CRITIQUE"
    r = restant / total
    if r <= SEUIL_CRITIQUE: return "CRITIQUE"
    if r <= SEUIL_SAFE:     return "DANGER"
    return "SAFE"


def lancer(cmd, env_extra=None):
    """Lance un script enfant avec env vars optionnelles."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, env=env, cwd=BASE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"Échec : {' '.join(cmd)}")
    return r.stdout


# ════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Vérifier la présence des modèles entraînés
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 1 — Vérification des modèles entraînés")

modeles_path = {h: os.path.join(OUT_TRAIN, f"model_rf_{h}.pkl") for h in ("7j","14j","30j")}
for h, p in modeles_path.items():
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} introuvable. Lance d'abord le pipeline training :\n"
            f"  python 'load and explore.py'  →  python feature_engineering.py  →  python train_models.py"
        )
    ok(f"model_rf_{h}.pkl présent")


# ════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Générer la base holdout si absente
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 2 — Génération de la base holdout (seed=999)")

regen_h = os.path.join(DATA_HOLDOUT, "_regen_holdout.py")
if not os.path.exists(os.path.join(DATA_HOLDOUT, "budget_history.json")):
    info("Base holdout absente → génération …")
    lancer([sys.executable, regen_h])
else:
    info("Base holdout déjà présente → régénération pour forcer la fraîcheur")
    lancer([sys.executable, regen_h])
ok("data_holdout/*.json prêts")


# ════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Nettoyage + features sur la base holdout
#          (chemins surchargés par variables d'environnement)
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 3 — Construction des features holdout")

env_h = {"BUDGET_DATA": DATA_HOLDOUT, "BUDGET_OUT": OUT_HOLDOUT}

info("load and explore.py  →  output_holdout/clean_*.json")
lancer([sys.executable, "load and explore.py"], env_extra=env_h)

# events_calendar.json est requis par feature_engineering.py :
# on le copie depuis output/ vers output_holdout/ s'il n'existe pas.
import shutil
src_cal = os.path.join(OUT_TRAIN, "events_calendar.json")
dst_cal = os.path.join(OUT_HOLDOUT, "events_calendar.json")
if os.path.exists(src_cal) and not os.path.exists(dst_cal):
    shutil.copy(src_cal, dst_cal)
    info("events_calendar.json copié depuis output/")

info("feature_engineering.py  →  output_holdout/features.csv")
lancer([sys.executable, "feature_engineering.py"], env_extra=env_h)
ok("Features holdout construites")


# ════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Charger modèles + features holdout
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 4 — Chargement modèles + features holdout")

with open(os.path.join(OUT_TRAIN, "feature_cols.json")) as f:
    FEATURES = json.load(f)

df_h = pd.read_csv(os.path.join(OUT_HOLDOUT, "features.csv"),
                   parse_dates=["debut_semaine"])
# Quota total pour la classification (reconstruit via JOIN)
with open(os.path.join(OUT_HOLDOUT, "clean_groupes.json")) as f:
    grp = pd.DataFrame(json.load(f))[["id","quota"]].rename(
        columns={"id":"groupe_id","quota":"_quota_total"})
df_h = df_h.merge(grp, on="groupe_id", how="left")

info(f"Holdout : {len(df_h)} lignes  "
     f"({df_h['debut_semaine'].min().date()} → {df_h['debut_semaine'].max().date()})")

X_h = np.nan_to_num(df_h[FEATURES].values, nan=0.0)

modeles = {h: joblib.load(p) for h, p in modeles_path.items()}
ok("3 modèles chargés (7j, 14j, 30j)")


# ════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Métriques de régression sur le holdout
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 5 — Régression sur le holdout (MAE / RMSE / MAPE)")

cibles = {"7j":"target_conso_7j", "14j":"target_conso_14j", "30j":"target_conso_30j"}

print()
print("  ┌──────────┬──────────────┬──────────────┬──────────────┐")
print("  │ Horizon  │   MAE (cr)   │   RMSE (cr)  │     MAPE     │")
print("  ├──────────┼──────────────┼──────────────┼──────────────┤")

resultats = {}
for h, col in cibles.items():
    m = modeles[h]
    y_reel = np.nan_to_num(df_h[col].values, nan=0.0)
    y_pred = np.maximum(0, m.predict(X_h))

    # Prédiction prudente pour la classification
    par_arbre = np.array([t.predict(X_h) for t in m.estimators_])
    y_prudent = np.maximum(0, par_arbre.mean(axis=0) + ALPHA_PRUDENT * par_arbre.std(axis=0))

    mae  = mean_absolute_error(y_reel, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_reel, y_pred)))
    mask = y_reel > 0
    mape = float(np.abs((y_reel[mask]-y_pred[mask])/y_reel[mask]).mean()*100) if mask.sum()>0 else float("nan")

    resultats[h] = {"y_reel":y_reel, "y_pred":y_pred, "y_prudent":y_prudent,
                    "mae":mae, "rmse":rmse, "mape":mape}

    print(f"  │  {h:<7} │  {mae:>10.2f}  │  {rmse:>10.2f}  │  {mape:>9.1f}%  │")
print("  └──────────┴──────────────┴──────────────┴──────────────┘")


# ════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Classification du risque sur le holdout (horizon 30j)
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 6 — Classification SAFE / DANGER / CRITIQUE sur le holdout")

p30 = resultats["30j"]
ql  = df_h["budget_quota_libre"].values
qt  = df_h["_quota_total"].values

y_reel_cls = np.array([classifier_risque(float(ql[i]-p30["y_reel"][i]),    float(qt[i])) for i in range(len(ql))])
y_pred_cls = np.array([classifier_risque(float(ql[i]-p30["y_prudent"][i]), float(qt[i])) for i in range(len(ql))])

print()
info(f"Distribution réelle  : {dict(Counter(y_reel_cls))}")
info(f"Distribution prédite : {dict(Counter(y_pred_cls))}")
print()

acc  = accuracy_score(y_reel_cls, y_pred_cls)
prec = precision_score(y_reel_cls, y_pred_cls, average="weighted", zero_division=0)
rec  = recall_score(y_reel_cls, y_pred_cls, average="weighted", zero_division=0)
f1   = f1_score(y_reel_cls, y_pred_cls, average="weighted", zero_division=0)

print(f"  Accuracy   : {acc:>6.1%}")
print(f"  Precision  : {prec:>6.1%}")
print(f"  Recall     : {rec:>6.1%}")
print(f"  F1-score   : {f1:>6.1%}")
print()
print("  Détail par classe :")
rep = classification_report(y_reel_cls, y_pred_cls, labels=CLASSES, zero_division=0)
for line in rep.split("\n"):
    print(f"    {line}")


# ════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Comparaison training vs holdout
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 7 — Comparaison training vs holdout")

# Métriques training (si evaluation_report existe, on extrait à la volée ;
# sinon on les recalcule à partir du features.csv training)
try:
    with open(os.path.join(OUT_TRAIN, "training_results.json")) as f:
        res_train = json.load(f)

    print()
    print("  ┌──────────┬─────────────────────┬─────────────────────┐")
    print("  │ Horizon  │    TEST (training)  │      HOLDOUT        │")
    print("  │          │       MAE   MAPE    │       MAE   MAPE    │")
    print("  ├──────────┼─────────────────────┼─────────────────────┤")
    for h in ("7j","14j","30j"):
        mt = res_train[h]["metriques_test"]
        mh = resultats[h]
        print(f"  │  {h:<7} │  {mt['mae']:>7.1f}  {mt['mape']:>5.1f}%    "
              f"│  {mh['mae']:>7.1f}  {mh['mape']:>5.1f}%    │")
    print("  └──────────┴─────────────────────┴─────────────────────┘")
    print()
    info("Si les deux colonnes sont proches → le modèle généralise (crédibilité OK).")
    info("Si holdout >> training → surapprentissage, modèle à retravailler.")
except Exception as e:
    warn(f"Comparaison training indisponible : {e}")


# ════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ════════════════════════════════════════════════════════════════
rapport = {
    "regression": {
        h: {"mae": round(resultats[h]["mae"],2),
            "rmse": round(resultats[h]["rmse"],2),
            "mape": round(resultats[h]["mape"],2)}
        for h in resultats
    },
    "classification": {
        "accuracy":  round(acc,4),
        "precision": round(prec,4),
        "recall":    round(rec,4),
        "f1":        round(f1,4),
        "distribution_reelle":  dict(Counter(y_reel_cls.tolist())),
        "distribution_predite": dict(Counter(y_pred_cls.tolist())),
    },
    "holdout_info": {
        "n_lignes": int(len(df_h)),
        "date_debut": str(df_h["debut_semaine"].min().date()),
        "date_fin":   str(df_h["debut_semaine"].max().date()),
        "seed":       999,
    },
}
with open(os.path.join(OUT_HOLDOUT, "holdout_report.json"), "w", encoding="utf-8") as f:
    json.dump(rapport, f, indent=2, ensure_ascii=False)
ok("output_holdout/holdout_report.json sauvegardé")

print()
print("══════════════════════════════════════════════════════════════════")
print("  VALIDATION HOLDOUT TERMINÉE")
print("══════════════════════════════════════════════════════════════════")
