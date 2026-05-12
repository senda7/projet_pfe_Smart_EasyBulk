import os, json, joblib
import pandas as pd
import numpy  as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import mean_absolute_error, mean_squared_error

# ── Chemins ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "output")

# ── Configuration du modèle ───────────────────────────────────────
# Ces hyperparamètres ont été choisis pour équilibrer
# précision et généralisation (éviter le surapprentissage).
CONFIG_RF = dict(
    n_estimators      = 200,   # nombre d'arbres : plus = meilleur mais plus lent
    # B2 : max_depth réduit de 10 → 7 et min_samples_leaf augmenté de 3 → 5
    # pour réduire le surapprentissage (écart train/test ~47% sur 14j et 30j
    # avec l'ancienne configuration). Cible : ramener l'écart < 30%.
    max_depth         = 7,     # profondeur max : limite la complexité
    min_samples_leaf  = 5,     # min. observations par feuille : évite le surfit
    min_samples_split = 5,     # min. observations pour diviser un nœud
    max_features      = "sqrt",# sous-ensemble aléatoire de features par nœud
    random_state      = 42,    # reproductibilité
    n_jobs            = -1,    # utilise tous les cœurs disponibles
)

# B5 : taille du test adaptative — constantes centralisées dans config.py
# Raison : sur un petit historique (ex : 15 semaines au total), un N fixe de 12
# laisserait seulement 3 semaines en train → modèle inutilisable.
from config import N_SEMAINES_TEST_CIBLE, RATIO_TEST_MAX, N_TRAIN_MIN

# Horizons de prédiction
HORIZONS = {
    "7j" : "target_conso_7j",
    "14j": "target_conso_14j",
    "30j": "target_conso_30j",
}

# ── Utilitaires ───────────────────────────────────────────────────
_log = []

def log(msg=""):
    print(msg)
    _log.append(str(msg))

def title(t):
    log(); log("┌" + "─"*64 + "┐")
    log(f"│  {t:<62}│")
    log("└" + "─"*64 + "┘")

def ok(msg):    log(f"  ✓  {msg}")
def info(msg):  log(f"  ·  {msg}")

def calculer_metriques(y_reel, y_predit, label):
    """Calcule MAE, RMSE et MAPE et les affiche proprement."""
    mae  = mean_absolute_error(y_reel, y_predit)
    rmse = np.sqrt(mean_squared_error(y_reel, y_predit))
    mask = y_reel > 0
    mape = float(np.abs((y_reel[mask]-y_predit[mask])/y_reel[mask]).mean()*100) \
           if mask.sum() > 0 else float("nan")
    log(f"  │  {label:<10}  MAE={mae:>8.2f} cr   RMSE={rmse:>8.2f} cr   MAPE={mape:>6.1f}%")
    return {"mae": round(mae,4), "rmse": round(rmse,4), "mape": round(mape,2)}

# ════════════════════════════════════════════════════════════════
# ÉTAPE 1  —  CHARGEMENT ET SPLIT
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 1  —  CHARGEMENT ET SPLIT TEMPOREL")

for chemin in [os.path.join(OUT, "features.csv"),
               os.path.join(OUT, "feature_cols.json")]:
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"{os.path.basename(chemin)} introuvable.\n"
            f"  → Lancez d'abord : python feature_engineering.py"
        )

df = pd.read_csv(
    os.path.join(OUT, "features.csv"),
    parse_dates=["debut_semaine"]
).sort_values(["groupe_id","debut_semaine"]).reset_index(drop=True)

with open(os.path.join(OUT, "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)

info(f"Données chargées  :  {len(df)} lignes  |  "
     f"{df['groupe_id'].nunique()} groupes  |  "
     f"{len(FEATURE_COLS)} features")
info(f"Période couverte  :  {df['debut_semaine'].min().date()}  →  "
     f"{df['debut_semaine'].max().date()}")

# Split temporel strict
# IMPORTANT : on sépare par date, jamais aléatoirement.
# Entraîner sur le futur et tester sur le passé serait une fuite de données.
#
# P2 : taille de test adaptative — on borne par 20 % du total disponible.
n_semaines_totales = df["debut_semaine"].nunique()
n_semaines_test    = max(1, min(
    N_SEMAINES_TEST_CIBLE,
    int(n_semaines_totales * RATIO_TEST_MAX)
))
info(f"Taille test calculée : {n_semaines_test} semaines "
     f"(sur {n_semaines_totales} disponibles)")

date_coupure = df["debut_semaine"].max() - pd.Timedelta(weeks=n_semaines_test)
df_train     = df[df["debut_semaine"] <= date_coupure].copy()
df_test      = df[df["debut_semaine"] >  date_coupure].copy()

log()
log("  ┌" + "─"*60 + "┐")
log(f"  │  {'SPLIT TEMPOREL':<58}│")
log("  ├" + "─"*60 + "┤")
log(f"  │  Train  :  {df_train['debut_semaine'].min().date()}  →  "
    f"{df_train['debut_semaine'].max().date()}   ({len(df_train):>4} lignes)  │")
log(f"  │  Test   :  {df_test['debut_semaine'].min().date()}   →  "
    f"{df_test['debut_semaine'].max().date()}   ({len(df_test):>4} lignes)  │")
log("  └" + "─"*60 + "┘")

if len(df_train) < N_TRAIN_MIN:
    raise ValueError(
        f"Données d'entraînement insuffisantes : {len(df_train)} lignes "
        f"(minimum requis : {N_TRAIN_MIN}).\n"
        f"  Vérifiez que budget_history couvre au moins "
        f"{N_TRAIN_MIN + n_semaines_test} semaines au total."
    )
if len(df_test) == 0:
    raise ValueError(
        "Jeu de test vide après le split. Augmentez la période d'historique."
    )

X_train = np.nan_to_num(df_train[FEATURE_COLS].values, nan=0.0)
X_test  = np.nan_to_num(df_test[FEATURE_COLS].values,  nan=0.0)

# ════════════════════════════════════════════════════════════════
# ÉTAPE 2  —  ENTRAÎNEMENT PAR HORIZON
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 2  —  ENTRAÎNEMENT RANDOM FOREST  (3 horizons)")

resultats     = {}
importances   = {}

for horizon, colonne_cible in HORIZONS.items():

    log()
    log("  ┌" + "─"*60 + "┐")
    log(f"  │  Horizon : {horizon}  —  '{colonne_cible}'   "
        f"{'':>{60 - len(horizon) - len(colonne_cible) - 14}}│")
    log("  ├" + "─"*60 + "┤")

    y_train = np.nan_to_num(df_train[colonne_cible].values, nan=0.0)
    y_test  = np.nan_to_num(df_test[colonne_cible].values,  nan=0.0)

    # Entraînement
    modele = RandomForestRegressor(**CONFIG_RF)
    modele.fit(X_train, y_train)

    # Prédictions — on force à 0 les valeurs négatives
    # (une consommation ne peut pas être négative)
    pred_train = np.maximum(0, modele.predict(X_train))
    pred_test  = np.maximum(0, modele.predict(X_test))

    # Métriques
    log(f"  │  {'':10}  {'MAE':>14}  {'RMSE':>14}  {'MAPE':>8}")
    log("  ├" + "─"*60 + "┤")
    m_train = calculer_metriques(y_train, pred_train, "Train")
    m_test  = calculer_metriques(y_test,  pred_test,  "Test")
    log("  └" + "─"*60 + "┘")

    # Importance des features (top 5 affichées)
    imp = pd.Series(modele.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)
    importances[horizon] = imp.to_dict()

    log(f"\n  Top 5 features les plus influentes (horizon {horizon}) :")
    for rang, (feature, valeur) in enumerate(imp.head(5).items(), 1):
        famille = feature.split("_")[0].upper()
        barre   = "█" * int(valeur * 80)
        log(f"    {rang}.  [{famille:<8}]  {feature:<42}  {valeur:.4f}  {barre}")

    # Sauvegarde du modèle
    chemin_modele = os.path.join(OUT, f"model_rf_{horizon}.pkl")
    joblib.dump(modele, chemin_modele)
    ok(f"model_rf_{horizon}.pkl  →  modèle sauvegardé")

    resultats[horizon] = {
        "metriques_train" : m_train,
        "metriques_test"  : m_test,
        "top10_features"  : imp.head(10).to_dict(),
    }

# ════════════════════════════════════════════════════════════════
# ÉTAPE 3  —  RÉSUMÉ
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 3  —  RÉSUMÉ DES PERFORMANCES")

log()
log("  Les métriques du jeu de TEST mesurent la qualité réelle du modèle")
log("  sur des données qu'il n'a jamais vues pendant l'entraînement.")
log()
log("  ┌──────────┬──────────────────────┬──────────────────────┐")
log("  │ Horizon  │   Train              │   Test               │")
log("  │          │   MAE      MAPE      │   MAE      MAPE      │")
log("  ├──────────┼──────────────────────┼──────────────────────┤")
for h, r in resultats.items():
    tr = r["metriques_train"]
    te = r["metriques_test"]
    log(f"  │  {h:<7} │  {tr['mae']:>7.2f} cr  {tr['mape']:>6.1f}%  │"
        f"  {te['mae']:>7.2f} cr  {te['mape']:>6.1f}%  │")
log("  └──────────┴──────────────────────┴──────────────────────┘")
log()
log("  Lecture : MAE = erreur moyenne en crédits")
log("            MAPE = erreur relative en % (plus c'est bas, mieux c'est)")

# ── Sauvegarde des résultats (utilisés par 04 et 05) ─────────────
with open(os.path.join(OUT, "training_results.json"), "w", encoding="utf-8") as f:
    json.dump(resultats, f, indent=2, ensure_ascii=False)
ok("training_results.json  →  métriques et importances sauvegardées")

with open(os.path.join(OUT, "training_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_log))
ok("training_report.txt   →  rapport complet d'entraînement")

log()
log("══════════════════════════════════════════════════════════════════")
log("  ENTRAÎNEMENT TERMINÉ  —  lancer ensuite : python evaluate.py")
log("══════════════════════════════════════════════════════════════════")