"""
╔══════════════════════════════════════════════════════════════════╗
║  04_evaluate.py                                                  ║
║                                                                  ║
║  RÔLE : Valider scientifiquement la qualité du modèle selon     ║
║         les deux dimensions du problème.                         ║
║                                                                  ║
║  PARTIE A  —  ÉVALUATION RÉGRESSION                              ║
║  Le modèle prédit un chiffre (crédits consommés).               ║
║  Métriques adaptées : MAE, RMSE, MAPE                           ║
║                                                                  ║
║  accuracy / precision / recall / f1 sont INVALIDES ici car      ║
║  ces métriques nécessitent des classes discrètes (0/1/2).       ║
║  Un chiffre comme 87.3 cr n'est pas une classe.                 ║
║                                                                  ║
║  PARTIE B  —  ÉVALUATION CLASSIFICATION DU RISQUE               ║
║  La classification SAFE/DANGER/CRITIQUE est dérivée du          ║
║  résultat de la régression :                                     ║
║    risque_reel   = classifier(quota_free − conso_réelle)        ║
║    risque_prédit = classifier(quota_free − conso_prédite)       ║
║  accuracy / precision / recall / f1 sont VALIDES ici.           ║
║                                                                  ║
║  ENTRÉE  → output/features.csv                                   ║
║            output/model_rf_*.pkl                                 ║
║            output/training_results.json                          ║
║            output/feature_cols.json                              ║
║                                                                  ║
║  SORTIE  → output/plot_prediction_vs_reel.png                    ║
║            output/plot_importance_features.png                   ║
║            output/plot_residus.png                               ║
║            output/plot_mae_train_vs_test.png                     ║
║            output/plot_classification_risque.png                 ║
║            output/evaluation_report.txt                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, json, joblib
import pandas as pd
import numpy  as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "output")

# B5 : constantes centralisées dans config.py
from config import (
    N_SEMAINES_TEST_CIBLE, RATIO_TEST_MAX,
    SEUIL_SAFE, SEUIL_CRITIQUE, ALPHA_PRUDENT, CLASSES_RISQUE,
)

HORIZONS = {"7j":"target_conso_7j","14j":"target_conso_14j","30j":"target_conso_30j"}
C_REEL="#1D9E75"; C_PREDIT="#378ADD"; C_ECART="#E24B4A"
C_FAMILLE={"conso":"#378ADD","tendance":"#1D9E75","campagne":"#EF9F27","budget":"#E24B4A","saison":"#9F77DD","evenement":"#9F77DD","jours":"#888780"}

_log=[]
def log(msg=""):
    print(msg); _log.append(str(msg))
def title(t):
    log(); log("┌"+"─"*64+"┐"); log(f"│  {t:<62}│"); log("└"+"─"*64+"┘")
def ok(msg):   log(f"  ✓  {msg}")
def info(msg): log(f"  ·  {msg}")
def warn(msg): log(f"  ⚠  {msg}")
def sauvegarder(fig,nom):
    fig.savefig(os.path.join(OUT,nom),dpi=130,bbox_inches="tight"); plt.close(fig); ok(nom)
def metriques_reg(y_r,y_p):
    mae=mean_absolute_error(y_r,y_p)
    rmse=np.sqrt(mean_squared_error(y_r,y_p))
    mask=y_r>0
    mape=float(np.abs((y_r[mask]-y_p[mask])/y_r[mask]).mean()*100) if mask.sum()>0 else float("nan")
    return mae,rmse,mape
def classifier_risque(restant,total):
    if total<=0 or restant<=0: return "CRITIQUE"
    r=restant/total
    if r<=SEUIL_CRITIQUE: return "CRITIQUE"
    if r<=SEUIL_SAFE: return "DANGER"
    return "SAFE"

# ── Chargement ────────────────────────────────────────────────────
title("CHARGEMENT")
REQUIS = ["features.csv", "feature_cols.json", "training_results.json", "clean_groupes.json"]
for f in REQUIS:
    if not os.path.exists(os.path.join(OUT, f)):
        raise FileNotFoundError(
            f"{f} introuvable dans output/. Lancez d'abord dans l'ordre : "
            "python 'load and explore.py' → python feature_engineering.py → "
            "python train_models.py"
        )
df=pd.read_csv(os.path.join(OUT,"features.csv"),parse_dates=["debut_semaine"]).sort_values(["groupe_id","debut_semaine"]).reset_index(drop=True)
with open(os.path.join(OUT,"feature_cols.json")) as f: FEATURES=json.load(f)
with open(os.path.join(OUT,"training_results.json")) as f: res_train=json.load(f)

# B1 : budget_quota_total n'est plus dans features.csv (option B — 10 features).
#      On le reconstruit depuis clean_groupes.json pour la classification SAFE/DANGER/CRITIQUE.
with open(os.path.join(OUT,"clean_groupes.json")) as f:
    _grp = pd.DataFrame(json.load(f))[["id","quota"]].rename(columns={"id":"groupe_id","quota":"_quota_total"})
df = df.merge(_grp, on="groupe_id", how="left")

# M1 : N_SEMAINES_TEST adaptatif selon volumétrie (cohérent avec train_models.py)
n_semaines_totales = df["debut_semaine"].nunique()
N_SEMAINES_TEST    = max(1, min(N_SEMAINES_TEST_CIBLE, int(n_semaines_totales * RATIO_TEST_MAX)))
info(f"N_semaines_test = {N_SEMAINES_TEST}  (sur {n_semaines_totales} semaines totales)")

date_coup=df["debut_semaine"].max()-pd.Timedelta(weeks=N_SEMAINES_TEST)
df_test=df[df["debut_semaine"]>date_coup].copy()
if len(df_test) == 0:
    raise RuntimeError("df_test est vide — données insuffisantes pour évaluer.")
X_test=np.nan_to_num(df_test[FEATURES].values,nan=0.0)
info(f"Test : {df_test['debut_semaine'].min().date()} → {df_test['debut_semaine'].max().date()} ({len(df_test)} lignes)")
preds={}
for h,col in HORIZONS.items():
    mp=os.path.join(OUT,f"model_rf_{h}.pkl")
    if not os.path.exists(mp): warn(f"model_rf_{h}.pkl absent"); continue
    m=joblib.load(mp); yr=np.nan_to_num(df_test[col].values,nan=0.0); yp=np.maximum(0,m.predict(X_test))
    # Prédiction prudente : moyenne + ALPHA × std des arbres (borne haute).
    # Sert uniquement à la classification SAFE/DANGER/CRITIQUE, pas à la régression.
    preds_par_arbre = np.array([t.predict(X_test) for t in m.estimators_])
    yp_prudent = np.maximum(0, preds_par_arbre.mean(axis=0) + ALPHA_PRUDENT * preds_par_arbre.std(axis=0))
    mae,rmse,mape=metriques_reg(yr,yp)
    preds[h]={"y_reel":yr,"y_predit":yp,"y_predit_prudent":yp_prudent,"model":m,"mae":mae,"rmse":rmse,"mape":mape}
    ok(f"Horizon {h} : MAE={mae:.1f} cr  RMSE={rmse:.1f} cr  MAPE={mape:.1f}%")

# ════════════════════════════════════════════════════════════════
# PARTIE A  —  RÉGRESSION
# ════════════════════════════════════════════════════════════════
title("PARTIE A  —  ÉVALUATION RÉGRESSION  (MAE / RMSE / MAPE)")
info("Ces métriques sont les SEULES valides pour un modèle de régression.")
info("accuracy/f1/recall sont invalides ici — le modèle prédit un chiffre, pas une classe.")
log()
log("  ┌──────────┬──────────────┬──────────────┬──────────────┐")
log("  │ Horizon  │   MAE (cr)   │   RMSE (cr)  │     MAPE     │")
log("  ├──────────┼──────────────┼──────────────┼──────────────┤")
for h in HORIZONS:
    if h in preds:
        p=preds[h]
        log(f"  │  {h:<7} │  {p['mae']:>10.2f}  │  {p['rmse']:>10.2f}  │  {p['mape']:>9.1f}%  │")
log("  └──────────┴──────────────┴──────────────┴──────────────┘")
log(); log("  · MAE  = erreur moyenne en crédits (±X cr par prédiction)")
log("  · RMSE = si RMSE >> MAE → quelques prédictions très mauvaises")
log("  · MAPE = erreur relative en % (comparaison entre groupes)")

# ════════════════════════════════════════════════════════════════
# PARTIE B  —  CLASSIFICATION DU RISQUE
# ════════════════════════════════════════════════════════════════
title("PARTIE B  —  ÉVALUATION CLASSIFICATION  (accuracy / precision / recall / f1)")
info("On dérive les labels SAFE/DANGER/CRITIQUE depuis la régression :")
info("  risque_reel   = classifier( quota_free - conso_reelle_30j )")
info("  risque_predit = classifier( quota_free - conso_predite_30j )")
info("Les métriques de classification sont VALIDES ici car les sorties sont des classes.")
log()

metriques_classif = {}

if "30j" in preds:
    p30 = preds["30j"]
    ql = df_test["budget_quota_libre"].values
    qt = df_test["_quota_total"].values  # B1 : reconstruit via JOIN clean_groupes.json

    # Classification dérivée d'une conso PRUDENTE (mean + 0.84·std des arbres)
    # pour maximiser le recall CRITIQUE : mieux vaut une fausse alerte qu'un
    # groupe qui tombe à 0 sans prévenir.
    y_reel_cls  = np.array([classifier_risque(float(ql[i]-p30["y_reel"][i]),          float(qt[i])) for i in range(len(ql))])
    y_pred_cls  = np.array([classifier_risque(float(ql[i]-p30["y_predit_prudent"][i]),float(qt[i])) for i in range(len(ql))])

    from collections import Counter
    log(f"  Distribution réelle  : {dict(Counter(y_reel_cls))}")
    log(f"  Distribution prédite : {dict(Counter(y_pred_cls))}")
    log()

    acc  = accuracy_score(y_reel_cls,  y_pred_cls)
    prec = precision_score(y_reel_cls, y_pred_cls, average="weighted", zero_division=0)
    rec  = recall_score(y_reel_cls,    y_pred_cls, average="weighted", zero_division=0)
    f1   = f1_score(y_reel_cls,        y_pred_cls, average="weighted", zero_division=0)

    log("  ┌────────────────────────────────────────────────────────────┐")
    log(f"  │  Accuracy   : {acc:>6.1%}  — % de classes correctement prédites  │")
    log(f"  │  Precision  : {prec:>6.1%}  — parmi CRITIQUE prédit, % vrais CRITIQUE│")
    log(f"  │  Recall     : {rec:>6.1%}  — parmi CRITIQUE réel, % correctement vus│")
    log(f"  │  F1-score   : {f1:>6.1%}  — équilibre precision/recall          │")
    log("  └────────────────────────────────────────────────────────────┘")
    log()

    # Rapport par classe
    log("  Détail par classe :")
    rep = classification_report(y_reel_cls, y_pred_cls, labels=CLASSES_RISQUE, zero_division=0)
    for l in rep.split("\n"): log(f"    {l}")
    log()
    log("  ⚠  MÉTRIQUE CLÉ MÉTIER : recall de la classe CRITIQUE")
    log("     Manquer un CRITIQUE = groupe qui tombe à 0 sans alerte")
    log("     Un recall CRITIQUE faible est plus grave qu'une accuracy faible")

    metriques_classif = {"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4)}

    # Graphe matrice de confusion
    cm = confusion_matrix(y_reel_cls, y_pred_cls, labels=CLASSES_RISQUE)
    fig,ax = plt.subplots(figsize=(7,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES_RISQUE)
    disp.plot(ax=ax, colorbar=False, cmap=plt.cm.Blues, values_format="d")
    ax.set_title("Matrice de confusion — Classification du risque\n(jeu de test, horizon 30j)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Risque PRÉDIT par le modèle", fontsize=9)
    ax.set_ylabel("Risque RÉEL (historique)", fontsize=9)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    sauvegarder(fig, "plot_classification_risque.png")
else:
    warn("Horizon 30j absent — évaluation classification ignorée")

# ════════════════════════════════════════════════════════════════
# GRAPHES RÉGRESSION
# ════════════════════════════════════════════════════════════════
title("GRAPHES RÉGRESSION")

# Graphe 1 : Prédiction vs Réel
fig,axes = plt.subplots(3,1,figsize=(13,12))
fig.suptitle("Random Forest — Prédiction vs Consommation Réelle (jeu de test)",fontsize=13,fontweight="bold")
for ax,(h,col) in zip(axes,HORIZONS.items()):
    if h not in preds: ax.set_visible(False); continue
    p=preds[h]; x=range(len(p["y_reel"]))
    ax.plot(x,p["y_reel"],  color=C_REEL,  lw=2.0,label="Réel",  zorder=3)
    ax.plot(x,p["y_predit"],color=C_PREDIT,lw=2.0,label="Prédit",linestyle="--",zorder=3)
    ax.fill_between(x,p["y_reel"],p["y_predit"],alpha=0.12,color=C_ECART,label="Écart")
    ax.set_title(f"Horizon {h}    MAE={p['mae']:.1f} cr    RMSE={p['rmse']:.1f} cr    MAPE={p['mape']:.1f}%",fontsize=10,pad=6)
    ax.set_ylabel("Crédits"); ax.legend(loc="upper right",fontsize=8)
    ax.grid(True,alpha=0.2,linestyle="--"); ax.set_facecolor("#f9f9f9")
axes[-1].set_xlabel("Semaine (jeu de test)")
plt.tight_layout(); sauvegarder(fig,"plot_prediction_vs_reel.png")

# Graphe 2 : Feature Importance
if "30j" in preds:
    imp=pd.Series(preds["30j"]["model"].feature_importances_,index=FEATURES).sort_values(ascending=True)
    colors=[C_FAMILLE.get(f.split("_")[0],"#888780") for f in imp.index]
    noms=[f.replace("_"," ") for f in imp.index]
    fig,ax=plt.subplots(figsize=(11,8))
    ax.barh(noms,imp.values,color=colors,edgecolor="none",height=0.7)
    ax.set_title("Importance des features — Random Forest (horizon 30j)",fontsize=11,fontweight="bold",pad=10)
    ax.set_xlabel("Importance relative"); ax.grid(True,axis="x",alpha=0.2,linestyle="--"); ax.set_facecolor("#f9f9f9")
    leg=[mpatches.Patch(color=c,label=f"[{k}] {d}") for k,c,d in [
        ("conso","#378ADD","Consommation passée"),("tendance","#1D9E75","Tendance"),
        ("campagne","#EF9F27","Activité campagnes"),("budget","#E24B4A","État quota"),
        ("saison","#9F77DD","Calendrier / événement"),("jours","#888780","Rythme avant zéro")]]
    ax.legend(handles=leg,loc="lower right",fontsize=8); plt.tight_layout(); sauvegarder(fig,"plot_importance_features.png")

# Graphe 3 : Résidus
if "30j" in preds:
    p=preds["30j"]; res=p["y_reel"]-p["y_predit"]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Analyse des résidus — horizon 30j",fontsize=11,fontweight="bold")
    axes[0].hist(res,bins=22,color=C_PREDIT,edgecolor="white",alpha=0.85)
    axes[0].axvline(0,color=C_ECART,lw=1.8,linestyle="--",label="Résidu=0")
    axes[0].axvline(res.mean(),color="#888780",lw=1.2,linestyle=":",label=f"Moy={res.mean():.1f}")
    axes[0].set_title("Distribution des erreurs"); axes[0].set_xlabel("Erreur (cr)"); axes[0].legend(fontsize=8)
    axes[0].grid(True,alpha=0.2); axes[0].set_facecolor("#f9f9f9")
    axes[1].scatter(p["y_predit"],res,color=C_PREDIT,alpha=0.45,s=25)
    axes[1].axhline(0,color=C_ECART,lw=1.8,linestyle="--")
    axes[1].set_title("Résidus vs Valeurs prédites"); axes[1].set_xlabel("Prédit (cr)"); axes[1].set_ylabel("Résidu")
    axes[1].grid(True,alpha=0.2); axes[1].set_facecolor("#f9f9f9")
    plt.tight_layout(); sauvegarder(fig,"plot_residus.png")

# Graphe 4 : MAE Train vs Test
h_list=[h for h in res_train]; mt=[res_train[h]["metriques_train"]["mae"] for h in h_list]; mte=[res_train[h]["metriques_test"]["mae"] for h in h_list]
x=np.arange(len(h_list)); w=0.35
fig,ax=plt.subplots(figsize=(9,5))
b1=ax.bar(x-w/2,mt, w,label="Train",color=C_PREDIT,alpha=0.85)
b2=ax.bar(x+w/2,mte,w,label="Test", color="#888780",alpha=0.85)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f"{b.get_height():.0f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
ax.set_title("MAE — Train vs Test par horizon\n(un grand écart = surapprentissage)",fontsize=11,fontweight="bold")
ax.set_xlabel("Horizon"); ax.set_ylabel("MAE (crédits)"); ax.set_xticks(x); ax.set_xticklabels(h_list)
ax.legend(fontsize=9); ax.grid(True,axis="y",alpha=0.2,linestyle="--"); ax.set_facecolor("#f9f9f9")
plt.tight_layout(); sauvegarder(fig,"plot_mae_train_vs_test.png")

# ── Rapport final ────────────────────────────────────────────────
title("RAPPORT FINAL")
log()
log("  PARTIE A  —  RÉGRESSION")
log("  ┌──────────┬────────────┬────────────┬──────────┐")
log("  │ Horizon  │  MAE (cr)  │  RMSE (cr) │   MAPE   │")
log("  ├──────────┼────────────┼────────────┼──────────┤")
for h in HORIZONS:
    if h in preds:
        p=preds[h]
        log(f"  │  {h:<7} │ {p['mae']:>10.2f} │ {p['rmse']:>10.2f} │ {p['mape']:>7.1f}% │")
log("  └──────────┴────────────┴────────────┴──────────┘")
if metriques_classif:
    log()
    log("  PARTIE B  —  CLASSIFICATION (SAFE/DANGER/CRITIQUE)")
    log(f"  Accuracy  : {metriques_classif['accuracy']:.1%}")
    log(f"  Precision : {metriques_classif['precision']:.1%}")
    log(f"  Recall    : {metriques_classif['recall']:.1%}")
    log(f"  F1-score  : {metriques_classif['f1']:.1%}")
    log("  ⚠  Recall CRITIQUE = métrique la plus importante")
log()
for g in ["plot_prediction_vs_reel.png","plot_importance_features.png","plot_residus.png","plot_mae_train_vs_test.png","plot_classification_risque.png"]:
    log(f"  →  {g}")
with open(os.path.join(OUT,"evaluation_report.txt"),"w",encoding="utf-8") as f:
    f.write("\n".join(_log))
ok("evaluation_report.txt")
log(); log("══════════════════════════════════════════════════════════════════")
log("  ÉVALUATION TERMINÉE  —  lancer ensuite : python predict.py")
log("══════════════════════════════════════════════════════════════════")