"""
╔══════════════════════════════════════════════════════════════════╗
║  01_load_and_explore.py                                          ║
║                                                                  ║
║  RÔLE : Charger les données brutes, détecter et corriger         ║
║         toutes les anomalies avant tout traitement.              ║
║                                                                  ║
║  ENTRÉE  → data/groupes.json                                     ║
║            data/campagnes.json                                   ║
║            data/budget_history.json                              ║
║                                                                  ║
║  SORTIE  → output/clean_groupes.json       (groupes validés)     ║
║            output/clean_campagnes.json     (campagnes validées)  ║
║            output/clean_budget_history.json (historique propre)  ║
║            output/cleaning_report.txt      (rapport nettoyage)   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json, os
import pandas as pd
import numpy  as np
from datetime import datetime

# ── Chemins ──────────────────────────────────────────────────────
# Les chemins peuvent être surchargés par variables d'environnement pour
# permettre une validation sur base holdout (cf. validate_holdout.py).
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.environ.get("BUDGET_DATA", os.path.join(BASE, "data"))
OUT    = os.environ.get("BUDGET_OUT",  os.path.join(BASE, "output"))
os.makedirs(OUT, exist_ok=True)

# ── Valeurs de référence (status_id) ─────────────────────────────
STATUS_GROUPE_ACTIF      = 1   # groupe opérationnel
STATUS_BUDGET_RECHARGE   = 1   # ajout de budget par le Superadmin
STATUS_BUDGET_CONSOMME   = 2   # consommation par une campagne

# ── Utilitaires d'affichage ───────────────────────────────────────
_log = []

def log(msg=""):
    print(msg)
    _log.append(str(msg))

def title(t):
    log(); log("┌" + "─"*64 + "┐")
    log(f"│  {t:<62}│")
    log("└" + "─"*64 + "┘")

def ok(msg):    log(f"  ✓  {msg}")
def fixed(msg): log(f"  ✎  CORRIGÉ  │ {msg}")
def warn(msg):  log(f"  ⚠  ATTENTION │ {msg}")
def info(msg):  log(f"  ·  {msg}")

def save_clean(df, name):
    df.to_json(os.path.join(OUT, f"clean_{name}.json"),
               orient="records", date_format="iso", force_ascii=False, indent=2)
    df.to_csv(os.path.join(OUT, f"clean_{name}.csv"), index=False, encoding="utf-8")
    ok(f"clean_{name}.json  →  {len(df)} lignes")

def load_json(name):
    with open(os.path.join(DATA, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def load_from_mysql():
    """
    Lit les 4 tables depuis MySQL `easybulk`.
    Retourne (df_groupes, df_camps, df_budget, df_ctp) ou None si MySQL down.
    """
    try:
        import pymysql
        conn = pymysql.connect(
            host="localhost", port=3306, user="root", password="",
            database="easybulk", charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        warn(f"MySQL inaccessible : {e}")
        return None

    try:
        with conn.cursor() as cur:
            # groupe : renomme created_at/updated_at → camelCase pour rester compatible
            cur.execute("""
                SELECT id, name, description, quota, quotaLoked, quotaFree,
                       status_id, organization_id, admin_id,
                       created_at AS createdAt, updated_at AS updatedAt
                  FROM groupe
            """)
            df_g = pd.DataFrame(cur.fetchall())

            cur.execute("""
                SELECT id, groupe_id, modificationDate, amount, status_id
                  FROM budget_history
            """)
            df_b = pd.DataFrame(cur.fetchall())
            # convertit Date Python → string ISO comme dans JSON
            if not df_b.empty and "modificationDate" in df_b.columns:
                df_b["modificationDate"] = df_b["modificationDate"].astype(str)

            cur.execute("""
                SELECT id, libelle, description, dateDebut, dateFin, dureValidite,
                       cost, budgetUsed, nbrPage, countContact,
                       deactivatedByGroup, hash, meOnly,
                       lastUpdateStatusAt, createdAt, updatedAt,
                       status_id, campaign_type_permission_id
                  FROM campagne
            """)
            df_c = pd.DataFrame(cur.fetchall())

            cur.execute("""
                SELECT id, enabled, groupe_id, campaign_type
                  FROM campaign_type_permission
            """)
            df_ctp = pd.DataFrame(cur.fetchall())

        return df_g, df_c, df_b, df_ctp
    finally:
        try: conn.close()
        except Exception: pass


# ════════════════════════════════════════════════════════════════
# ÉTAPE 1  —  CHARGEMENT (MySQL d'abord, fallback JSON)
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 1  —  CHARGEMENT DES SOURCES")

_loaded = load_from_mysql()
if _loaded is not None:
    df_groupes, df_camps, df_budget, df_ctp = _loaded
    SOURCE = "MySQL easybulk"
    ok("Source : MySQL `easybulk` (table groupe / budget_history / campagne / campaign_type_permission)")
else:
    info("Fallback : chargement depuis data/*.json")
    df_groupes  = pd.DataFrame(load_json("groupes"))
    df_camps    = pd.DataFrame(load_json("campagnes"))
    df_budget   = pd.DataFrame(load_json("budget_history"))
    df_ctp      = pd.DataFrame(load_json("campaign_type_permission"))
    SOURCE = "data/*.json"

# JOIN campagne ↔ campaign_type_permission pour obtenir groupe_id sur les campagnes
df_ctp_join = df_ctp[["id", "groupe_id", "campaign_type"]].rename(
    columns={"id": "campaign_type_permission_id"}
)
df_camps = df_camps.merge(df_ctp_join, on="campaign_type_permission_id", how="left")

info(f"groupes                          :  {len(df_groupes):>4} lignes  (source : {SOURCE})")
info(f"campagnes                        :  {len(df_camps):>4} lignes")
info(f"budget_history                   :  {len(df_budget):>4} lignes")
info(f"campaign_type_permission         :  {len(df_ctp):>4} lignes (JOIN → groupe_id)")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 2  —  NETTOYAGE  :  GROUPES
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 2  —  NETTOYAGE  :  GROUPES")

n_avant = len(df_groupes)

# 2.1  Typage numérique des champs budget
for col in ["quota", "quotaLoked", "quotaFree"]:
    df_groupes[col] = pd.to_numeric(df_groupes[col], errors="coerce").fillna(0).astype(int)

# 2.2  Vérification de l'invariant fondamental
#      quota_free doit toujours égaler  quota_total - quota_locked
#      Si ce n'est pas le cas, quota_free est recalculé.
brisé = df_groupes[
    df_groupes["quotaFree"] != (df_groupes["quota"] - df_groupes["quotaLoked"])
]
if len(brisé):
    fixed(f"{len(brisé)} groupe(s) : invariant quota_free = quota_total - quota_locked brisé → recalculé")
    df_groupes["quotaFree"] = df_groupes["quota"] - df_groupes["quotaLoked"]
else:
    ok("Invariant  quota_free + quota_locked = quota_total  →  vérifié sur tous les groupes")

# 2.3  Valeurs négatives impossibles
for col in ["quota", "quotaLoked"]:
    négatifs = (df_groupes[col] < 0).sum()
    if négatifs:
        fixed(f"{négatifs} valeur(s) négatives dans '{col}' → remplacées par 0")
        df_groupes.loc[df_groupes[col] < 0, col] = 0

if (df_groupes["quotaFree"] < 0).any():
    warn(f"{(df_groupes['quotaFree'] < 0).sum()} groupe(s) avec quotaFree < 0  "
         f"(groupe en dépassement budgétaire)")

# 2.4  Doublons sur l'identifiant
doublons = df_groupes.duplicated(subset=["id"], keep="last").sum()
if doublons:
    fixed(f"{doublons} doublon(s) sur groupe.id → supprimés (conservé : dernière occurrence)")
    df_groupes = df_groupes.drop_duplicates(subset=["id"], keep="last")

# 2.5  Flag statut actif/inactif pour les filtres suivants
df_groupes["est_actif"] = (df_groupes["status_id"] == STATUS_GROUPE_ACTIF).astype(int)
inactifs = (df_groupes["est_actif"] == 0).sum()
if inactifs:
    info(f"{inactifs} groupe(s) inactif(s) — conservés avec flag est_actif=0")

GROUPES_VALIDES = set(df_groupes["id"])
ok(f"Groupes valides après nettoyage : {sorted(GROUPES_VALIDES)}")
info(f"Lignes avant : {n_avant}  →  après : {len(df_groupes)}")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 3  —  NETTOYAGE  :  HISTORIQUE BUDGET
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 3  —  NETTOYAGE  :  HISTORIQUE BUDGET")

n_avant = len(df_budget)

# 3.1  Dates
df_budget["modificationDate"] = pd.to_datetime(
    df_budget["modificationDate"], errors="coerce"
)
dates_invalides = df_budget["modificationDate"].isna().sum()
if dates_invalides:
    fixed(f"{dates_invalides} date(s) illisible(s) → lignes supprimées")
    df_budget = df_budget.dropna(subset=["modificationDate"])

aujourd_hui = pd.Timestamp(datetime.today().date())
dates_futures = (df_budget["modificationDate"] > aujourd_hui).sum()
if dates_futures:
    fixed(f"{dates_futures} ligne(s) avec date future → supprimées")
    df_budget = df_budget[df_budget["modificationDate"] <= aujourd_hui]

# 3.2  Montants nuls — inutilisables
montants_nuls = (df_budget["amount"] == 0).sum()
if montants_nuls:
    fixed(f"{montants_nuls} ligne(s) avec montant = 0 → supprimées (aucune opération réelle)")
    df_budget = df_budget[df_budget["amount"] != 0]

# 3.3  Cohérence signe / type d'opération
#      RECHARGE     (status=1) → montant doit être positif (ajout de crédit)
#      CONSOMMATION (status=2) → montant doit être négatif (retrait de crédit)
recharges_neg   = df_budget[(df_budget["status_id"] == STATUS_BUDGET_RECHARGE)   & (df_budget["amount"] < 0)]
consomm_pos     = df_budget[(df_budget["status_id"] == STATUS_BUDGET_CONSOMME)   & (df_budget["amount"] > 0)]

if len(recharges_neg):
    fixed(f"{len(recharges_neg)} recharge(s) avec montant négatif → signe inversé")
    df_budget.loc[recharges_neg.index, "amount"] *= -1

if len(consomm_pos):
    fixed(f"{len(consomm_pos)} consommation(s) avec montant positif → signe inversé")
    df_budget.loc[consomm_pos.index, "amount"] *= -1

# 3.4  Lignes dont le groupe_id n'existe plus (orphelines)
orphelines = (~df_budget["groupe_id"].isin(GROUPES_VALIDES)).sum()
if orphelines:
    fixed(f"{orphelines} ligne(s) orpheline(s) (groupe_id inexistant) → supprimées")
    df_budget = df_budget[df_budget["groupe_id"].isin(GROUPES_VALIDES)]

# 3.5  Doublons exacts (même groupe + même date + même type d'opération)
doublons = df_budget.duplicated(
    subset=["groupe_id","modificationDate","status_id"], keep="last"
).sum()
if doublons:
    fixed(f"{doublons} doublon(s) strict(s) → supprimés")
    df_budget = df_budget.drop_duplicates(
        subset=["groupe_id","modificationDate","status_id"], keep="last"
    )

# 3.6  Détection outliers sur les consommations (seuil conservateur : 3×IQR)
#      Un outlier est conservé mais plafonné pour ne pas biaiser les moyennes.
conso = df_budget[df_budget["status_id"] == STATUS_BUDGET_CONSOMME].copy()
if len(conso) > 4:
    valeurs_abs = conso["amount"].abs()
    Q1, Q3      = valeurs_abs.quantile(0.25), valeurs_abs.quantile(0.75)
    plafond     = Q3 + 3*(Q3 - Q1)
    outliers    = conso[valeurs_abs > plafond]
    if len(outliers):
        warn(f"{len(outliers)} consommation(s) hors norme (> {plafond:.0f} cr) → "
             f"plafonnées (conservées mais limitées pour éviter de biaiser le modèle)")
        df_budget.loc[outliers.index, "amount"] = -plafond

# 3.7  Colonnes temporelles dérivées (utiles pour le feature engineering)
df_budget["numero_semaine"] = df_budget["modificationDate"].dt.isocalendar().week.astype(int)
df_budget["annee"]          = df_budget["modificationDate"].dt.year

ok(f"Lignes avant : {n_avant}  →  après : {len(df_budget)}")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 4  —  NETTOYAGE  :  CAMPAGNES
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 4  —  NETTOYAGE  :  CAMPAGNES")

n_avant = len(df_camps)

# 4.1  Dates
df_camps["dateDebut"] = pd.to_datetime(df_camps["dateDebut"], utc=True, errors="coerce")
df_camps["dateFin"]   = pd.to_datetime(df_camps["dateFin"],   utc=True, errors="coerce")

dates_invalides = df_camps["dateDebut"].isna().sum()
if dates_invalides:
    fixed(f"{dates_invalides} campagne(s) sans date de début valide → supprimées")
    df_camps = df_camps.dropna(subset=["dateDebut"])

# Les campagnes futures ne sont pas encore réalisées : on les exclut du training
aujourd_hui_tz = pd.Timestamp(datetime.today().date(), tz="UTC")
futures = (df_camps["dateDebut"] > aujourd_hui_tz).sum()
if futures:
    fixed(f"{futures} campagne(s) future(s) → exclues (non encore réalisées)")
    df_camps = df_camps[df_camps["dateDebut"] <= aujourd_hui_tz]

# dateFin < dateDebut — incohérence logique
if df_camps["dateFin"].notna().any():
    incoherent = df_camps[
        df_camps["dateFin"].notna() & (df_camps["dateFin"] < df_camps["dateDebut"])
    ]
    if len(incoherent):
        fixed(f"{len(incoherent)} campagne(s) dateFin < dateDebut → dateFin remise à vide")
        df_camps.loc[incoherent.index, "dateFin"] = pd.NaT

# 4.2  Champs numériques
for col in ["cost", "budgetUsed", "countContact", "nbrPage"]:
    df_camps[col] = pd.to_numeric(df_camps[col], errors="coerce").fillna(0).astype(int)

# 4.3  Campagnes sans contenu utile
campagnes_vides = ((df_camps["countContact"] == 0) | (df_camps["cost"] == 0)).sum()
if campagnes_vides:
    fixed(f"{campagnes_vides} campagne(s) vide(s) (0 contact ou 0 coût) → supprimées")
    df_camps = df_camps[(df_camps["countContact"] > 0) & (df_camps["cost"] > 0)]

# 4.4  Budget consommé > coût estimé — impossible
depassement = (df_camps["budgetUsed"] > df_camps["cost"]).sum()
if depassement:
    fixed(f"{depassement} campagne(s) : budgetUsed > cost → plafonné au coût estimé")
    masque = df_camps["budgetUsed"] > df_camps["cost"]
    df_camps.loc[masque, "budgetUsed"] = df_camps.loc[masque, "cost"]

# 4.5  Campagnes dont le groupe n'existe plus
orphelines = (~df_camps["groupe_id"].isin(GROUPES_VALIDES)).sum()
if orphelines:
    fixed(f"{orphelines} campagne(s) orpheline(s) (groupe supprimé) → supprimées")
    df_camps = df_camps[df_camps["groupe_id"].isin(GROUPES_VALIDES)]

# 4.6  Colonnes temporelles dérivées
df_camps["numero_semaine"] = df_camps["dateDebut"].dt.isocalendar().week.astype(int)
df_camps["annee"]          = df_camps["dateDebut"].dt.year
df_camps["mois"]           = df_camps["dateDebut"].dt.month

if "event" not in df_camps.columns:
    df_camps["event"] = "normal"
df_camps["event"] = df_camps["event"].fillna("normal")

ok(f"Lignes avant : {n_avant}  →  après : {len(df_camps)}")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 5  —  VÉRIFICATION CROISÉE
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 5  —  VÉRIFICATION CROISÉE (cohérence entre les 3 tables)")

log()
log(f"  {'ID':<5} {'Nom groupe':<22} {'Statut':<10} "
    f"{'Hist. budget':>13} {'Campagnes':>10}")
log("  " + "─"*62)
for gid in sorted(GROUPES_VALIDES):
    g       = df_groupes[df_groupes["id"] == gid].iloc[0]
    nb_bh   = (df_budget["groupe_id"] == gid).sum()
    nb_cp   = (df_camps["groupe_id"]  == gid).sum()
    statut  = "actif" if g["est_actif"] else "inactif"
    log(f"  {gid:<5} {g['name']:<22} {statut:<10} {nb_bh:>13} {nb_cp:>10}")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 6  —  SAUVEGARDE
# ════════════════════════════════════════════════════════════════
title("ÉTAPE 6  —  SAUVEGARDE DES FICHIERS NETTOYÉS")

save_clean(df_groupes, "groupes")
save_clean(df_camps,   "campagnes")
save_clean(df_budget,  "budget_history")

with open(os.path.join(OUT, "cleaning_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_log))
ok("cleaning_report.txt  →  rapport de nettoyage complet")

log()
log("══════════════════════════════════════════════════════════════════")
log("  NETTOYAGE TERMINÉ  —  lancer ensuite : python feature_engineering.py")
log("══════════════════════════════════════════════════════════════════")