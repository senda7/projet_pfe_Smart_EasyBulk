import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import date as Date

# Import propre du module calendrier (plus d'import dynamique fragile)
from events_calendar import get_evenement_pour_date, get_prochain_evenement

# ── Chemins ──────────────────────────────────────────────────────
# OUT surchargeable via BUDGET_OUT pour générer les features d'un jeu
# holdout sans écraser celles du training (cf. validate_holdout.py).
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.environ.get("BUDGET_OUT", os.path.join(BASE, "output"))

# ── Paramètres ───────────────────────────────────────────────────
TIMEZONE_LOCAL = "Africa/Tunis"   # M3 : utiliser l'heure locale pour l'hebdo
FENETRE_CAL_JOURS = 7             # Approche C : ±7 jours autour de la date cible
HORIZON_PROCHAIN_EVT = 30         # B.1 : cherche le prochain événement dans 30 jours

# ── Logger simple ────────────────────────────────────────────────
def log(msg=""): print(msg)
def ok(m):      log(f"  ✓  {m}")
def info(m):    log(f"  ·  {m}")
def warn(m):    log(f"  ⚠  {m}")
def title(t):
    log(); log("┌" + "─"*64 + "┐")
    log(f"│  {t:<62}│")
    log("└" + "─"*64 + "┘")


# ═══════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def charger_clean(nom):
    chemin = os.path.join(OUT, f"clean_{nom}.json")
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"{chemin} introuvable.\n"
            f"  → Lancez d'abord : python 01_load_and_explore.py"
        )
    return pd.read_json(chemin, orient="records")


def charger_calendrier():
    chemin = os.path.join(OUT, "events_calendar.json")
    if not os.path.exists(chemin):
        warn("events_calendar.json absent — lancez d'abord events_calendar.py")
        return {}
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def distance_calendaire(d_hist: pd.Timestamp, d_ref: pd.Timestamp) -> int:
    """
    Distance en jours entre deux dates en ignorant l'année.
    Gère le wrap-around : (31/12, 1/1) = 1 jour.
    Utilisée pour l'approche C (fenêtre calendaire).
    """
    doy_hist = d_hist.dayofyear
    doy_ref  = d_ref.dayofyear
    diff = abs(doy_hist - doy_ref)
    return min(diff, 365 - diff)


# ═══════════════════════════════════════════════════════════════
# C1 — RECONSTRUCTION DU QUOTA HISTORIQUE
# ═══════════════════════════════════════════════════════════════
# Au lieu d'utiliser le quota ACTUEL du groupe (fuite),
# on reconstruit le quota à chaque date via budget_history :
#   quota_total(t)   = Σ recharges (status=1)   jusqu'à t
#   quota_libre(t)   = Σ tous les amounts (status=1 et 2) jusqu'à t
#   quota_locked(t)  = quota_total(t) - quota_libre(t)
#
# Approximation pragmatique : ignore l'état intermédiaire "campagne créée
# mais non encore consommée". Suffisant pour l'entraînement cohérent.

def construire_historique_budget(df_budget: pd.DataFrame) -> dict:
    """
    Prépare pour chaque groupe une table triée avec cumsum des opérations.
    Retour : {groupe_id: DataFrame(modificationDate, total_cumul, libre_cumul)}
    """
    historique = {}
    for gid in df_budget["groupe_id"].unique():
        sub = (
            df_budget[df_budget["groupe_id"] == gid]
            .sort_values("modificationDate")
            .reset_index(drop=True)
            .copy()
        )
        # total_cumul = Σ recharges uniquement (les conso ne changent pas le total)
        recharges_seules = sub["amount"].where(sub["status_id"] == 1, 0)
        sub["total_cumul"] = recharges_seules.cumsum()
        # libre_cumul = balance reconstruite avec un plancher à 0 : dans le système
        # réel, le moteur SMS bloque les envois dès que le solde atteint 0, donc
        # la balance ne peut jamais devenir négative. Sans ce floor, les données
        # de test peuvent afficher un solde à −40 000 cr et tout classer CRITIQUE.
        solde = 0.0
        libres = []
        for amount in sub["amount"].values:
            solde = max(0.0, solde + float(amount))
            libres.append(solde)
        sub["libre_cumul"] = libres
        historique[gid] = sub[["modificationDate", "total_cumul", "libre_cumul"]]
    return historique


def etat_budget_a(historique: dict, gid: int, date_cible: pd.Timestamp):
    """
    Retourne (quota_total, quota_libre, quota_locked) estimés pour le groupe gid
    à la date_cible (dernière opération strictement avant).
    """
    sub = historique.get(gid)
    if sub is None or sub.empty:
        return 0.0, 0.0, 0.0

    masque = sub["modificationDate"] < date_cible
    if not masque.any():
        return 0.0, 0.0, 0.0

    idx = sub.index[masque][-1]
    total  = float(sub.at[idx, "total_cumul"])
    libre  = float(sub.at[idx, "libre_cumul"])
    locked = max(0.0, total - libre)
    return total, libre, locked


# ═══════════════════════════════════════════════════════════════
# APPROCHE C — HISTORIQUE SAISONNIER PAR FENÊTRE CALENDAIRE
# ═══════════════════════════════════════════════════════════════

def historique_calendaire(grp_serie: pd.DataFrame, date_cible: pd.Timestamp,
                          fenetre_jours: int = FENETRE_CAL_JOURS) -> float:
    """
    Moyenne de consommation des semaines des années précédentes dont la date
    de début est à ±fenetre_jours (jours calendaires) de date_cible.
    Robuste aux bordures d'année (contrairement au matching ISO).
    """
    annee_cible = date_cible.year
    # Ne regarder que le passé strict (pas de fuite)
    passe_strict = grp_serie[grp_serie["debut_semaine"] < date_cible]
    # Exclure l'année en cours pour vraiment comparer entre années
    passe_autres_annees = passe_strict[
        passe_strict["debut_semaine"].dt.year < annee_cible
    ]
    if passe_autres_annees.empty:
        return np.nan

    distances = passe_autres_annees["debut_semaine"].apply(
        lambda d: distance_calendaire(d, date_cible)
    )
    dans_fenetre = passe_autres_annees[distances <= fenetre_jours]
    if dans_fenetre.empty:
        return np.nan
    return float(dans_fenetre["consommation_semaine"].mean())


# ═══════════════════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════════════════
title("CHARGEMENT DES DONNÉES NETTOYÉES")

df_groupes = charger_clean("groupes")
df_camps   = charger_clean("campagnes")
df_budget  = charger_clean("budget_history")
calendrier = charger_calendrier()

df_budget["modificationDate"] = pd.to_datetime(df_budget["modificationDate"], utc=True, errors="coerce")
df_camps["dateDebut"]         = pd.to_datetime(df_camps["dateDebut"],         utc=True, errors="coerce")

ok(f"Groupes     : {len(df_groupes)}")
ok(f"Campagnes   : {len(df_camps)}  ({df_camps['groupe_id'].nunique()} groupes concernés)")
ok(f"Historique  : {len(df_budget)} opérations")
ok(f"Calendrier  : {sum(calendrier.get('meta', {}).values())} événements chargés")


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 1 — SÉRIE TEMPORELLE HEBDOMADAIRE
# ═══════════════════════════════════════════════════════════════
title("ÉTAPE 1 — SÉRIE HEBDOMADAIRE PAR GROUPE")

# On ne garde que les consommations réelles (status=2) pour la série
consommations = df_budget[df_budget["status_id"] == 2].copy()
consommations["montant_absolu"] = consommations["amount"].abs()

# M3 : convertir en heure locale AVANT le découpage hebdo
# Cela évite qu'une opération dimanche 23h locale soit classée en lundi UTC
consommations["date_locale"] = consommations["modificationDate"].dt.tz_convert(TIMEZONE_LOCAL)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    consommations["debut_semaine"] = (
        consommations["date_locale"]
        .dt.to_period("W")
        .apply(lambda p: p.start_time)
        .dt.tz_localize(TIMEZONE_LOCAL)
    )

# Somme par (groupe, semaine)
serie_hebdo = (
    consommations
    .groupby(["groupe_id", "debut_semaine"])
    .agg(consommation_semaine=("montant_absolu", "sum"))
    .reset_index()
)

# Remplir les semaines sans conso avec 0
serie_complete = []
groupes_sans_historique = []

for gid in df_groupes["id"].unique():
    sous_serie = serie_hebdo[serie_hebdo["groupe_id"] == gid]
    if sous_serie.empty:
        groupes_sans_historique.append(gid)
        continue
    toutes_semaines = pd.DataFrame({
        "debut_semaine": pd.date_range(
            sous_serie["debut_semaine"].min(),
            sous_serie["debut_semaine"].max(),
            freq="W-MON", tz=TIMEZONE_LOCAL
        )
    })
    toutes_semaines["groupe_id"] = gid
    complet = toutes_semaines.merge(sous_serie, on=["groupe_id", "debut_semaine"], how="left")
    complet["consommation_semaine"] = complet["consommation_semaine"].fillna(0)
    serie_complete.append(complet)

if groupes_sans_historique:
    warn(f"{len(groupes_sans_historique)} groupe(s) sans historique : {groupes_sans_historique}")
    warn("  → Ces groupes n'auront pas de prédiction. Ajouter du fallback si besoin.")

serie = (
    pd.concat(serie_complete, ignore_index=True)
    .sort_values(["groupe_id", "debut_semaine"])
    .reset_index(drop=True)
)

# C2 : utiliser l'année ISO (cohérente avec numero_semaine ISO)
iso = serie["debut_semaine"].dt.isocalendar()
serie["numero_semaine"] = iso.week.astype(int)
serie["annee"]          = iso.year.astype(int)   # ← ISO, plus grégorien
serie["mois"]           = serie["debut_semaine"].dt.month

ok(f"Série hebdomadaire : {len(serie)} lignes "
   f"({serie['groupe_id'].nunique()} groupes, "
   f"{serie['annee'].min()}–{serie['annee'].max()})")


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 2 — PRÉPARATION HISTORIQUE BUDGET (C1)
# ═══════════════════════════════════════════════════════════════
title("ÉTAPE 2 — RECONSTRUCTION DU QUOTA HISTORIQUE (C1)")

historique_budget = construire_historique_budget(df_budget)
ok(f"Historique budget construit pour {len(historique_budget)} groupe(s)")


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 3 — CONSTRUCTION DES FEATURES
# ═══════════════════════════════════════════════════════════════
title("ÉTAPE 3 — CALCUL DES FEATURES PAR (GROUPE × SEMAINE)")

lignes = []

for gid in serie["groupe_id"].unique():

    grp_serie = serie[serie["groupe_id"] == gid].reset_index(drop=True)
    n = len(grp_serie)

    for i in range(n):

        ligne   = grp_serie.iloc[i]
        d_courante = ligne["debut_semaine"]
        f = {}

        # ── Métadonnées ──────────────────────────────────────────
        f["groupe_id"]      = gid
        f["debut_semaine"]  = d_courante
        f["numero_semaine"] = int(ligne["numero_semaine"])
        f["annee"]          = int(ligne["annee"])
        f["mois"]           = int(ligne["mois"])
        f["_conso_reelle"]  = float(ligne["consommation_semaine"])

        # Helper : passé récent
        def passe(nb_sem):
            return grp_serie.iloc[max(0, i - nb_sem):i]["consommation_semaine"]

        # ════ [CONSO] — Consommation passée (2 features) ════════
        f["conso_semaine_1"]    = float(passe(1).sum())
        f["conso_moyenne_4sem"] = float(passe(4).mean()) if len(passe(4)) > 0 else 0.0

        # ════ [TENDANCE] — Évolution récente (1 feature) ════════
        # M2 : ratio_accel = 0 si pas de référence (au lieu de conso/1.0)
        if i >= 1 and f["conso_moyenne_4sem"] > 0:
            f["tendance_ratio_accel"] = float(
                grp_serie.iloc[i-1]["consommation_semaine"] / f["conso_moyenne_4sem"]
            )
        else:
            f["tendance_ratio_accel"] = 0.0

        # ════ [CAMPAGNE] — Activité récente (1 feature) ═════════
        fenetre_30j = d_courante - pd.Timedelta(days=30)
        camps_30j = df_camps[
            (df_camps["groupe_id"] == gid) &
            (df_camps["dateDebut"] >= fenetre_30j) &
            (df_camps["dateDebut"] <  d_courante)
        ]
        f["campagne_nb_30j"] = len(camps_30j)

        # ════ [BUDGET] — État du quota RECONSTRUIT (C1, 2 features) ══
        # quota_total est calculé localement pour obtenir le ratio ;
        # on ne l'expose PAS comme feature (redondant avec quota_libre
        # + ratio, et corrélé à la taille absolue du groupe).
        quota_total, quota_libre, _ = etat_budget_a(
            historique_budget, gid, d_courante
        )
        f["budget_quota_libre"] = float(quota_libre)
        f["budget_ratio_libre"] = (
            float(quota_libre / quota_total) if quota_total > 0 else 0.0
        )

        # ════ [SAISON] — Calendrier + historique saisonnier (2) ══
        d_date = d_courante.date() if hasattr(d_courante, "date") else Date.fromisoformat(str(d_courante)[:10])
        evt = get_evenement_pour_date(calendrier, d_date, gid)
        f["saison_multiplicateur"] = float(evt["multiplicateur"])

        # APPROCHE C : historique saisonnier par fenêtre calendaire ±7 jours
        hist_calend = historique_calendaire(grp_serie, d_courante, FENETRE_CAL_JOURS)
        if np.isnan(hist_calend):
            f["saison_conso_historique_semaine"] = f["conso_moyenne_4sem"]
        else:
            f["saison_conso_historique_semaine"] = hist_calend

        # ════ [PROCHAIN] — Prochain événement (1 feature) ═══════
        # BUG-FIX (B1) : on passe groupe_id pour que les anniversaires
        # spécifiques au groupe soient pris en compte. Sans ça, tous les
        # groupes voyaient les mêmes événements globaux.
        prochain = get_prochain_evenement(
            calendrier, d_date, HORIZON_PROCHAIN_EVT, groupe_id=gid
        )
        if prochain["nom"] != "Aucun":
            _prochain_mult = float(prochain["multiplicateur"])
            f["evenement_prochain_jours"] = float(prochain["dans_jours"])
        else:
            _prochain_mult = 1.0
            f["evenement_prochain_jours"] = 999.0

        # ════ [RYTHME] — B.2 : Rythme d'épuisement (1 feature) ═══
        # Signal DIRECT pour l'alerte : jours avant quota=0 au rythme
        # actuel, pondéré par l'intensité du prochain événement.
        conso_journaliere = f["conso_moyenne_4sem"] / 7 if f["conso_moyenne_4sem"] > 0 else 0.0
        conso_journaliere_evt = conso_journaliere * _prochain_mult
        if conso_journaliere_evt > 0 and quota_libre > 0:
            f["jours_avant_zero_rythme_event"] = min(999.0, float(quota_libre / conso_journaliere_evt))
        else:
            f["jours_avant_zero_rythme_event"] = 999.0

        # ════ TARGETS — à prédire ═════════════════════════════════
        f["target_conso_7j"]  = float(grp_serie.iloc[i+1]["consommation_semaine"])           if i+1 < n else np.nan
        f["target_conso_14j"] = float(grp_serie.iloc[i+1:i+3]["consommation_semaine"].sum()) if i+2 < n else np.nan
        # 30j = somme des 4 semaines suivantes (28j, approximation hebdo
        # du cycle mensuel de recharge — le plus proche en grain hebdo).
        f["target_conso_30j"] = float(grp_serie.iloc[i+1:i+5]["consommation_semaine"].sum()) if i+4 < n else np.nan

        lignes.append(f)

df_features = pd.DataFrame(lignes)


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 4 — NETTOYAGE FINAL
# ═══════════════════════════════════════════════════════════════
title("ÉTAPE 4 — NETTOYAGE FINAL")

n_avant = len(df_features)
df_features = df_features.dropna(
    subset=["target_conso_7j", "target_conso_14j", "target_conso_30j"]
)
info(f"{n_avant - len(df_features)} lignes retirées (fin de série, target inconnue)")

FEATURE_COLS = [
    c for c in df_features.columns
    if c.startswith(("conso_", "tendance_", "campagne_", "budget_",
                     "saison_", "evenement_", "jours_"))
]

df_features[FEATURE_COLS] = df_features[FEATURE_COLS].fillna(0).round(4)

ok(f"Features finales : {len(df_features)} lignes × {len(FEATURE_COLS)} features")


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 5 — SAUVEGARDE
# ═══════════════════════════════════════════════════════════════
title("ÉTAPE 5 — SAUVEGARDE")

df_features.to_csv(os.path.join(OUT, "features.csv"), index=False, encoding="utf-8")
with open(os.path.join(OUT, "feature_cols.json"), "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

ok("features.csv         →  table d'apprentissage complète")
ok("feature_cols.json    →  liste des features")

# Résumé par famille (jeu réduit — 10 features)
FAMILLES = {
    "[CONSO]     Consommation passée      ": [c for c in FEATURE_COLS if c.startswith("conso_")],
    "[TENDANCE]  Évolution récente        ": [c for c in FEATURE_COLS if c.startswith("tendance_")],
    "[CAMPAGNE]  Activité campagnes       ": [c for c in FEATURE_COLS if c.startswith("campagne_")],
    "[BUDGET]    État quota (reconstruit) ": [c for c in FEATURE_COLS if c.startswith("budget_")],
    "[SAISON]    Calendrier événements    ": [c for c in FEATURE_COLS if c.startswith("saison_")],
    "[PROCHAIN]  Prochain événement       ": [c for c in FEATURE_COLS if c.startswith("evenement_")],
    "[RYTHME]    B.2 — rythme épuisement  ": [c for c in FEATURE_COLS if c.startswith("jours_")],
}

log()
log("  Features construites par famille :")
log("  " + "─"*66)
for famille, cols in FAMILLES.items():
    log(f"  {famille} ({len(cols)} var.)")
    for c in cols:
        log(f"     · {c}")
log("  " + "─"*66)
log("  Targets à prédire :")
for t in ["target_conso_7j", "target_conso_14j", "target_conso_30j"]:
    log(f"     · {t}")
log()
log("══════════════════════════════════════════════════════════════════")
log("  FEATURE ENGINEERING TERMINÉ  —  lancer ensuite : python train_models.py")
log("══════════════════════════════════════════════════════════════════")
