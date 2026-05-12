"""
╔══════════════════════════════════════════════════════════════════╗
║  05_predict.py                                                   ║
║                                                                  ║
║  RÔLE : Générer les prédictions de quota pour TOUS les groupes   ║
║         actifs (comportement par défaut) ou pour un groupe       ║
║         spécifique si demandé.                                   ║
║                                                                  ║
║  CE QUE CE SCRIPT FAIT POUR CHAQUE GROUPE                        ║
║  1. Récupère l'état actuel du quota (total, verrouillé, libre)   ║
║  2. Charge le vecteur de features de la dernière semaine connue  ║
║  3. Demande au modèle Random Forest : "combien va-t-il           ║
║     consommer dans les H prochains jours ?"                      ║
║  4. Calcule le budget restant prédit                             ║
║  5. Classe le niveau de risque : SAFE / DANGER / CRITIQUE        ║
║  6. Recommande le montant A à ajouter si nécessaire              ║
║  7. Estime le nombre de jours avant épuisement du quota          ║
║                                                                  ║
║  ENTRÉE  → output/clean_groupes.json                             ║
║            output/features.csv                                   ║
║            output/feature_cols.json                              ║
║            output/model_rf_7j.pkl                                ║
║            output/model_rf_14j.pkl                               ║
║            output/model_rf_30j.pkl                               ║
║                                                                  ║
║  SORTIE  → affichage console (tous les groupes ou un seul)       ║
║            output/predictions_tous_groupes.json  (si --save)     ║
║            output/prediction_groupe{id}.json     (si --save)     ║
║                                                                  ║
║  USAGE                                                           ║
║    Tous les groupes, horizon 30j (défaut) :                      ║
║      python 05_predict.py                                        ║
║                                                                  ║
║    Tous les groupes, horizon 7j :                                ║
║      python 05_predict.py --horizon 7                            ║
║                                                                  ║
║    Un seul groupe :                                              ║
║      python 05_predict.py --groupe_id 2 --horizon 14             ║
║                                                                  ║
║    Avec sauvegarde JSON :                                        ║
║      python 05_predict.py --horizon 30 --save                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, argparse, warnings
from datetime import date, timedelta
import joblib
import numpy  as np
import pandas as pd
from events_calendar import get_prochain_evenement, get_evenement_pour_date
# B5 : seuils centralisés dans config.py (source unique de vérité)
from config import (
    SEUIL_SAFE, SEUIL_CRITIQUE, MARGE_SECURITE,
    SEUIL_TENDANCE, ALPHA_PRUDENT,
)
warnings.filterwarnings("ignore")

# ── Chemins ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "output")

# ── Icônes d'affichage ────────────────────────────────────────────
ICONE_RISQUE   = {"SAFE": "✅  SAFE", "DANGER": "⚠️   DANGER", "CRITIQUE": "🔴  CRITIQUE"}
ICONE_TENDANCE = {"hausse": "↗  hausse", "stable": "→  stable", "baisse": "↘  baisse"}

# ── Cache mémoire (évite de recharger les fichiers à chaque appel) ─
_cache = {}

# ════════════════════════════════════════════════════════════════
# CHARGEMENT DES RESSOURCES
# ════════════════════════════════════════════════════════════════

def charger_ressources():
    """Charge les modèles, features et groupes en mémoire une seule fois."""
    if _cache:
        return _cache

    fichiers_requis = {
        "features.csv"         : "feature_engineering.py",
        "feature_cols.json"    : "feature_engineering.py",
        "clean_groupes.json"   : "load and explore.py",
    }
    for fichier, script in fichiers_requis.items():
        chemin = os.path.join(OUT, fichier)
        if not os.path.exists(chemin):
            raise FileNotFoundError(
                f"{fichier} introuvable.\n"
                f"  → Lancez d'abord : python '{script}'"
            )

    _cache["df_features"] = pd.read_csv(
        os.path.join(OUT, "features.csv"),
        parse_dates=["debut_semaine"]
    )
    _cache["feature_cols"] = json.load(open(os.path.join(OUT, "feature_cols.json")))
    _cache["df_groupes"]   = pd.read_json(
        os.path.join(OUT, "clean_groupes.json"), orient="records"
    )

    # Calendrier : optionnel (dégrade gracieusement si absent)
    chemin_cal = os.path.join(OUT, "events_calendar.json")
    if os.path.exists(chemin_cal):
        with open(chemin_cal, encoding="utf-8") as f:
            _cache["calendrier"] = json.load(f)
    else:
        _cache["calendrier"] = None

    # Chargement des 3 modèles (7j, 14j, 30j)
    modeles = {}
    for h in ["7j", "14j", "30j"]:
        chemin_model = os.path.join(OUT, f"model_rf_{h}.pkl")
        if os.path.exists(chemin_model):
            modeles[h] = joblib.load(chemin_model)

    if not modeles:
        raise FileNotFoundError(
            "Aucun modèle trouvé dans output/.\n"
            "  → Lancez d'abord : python train_models.py"
        )
    _cache["modeles"] = modeles
    return _cache

# ════════════════════════════════════════════════════════════════
# FONCTIONS DE CALCUL
# ════════════════════════════════════════════════════════════════

def obtenir_vecteur_features(df_features, feature_cols, groupe_id):
    """
    Récupère le dernier vecteur de features connu pour ce groupe.
    C'est le point de départ de la prédiction.
    """
    données_groupe = df_features[df_features["groupe_id"] == groupe_id]
    données_groupe = données_groupe.sort_values("debut_semaine")

    if données_groupe.empty:
        raise ValueError(
            f"Aucune donnée historique pour groupe_id={groupe_id}.\n"
            f"  Vérifiez que ce groupe existe dans les données."
        )
    dernière_ligne = données_groupe.iloc[-1]
    X = dernière_ligne[feature_cols].fillna(0).values.reshape(1, -1)
    return X, dernière_ligne

def classifier_risque(budget_restant, quota_total):
    """
    Classifie le niveau de risque selon le ratio budget restant / quota total.
    SAFE     : ratio > 30%
    DANGER   : 5% < ratio ≤ 30%
    CRITIQUE : ratio ≤ 5% ou budget déjà épuisé
    """
    if quota_total <= 0 or budget_restant <= 0:
        return "CRITIQUE"
    ratio = budget_restant / quota_total
    if ratio <= SEUIL_CRITIQUE:
        return "CRITIQUE"
    if ratio <= SEUIL_SAFE:
        return "DANGER"
    return "SAFE"

def calculer_montant_A(conso_prevue, quota_free, quota_total):
    """
    Retourne deux montants complémentaires pour cohérence avec le niveau
    de risque :

      A_minimum     : de quoi SURVIVRE l'horizon + marge 15% du quota.
                      Si 0  → le groupe tient sans recharge.

      A_recommande  : de quoi rester au-dessus du seuil SAFE (30%) après
                      consommation prévue. C'est le montant à ajouter
                      pour SORTIR de la zone DANGER/CRITIQUE.

    Les deux peuvent diverger quand le groupe survit (A_min = 0) mais
    termine le cycle avec un ratio < SEUIL_SAFE (DANGER).
    """
    marge_survie = quota_total * MARGE_SECURITE
    A_min        = max(0.0, conso_prevue - quota_free + marge_survie)

    # A_recommande : ramener le restant à SEUIL_SAFE * quota_total
    cible_safe   = quota_total * SEUIL_SAFE
    A_reco       = max(0.0, conso_prevue - quota_free + cible_safe)

    return round(A_min, 2), round(A_reco, 2), round(marge_survie, 2)

def calculer_tendance(dernière_ligne, conso_moyenne_4sem):
    """
    Évalue si la consommation récente est en hausse, stable ou baisse
    en comparant conso_semaine_1 à la moyenne 4 semaines.
    B3 : tendance_diff_1sem n'est plus exposé dans les 10 features → on
         reconstruit le delta depuis les features retenues.
    """
    conso_s1 = float(dernière_ligne.get("conso_semaine_1", 0))
    diff     = conso_s1 - conso_moyenne_4sem
    seuil    = conso_moyenne_4sem * SEUIL_TENDANCE if conso_moyenne_4sem > 0 else 1.0
    if diff > seuil:
        return "hausse", round(diff, 2)
    if diff < -seuil:
        return "baisse", round(diff, 2)
    return "stable", round(diff, 2)

def détecter_événements(date_début, horizon_jours, calendrier=None, groupe_id=None):
    """
    B2 : délègue à events_calendar (source unique de vérité, calendrier
         glissant). Pas de hardcoding d'années.

    BUG-FIX : on transmet groupe_id pour que les anniversaires spécifiques
    à chaque groupe soient bien pris en compte. Sans ça, tous les groupes
    voyaient exactement les mêmes événements (uniquement les communs :
    Ramadan, Aïd, Réveillon, Fête nationale).
    """
    if calendrier is None:
        return ["Aucun"]
    if isinstance(date_début, pd.Timestamp):
        d0 = date_début.date()
    elif isinstance(date_début, date):
        d0 = date_début
    else:
        d0 = pd.Timestamp(date_début).date()
    noms = []
    for delta in range(horizon_jours):
        evt = get_evenement_pour_date(
            calendrier, d0 + timedelta(days=delta), groupe_id=groupe_id
        )
        if evt["nom"] != "normal":
            noms.append(evt["nom"])
    noms = list(dict.fromkeys(noms))
    return noms if noms else ["Aucun"]

def estimer_jours_avant_zero(quota_libre, conso_moyenne_4sem, conso_prevue, horizon):
    """
    Donne deux estimations complémentaires du temps avant épuisement :
    - Estimation historique : basée sur le rythme moyen des 4 dernières semaines
    - Estimation prédite   : basée sur la consommation prévue par le modèle
    """
    conso_journalière = (conso_moyenne_4sem / 7.0) if conso_moyenne_4sem > 0 else 0.001
    jours_historique  = round(quota_libre / conso_journalière, 1)

    if conso_prevue > 0:
        jours_predit = round((quota_libre / conso_prevue) * horizon, 1)
    else:
        jours_predit = float("inf")

    return jours_historique, jours_predit

# ════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE DE PRÉDICTION
# ════════════════════════════════════════════════════════════════

def predire(groupe_id: int, horizon: int, verbose: bool = True) -> dict:
    """
    Prédit la consommation future et génère le résultat métier complet
    pour un groupe donné.

    Paramètres
    ──────────
    groupe_id  :  identifiant du groupe (colonne 'id' dans la table Groupe)
    horizon    :  7, 14 ou 30 (jours)
    verbose    :  True pour afficher le rapport formaté en console

    Retour
    ──────
    dict contenant tous les indicateurs métier
    """

    # ── Validation ────────────────────────────────────────────
    correspondance_horizon = {7: "7j", 14: "14j", 30: "30j"}
    if horizon not in correspondance_horizon:
        raise ValueError(
            f"Horizon invalide : {horizon}.\n"
            f"  Valeurs acceptées : 7, 14 ou 30 (jours)"
        )
    clé_horizon = correspondance_horizon[horizon]

    # ── Ressources ────────────────────────────────────────────
    res         = charger_ressources()
    df_features = res["df_features"]
    feature_cols = res["feature_cols"]
    df_groupes  = res["df_groupes"]
    modeles     = res["modeles"]

    if clé_horizon not in modeles:
        raise ValueError(
            f"Le modèle pour l'horizon {clé_horizon} est absent.\n"
            f"  → Relancez : python train_models.py"
        )

    # ── Données du groupe ─────────────────────────────────────
    lignes = df_groupes[df_groupes["id"] == groupe_id]
    if lignes.empty:
        raise ValueError(f"groupe_id={groupe_id} introuvable.")

    g            = lignes.iloc[0]
    nom_groupe   = str(g["name"])
    quota_total  = int(g["quota"])

    # ── Vecteur d'entrée du modèle ────────────────────────────
    X, dernière_ligne = obtenir_vecteur_features(df_features, feature_cols, groupe_id)

    # P4-bis : on prend le quota libre RÉEL calculé par feature_engineering
    # (cumsum + plancher à 0 sur l'historique budget_history) au lieu de la
    # valeur statique 2023 stockée dans clean_groupes.json. Ça aligne predict.py
    # avec predict_all_to_mysql.py et évite la divergence terminal vs HTML.
    quota_libre  = int(round(float(dernière_ligne.get("budget_quota_libre",
                                    quota_total - g.get("quotaLoked", 0)))))
    quota_verrou = max(0, quota_total - quota_libre)

    # ── Prédiction ────────────────────────────────────────────
    modele = modeles[clé_horizon]
    preds_par_arbre = np.array([t.predict(X)[0] for t in modele.estimators_])
    conso_prevue    = round(max(0.0, float(preds_par_arbre.mean())), 2)
    # Borne haute prudente : sert à la classification du risque et au calcul
    # du montant A recommandé. Plus elle est élevée, plus on tend vers CRITIQUE.
    conso_prudente  = round(max(0.0, float(preds_par_arbre.mean() + ALPHA_PRUDENT * preds_par_arbre.std())), 2)

    # ── Calculs métier ────────────────────────────────────────
    budget_restant  = round(quota_libre - conso_prevue, 2)
    niveau_risque   = classifier_risque(quota_libre - conso_prudente, quota_total)
    A_min, A_reco, marge = calculer_montant_A(conso_prudente, quota_libre, quota_total)

    conso_moy_4sem  = float(dernière_ligne.get("conso_moyenne_4sem", conso_prevue/horizon*7))
    jours_hist, jours_pred = estimer_jours_avant_zero(
        quota_libre, conso_moy_4sem, conso_prevue, horizon
    )

    quota_épuisé    = budget_restant <= 0
    tendance_label, tendance_val = calculer_tendance(dernière_ligne, conso_moy_4sem)
    événements      = détecter_événements(
        pd.Timestamp.now(), horizon, res.get("calendrier"), groupe_id=groupe_id
    )

    # ── Résultat structuré ────────────────────────────────────
    résultat = {
        # Identification
        "groupe_id"                 : groupe_id,
        "groupe_nom"                : nom_groupe,
        "horizon_jours"             : horizon,

        # État actuel du quota (avant prédiction)
        "quota_total"               : quota_total,
        "quota_verrouille"          : quota_verrou,
        "quota_libre"               : quota_libre,

        # Résultat de la prédiction
        "consommation_prevue_credits"   : conso_prevue,
        "consommation_prudente_credits" : conso_prudente,    # mean + 0.84·std
        "budget_restant_predit"         : budget_restant,

        # Sortie métier principale
        "niveau_risque"              : niveau_risque,
        "montant_A_minimum"          : A_min,    # survivre l'horizon + marge 15%
        "montant_A_recommande"       : A_reco,   # rester au-dessus du seuil SAFE (30%)
        "marge_securite_incluse"     : marge,

        # Indicateurs temporels
        "quota_atteint_zero"         : quota_épuisé,
        "jours_avant_zero_historique": jours_hist,
        "jours_avant_zero_predit"    : jours_pred,
        "semaines_avant_zero"        : round(jours_hist / 7, 1),

        # Contexte
        "tendance"                   : tendance_label,
        "tendance_variation"         : tendance_val,
        "evenements_dans_horizon"    : événements,
    }

    # ── Affichage rapport console ─────────────────────────────
    if verbose:
        print()
        print("╔" + "═"*56 + "╗")
        print(f"║  PRÉDICTION QUOTA  —  {nom_groupe.upper():<33}║")
        print("╠" + "═"*56 + "╣")
        print(f"║  Groupe ID                 : {groupe_id:<26}║")
        print(f"║  Horizon de prédiction     : {horizon} jours{'':<22}║")
        print("╠" + "═"*56 + "╣")
        print(f"║  ÉTAT ACTUEL DU QUOTA{'':<35}║")
        print(f"║  ──────────────────────────────────────────────────  ║")
        print(f"║  quota_total               : {quota_total:>10} crédits{'':<12}║")
        print(f"║  quota_verrouillé          : {quota_verrou:>10} crédits  (campagnes en cours)║")
        print(f"║  quota_libre               : {quota_libre:>10} crédits  (disponible)    ║")
        # B8 : on affiche EXPLICITEMENT les deux estimations pour que
        # l'utilisateur comprenne pourquoi un risque CRITIQUE peut apparaître
        # alors que le "budget restant prévu" semble positif.
        budget_restant_prudent = round(quota_libre - conso_prudente, 2)
        print("╠" + "═"*56 + "╣")
        print(f"║  PRÉDICTION  (Random Forest){'':<28}║")
        print(f"║  ──────────────────────────────────────────────────  ║")
        print(f"║  Consommation prévue (moy.) : {conso_prevue:>8.1f} crédits{'':<14}║")
        print(f"║  Borne haute (80% scénarios): {conso_prudente:>8.1f} crédits{'':<14}║")
        print(f"║  Budget restant (prévu)     : {budget_restant:>8.1f} crédits{'':<14}║")
        print(f"║  Budget restant (prudent)   : {budget_restant_prudent:>8.1f} crédits{'':<14}║")
        print("╠" + "═"*56 + "╣")
        print(f"║  RÉSULTAT MÉTIER{'':<40}║")
        print(f"║  ──────────────────────────────────────────────────  ║")
        print(f"║  Niveau de risque           :  {ICONE_RISQUE[niveau_risque]:<25}║")
        print(f"║  (basé sur la borne prudente, pas sur la moyenne){'':<5}║")
        print(f"║  A minimum (survivre {horizon}j)   : {A_min:>8.1f} crédits{'':<14}║")
        print(f"║  A recommandé (rester SAFE) : {A_reco:>8.1f} crédits{'':<14}║")
        #print(f"║  (marge sécurité incluse    : {marge:>8.1f} crédits){'':<12}║")#
        print("╠" + "═"*56 + "╣")
        print(f"║  INDICATEURS TEMPORELS{'':<34}║")
        print(f"║  ──────────────────────────────────────────────────  ║")
        if quota_épuisé:
            print(f"║  ⚠  Quota déjà épuisé — recharge urgente requise{'':<5}║")
        else:
            print(f"║  Jours avant quota 0 (prédit): ~{jours_pred:<5.0f} jours{'':<17}║")
            print(f"║  Semaines avant quota 0      : ~{jours_hist/7:<5.1f} semaines{'':<14}║")
        print("╠" + "═"*56 + "╣")
        print(f"║  CONTEXTE{'':<47}║")
        print(f"║  ──────────────────────────────────────────────────  ║")
        print(f"║  Tendance récente           :  {ICONE_TENDANCE[tendance_label]:<25}║")
        #print(f"║  Événements dans l'horizon  :  {', '.join(événements):<25}║")#
        print("╚" + "═"*56 + "╝")
        print()

    return résultat

# ════════════════════════════════════════════════════════════════
# TABLEAU DE SYNTHÈSE — tous les groupes
# ════════════════════════════════════════════════════════════════

def afficher_synthese(liste_résultats):
    """Affiche un tableau récapitulatif pour tous les groupes."""
    ICONE = {"SAFE":"✅","DANGER":"⚠️ ","CRITIQUE":"🔴"}

    print()
    print("╔" + "═"*88 + "╗")
    print(f"║  {'SYNTHÈSE — TOUS LES GROUPES':<86}║")
    print("╠" + "═"*22 + "╦" + "═"*12 + "╦" + "═"*15 + "╦" + "═"*15 + "╦" + "═"*15 + "╦" + "═"*7 + "╣")
    print(f"║  {'Groupe':<20}  ║ {'Risque':<10} ║ {'Conso prévue':>13} ║ {'A minimum':>13} ║ {'A recommandé':>13} ║ {'J→0':>5} ║")
    print("╠" + "═"*22 + "╬" + "═"*12 + "╬" + "═"*15 + "╬" + "═"*15 + "╬" + "═"*15 + "╬" + "═"*7 + "╣")
    for r in liste_résultats:
        ic  = ICONE.get(r["niveau_risque"], "")
        j0  = f"~{r['jours_avant_zero_historique']:.0f}"
        print(f"║  {r['groupe_nom']:<20}  ║ {ic} {r['niveau_risque']:<8} "
              f"║ {r['consommation_prevue_credits']:>11.1f} cr "
              f"║ {r['montant_A_minimum']:>11.1f} cr "
              f"║ {r['montant_A_recommande']:>11.1f} cr "
              f"║ {j0:>5} ║")
    print("╚" + "═"*22 + "╩" + "═"*12 + "╩" + "═"*15 + "╩" + "═"*15 + "╩" + "═"*15 + "╩" + "═"*7 + "╝")
    print()

# ════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Prédiction quota EasyBulk — Random Forest",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--groupe_id", type=int, default=None,
        help="Prédire pour un seul groupe (ex: --groupe_id 2)\n"
             "Si non précisé : prédit pour TOUS les groupes actifs."
    )
    parser.add_argument(
        "--horizon", type=int, default=30, choices=[7, 14, 30],
        help="Horizon de prédiction en jours (défaut : 30)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Sauvegarder les résultats en JSON dans output/"
    )
    args = parser.parse_args()

    res = charger_ressources()

    if args.groupe_id is not None:
        # ── Mode : un seul groupe ────────────────────────────
        résultat = predire(args.groupe_id, args.horizon, verbose=True)

        if args.save:
            nom_fichier = f"prediction_groupe{args.groupe_id}_{args.horizon}j.json"
            with open(os.path.join(OUT, nom_fichier), "w", encoding="utf-8") as f:
                json.dump(résultat, f, indent=2, ensure_ascii=False)
            print(f"  ✓  Résultat sauvegardé  →  {nom_fichier}")

    else:
        # ── Mode par défaut : TOUS les groupes actifs ────────
        df_groupes = res["df_groupes"]

        # Filtrer sur les groupes actifs uniquement
        if "est_actif" in df_groupes.columns:
            groupes_actifs = df_groupes[df_groupes["est_actif"] == 1]
        else:
            groupes_actifs = df_groupes

        ids_groupes = groupes_actifs["id"].tolist()

        print(f"\n  Prédiction pour {len(ids_groupes)} groupe(s) actif(s) "
              f"— horizon {args.horizon}j\n")

        tous_résultats = []
        for gid in ids_groupes:
            try:
                r = predire(gid, args.horizon, verbose=True)
                tous_résultats.append(r)
            except Exception as e:
                print(f"  ⚠  Groupe {gid} — erreur : {e}\n")

        # Tableau de synthèse final
        if tous_résultats:
            afficher_synthese(tous_résultats)

        if args.save and tous_résultats:
            nom_fichier = f"predictions_tous_groupes_{args.horizon}j.json"
            with open(os.path.join(OUT, nom_fichier), "w", encoding="utf-8") as f:
                json.dump(tous_résultats, f, indent=2, ensure_ascii=False)
            print(f"  ✓  Résultats sauvegardés  →  {nom_fichier}")