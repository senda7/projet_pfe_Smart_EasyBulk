"""
config.py
──────────────────────────────────────────────────────────────────
Constantes partagées par tout le pipeline ML.

Source unique de vérité : modifier ICI propage automatiquement à
predict.py, evaluate.py, validate_holdout.py.

Avant la centralisation, ces valeurs étaient dupliquées dans 3 fichiers
avec un risque de dérive (un seuil modifié dans evaluate.py mais pas
dans predict.py → métriques d'évaluation incohérentes avec la
classification réelle en production).
"""

# ── Seuils de classification du risque ───────────────────────────
# Ratio = budget_restant / quota_total
SEUIL_SAFE     = 0.30   # ratio > 30%        → SAFE
SEUIL_CRITIQUE = 0.05   # ratio ≤ 5%         → CRITIQUE
                        # entre les deux     → DANGER

# ── Marge de sécurité dans le calcul du montant A ────────────────
# A_min = max(0, conso_prevue − quota_libre + quota_total × MARGE_SECURITE)
MARGE_SECURITE = 0.15   # 15 % du quota total comme tampon de survie

# ── Borne haute de prédiction (prudent prediction) ───────────────
# conso_prudente = mean(arbres) + ALPHA_PRUDENT × std(arbres)
# Utilisée UNIQUEMENT pour la classification du risque (pas pour
# l'estimation de la consommation affichée à l'utilisateur).
# 0.84 ≈ quantile 80 % d'une loi normale — choix empirique pour
# maximiser le recall de la classe CRITIQUE.
ALPHA_PRUDENT = 0.84

# ── Seuil de variation pour la tendance (hausse / stable / baisse) ─
SEUIL_TENDANCE = 0.20   # variation > 20 % de la moyenne 4 semaines

# ── Classes de sortie de la classification du risque ─────────────
CLASSES_RISQUE = ["SAFE", "DANGER", "CRITIQUE"]

# ── Split temporel (cohérent train_models.py / evaluate.py) ──────
N_SEMAINES_TEST_CIBLE = 12
RATIO_TEST_MAX        = 0.20
N_TRAIN_MIN           = 20
