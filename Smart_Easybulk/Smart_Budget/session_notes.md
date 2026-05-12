# Session Notes — Chapitre 4 : Smart Budget Consumer Predictor

## Structure finale de la section Modélisation (CRISP-DM)

```
V. Modélisation
├── 1. Choix du modèle
├── 2. Conception du plan de test
├── 3. Entraînement et prédiction
└── 4. Classification du niveau de risque
```

---

## 1. Choix du modèle

### Paragraphe validé
> Pour répondre à la problématique de prédiction de consommation de crédits SMS, nous avons retenu l'algorithme Random Forest Regressor. Ce choix repose sur trois considérations principales. Premièrement, cet algorithme est particulièrement adapté aux données tabulaires hétérogènes combinant des ratios budgétaires, des comptages de campagnes et des multiplicateurs saisonniers. Deuxièmement, sa nature ensembliste lui confère une bonne résistance au surapprentissage sans nécessiter de normalisation préalable des données. Troisièmement, il permet d'accéder individuellement aux prédictions de chaque arbre, ce qui permet de calculer une borne haute prudente utilisée pour la classification du niveau de risque.

### Points clés
- Pas d'encodage dans le projet (toutes les features sont déjà numériques)
- Pas de modèle de classification séparé — c'est un **RF Regressor** uniquement
- La classification SAFE/DANGER/CRITIQUE est dérivée par règles mathématiques, pas par apprentissage

---

## 2. Conception du plan de test

### Paragraphe validé
> La conception du plan de test repose sur un découpage temporel strict des données, une exigence fondamentale pour tout modèle de prédiction sur séries temporelles (Hyndman & Athanasopoulos, 2018). Contrairement à un split aléatoire classique, qui risquerait d'entraîner une fuite d'information, le split temporel garantit que toutes les semaines du jeu d'entraînement précèdent celles du jeu de test. Les semaines antérieures à la date de coupure constituent le jeu d'entraînement, et les 12 semaines suivantes forment le jeu de test (plafonné à 20 % de l'historique total disponible).

### Référence
- Hyndman, R.J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.

### Schéma
- Fichier : `output/schema_split_temporel.png`
- Timeline bleu (Train : Jan 2024 → Sep 2025) / rouge (Test : Oct → Déc 2025, 12 semaines)

---

## 3. Entraînement et prédiction

### Entraînement — paragraphe
> Trois modèles Random Forest Regressor indépendants sont entraînés, chacun spécialisé sur un horizon temporel distinct : 7, 14 et 30 jours. Ils partagent les mêmes 10 variables d'entrée et les mêmes hyperparamètres (n_estimators = 200, max_depth = 7, min_samples_leaf = 5), mais diffèrent par leur variable cible. Une fois entraînés, les trois modèles sont sérialisés dans des fichiers .pkl distincts.

### Prédiction — paragraphe validé
> Pour un groupe donné, le modèle interroge individuellement chacun des 200 arbres et collecte leurs prédictions séparées. La consommation estimée correspond à la moyenne de ces prédictions, forcée à zéro si négative. En parallèle, une borne haute prudente est calculée : consommation_prudente = moyenne + 0,84 × écart-type. Le coefficient 0,84 correspond au quantile 80 % d'une loi normale, ce qui signifie que la borne prudente est supérieure à la prédiction centrale dans quatre cas sur cinq.

### Points techniques importants
- La moyenne des arbres est calculée dans **predict.py**, pas dans train_models.py
- `preds_par_arbre = np.array([t.predict(X)[0] for t in modele.estimators_])`
- `conso_prevue = max(0.0, preds_par_arbre.mean())`
- `conso_prudente = max(0.0, preds_par_arbre.mean() + 0.84 * preds_par_arbre.std())`

### Règles métier (dans predict.py — pas du ML)
- `A_minimum = max(0, conso_prudente − quota_libre + 15% × quota_total)` → survivre l'horizon
- `A_recommandé = max(0, conso_prudente − quota_libre + 30% × quota_total)` → rester SAFE
- `Budget restant prédit = quota_libre − conso_prévue`
- `Jours avant épuisement` = calculé depuis historique + prédit

### Schémas
- `output/schema_prediction_corrige.png` — diagramme final de la prédiction
  - **Modèle entraîné** : RF Regressor → Consommation prévue + Borne haute prudente
  - **Règles métier** : Budget restant, Jours avant épuisement, Montant A recommandé
- `output/schema_regression_v2.png` — architecture interne du RF (10 features → 200 arbres → agrégation → 3 sorties)

---

## 4. Classification du niveau de risque

### Principe
Ce n'est **pas un modèle de classification** — ce sont des règles mathématiques appliquées sur la borne prudente.

```
ratio = (quota_libre − conso_prudente) / quota_total

ratio > 30%          → SAFE
5% < ratio ≤ 30%     → DANGER
ratio ≤ 5%           → CRITIQUE
```

### Avantage de cette approche
Les seuils restent modifiables librement dans `config.py` sans réentraîner le modèle.

### Schémas draw.io disponibles
- `schema_regression.drawio` — architecture du modèle de régression
- `schema_classification.drawio` — architecture de la classification

---

## Fichiers modifiés

| Fichier | Action |
|---------|--------|
| `rapport-pfe-v2.docx` | Section Modélisation restructurée |
| `output/schema_split_temporel.png` | Nouveau schéma |
| `output/schema_regression_v2.png` | Nouveau schéma |
| `output/schema_prediction_corrige.png` | Nouveau schéma |
| `schema_regression.drawio` | Fichier draw.io |
| `schema_classification.drawio` | Fichier draw.io |

---

## Structure complète du Chapitre 4 (état actuel)

```
Introduction
Compréhension métier
Compréhension des données
Préparation des données
  ├── Nettoyage des données
  │     ├── Mise en cohérence des données de groupes
  │     ├── Assainissement de l'historique budgétaire
  │     ├── Traitement et filtrage des campagnes
  │     └── Contrôle de la cohérence entre fichiers
  ├── Construction de la série temporelle hebdomadaire
  ├── Reconstruction du quota historique
  └── Feature Engineering
Modélisation
  ├── Choix du modèle ✓
  ├── Conception du plan de test ✓
  ├── Entraînement et prédiction ✓
  └── Classification du niveau de risque (à développer)
Evaluation
  ├── Évaluation de la précision des prédictions
  ├── Évaluation de la classification du niveau de risque
  └── Analyse des résultats et interprétation
Déploiement (à écrire)
Conclusion (à écrire)
```

---

## À faire

- [ ] Paragraphe : Classification du niveau de risque (règles + formule ratio)
- [ ] Insérer les nouveaux paragraphes et schémas dans le docx
- [ ] Exporter schema_classification.drawio → PNG et insérer dans le docx
- [ ] Écrire section Déploiement
- [ ] Écrire Conclusion du chapitre
