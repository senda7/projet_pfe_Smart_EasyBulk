# Smart SMS Predictor — API

Backend Flask qui expose les données MySQL au frontend HTML (`v3_enhanced.html`).

## Installation (1ère fois)

```bash
# Depuis le dossier racine du projet :
cd C:\Users\nour2\Downloads\PythonProject\budget

# Active le venv Python existant :
.venv\Scripts\activate

# Installe les dépendances API :
pip install -r api/requirements.txt
```

## Préparer la base de données

1. Lance **XAMPP Control Panel**
2. Démarre **Apache** et **MySQL**
3. Ouvre phpMyAdmin : http://localhost/phpmyadmin
4. Onglet "Importer" → choisis le fichier `sql/schema_v2.sql`
5. Clique "Exécuter"
6. Vérifie que la base `budget` apparaît avec **9 tables** (organization, groupe, budget_history, campagne, prm_status, prm_campaign_type, campaign_type_permission, **predictions**, **notifications**)

## Lancer l'API

```bash
cd api
python app.py
```

Sortie attendue :
```
══════════════════════════════════════════════════════════
  Smart SMS Predictor — Budget API
══════════════════════════════════════════════════════════
  ·  Vérification connexion MySQL …
  ✓  MySQL connecté
  ·  API en écoute sur http://localhost:5000
  ·  Test rapide : http://localhost:5000/health
══════════════════════════════════════════════════════════
```

## Tester

Ouvre dans le navigateur :

| URL | Attendu |
|---|---|
| http://localhost:5000/ | JSON descriptif de l'API |
| http://localhost:5000/health | `{ "api": "ok", "db": "ok" }` |
| http://localhost:5000/groupes | Liste des 3 groupes (Commercial, Marketing, RH) |
| http://localhost:5000/groupes/1 | Détail du groupe Commercial |

## Endpoints disponibles

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Statut API + DB |
| GET | `/groupes` | Liste des groupes |
| GET | `/groupes/<id>` | Détail d'un groupe |

À venir :
- POST `/groupes` (créer)
- PUT `/groupes/<id>` (modifier)
- DELETE `/groupes/<id>`
- POST `/groupes/<id>/recharge`
- GET `/dashboard`
- GET `/notifications`
- POST `/predictions/run`

## Configuration

Si tu changes le mot de passe MySQL ou le port, édite `api/config.py` ou définis ces variables d'environnement :

```bash
set BUDGET_DB_HOST=localhost
set BUDGET_DB_PORT=3306
set BUDGET_DB_USER=root
set BUDGET_DB_PASSWORD=
set BUDGET_DB_NAME=budget
set BUDGET_API_PORT=5000
```
