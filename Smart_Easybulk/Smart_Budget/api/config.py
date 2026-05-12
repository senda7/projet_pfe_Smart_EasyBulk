"""
config.py — Paramètres de l'API.
Modifiable via variables d'environnement (.env) si besoin.

XAMPP par défaut :
  - host       : localhost
  - port MySQL : 3306
  - user       : root
  - password   : '' (vide par défaut sur XAMPP local)
  - database   : budget
"""
import os

# ── BASE DE DONNÉES (XAMPP local par défaut) ──────────────────────
DB_CONFIG = {
    "host":     os.environ.get("BUDGET_DB_HOST", "localhost"),
    "port":     int(os.environ.get("BUDGET_DB_PORT", 3306)),
    "user":     os.environ.get("BUDGET_DB_USER", "root"),
    "password": os.environ.get("BUDGET_DB_PASSWORD", ""),
    "database": os.environ.get("BUDGET_DB_NAME", "budget"),
    "charset":  "utf8mb4",
    "use_unicode": True,
    "autocommit": False,
}

# ── API ───────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"     # accessible aussi en réseau local
API_PORT = int(os.environ.get("BUDGET_API_PORT", 5000))
DEBUG    = True          # passe à False en production

# ── SEUILS MÉTIER (cohérents avec config.py du pipeline ML) ───────
SEUIL_SAFE     = 0.30
SEUIL_CRITIQUE = 0.05
MARGE_SECURITE = 0.15
ALPHA_PRUDENT  = 0.84
