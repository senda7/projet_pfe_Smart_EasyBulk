"""
app.py — Point d'entrée de l'API Flask Smart SMS Predictor.

Lance avec :
    cd budget/api
    python app.py

Puis teste depuis le navigateur ou Postman :
    GET http://localhost:5000/health
    GET http://localhost:5000/groupes
"""
import os
import sys

# Ajoute la racine du projet au PYTHONPATH pour pouvoir importer
# predict.py (qui contient la fonction predire() utilisée par les routes).
# IMPORTANT : on utilise append (et pas insert(0)) parce qu'il existe
# déjà un config.py à la racine du projet (pour le pipeline ML), et
# nous on veut que `from config import ...` résolve d'abord vers
# api/config.py (notre fichier API). Avec append, api/ reste prioritaire.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from flask import Flask, jsonify
from flask_cors import CORS

from config import API_HOST, API_PORT, DEBUG
from db import test_connection
from routes import groupes as groupes_route


def create_app() -> Flask:
    app = Flask(__name__)

    # CORS large pour la démo (le frontend peut tourner sur file://, port 5500, etc.)
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Blueprints (= modules d'endpoints)
    app.register_blueprint(groupes_route.bp)

    # ── Endpoints racine ──────────────────────────────────────
    @app.route("/")
    def index():
        return jsonify({
            "name":    "Smart SMS Predictor — Budget API",
            "status":  "ok",
            "version": "0.1.0",
            "endpoints": [
                "GET  /health",
                "GET  /groupes",
                "GET  /groupes/<id>",
            ],
        })

    @app.route("/health")
    def health():
        db_ok = test_connection()
        return jsonify({
            "api": "ok",
            "db":  "ok" if db_ok else "down",
        }), (200 if db_ok else 503)

    # ── Gestion d'erreurs JSON propres ────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not Found", "detail": str(e.description)}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Server Error", "detail": str(e)}), 500

    return app


if __name__ == "__main__":
    print()
    print("══════════════════════════════════════════════════════════")
    print("  Smart SMS Predictor — Budget API")
    print("══════════════════════════════════════════════════════════")

    print("  ·  Vérification connexion MySQL …")
    if test_connection():
        print("  ✓  MySQL connecté")
    else:
        print("  ⚠  MySQL inaccessible — vérifie XAMPP (port 3306)")
        print("     L'API démarre quand même en mode dégradé.")

    print(f"  ·  API en écoute sur http://localhost:{API_PORT}")
    print(f"  ·  Test rapide : http://localhost:{API_PORT}/health")
    print("══════════════════════════════════════════════════════════")
    print()

    app = create_app()
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
