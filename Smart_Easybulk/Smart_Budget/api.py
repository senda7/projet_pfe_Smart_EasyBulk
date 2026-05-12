"""
api.py  — Budget ML  ↔  v3_enhanced.html
──────────────────────────────────────────────────────────────────
USAGE :
    pip install flask flask-cors pymysql
    python api.py

ENDPOINTS :
    GET  /groupes          → liste groupes + prédictions ML
    POST /groupes          → crée un groupe (MySQL + JSON local)
    GET  /groupes/<id>     → détail un groupe
    GET  /health           → statut API (topbar HTML)
"""

import os, sys, json, math
from datetime import date, timedelta
import numpy  as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from predict import charger_ressources, predire
from config  import ALPHA_PRUDENT

app  = Flask(__name__)
CORS(app)

OUT  = os.path.join(BASE, "output")
DATA = os.path.join(BASE, "data")

# ════════════════════════════════════════════════════════════════
# CONFIG MYSQL (adapter selon ton XAMPP)
# ════════════════════════════════════════════════════════════════
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 3306,
    "user"    : "root",        # utilisateur XAMPP par défaut
    "password": "",            # mot de passe XAMPP (vide par défaut)
    "database": "easybulk",    # nom de ta base de données
    "charset" : "utf8mb4",
}

def get_db():
    """Retourne une connexion MySQL (None si XAMPP inaccessible)."""
    try:
        import pymysql
        conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
        return conn
    except Exception as e:
        print(f"  ⚠  MySQL inaccessible : {e}")
        return None


# ════════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════════

def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 999
    return v

def _predire_safe(gid, horizon):
    try:
        return predire(gid, horizon, verbose=False)
    except Exception as e:
        print(f"  ⚠  predire({gid}, {horizon}j) → {e}")
        return {}

def _conso_prudente(gid, res):
    try:
        modele = res["modeles"].get("30j")
        cols   = res["feature_cols"]
        df     = res["df_features"]
        if modele is None: return 0.0
        sub = df[df["groupe_id"] == gid].sort_values("debut_semaine")
        if sub.empty: return 0.0
        X = sub.iloc[[-1]][cols].fillna(0).values
        par_arbre = np.array([t.predict(X)[0] for t in modele.estimators_])
        return round(max(0.0, float(par_arbre.mean() + ALPHA_PRUDENT * par_arbre.std())), 2)
    except Exception:
        return 0.0

def _historique(df_features, gid, nb=26):
    if "_conso_reelle" not in df_features.columns: return []
    sub = (df_features[df_features["groupe_id"] == gid]
           .sort_values("debut_semaine").tail(nb))
    return [round(float(v), 0) for v in sub["_conso_reelle"].fillna(0)]

def _evenements(p30):
    noms = p30.get("evenements_dans_horizon", [])
    if not noms or noms == ["Aucun"]: return []
    return [{"nom": n, "type": "calendrier",
             "date": str(date.today() + timedelta(days=10)), "mult": 1.5}
            for n in noms]

def _build_groupe(row, res):
    """
    Construit l'objet groupe complet renvoyé au HTML.

    PRINCIPE : api.py ne calcule rien. Toutes les valeurs (quota, conso,
    risque, A_min, A_reco, J→0…) viennent directement de predict.py.
    api.py n'est qu'une passerelle vers le HTML.
    """
    gid       = int(row["id"])
    est_actif = int(row.get("est_actif", 1)) == 1

    # 1. Appelle predict.py sur les 3 horizons
    p7  = _predire_safe(gid,  7)
    p14 = _predire_safe(gid, 14)
    p30 = _predire_safe(gid, 30)
    prud   = _conso_prudente(gid, res)
    risque = p30.get("niveau_risque", "SAFE")

    # 2. Quotas : on PREND ce que predict.py renvoie (valeurs réelles
    #    calculées via cumsum + plancher 0 sur budget_history).
    #    Sans ce mapping, api.py enverrait des valeurs statiques 2023
    #    incohérentes avec les J→0 calculés à partir du quota réel.
    if p30 and "quota_libre" in p30:
        quota_total = int(p30.get("quota_total",      row.get("quota", 0)))
        quota_libre = int(p30.get("quota_libre",      0))
        quota_verr  = int(p30.get("quota_verrouille", 0))
    else:
        # Fallback si predict.py a échoué (ex: features manquantes)
        quota_total = int(row.get("quota", 0))
        quota_verr  = int(row.get("quotaLoked", 0))
        quota_libre = quota_total - quota_verr

    return {
        "id"               : gid,
        "nom"              : str(row.get("name", f"Groupe {gid}")),
        "description"      : str(row.get("description", "")),
        "is_active"        : est_actif,
        "quota_total"      : quota_total,
        "quota_verrouille" : quota_verr,
        "quota_libre"      : quota_libre,
        "risque"           : risque,
        "predictions": {
            "7j" : {"conso_prevue": _clean(p7.get("consommation_prevue_credits", 0))},
            "14j": {"conso_prevue": _clean(p14.get("consommation_prevue_credits", 0))},
            "30j": {
                "conso_prevue"    : _clean(p30.get("consommation_prevue_credits", 0)),
                "conso_prudente"  : _clean(prud),
                "a_min"           : _clean(p30.get("montant_A_minimum",    0)),
                "a_reco"          : _clean(p30.get("montant_A_recommande",  0)),
                "jours_avant_zero": _clean(p30.get("jours_avant_zero_predit", 999)),
                "niveau_risque"   : risque,
                "tendance"        : p30.get("tendance", "stable"),
                "budget_restant"  : _clean(p30.get("budget_restant_predit", 0)),
            },
        },
        "historique": _historique(res["df_features"], gid),
        "evenements" : _evenements(p30),
        "campagnes"  : [],
    }


# ════════════════════════════════════════════════════════════════
# HELPERS JSON LOCAL (fallback si MySQL down)
# ════════════════════════════════════════════════════════════════

def _lire_groupes_json():
    """Lit data/groupes.json (fichier source du pipeline Python)."""
    path = os.path.join(DATA, "groupes.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _ecrire_groupes_json(data):
    """Écrit data/groupes.json et régénère clean_groupes.json."""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "groupes.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Régénérer clean_groupes.json en mémoire (version simplifiée)
    df = pd.DataFrame(data)
    if "status_id" in df.columns:
        df["est_actif"] = (df["status_id"] == 1).astype(int)
    else:
        df["est_actif"] = 1
    if "quotaFree" not in df.columns:
        df["quotaFree"] = df["quota"] - df.get("quotaLoked", 0)
    clean_path = os.path.join(OUT, "clean_groupes.json")
    os.makedirs(OUT, exist_ok=True)
    df.to_json(clean_path, orient="records", force_ascii=False, indent=2)
    print(f"  ✓  clean_groupes.json mis à jour ({len(df)} groupes)")


def _prochain_id_json():
    groupes = _lire_groupes_json()
    if not groupes: return 1
    return max(int(g.get("id", 0)) for g in groupes) + 1


# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """
    Page d'accueil — confirme que l'API tourne.
    Ouvrir v3_enhanced.html dans le navigateur, pas cette URL.
    """
    return """
    <html><head><title>Budget ML API</title>
    <style>
      body{font-family:monospace;background:#0b0f18;color:#c9d8f0;
           display:flex;align-items:center;justify-content:center;
           height:100vh;margin:0}
      .box{background:#111827;border:1px solid #1e2d45;border-radius:12px;
           padding:2rem 2.5rem;max-width:480px;text-align:center}
      h2{color:#3b82f6;margin-bottom:.5rem}
      .ok{color:#10b981;font-size:2rem;margin-bottom:1rem}
      a{color:#3b82f6;text-decoration:none}
      a:hover{text-decoration:underline}
      .ep{background:#0b0f18;border-radius:6px;padding:.4rem .8rem;
          display:block;margin:.4rem 0;color:#6ee7b7;font-size:.9rem}
    </style></head><body>
    <div class="box">
      <div class="ok">✓</div>
      <h2>Budget ML API</h2>
      <p style="color:#5a7399;margin-bottom:1.5rem">
        L'API tourne correctement.<br>
        Ouvre <strong>interface.html</strong> dans ton navigateur.
      </p>
      <div style="text-align:left">
        <span class="ep">GET  /groupes</span>
        <span class="ep">POST /groupes</span>
        <span class="ep">GET  /groupes/&lt;id&gt;</span>
        <span class="ep">GET  /health</span>
      </div>
    </div>
    </body></html>
    """, 200


@app.route("/groupes", methods=["GET"])
def get_groupes():
    """
    Liste complète des groupes avec prédictions ML.

    SOURCE DES DONNÉES :
      • Liste des groupes  →  MySQL (source de vérité)
      • Prédictions ML     →  predict.py (via _build_groupe)

    Si MySQL est éteint, fallback sur predict.py (clean_groupes.json).
    """
    try:
        # 1. Liste des groupes : MySQL d'abord, sinon fallback JSON
        rows = []
        conn = get_db()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, name, description,
                               quota, quotaLoked, quotaFree,
                               status_id, organization_id, admin_id
                        FROM groupe
                        WHERE status_id = 1   -- actifs uniquement
                        ORDER BY id
                    """)
                    rows = cur.fetchall()  # DictCursor → liste de dicts
                # Ajoute le champ 'est_actif' attendu par _build_groupe
                for r in rows:
                    r["est_actif"] = 1
            finally:
                conn.close()
            print(f"  ·  {len(rows)} groupes lus depuis MySQL")
        else:
            # Fallback JSON si XAMPP éteint
            print("  ⚠  MySQL inaccessible → fallback clean_groupes.json")
            res_fb = charger_ressources()
            df_fb  = res_fb["df_groupes"]
            if "est_actif" in df_fb.columns:
                df_fb = df_fb[df_fb["est_actif"] == 1]
            rows = df_fb.to_dict("records")

        # 2. Ressources ML (toujours via predict.py)
        res = charger_ressources()

        # 3. Pour chaque groupe : enrichi avec prédictions ML
        result = []
        for row in rows:
            g = _build_groupe(row, res)
            print(f"  ✓  {g['nom']:<22} {g['risque']}")
            result.append(g)
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/groupes", methods=["POST"])
def post_groupe():
    """
    Crée un nouveau groupe.
    1. Tente de l'enregistrer dans MySQL (XAMPP).
    2. Met à jour data/groupes.json + output/clean_groupes.json.
    3. Retourne le groupe créé avec id.

    Body JSON attendu :
      { nom, budget, description, administrateur,
        enteteAlpha: [...], typesCampagne: [...] }
    """
    body = request.get_json(force=True) or {}

    nom    = (body.get("nom") or "").strip()
    budget = int(body.get("budget") or 0)
    desc   = (body.get("description") or "").strip()
    admin  = (body.get("administrateur") or "").strip()

    if not nom or budget <= 0:
        return jsonify({"error": "nom et budget requis"}), 400

    new_id   = None
    mysql_ok = False

    # ── Tentative MySQL ────────────────────────────────────────
    conn = get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO groupe (name, quota, quotaLoked, quotaFree, status_id, description)
                    VALUES (%s, %s, 0, %s, 1, %s)
                """, (nom, budget, budget, desc))
                new_id = cur.lastrowid
            conn.commit()
            mysql_ok = True
            print(f"  ✓  MySQL : groupe '{nom}' inséré (id={new_id})")
        except Exception as e:
            print(f"  ⚠  MySQL INSERT échoué : {e}")
            conn.rollback()
        finally:
            conn.close()

    # ── Fallback JSON local ────────────────────────────────────
    if not mysql_ok:
        new_id = _prochain_id_json()
        print(f"  ·  MySQL indisponible → JSON local (id={new_id})")

    # ── Mise à jour data/groupes.json ─────────────────────────
    groupes = _lire_groupes_json()
    nouveau = {
        "id"         : new_id,
        "name"       : nom,
        "quota"      : budget,
        "quotaLoked" : 0,
        "quotaFree"  : budget,
        "status_id"  : 1,
        "description": desc,
    }
    # Éviter les doublons si déjà inséré via MySQL
    if not any(g.get("id") == new_id for g in groupes):
        groupes.append(nouveau)
        _ecrire_groupes_json(groupes)

    # ── Invalider le cache de predict.py ─────────────────────
    from predict import _cache
    _cache.clear()

    return jsonify({
        "status"   : "ok",
        "id"       : new_id,
        "mysql"    : mysql_ok,
        "groupe"   : {
            "id"               : new_id,
            "nom"              : nom,
            "description"      : desc,
            "is_active"        : True,
            "quota_total"      : budget,
            "quota_verrouille" : 0,
            "quota_libre"      : budget,
            "risque"           : "SAFE",
            "predictions"      : {
                "7j" : {"conso_prevue": 0},
                "14j": {"conso_prevue": 0},
                "30j": {"conso_prevue": 0, "conso_prudente": 0,
                        "a_min": 0, "a_reco": 0,
                        "jours_avant_zero": 999, "niveau_risque": "SAFE",
                        "tendance": "stable", "budget_restant": budget},
            },
            "historique": [],
            "evenements" : [],
            "campagnes"  : [],
        }
    }), 201


@app.route("/groupes/<int:gid>", methods=["GET"])
def get_groupe_detail(gid):
    """Détail d'un groupe — lit l'identité depuis MySQL, les prédictions via predict.py."""
    try:
        row = None
        # 1. Identité du groupe : MySQL d'abord
        conn = get_db()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, name, description,
                               quota, quotaLoked, quotaFree,
                               status_id, organization_id, admin_id
                        FROM groupe WHERE id = %s
                    """, (gid,))
                    row = cur.fetchone()
                if row:
                    row["est_actif"] = 1 if row.get("status_id", 1) == 1 else 0
            finally:
                conn.close()

        # 2. Fallback JSON si MySQL down ou groupe absent
        res = charger_ressources()
        if not row:
            df = res["df_groupes"]
            sub = df[df["id"] == gid]
            if sub.empty:
                return jsonify({"error": f"groupe_id={gid} introuvable"}), 404
            row = sub.iloc[0].to_dict()

        return jsonify(_build_groupe(row, res))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
@app.route("/api/health")
def health():
    try:
        res = charger_ressources()
        return jsonify({
            "status"              : "ok",
            "model_version"       : "RF-v1",
            "n_features_xgb"      : len(res.get("feature_cols", [])),
            "n_features_rsf"      : 0,
            "trained_at"          : "voir training_report.txt",
            "is_stale"            : False,
            "label_map"           : {"SAFE": 0, "DANGER": 1, "CRITIQUE": 2},
            "score_thresholds"    : {},
            "thresholds_source"   : "pkg",
            "fusion_params_source": "pkg",
            "rsf_scaler"          : "none",
            "avail_threshold"     : None,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   API Budget ML  ←→  interface.html                  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  GET  /groupes      → liste ML                       ║")
    print("║  POST /groupes      → créer groupe (MySQL + JSON)    ║")
    print("║  GET  /groupes/<id> → détail                         ║")
    print("║  GET  /health       → statut                         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── AUTO-INIT MYSQL ───────────────────────────────────
    # Crée les tables dans `easybulk` si elles n'existent pas,
    # et importe data/*.json si la table `groupe` est vide.
    print("──── Initialisation MySQL ───────────────────────")
    from init_db import init_db_and_seed
    init_db_and_seed(get_db)
    print("─────────────────────────────────────────────────")
    print()

    app.run(debug=True, port=5000)