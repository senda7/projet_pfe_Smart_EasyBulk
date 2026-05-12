"""
api.py  — Budget ML  ↔  test_easybulk.html   VERSION CORRIGÉE
──────────────────────────────────────────────────────────────────
CORRECTION :
  • Au démarrage, sync clean_groupes.json depuis MySQL
    → predict.py voit les vrais groupes (Prisons Nord/Centre/Sud...)
    → plus de groupes statiques Marketing/RH/Commercial

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

from predict import charger_ressources, predire, _cache as predict_cache
from config  import ALPHA_PRUDENT

app  = Flask(__name__)
CORS(app)

OUT  = os.path.join(BASE, "output")
DATA = os.path.join(BASE, "data")

# ════════════════════════════════════════════════════════════════
# CONFIG MYSQL
# ════════════════════════════════════════════════════════════════
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 3306,
    "user"    : "root",
    "password": "",
    "database": "easybulk",
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
# SYNC MySQL → clean_groupes.json  (FIX PRINCIPAL)
# ════════════════════════════════════════════════════════════════

def sync_clean_groupes_from_mysql():
    """
    Lit les groupes depuis MySQL et écrase clean_groupes.json.
    Appelé au démarrage de l'API → predict.py voit les vrais groupes.
    """
    conn = get_db()
    if not conn:
        print("  ⚠  sync_clean_groupes : MySQL inaccessible, clean_groupes.json inchangé")
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, description,
                       quota, quotaLoked, quotaFree,
                       status_id, organization_id, admin_id
                FROM groupe
                ORDER BY id
            """)
            rows = cur.fetchall()

        if not rows:
            print("  ⚠  sync_clean_groupes : table groupe vide")
            return False

        # Construire le DataFrame compatible avec predict.py
        df = pd.DataFrame(rows)
        df["est_actif"]  = (df["status_id"] == 1).astype(int)
        # predict.py utilise quotaLoked / quotaFree (camelCase)
        if "quotaLoked" not in df.columns:
            df["quotaLoked"] = df.get("quota_loked", 0)
        if "quotaFree" not in df.columns:
            df["quotaFree"]  = df["quota"] - df["quotaLoked"]

        os.makedirs(OUT, exist_ok=True)
        clean_path = os.path.join(OUT, "clean_groupes.json")
        df.to_json(clean_path, orient="records", force_ascii=False, indent=2)
        print(f"  ✓  clean_groupes.json synchro depuis MySQL ({len(df)} groupes)")
        for _, r in df.iterrows():
            print(f"      #{int(r['id'])} {str(r['name']):<22}  quota={int(r['quota']):>8}  libre={int(r.get('quotaFree', r['quota'])):>8}")
        return True

    except Exception as e:
        print(f"  ⚠  sync_clean_groupes erreur : {e}")
        return False
    finally:
        conn.close()


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
    Toutes les valeurs viennent de predict.py (quota réel via cumsum).
    Si predict.py échoue (groupe sans features), on utilise les données MySQL.
    """
    gid       = int(row["id"])
    est_actif = int(row.get("est_actif", 1)) == 1

    # 1. Prédictions ML via predict.py
    p7  = _predire_safe(gid,  7)
    p14 = _predire_safe(gid, 14)
    p30 = _predire_safe(gid, 30)
    prud   = _conso_prudente(gid, res)
    risque = p30.get("niveau_risque", "SAFE")

    # 2. Quotas — depuis predict.py si disponible, sinon MySQL direct
    if p30 and "quota_libre" in p30:
        quota_total = int(p30.get("quota_total",      row.get("quota", 0)))
        quota_libre = int(p30.get("quota_libre",      0))
        quota_verr  = int(p30.get("quota_verrouille", 0))
    else:
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
    path = os.path.join(DATA, "groupes.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _ecrire_groupes_json(data):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "groupes.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
        Ouvre <strong>test_easybulk.html</strong> dans ton navigateur.
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
    try:
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
                        WHERE status_id = 1
                        ORDER BY id
                    """)
                    rows = cur.fetchall()
                for r in rows:
                    r["est_actif"] = 1
            finally:
                conn.close()
            print(f"  ·  {len(rows)} groupes lus depuis MySQL")
        else:
            print("  ⚠  MySQL inaccessible → fallback clean_groupes.json")
            res_fb = charger_ressources()
            df_fb  = res_fb["df_groupes"]
            if "est_actif" in df_fb.columns:
                df_fb = df_fb[df_fb["est_actif"] == 1]
            rows = df_fb.to_dict("records")

        res = charger_ressources()

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
    body = request.get_json(force=True) or {}

    nom    = (body.get("nom") or "").strip()
    budget = int(body.get("budget") or 0)
    desc   = (body.get("description") or "").strip()
    admin  = (body.get("administrateur") or "").strip()

    if not nom or budget <= 0:
        return jsonify({"error": "nom et budget requis"}), 400

    new_id   = None
    mysql_ok = False

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

    if not mysql_ok:
        new_id = _prochain_id_json()
        print(f"  ·  MySQL indisponible → JSON local (id={new_id})")

    # Mettre à jour data/groupes.json ET resync clean_groupes.json
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
    if not any(g.get("id") == new_id for g in groupes):
        groupes.append(nouveau)
        _ecrire_groupes_json(groupes)

    # Resync clean_groupes.json depuis MySQL (inclut le nouveau groupe)
    sync_clean_groupes_from_mysql()

    # Invalider le cache de predict.py → sera rechargé au prochain appel
    predict_cache.clear()

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
    try:
        row = None
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
    print("║   API Budget ML  ←→  test_easybulk.html              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  GET  /groupes      → liste ML                       ║")
    print("║  POST /groupes      → créer groupe (MySQL + JSON)    ║")
    print("║  GET  /groupes/<id> → détail                         ║")
    print("║  GET  /health       → statut                         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── SYNC MySQL → clean_groupes.json au démarrage ──────────
    print("──── Sync MySQL → clean_groupes.json ────────────")
    sync_clean_groupes_from_mysql()
    # Invalider le cache predict.py → recharge avec les nouveaux groupes
    predict_cache.clear()
    print("─────────────────────────────────────────────────")
    print()

    app.run(debug=True, port=5000)