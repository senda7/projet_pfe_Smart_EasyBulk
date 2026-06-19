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
# CONFIG MYSQL — lit les variables d'environnement (Docker)
# avec valeurs par défaut pour le développement local (XAMPP).
# ════════════════════════════════════════════════════════════════
DB_CONFIG = {
    "host"    : os.environ.get("DB_HOST",     "localhost"),
    "port"    : int(os.environ.get("DB_PORT", "3306")),
    "user"    : os.environ.get("DB_USER",     "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME",     "easybulk"),
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


# UTILITAIRES

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

def _fetch_predictions_from_db(gid):

    conn = get_db()
    if not conn:
        return {}
    out = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT horizon_jours, conso_prevue, conso_prudente,
                       niveau_risque, a_min, a_reco, jours_avant_zero,
                       quota_libre_snapshot, created_at
                FROM predictions
                WHERE groupe_id = %s
                ORDER BY horizon_jours, created_at DESC
            """, (gid,))
            for r in cur.fetchall():
                h = int(r["horizon_jours"])
                if h in out:           # garde la plus récente seulement
                    continue
                out[h] = r
    except Exception as e:
        print(f"  ⚠  fetch predictions ({gid}): {e}")
    finally:
        try: conn.close()
        except Exception: pass
    return out


def _build_groupe(row, res):

    gid       = int(row["id"])
    est_actif = int(row.get("est_actif", 1)) == 1

    # ── 1. Tentative LECTURE MySQL.predictions (mode BATCH) ──────
    db_preds = _fetch_predictions_from_db(gid)
    if 30 in db_preds:
        # ✓ Le scheduler est passé, on a des données fraîches en BDD
        p30db = db_preds[30]
        p14db = db_preds.get(14)
        p7db  = db_preds.get(7)

        quota_total = int(row.get("quota", 0))
        quota_libre = int(p30db["quota_libre_snapshot"] or 0)
        quota_verr  = max(0, quota_total - quota_libre)
        risque      = p30db["niveau_risque"]

        # Historique + événements viennent toujours de predict.py
        # (df_features chargé en mémoire via charger_ressources)
        hist  = _historique(res["df_features"], gid)
        evts  = []                # facultatif : compatible avec mode live

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
                "7j" : {"conso_prevue": float(p7db["conso_prevue"])  if p7db  else 0.0},
                "14j": {"conso_prevue": float(p14db["conso_prevue"]) if p14db else 0.0},
                "30j": {
                    "conso_prevue"    : float(p30db["conso_prevue"]),
                    "conso_prudente"  : float(p30db["conso_prudente"]),
                    "a_min"           : float(p30db["a_min"]    or 0),
                    "a_reco"          : float(p30db["a_reco"]   or 0),
                    "jours_avant_zero": int(  p30db["jours_avant_zero"] or 999),
                    "niveau_risque"   : risque,
                    "tendance"        : "stable",
                    "budget_restant"  : quota_libre - float(p30db["conso_prevue"]),
                },
            },
            "historique" : hist,
            "evenements" : evts,
            "campagnes"  : [],
            "_source"    : "mysql.predictions",   # debug
        }

    # ── 2. Fallback LIVE : predict.py appelé directement ─────────
    p7  = _predire_safe(gid,  7)
    p14 = _predire_safe(gid, 14)
    p30 = _predire_safe(gid, 30)
    prud   = _conso_prudente(gid, res)
    risque = p30.get("niveau_risque", "SAFE")

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
        "_source"    : "predict.py-live",   # debug
    }

#========== HELPERS JSON LOCAL (fallback si MySQL down)===========


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


#=========== ENDPOINTS=================


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
    # ── Tentative MySQL avec vérification budget organisation ──
    conn = get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                # 1. Vérifie le budget org disponible (organization_id=1 par défaut)
                cur.execute("SELECT quota FROM organization WHERE id = 1")
                row = cur.fetchone()
                if not row:
                    conn.close()
                    return jsonify({"error": "Organisation 1 introuvable"}), 404
                org_budget = int(row["quota"] or 0)

                if budget > org_budget:
                    conn.close()
                    return jsonify({
                        "error": f"Budget organisation insuffisant : "
                                 f"demandé {budget} cr, disponible {org_budget} cr"
                    }), 400

                # 2. INSERT groupe
                cur.execute("""
                    INSERT INTO groupe (name, quota, quotaLoked, quotaFree, status_id, description)
                    VALUES (%s, %s, 0, %s, 1, %s)
                """, (nom, budget, budget, desc))
                new_id = cur.lastrowid

                # 3. Décrémente le budget de l'organisation
                cur.execute(
                    "UPDATE organization SET quota = quota - %s WHERE id = 1",
                    (budget,)
                )

            conn.commit()
            mysql_ok = True
            print(f"  ✓  MySQL : groupe '{nom}' inséré (id={new_id}), "
                  f"budget org : {org_budget} → {org_budget - budget}")
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


# ════════════════════════════════════════════════════════════════
# POST /groupes/<id>/recharger
#   1. INSERT dans MySQL.budget_history (statut RECHARGE)
#   2. UPDATE groupe.quotaFree (visible immédiatement)
#   3. Append data/budget_history.json (pour que ML pipeline le voie)
#   4. Régénère features.csv puis relance predict_all_to_mysql
#      → MySQL.predictions est mis à jour → le risque change !
#
# Durée totale : ~30 sec (subprocess + ML). Réponse synchrone.
# ════════════════════════════════════════════════════════════════
@app.route("/groupes/<int:gid>/recharge",  methods=["PUT"])
@app.route("/groupes/<int:gid>/recharger", methods=["POST"])
def recharger_groupe(gid):
    """
    Recharge un groupe — VERSION SIMPLE.

    Comportement :
      • quota_total      += montant
      • quotaFree        += montant
      • quotaLoked       INCHANGÉ
      • predictions      : niveau_risque='SAFE', a_min=0, a_reco=0
      • notifications    : supprimées pour ce groupe
      • budget_history   : +1 ligne (status_id=1, RECHARGE)
    """
    montant = int((request.get_json(force=True) or {}).get("montant", 0))
    if montant <= 0:
        return jsonify({"error": "montant invalide"}), 400

    conn = get_db()
    if not conn:
        return jsonify({"error": "MySQL inaccessible"}), 503

    try:
        with conn.cursor() as cur:
            # 0. Vérifie le budget org disponible
            cur.execute("SELECT quota FROM organization WHERE id = 1")
            row = cur.fetchone()
            if not row:
                conn.close()
                return jsonify({"error": "Organisation 1 introuvable"}), 404
            org_budget = int(row["quota"] or 0)

            if montant > org_budget:
                conn.close()
                return jsonify({
                    "error": f"Budget organisation insuffisant : "
                             f"recharge {montant} cr, disponible {org_budget} cr"
                }), 400

            # 1. Trace de la recharge
            cur.execute("""
                INSERT INTO budget_history (groupe_id, modificationDate, amount, status_id)
                VALUES (%s, CURDATE(), %s, 1)
            """, (gid, montant))

            # 2. Quota total + libre, quotaLoked inchangé
            cur.execute("""
                UPDATE groupe
                   SET quota     = quota     + %s,
                       quotaFree = quotaFree + %s
                 WHERE id = %s
            """, (montant, montant, gid))

            # 2bis. Décrémente le budget de l'organisation
            cur.execute(
                "UPDATE organization SET quota = quota - %s WHERE id = 1",
                (montant,)
            )

            # 3. Prédictions → SAFE, A_min=0, A_reco=0, J→0 recalculé
            #    Formule J→0 cohérente avec predict.py :
            #       jours_avant_zero = (nouveau_quota_libre / conso_prevue) × horizon
            cur.execute("""
                UPDATE predictions
                   SET niveau_risque        = 'SAFE',
                       a_min                = 0,
                       a_reco               = 0,
                       jours_avant_zero     = CASE
                           WHEN conso_prevue <= 0 THEN 999
                           ELSE LEAST(999, CAST(ROUND(
                                (quota_libre_snapshot + %s)
                                / conso_prevue * horizon_jours
                           ) AS UNSIGNED))
                       END,
                       quota_libre_snapshot = quota_libre_snapshot + %s
                 WHERE groupe_id = %s
            """, (montant, montant, gid))

            # 4. Notifications de ce groupe supprimées
            cur.execute("DELETE FROM notifications WHERE groupe_id = %s", (gid,))

        conn.commit()
        print(f"  ✓  Recharge groupe #{gid} : +{montant} cr → SAFE")
        return jsonify({
            "status":    "ok",
            "groupe_id": gid,
            "montant":   montant,
            "message":   f"Recharge de {montant} cr enregistrée — groupe SAFE",
        })

    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(f"  ⚠  Recharge échouée : {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        try: conn.close()
        except Exception: pass


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
# SCHEDULER (APScheduler) — calculs ML automatiques
# ════════════════════════════════════════════════════════════════
# Tâche 1 : prédictions chaque jour à 00:00 (minuit)
# Tâche 2 : retrain complet chaque dimanche à 02:00
# + Endpoints manuels POST /retrain et POST /predict-now pour démo

def _job_daily_predict():
    """Tâche planifiée : calcule les prédictions et les écrit en BDD."""
    print()
    print("┌─ SCHEDULER ─────────────────────────────────────")
    print("│ Tâche QUOTIDIENNE : calcul des prédictions ML")
    print("└─────────────────────────────────────────────────")
    try:
        from predict_all_to_mysql import main as run_predict_all
        run_predict_all()
    except Exception as e:
        print(f"  ⚠  Tâche prédictions échouée : {e}")


def _job_weekly_retrain():
    """Tâche planifiée : régénère features + réentraîne modèles + prédictions."""
    import subprocess
    print()
    print("┌─ SCHEDULER ─────────────────────────────────────")
    print("│ Tâche HEBDOMADAIRE : retrain complet")
    print("└─────────────────────────────────────────────────")
    steps = [
        ("load_and_explore.py",    "Nettoyage et chargement des données")
        ("feature_engineering.py", "Régénération features.csv"),
        ("train_models.py",        "Entraînement des modèles RF"),
    ]
    for script, label in steps:
        print(f"  ▶  {label}…")
        r = subprocess.run([sys.executable, os.path.join(BASE, script)],
                          cwd=BASE, capture_output=False)
        if r.returncode != 0:
            print(f"  ⚠  {script} a échoué")
            return
        print(f"  ✓  {script} terminé")
    # Puis recalcule les prédictions avec les nouveaux modèles
    _job_daily_predict()


# ── Endpoints manuels ─────────────────────────────────────────
@app.route("/retrain", methods=["POST"])
def manual_retrain():
    """Lance MAINTENANT un retrain complet (features + models + predictions)."""
    try:
        _job_weekly_retrain()
        return jsonify({"status": "ok", "message": "Retrain complet terminé"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict-now", methods=["POST"])
def manual_predict_now():
    """Lance MAINTENANT le calcul des prédictions (sans retrain)."""
    try:
        _job_daily_predict()
        return jsonify({"status": "ok", "message": "Prédictions recalculées"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def _start_scheduler():
    """Démarre APScheduler avec les 2 tâches automatiques."""
    from apscheduler.schedulers.background import BackgroundScheduler
    sched = BackgroundScheduler(daemon=True)

    sched.add_job(_job_daily_predict,   trigger="cron", hour=0,  minute=0,
                  id="daily_predict",  replace_existing=True)
    sched.add_job(_job_weekly_retrain,  trigger="cron", day_of_week="sun",
                  hour=2, minute=0,
                  id="weekly_retrain", replace_existing=True)
    sched.start()

    print()
    print("──── Scheduler ML démarré ───────────────────────")
    for j in sched.get_jobs():
        print(f"  ·  {j.id:<20} prochain : {j.next_run_time}")
    print("─────────────────────────────────────────────────")


@app.route("/organization", methods=["GET"])
@app.route("/api/organization", methods=["GET"])
def get_organization():
    """
    Retourne le budget restant de l'organisation (id=1 par défaut).
    Utilisé par le HTML pour afficher le solde dans la topbar/dashboard.
    """
    conn = get_db()
    if not conn:
        return jsonify({"error": "MySQL inaccessible"}), 503
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, description, quota AS budget_libre
                  FROM organization WHERE id = 1
            """)
            row = cur.fetchone()
        if not row:
            return jsonify({"error": "Organisation 1 introuvable"}), 404
        return jsonify({
            "id":           row["id"],
            "name":         row["name"],
            "description":  row["description"],
            "budget_libre": int(row["budget_libre"] or 0),
        })
    finally:
        try: conn.close()
        except Exception: pass


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
    print("║  POST /retrain      → retrain manuel (features+ML)   ║")
    print("║  POST /predict-now  → recalcul prédictions manuel    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── AUTO-INIT MYSQL ───────────────────────────────────
    print("──── Initialisation MySQL ───────────────────────")
    from init_db import init_db_and_seed
    init_db_and_seed(get_db)
    print("─────────────────────────────────────────────────")

    # ── AUTO-INIT PRÉDICTIONS ─────────────────────────────
    # Si la table predictions est vide, on lance un calcul immédiat
    # pour que l'interface ait des données dès le premier démarrage.
    conn_check = get_db()
    if conn_check:
        try:
            with conn_check.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS n FROM predictions")
                n = cur.fetchone()["n"]
        finally:
            conn_check.close()
        if n == 0:
            print()
            print("──── Premier démarrage : calcul initial des prédictions ─")
            _job_daily_predict()
            print("─────────────────────────────────────────────────")

    # ── SCHEDULER (tâches automatiques) ───────────────────
    _start_scheduler()

    print()
    # use_reloader=False : sinon APScheduler démarre 2 fois en mode debug
    app.run(debug=True, port=5000, use_reloader=False)