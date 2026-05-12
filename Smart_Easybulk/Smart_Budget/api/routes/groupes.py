"""
routes/groupes.py — Endpoints CRUD sur la ressource "groupe".

Pour chaque groupe lu depuis MySQL, on appelle directement la fonction
`predire()` du module `predict.py` (à la racine du projet).
Source unique de vérité : le terminal `python predict.py` et le HTML
affichent EXACTEMENT les mêmes chiffres parce qu'ils utilisent
la même fonction Python.
"""
import json
from flask import Blueprint, jsonify, request, abort
from db import get_cursor

# predire() = MÊME fonction que celle qu'on appelle en CLI :
#     python predict.py --groupe_id X --horizon Y
# (cf. app.py qui ajoute la racine du projet au sys.path)
from predict import predire

bp = Blueprint("groupes", __name__, url_prefix="/groupes")


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def _ml_for_groupe(groupe_id: int) -> dict:
    """
    Appelle predire() sur les 3 horizons (7j, 14j, 30j) pour un groupe.
    Retourne un dict avec :
      - quota_total / quota_libre / quota_verrouille (valeurs réelles
        calculées par feature_engineering, identiques au terminal)
      - risque (du 30j)
      - predictions = {7j: {...}, 14j: {...}, 30j: {...}}

    Si predict.py échoue (ex: features manquantes pour ce groupe),
    on retourne un dict vide → le caller utilise le fallback MySQL.
    """
    horizons = {}
    common   = {}
    for h in (7, 14, 30):
        try:
            r = predire(groupe_id, h, verbose=False)
            # Quota = la valeur RÉELLE calculée par predict.py
            common.setdefault("quota_total",      int(r["quota_total"]))
            common.setdefault("quota_libre",      int(r["quota_libre"]))
            common.setdefault("quota_verrouille", int(r["quota_verrouille"]))
            # Jours avant 0 (peut être inf si conso=0)
            jours_zero = r.get("jours_avant_zero_predit", 999)
            if jours_zero == float("inf"):
                jours_zero = 999
            horizons[f"{h}j"] = {
                "conso_prevue":     round(float(r["consommation_prevue_credits"]),    2),
                "conso_prudente":   round(float(r["consommation_prudente_credits"]),  2),
                "niveau_risque":    r["niveau_risque"],
                "a_min":            round(float(r["montant_A_minimum"]),    2),
                "a_reco":           round(float(r["montant_A_recommande"]), 2),
                "jours_avant_zero": int(min(jours_zero, 999)),
            }
        except Exception as e:
            print(f"  [predire gid={groupe_id} h={h}j] échoué : {e}")

    if "30j" in horizons:
        common["risque"] = horizons["30j"]["niveau_risque"]
    common["predictions"] = horizons
    return common


def _serialize_groupe(row: dict) -> dict:
    """Convertit la ligne SQL brute → dict de base (sans ML)."""
    entete = row.get("entete_alpha")
    if isinstance(entete, str):
        try: entete = json.loads(entete)
        except Exception: entete = []
    types = row.get("type_campagne")
    if isinstance(types, str):
        try: types = json.loads(types)
        except Exception: types = []

    return {
        "id":               row["id"],
        "nom":              row["name"],
        "description":      row["description"] or "",
        "quota_total":      int(row["quota"] or 0),
        "quota_verrouille": int(row["quota_loked"] or 0),
        "quota_libre":      int(row["quota_free"] or 0),
        "entete_alpha":     entete or [],
        "type_campagne":    types or [],
        "is_active":        bool(row["is_active"]),
        "organization_id":  row["organization_id"],
        "admin_id":         row["admin_id"],
        # Risque et predictions seront overridés par le ML si disponible
        "risque":           "SAFE",
        "predictions":      {},
    }


# ──────────────────────────────────────────────────────────────────
# GET /groupes — liste avec prédictions ML
# ──────────────────────────────────────────────────────────────────
@bp.route("", methods=["GET"])
def list_groupes():
    """
    Liste tous les groupes ; pour chacun, appelle predire() (3 horizons).
    Filtrable par ?organisation_id=...
    """
    org_id = request.args.get("organisation_id", type=int)
    sql = """
        SELECT id, name, description, quota, quota_loked, quota_free,
               status_id, organization_id, admin_id,
               entete_alpha, type_campagne, is_active,
               created_at, updated_at
        FROM groupe
        WHERE 1=1
    """
    params = []
    if org_id is not None:
        sql += " AND organization_id = %s"
        params.append(org_id)
    sql += " ORDER BY id ASC"

    with get_cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    out = []
    for r in rows:
        base = _serialize_groupe(r)
        ml   = _ml_for_groupe(r["id"])
        # Le ML écrase quota_libre / risque / predictions s'il a réussi
        base.update(ml)
        out.append(base)
    return jsonify(out)


# ──────────────────────────────────────────────────────────────────
# GET /groupes/<id> — détail d'un groupe
# ──────────────────────────────────────────────────────────────────
@bp.route("/<int:groupe_id>", methods=["GET"])
def get_groupe(groupe_id: int):
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, name, description, quota, quota_loked, quota_free,
                   status_id, organization_id, admin_id,
                   entete_alpha, type_campagne, is_active,
                   created_at, updated_at
            FROM groupe WHERE id = %s
        """, (groupe_id,))
        row = cur.fetchone()

    if not row:
        abort(404, description=f"Groupe {groupe_id} introuvable")

    base = _serialize_groupe(row)
    ml   = _ml_for_groupe(groupe_id)
    base.update(ml)
    return jsonify(base)
