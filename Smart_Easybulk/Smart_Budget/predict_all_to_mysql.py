"""
predict_all_to_mysql.py
─────────────────────────────────────────────────────────────────────
Calcule les prédictions ML pour TOUS les groupes sur les 3 horizons
(7j / 14j / 30j) et les stocke dans MySQL.predictions.

ARCHITECTURE :
    predict.py      = CERVEAU ML (calcule les prédictions)
    predict_all_to_mysql.py = LIVREUR  (boucle + INSERT en BDD)

→ Aucune logique ML ici. On REUTILISE predire() de predict.py.

USAGE :
    Mode 1 — Script CLI :
        python predict_all_to_mysql.py

    Mode 2 — Module Python (appelé par api.py via APScheduler) :
        from predict_all_to_mysql import main
        main()
"""

import os
import sys
import pymysql

# Permet d'importer predire() de predict.py (même dossier)
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from predict import predire   # ← source unique de vérité


# ════════════════════════════════════════════════════════════════════
# CONFIG MYSQL (identique à api.py)
# ════════════════════════════════════════════════════════════════════
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "",
    "database": "easybulk",
    "charset":  "utf8mb4",
}

HORIZONS = [7, 14, 30]


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
def log(m=""): print(m, flush=True)
def title(t):
    log(); log("┌" + "─"*62 + "┐")
    log(f"│  {t:<60}│"); log("└" + "─"*62 + "┘")
def ok(m):   log(f"  ✓  {m}")
def info(m): log(f"  ·  {m}")
def warn(m): log(f"  ⚠  {m}")


def _connect():
    """Connexion MySQL — None si XAMPP éteint."""
    try:
        return pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    except Exception as e:
        warn(f"MySQL inaccessible : {e}")
        return None


# ════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE — utilisé en CLI et par APScheduler
# ════════════════════════════════════════════════════════════════════
def main():
    title("PRÉDICTIONS ML  →  MySQL.predictions")
    info("Source des calculs : predict.py (fonction predire())")
    info("Destination        : easybulk.predictions")

    conn = _connect()
    if not conn:
        return False

    try:
        # 1. Récupère la liste des groupes actifs
        with conn.cursor() as cur:
            cur.execute("SELECT id, name FROM groupe WHERE status_id = 1 ORDER BY id")
            groupes = cur.fetchall()
        info(f"{len(groupes)} groupes actifs trouvés")

        # 2. Vide les anciennes prédictions
        with conn.cursor() as cur:
            cur.execute("DELETE FROM predictions")
        info("Anciennes prédictions effacées")

        # 3. Calcule + INSERT pour chaque (groupe, horizon)
        sql_insert = """
            INSERT INTO predictions
            (groupe_id, horizon_jours, conso_prevue, conso_prudente,
             niveau_risque, a_min, a_reco, jours_avant_zero,
             quota_libre_snapshot)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        # Seuils (cohérents avec predict.py / config.py)
        SEUIL_SAFE = 0.30
        SEUIL_CRITIQUE = 0.05
        MARGE = 0.15

        n_inserted = 0
        for g in groupes:
            gid = int(g["id"])
            nom = g["name"]
            log()
            log(f"  Groupe #{gid:<2} {nom}")

            # ★ Récupère le quota RÉEL depuis MySQL (reflète les recharges)
            with conn.cursor() as cur:
                cur.execute("SELECT quota, quotaFree, quotaLoked "
                            "FROM groupe WHERE id = %s", (gid,))
                row_grp = cur.fetchone()
            if not row_grp:
                warn(f"  groupe {gid} introuvable en MySQL — ignoré")
                continue
            quota_total_actuel = float(row_grp["quota"] or 0)
            quota_libre_actuel = float(row_grp["quotaFree"] or 0)

            for h in HORIZONS:
                try:
                    # ML : conso prévue/prudente vient de predict.py (features.csv)
                    r = predire(gid, h, verbose=False)
                    conso_prevue = float(r["consommation_prevue_credits"])
                    conso_prudente = float(r["consommation_prudente_credits"])

                    # ★ Recalcule risque / A_min / A_reco / J→0 avec le quota RÉEL
                    #   (et non celui figé dans features.csv)
                    restant = quota_libre_actuel - conso_prudente
                    if quota_total_actuel <= 0 or restant <= 0:
                        niveau = "CRITIQUE"
                    else:
                        ratio = restant / quota_total_actuel
                        niveau = ("CRITIQUE" if ratio <= SEUIL_CRITIQUE
                                  else "DANGER" if ratio <= SEUIL_SAFE
                        else "SAFE")

                    a_min = max(0.0, conso_prudente - quota_libre_actuel
                                + quota_total_actuel * MARGE)
                    a_reco = max(0.0, conso_prudente - quota_libre_actuel
                                 + quota_total_actuel * SEUIL_SAFE)

                    if conso_prevue > 0:
                        jzero = min(999, int((quota_libre_actuel / conso_prevue) * h))
                    else:
                        jzero = 999

                    with conn.cursor() as cur:
                        cur.execute(sql_insert, (
                            gid, h,
                            conso_prevue,
                            conso_prudente,
                            niveau,
                            round(a_min, 2),
                            round(a_reco, 2),
                            jzero,
                            int(quota_libre_actuel),
                        ))
                    n_inserted += 1
                    log(f"     {h:>2}j  conso={conso_prevue:>7.0f}  "
                        f"prudente={conso_prudente:>7.0f}  "
                        f"risque={niveau:<9}  A_reco={a_reco:>7.0f}  "
                        f"J→0={jzero}  (quota MySQL={int(quota_libre_actuel)})")
                except Exception as e:
                    warn(f"    {h}j  ÉCHEC : {e}")

        # 4. Génère les notifications pour groupes à risque (30j seulement)
        log()
        title("Notifications automatiques (DANGER / CRITIQUE sur 30j)")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM notifications")
            cur.execute("""
                SELECT p.groupe_id, g.name, p.niveau_risque, p.conso_prevue, p.a_reco
                FROM predictions p
                JOIN groupe g ON g.id = p.groupe_id
                WHERE p.horizon_jours = 30
                  AND p.niveau_risque IN ('DANGER','CRITIQUE')
                ORDER BY p.niveau_risque DESC
            """)
            risques = cur.fetchall()
            for r in risques:
                type_n = "CRITICAL" if r["niveau_risque"] == "CRITIQUE" else "WARNING"
                msg = (f"⚠ {r['name']} : niveau {r['niveau_risque']} sur 30j "
                       f"(conso prévue {int(r['conso_prevue'])} cr, "
                       f"recharger {int(r['a_reco'])} cr)")
                cur.execute("""
                    INSERT INTO notifications (organization_id, groupe_id, type, message)
                    VALUES (1, %s, %s, %s)
                """, (r["groupe_id"], type_n, msg))
                ok(f"[{type_n:<8}] {msg}")
            if not risques:
                info("Aucun groupe à risque — aucune notification créée")

        conn.commit()

        # 5. Récap
        log()
        title("Récapitulatif")
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM predictions")
            info(f"predictions   : {cur.fetchone()['n']}")
            cur.execute("SELECT COUNT(*) AS n FROM notifications")
            info(f"notifications : {cur.fetchone()['n']}")
        log()
        log("══════════════════════════════════════════════════════════")
        log("  ✓  MySQL.predictions mis à jour. L'interface peut lire.")
        log("══════════════════════════════════════════════════════════")
        return True

    except Exception as e:
        import traceback; traceback.print_exc()
        warn(f"Erreur : {e}")
        return False
    finally:
        try: conn.close()
        except Exception: pass


if __name__ == "__main__":
    main()
