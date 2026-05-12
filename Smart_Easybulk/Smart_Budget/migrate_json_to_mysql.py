"""
migrate_json_to_mysql.py
─────────────────────────────────────────────────────────────────
Migre les fichiers JSON de data/ vers la base MySQL `budget`.

Effaces les seeds (Commercial/Marketing/RH du schema_v2.sql)
et insère les VRAIS groupes du training ML :
  • Prisons Nord / Centre / Sud / Est
  • Test / Dev
… avec tout leur historique (recharges + consommations + campagnes).

Lance :
    .venv\\Scripts\\python.exe migrate_json_to_mysql.py

Pré-requis :
    XAMPP démarré, MySQL accessible sur localhost:3306
    Base `budget` créée et schema_v2.sql importé
"""
import os
import sys
import json
from datetime import datetime
import mysql.connector
from mysql.connector import Error

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")

DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "",
    "database": "budget",
    "charset":  "utf8mb4",
}

# ── Logger ────────────────────────────────────────────────────────
def log(m=""): print(m)
def title(t):
    log(); log("┌" + "─"*60 + "┐")
    log(f"│  {t:<58}│")
    log("└" + "─"*60 + "┘")
def ok(m):   log(f"  ✓  {m}")
def info(m): log(f"  ·  {m}")
def warn(m): log(f"  ⚠  {m}")

# ── Utilitaires ───────────────────────────────────────────────────
def load_json(name):
    path = os.path.join(DATA_DIR, f"{name}.json")
    if not os.path.exists(path):
        warn(f"Fichier introuvable : {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def parse_date(s):
    """ISO ou date simple → datetime Python sans timezone."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        # MySQL DATETIME ne stocke pas la timezone → on la retire
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, AttributeError):
        try:
            return datetime.strptime(str(s), "%Y-%m-%d")
        except ValueError:
            return None

def parse_date_only(s):
    """Renvoie juste la date (pour budget_history.modification_date)."""
    dt = parse_date(s)
    return dt.date() if dt else None


# ══════════════════════════════════════════════════════════════════
def main():
    title("MIGRATION JSON → MySQL `budget`")

    # ── 1. Lecture des JSONs ──────────────────────────────────
    title("ÉTAPE 1 — Lecture des fichiers JSON")
    groupes_json = load_json("groupes")
    types_json   = load_json("prm_campaign_type")
    perms_json   = load_json("campaign_type_permission")
    budget_json  = load_json("budget_history")
    camps_json   = load_json("campagnes")

    info(f"groupes.json                     : {len(groupes_json):>5}")
    info(f"prm_campaign_type.json           : {len(types_json):>5}")
    info(f"campaign_type_permission.json    : {len(perms_json):>5}")
    info(f"budget_history.json              : {len(budget_json):>5}")
    info(f"campagnes.json                   : {len(camps_json):>5}")

    if not groupes_json:
        warn("groupes.json vide ou introuvable — abandon")
        sys.exit(1)

    # ── 2. Connexion MySQL ────────────────────────────────────
    title("ÉTAPE 2 — Connexion à MySQL")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor()
        ok(f"Connecté à {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    except Error as e:
        warn(f"Connexion échouée : {e}")
        warn("Vérifie que XAMPP > MySQL est démarré")
        sys.exit(1)

    # ── 3. Nettoyage des données existantes ───────────────────
    title("ÉTAPE 3 — Nettoyage")
    cur.execute("SET FOREIGN_KEY_CHECKS=0")
    tables_to_clear = [
        "notifications",
        "predictions",
        "campagne",
        "campaign_type_permission",
        "budget_history",
        "groupe",
        "organization",
        "prm_campaign_type",
    ]
    for t in tables_to_clear:
        cur.execute(f"DELETE FROM {t}")
        cur.execute(f"ALTER TABLE {t} AUTO_INCREMENT = 1")
        ok(f"vidée : {t}")
    cur.execute("SET FOREIGN_KEY_CHECKS=1")

    # ── 4. organization ───────────────────────────────────────
    title("ÉTAPE 4 — Insertion de l'organisation")
    cur.execute("""
        INSERT INTO organization (id, name, description, quota)
        VALUES (1, 'EasyBulk Demo', 'Organisation racine pour la démo PFE', 5000000)
    """)
    ok("organization #1  EasyBulk Demo")

    # ── 5. groupes (5 groupes Prisons + Test/Dev) ─────────────
    title("ÉTAPE 5 — Insertion des groupes")
    sql = """
        INSERT INTO groupe (id, name, description, quota, quota_loked, quota_free,
                            status_id, organization_id, admin_id,
                            entete_alpha, type_campagne, is_active,
                            created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for g in groupes_json:
        # Génère des défauts plausibles pour les nouveaux champs UI
        entete_default = [g["name"].split()[0][:10], "85811"]
        types_default  = ["classique", "transactionnelle"]
        cur.execute(sql, (
            g["id"], g["name"], g["description"],
            g["quota"], g.get("quotaLoked", 0), g.get("quotaFree", g["quota"]),
            g["status_id"], g["organization_id"], g.get("admin_id"),
            json.dumps(entete_default),
            json.dumps(types_default),
            True,
            parse_date(g.get("createdAt")), parse_date(g.get("updatedAt"))
        ))
        ok(f"groupe #{g['id']:<2} {g['name']:<20}  quota={g['quota']:>7}  libre={g.get('quotaFree', g['quota']):>7}")

    # ── 6. prm_campaign_type ──────────────────────────────────
    title("ÉTAPE 6 — Types de campagne")
    type_map = {}  # code (str) → id (int)
    for i, t in enumerate(types_json, start=1):
        cur.execute(
            "INSERT INTO prm_campaign_type (id, code, name) VALUES (%s, %s, %s)",
            (i, t["code"], t.get("value"))
        )
        type_map[t["code"]] = i
        ok(f"type #{i}  {t['code']}  ({t.get('value', '')})")

    # ── 7. campaign_type_permission ───────────────────────────
    title("ÉTAPE 7 — Permissions de campagne")
    sql = """
        INSERT INTO campaign_type_permission (id, enabled, groupe_id, campaign_type)
        VALUES (%s, %s, %s, %s)
    """
    for p in perms_json:
        ct = p["campaign_type"]
        ct_id = type_map.get(ct) if isinstance(ct, str) else int(ct)
        cur.execute(sql, (p["id"], bool(p.get("enabled", True)), p["groupe_id"], ct_id))
    ok(f"{len(perms_json)} permissions insérées")

    # ── 8. budget_history (gros volume) ───────────────────────
    title("ÉTAPE 8 — budget_history (recharges + consommations)")
    sql = """
        INSERT INTO budget_history (id, groupe_id, modification_date, amount, status_id)
        VALUES (%s, %s, %s, %s, %s)
    """
    batch = []
    for b in budget_json:
        d = parse_date_only(b.get("modificationDate"))
        if d is None:
            continue
        batch.append((b["id"], b["groupe_id"], d, b["amount"], b["status_id"]))

    # Insertion par paquets de 1000 (rapide et évite la limite max_allowed_packet)
    BATCH_SIZE = 1000
    for i in range(0, len(batch), BATCH_SIZE):
        chunk = batch[i:i+BATCH_SIZE]
        cur.executemany(sql, chunk)
    ok(f"{len(batch)} lignes insérées (par paquets de {BATCH_SIZE})")

    # ── 9. campagnes ──────────────────────────────────────────
    title("ÉTAPE 9 — campagnes")
    sql = """
        INSERT INTO campagne (id, libelle, description, date_debut, date_fin,
                              dure_validite, cost, budget_used, nbr_page, count_contact,
                              deactivated_by_group, hash, me_only,
                              last_updated_status_at, created_at, updated_at,
                              status_id, campaign_type_permission_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    batch = []
    for c in camps_json:
        batch.append((
            c["id"],
            c.get("libelle"),
            c.get("description"),
            parse_date(c.get("dateDebut")),
            parse_date(c.get("dateFin")),
            c.get("dureValidite"),
            c.get("cost"),
            c.get("budgetUsed"),
            c.get("nbrPage"),
            c.get("countContact"),
            bool(c.get("deactivatedByGroup", False)),
            c.get("hash", ""),
            bool(c.get("meOnly", False)),
            parse_date(c.get("lastUpdateStatusAt")),
            parse_date(c.get("createdAt")),
            parse_date(c.get("updatedAt")),
            c.get("status_id", 4),
            c.get("campaign_type_permission_id"),
        ))
    if batch:
        for i in range(0, len(batch), BATCH_SIZE):
            cur.executemany(sql, batch[i:i+BATCH_SIZE])
    ok(f"{len(batch)} campagnes insérées")

    # ── 10. Validation finale ─────────────────────────────────
    conn.commit()

    title("ÉTAPE 10 — Vérifications post-import")
    cur.execute("SELECT COUNT(*) FROM organization");           info(f"organization        : {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM groupe");                 info(f"groupe              : {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM prm_campaign_type");      info(f"prm_campaign_type   : {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM campaign_type_permission"); info(f"campaign_type_perm  : {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM budget_history");         info(f"budget_history      : {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM campagne");               info(f"campagne            : {cur.fetchone()[0]}")

    log()
    cur.execute("SELECT id, name, quota, quota_free FROM groupe ORDER BY id")
    log("  Groupes en base :")
    for row in cur.fetchall():
        log(f"    #{row[0]} {row[1]:<25}  quota={row[2]:>8}  libre={row[3]:>8}")

    cur.close()
    conn.close()

    title("MIGRATION TERMINÉE ✓")
    log()
    log("  Tu peux maintenant :")
    log("    1. Vérifier dans phpMyAdmin (rafraîchis la page)")
    log("    2. Lancer l'API : cd api && python app.py")
    log("    3. Tester http://localhost:5000/groupes  → tu verras les 5 groupes")
    log()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        warn(f"Erreur : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
