"""
migrate_json_to_mysql.py  — VERSION CORRIGÉE
─────────────────────────────────────────────────────────────────
Crée TOUTES les tables dans `easybulk` (Budget + SMS Contacts)
et migre les JSONs de data/ vers MySQL.

Tables créées :
  Budget  → organization, groupe, budget_history,
             prm_campaign_type, campaign_type_permission, campagne
  SMS     → contacts, contact_tags

Lance :
    python migrate_json_to_mysql.py

Pré-requis :
    XAMPP démarré, MySQL accessible sur localhost:3306
    Base `easybulk` créée dans phpMyAdmin
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
    "database": "easybulk",
    "charset":  "utf8mb4",
}

BATCH_SIZE = 1000

# ── Logger ────────────────────────────────────────────────────────
def log(m=""):  print(m)
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
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, AttributeError):
        try:
            return datetime.strptime(str(s), "%Y-%m-%d")
        except ValueError:
            return None

def parse_date_only(s):
    dt = parse_date(s)
    return dt.date() if dt else None


# ══════════════════════════════════════════════════════════════════
def main():
    title("MIGRATION JSON → MySQL `easybulk`  (Budget + SMS)")

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

    # ── 3. Création de TOUTES les tables ─────────────────────
    title("ÉTAPE 3 — Création des tables (si inexistantes)")
    cur.execute("SET FOREIGN_KEY_CHECKS=0")

    # ── Tables BUDGET ─────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS organization (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            name        VARCHAR(200),
            description TEXT,
            quota       BIGINT DEFAULT 0
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    ok("table: organization")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prm_campaign_type (
            id   INT AUTO_INCREMENT PRIMARY KEY,
            code VARCHAR(100),
            name VARCHAR(200)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    ok("table: prm_campaign_type")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS groupe (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            name            VARCHAR(200)  NOT NULL,
            description     TEXT,
            quota           BIGINT        DEFAULT 0,
            quota_loked     BIGINT        DEFAULT 0,
            quota_free      BIGINT        DEFAULT 0,
            quotaLoked      BIGINT        DEFAULT 0,
            quotaFree       BIGINT        DEFAULT 0,
            status_id       INT           DEFAULT 1,
            organization_id INT,
            admin_id        INT,
            entete_alpha    JSON,
            type_campagne   JSON,
            is_active       TINYINT       DEFAULT 1,
            created_at      DATETIME,
            updated_at      DATETIME
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    ok("table: groupe")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS budget_history (
            id                INT AUTO_INCREMENT PRIMARY KEY,
            groupe_id         INT,
            modification_date DATE,
            amount            BIGINT,
            status_id         INT DEFAULT 1
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    ok("table: budget_history")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS campaign_type_permission (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            enabled       TINYINT DEFAULT 1,
            groupe_id     INT,
            campaign_type INT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    ok("table: campaign_type_permission")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS campagne (
            id                          INT AUTO_INCREMENT PRIMARY KEY,
            libelle                     VARCHAR(300),
            description                 TEXT,
            date_debut                  DATETIME,
            date_fin                    DATETIME,
            dure_validite               INT,
            cost                        DECIMAL(15,2),
            budget_used                 DECIMAL(15,2),
            nbr_page                    INT,
            count_contact               INT,
            deactivated_by_group        TINYINT  DEFAULT 0,
            hash                        VARCHAR(500),
            me_only                     TINYINT  DEFAULT 0,
            last_updated_status_at      DATETIME,
            created_at                  DATETIME,
            updated_at                  DATETIME,
            status_id                   INT      DEFAULT 4,
            campaign_type_permission_id INT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    ok("table: campagne")

    # ── Tables SMS ────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            telephone  VARCHAR(20)  NOT NULL UNIQUE,
            nom        VARCHAR(100) DEFAULT '',
            prenom     VARCHAR(100) DEFAULT '',
            email      VARCHAR(200) DEFAULT '',
            pays       VARCHAR(10)  DEFAULT 'TN',
            date_ajout DATE         NOT NULL DEFAULT (CURDATE()),
            created_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    ok("table: contacts")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS contact_tags (
            contact_id INT          NOT NULL,
            tag        VARCHAR(100) NOT NULL,
            PRIMARY KEY (contact_id, tag),
            FOREIGN KEY (contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    ok("table: contact_tags")

    cur.execute("SET FOREIGN_KEY_CHECKS=1")
    conn.commit()

    # ── 4. Nettoyage tables Budget ────────────────────────────
    title("ÉTAPE 4 — Nettoyage des données Budget existantes")
    cur.execute("SET FOREIGN_KEY_CHECKS=0")
    tables_budget = [
        "campagne",
        "campaign_type_permission",
        "budget_history",
        "groupe",
        "organization",
        "prm_campaign_type",
    ]
    for t in tables_budget:
        cur.execute(f"DELETE FROM {t}")
        cur.execute(f"ALTER TABLE {t} AUTO_INCREMENT = 1")
        ok(f"vidée : {t}")
    cur.execute("SET FOREIGN_KEY_CHECKS=1")

    # ── 5. organization ───────────────────────────────────────
    title("ÉTAPE 5 — Insertion de l'organisation")
    cur.execute("""
        INSERT INTO organization (id, name, description, quota)
        VALUES (1, 'EasyBulk Demo', 'Organisation racine pour la démo PFE', 5000000)
    """)
    ok("organization #1  EasyBulk Demo")

    # ── 6. groupes ────────────────────────────────────────────
    title("ÉTAPE 6 — Insertion des groupes")
    sql = """
        INSERT INTO groupe (id, name, description, quota, quota_loked, quota_free,
                            quotaLoked, quotaFree,
                            status_id, organization_id, admin_id,
                            entete_alpha, type_campagne, is_active,
                            created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for g in groupes_json:
        entete_default = [g["name"].split()[0][:10], "85811"]
        types_default  = ["classique", "transactionnelle"]
        ql = g.get("quotaLoked", 0)
        qf = g.get("quotaFree", g["quota"])
        cur.execute(sql, (
            g["id"], g["name"], g.get("description", ""),
            g["quota"], ql, qf,
            ql, qf,
            g.get("status_id", 1), g.get("organization_id", 1), g.get("admin_id"),
            json.dumps(entete_default),
            json.dumps(types_default),
            True,
            parse_date(g.get("createdAt")), parse_date(g.get("updatedAt"))
        ))
        ok(f"groupe #{g['id']:<2} {g['name']:<22}  quota={g['quota']:>8}  libre={qf:>8}")

    # ── 7. prm_campaign_type ──────────────────────────────────
    title("ÉTAPE 7 — Types de campagne")
    type_map = {}
    for i, t in enumerate(types_json, start=1):
        cur.execute(
            "INSERT INTO prm_campaign_type (id, code, name) VALUES (%s, %s, %s)",
            (i, t["code"], t.get("value"))
        )
        type_map[t["code"]] = i
        ok(f"type #{i}  {t['code']}  ({t.get('value', '')})")

    # ── 8. campaign_type_permission ───────────────────────────
    title("ÉTAPE 8 — Permissions de campagne")
    sql = """
        INSERT INTO campaign_type_permission (id, enabled, groupe_id, campaign_type)
        VALUES (%s, %s, %s, %s)
    """
    for p in perms_json:
        ct    = p["campaign_type"]
        ct_id = type_map.get(ct) if isinstance(ct, str) else int(ct)
        cur.execute(sql, (p["id"], bool(p.get("enabled", True)), p["groupe_id"], ct_id))
    ok(f"{len(perms_json)} permissions insérées")

    # ── 9. budget_history ─────────────────────────────────────
    title("ÉTAPE 9 — budget_history (recharges + consommations)")
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
    for i in range(0, len(batch), BATCH_SIZE):
        cur.executemany(sql, batch[i:i+BATCH_SIZE])
    ok(f"{len(batch)} lignes insérées (par paquets de {BATCH_SIZE})")

    # ── 10. campagnes ─────────────────────────────────────────
    title("ÉTAPE 10 — campagnes")
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
            c["id"], c.get("libelle"), c.get("description"),
            parse_date(c.get("dateDebut")), parse_date(c.get("dateFin")),
            c.get("dureValidite"), c.get("cost"), c.get("budgetUsed"),
            c.get("nbrPage"), c.get("countContact"),
            bool(c.get("deactivatedByGroup", False)),
            c.get("hash", ""),
            bool(c.get("meOnly", False)),
            parse_date(c.get("lastUpdateStatusAt")),
            parse_date(c.get("createdAt")), parse_date(c.get("updatedAt")),
            c.get("status_id", 4), c.get("campaign_type_permission_id"),
        ))
    for i in range(0, len(batch), BATCH_SIZE):
        cur.executemany(sql, batch[i:i+BATCH_SIZE])
    ok(f"{len(batch)} campagnes insérées")

    # ── 11. Commit + Vérifications ────────────────────────────
    conn.commit()

    title("ÉTAPE 11 — Vérifications post-import")
    checks = [
        ("organization",            "organization"),
        ("groupe",                  "groupe"),
        ("prm_campaign_type",       "prm_campaign_type"),
        ("campaign_type_permission","campaign_type_perm"),
        ("budget_history",          "budget_history"),
        ("campagne",                "campagne"),
        ("contacts",                "contacts"),
        ("contact_tags",            "contact_tags"),
    ]
    for table, label in checks:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        info(f"{label:<25} : {cur.fetchone()[0]}")

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
    log("    1. Vérifier dans phpMyAdmin → easybulk (rafraîchis)")
    log("    2. Terminal 1 : cd Smart_Budget && python api.py")
    log("    3. Terminal 2 : cd Smat_SMS   && python predict_api.py")
    log("    4. Ouvrir test_easybulk.html → les deux dots verts ✓")
    log()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        warn(f"Erreur : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)