"""
init_db.py — Auto-initialisation de la base MySQL `easybulk`.
─────────────────────────────────────────────────────────────────────
Appelé par api.py au démarrage. Fait 2 choses :

  1. CRÉE LES TABLES si elles n'existent pas (CREATE TABLE IF NOT EXISTS)
     → Idempotent : peut être rappelé sans risque

  2. IMPORTE LES DONNÉES depuis data/*.json si la table `groupe` est vide
     → Une seule fois au premier démarrage

Si MySQL n'est pas accessible (XAMPP éteint), on log un warning et on
continue : l'API marche quand même en mode "lecture seule depuis predict.py"
(MySQL sert uniquement pour POST /groupes).
"""

import os
import json
from datetime import datetime

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")


# ════════════════════════════════════════════════════════════════════
# DDL  —  9 tables (CREATE TABLE IF NOT EXISTS)
# ════════════════════════════════════════════════════════════════════
DDL = [
    """
    CREATE TABLE IF NOT EXISTS prm_status (
        id         INT          NOT NULL AUTO_INCREMENT,
        code       VARCHAR(50)  NOT NULL,
        type       VARCHAR(50)  NOT NULL,
        value      VARCHAR(100) NULL,
        created_at DATETIME     NULL,
        updated_at DATETIME     NULL,
        PRIMARY KEY (id),
        UNIQUE KEY uk_status_code_type (code, type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS organization (
        id          INT          NOT NULL AUTO_INCREMENT,
        name        VARCHAR(100) NOT NULL UNIQUE,
        description VARCHAR(250) NULL,
        quota       BIGINT       NOT NULL DEFAULT 0,
        PRIMARY KEY (id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS groupe (
        id              INT          NOT NULL AUTO_INCREMENT,
        name            VARCHAR(100) NOT NULL,
        description     VARCHAR(250) NULL,
        quota           BIGINT       NOT NULL DEFAULT 0,
        quotaLoked      BIGINT       NULL DEFAULT 0,
        quotaFree       BIGINT       NULL,
        status_id       INT          NOT NULL DEFAULT 1,
        organization_id INT          NOT NULL DEFAULT 1,
        admin_id        INT          NULL,
        created_at      DATETIME     NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      DATETIME     NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        KEY idx_groupe_org (organization_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS budget_history (
        id                INT      NOT NULL AUTO_INCREMENT,
        groupe_id         INT      NOT NULL,
        modificationDate  DATE     NOT NULL,
        amount            BIGINT   NOT NULL,
        status_id         INT      NOT NULL,
        PRIMARY KEY (id),
        KEY idx_bh_groupe_date (groupe_id, modificationDate)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS prm_campaign_type (
        id   INT         NOT NULL AUTO_INCREMENT,
        code VARCHAR(50) NOT NULL UNIQUE,
        value VARCHAR(100) NULL,
        PRIMARY KEY (id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS campaign_type_permission (
        id            BIGINT       NOT NULL AUTO_INCREMENT,
        enabled       BOOLEAN      NOT NULL DEFAULT TRUE,
        groupe_id     INT          NOT NULL,
        campaign_type VARCHAR(50)  NOT NULL,
        PRIMARY KEY (id),
        UNIQUE KEY uk_ctp_groupe_type (groupe_id, campaign_type),
        KEY idx_ctp_groupe (groupe_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS campagne (
        id                          BIGINT       NOT NULL AUTO_INCREMENT,
        libelle                     VARCHAR(200) NULL,
        description                 VARCHAR(500) NULL,
        dateDebut                   DATETIME     NULL,
        dateFin                     DATETIME     NULL,
        dureValidite                INT          NULL,
        cost                        BIGINT       NULL,
        budgetUsed                  BIGINT       NULL,
        nbrPage                     INT          NULL,
        countContact                BIGINT       NULL,
        deactivatedByGroup          BOOLEAN      NOT NULL DEFAULT FALSE,
        hash                        VARCHAR(255) NULL,
        meOnly                      BOOLEAN      NULL,
        lastUpdateStatusAt          DATETIME     NULL,
        createdAt                   DATETIME     NULL,
        updatedAt                   DATETIME     NULL,
        status_id                   INT          NOT NULL DEFAULT 4,
        campaign_type_permission_id BIGINT       NULL,
        PRIMARY KEY (id),
        KEY idx_camp_date_debut (dateDebut)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id                  BIGINT        NOT NULL AUTO_INCREMENT,
        groupe_id           INT           NOT NULL,
        horizon_jours       INT           NOT NULL,
        conso_prevue        DECIMAL(12,2) NOT NULL,
        conso_prudente      DECIMAL(12,2) NOT NULL,
        niveau_risque       ENUM('SAFE','DANGER','CRITIQUE') NOT NULL,
        a_min               DECIMAL(12,2) NULL,
        a_reco              DECIMAL(12,2) NULL,
        jours_avant_zero    INT           NULL,
        quota_libre_snapshot BIGINT       NOT NULL,
        created_at          DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        KEY idx_pred_groupe (groupe_id, horizon_jours, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS notifications (
        id              BIGINT       NOT NULL AUTO_INCREMENT,
        organization_id INT          NOT NULL DEFAULT 1,
        groupe_id       INT          NULL,
        type            ENUM('INFO','WARNING','CRITICAL') NOT NULL,
        message         VARCHAR(500) NOT NULL,
        is_read         BOOLEAN      NOT NULL DEFAULT FALSE,
        created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        KEY idx_notif_org_unread (organization_id, is_read, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


# ════════════════════════════════════════════════════════════════════
# Données de référence (seeds minimaux : statuses + organisation 1)
# ════════════════════════════════════════════════════════════════════
SEED_REF = [
    ("INSERT IGNORE INTO prm_status (id, code, type, value) VALUES "
     "(1,'RECHARGE','BUDGET','Rechargement'),"
     "(2,'CONSOMMATION','BUDGET','Consommation'),"
     "(3,'DRAFT','CAMPAIGN','Brouillon'),"
     "(4,'SENT','CAMPAIGN','Envoyée'),"
     "(5,'SCHEDULED','CAMPAIGN','Planifiée')"),
    ("INSERT IGNORE INTO organization (id, name, description, quota) VALUES "
     "(1, 'EasyBulk Demo', 'Organisation racine', 5000000)"),
]


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
def _parse_date(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        try:
            return datetime.strptime(str(s), "%Y-%m-%d")
        except Exception:
            return None


def _load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════
# 1. CRÉATION DES TABLES
# ════════════════════════════════════════════════════════════════════
def _create_tables(conn):
    with conn.cursor() as cur:
        for ddl in DDL:
            cur.execute(ddl)
        for seed in SEED_REF:
            cur.execute(seed)
    conn.commit()


# ════════════════════════════════════════════════════════════════════
# 2. IMPORT DES DONNÉES JSON (si tables vides)
# ════════════════════════════════════════════════════════════════════
def _seed_from_json(conn):
    groupes      = _load_json("groupes.json")
    types        = _load_json("prm_campaign_type.json")
    perms        = _load_json("campaign_type_permission.json")
    budget_hist  = _load_json("budget_history.json")
    campagnes    = _load_json("campagnes.json")

    inserted = {}
    with conn.cursor() as cur:
        # 1. groupes (5 groupes Prisons + Test/Dev)
        for g in groupes:
            cur.execute("""
                INSERT IGNORE INTO groupe
                (id, name, description, quota, quotaLoked, quotaFree,
                 status_id, organization_id, admin_id, created_at, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                g["id"], g["name"], g.get("description", ""),
                g["quota"], g.get("quotaLoked", 0), g.get("quotaFree", g["quota"]),
                g.get("status_id", 1), g.get("organization_id", 1), g.get("admin_id"),
                _parse_date(g.get("createdAt")), _parse_date(g.get("updatedAt")),
            ))
        inserted["groupes"] = len(groupes)

        # 2. types de campagne
        for i, t in enumerate(types, start=1):
            cur.execute(
                "INSERT IGNORE INTO prm_campaign_type (id, code, value) VALUES (%s,%s,%s)",
                (i, t["code"], t.get("value"))
            )
        inserted["types"] = len(types)

        # 3. permissions
        for p in perms:
            cur.execute("""
                INSERT IGNORE INTO campaign_type_permission
                (id, enabled, groupe_id, campaign_type)
                VALUES (%s,%s,%s,%s)
            """, (p["id"], bool(p.get("enabled", True)),
                  p["groupe_id"], str(p["campaign_type"])))
        inserted["permissions"] = len(perms)

        # 4. budget_history (gros : INSERT en batch)
        batch = []
        for b in budget_hist:
            d = _parse_date(b.get("modificationDate"))
            if d is None:
                continue
            batch.append((b["id"], b["groupe_id"], d.date(),
                          b["amount"], b["status_id"]))
        if batch:
            cur.executemany("""
                INSERT IGNORE INTO budget_history
                (id, groupe_id, modificationDate, amount, status_id)
                VALUES (%s,%s,%s,%s,%s)
            """, batch)
        inserted["budget_history"] = len(batch)

        # 5. campagnes
        batch = []
        for c in campagnes:
            batch.append((
                c["id"], c.get("libelle"), c.get("description"),
                _parse_date(c.get("dateDebut")), _parse_date(c.get("dateFin")),
                c.get("dureValidite"),
                c.get("cost"), c.get("budgetUsed"),
                c.get("nbrPage"), c.get("countContact"),
                bool(c.get("deactivatedByGroup", False)),
                c.get("hash", ""),
                bool(c.get("meOnly", False)),
                _parse_date(c.get("lastUpdateStatusAt")),
                _parse_date(c.get("createdAt")),
                _parse_date(c.get("updatedAt")),
                c.get("status_id", 4),
                c.get("campaign_type_permission_id"),
            ))
        if batch:
            cur.executemany("""
                INSERT IGNORE INTO campagne
                (id, libelle, description, dateDebut, dateFin, dureValidite,
                 cost, budgetUsed, nbrPage, countContact, deactivatedByGroup,
                 hash, meOnly, lastUpdateStatusAt, createdAt, updatedAt,
                 status_id, campaign_type_permission_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, batch)
        inserted["campagnes"] = len(batch)

    conn.commit()
    return inserted


def _count_groupes(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM groupe")
        row = cur.fetchone()
        return row["n"] if isinstance(row, dict) else row[0]


# ════════════════════════════════════════════════════════════════════
# Point d'entrée — appelé par api.py
# ════════════════════════════════════════════════════════════════════
def init_db_and_seed(get_db_func):
    """
    Vérifie / crée les tables et peuple si vides.
    `get_db_func` : fonction sans argument qui renvoie une connexion MySQL
                    (ou None si XAMPP éteint).
    """
    conn = get_db_func()
    if not conn:
        print("  ⚠  MySQL inaccessible — init DB sautée (l'API tournera "
              "en lecture seule via predict.py).")
        return False

    try:
        # 1. Crée les tables manquantes
        _create_tables(conn)
        print("  ✓  Tables vérifiées (9 tables prêtes dans `easybulk`)")

        # 2. Importe les seeds si la table groupe est vide
        n = _count_groupes(conn)
        if n == 0:
            print("  ·  Table `groupe` vide → import depuis data/*.json …")
            stats = _seed_from_json(conn)
            print(f"     groupes        : {stats.get('groupes', 0):>5}")
            print(f"     types          : {stats.get('types', 0):>5}")
            print(f"     permissions    : {stats.get('permissions', 0):>5}")
            print(f"     budget_history : {stats.get('budget_history', 0):>5}")
            print(f"     campagnes      : {stats.get('campagnes', 0):>5}")
            print("  ✓  Données importées")
        else:
            print(f"  ·  {n} groupes déjà en base — pas d'import")

        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ⚠  Erreur init DB : {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass
