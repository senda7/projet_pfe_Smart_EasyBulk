-- ════════════════════════════════════════════════════════════════
-- Schéma minimal + données de test pour le pipeline ML budget
-- Dialecte  : MySQL 8 (InnoDB, utf8mb4)
-- Couverture: 2024-01-01 → 2026-03-31 (~27 mois, 3 groupes)
-- Volumétrie: ~350 lignes budget_history, ~60 campagnes
-- ════════════════════════════════════════════════════════════════

CREATE DATABASE IF NOT EXISTS easybulk_ml
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE easybulk_ml;

-- Pour relances idempotentes pendant les tests
SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS campagne;
DROP TABLE IF EXISTS campaign_type_permission;
DROP TABLE IF EXISTS prm_campaign_type;
DROP TABLE IF EXISTS budget_history;
DROP TABLE IF EXISTS groupe;
DROP TABLE IF EXISTS organization;
DROP TABLE IF EXISTS prm_status;
SET FOREIGN_KEY_CHECKS = 1;


-- ════════════════════════════════════════════════════════════════
-- PARTIE 1 — DDL (7 tables)
-- ════════════════════════════════════════════════════════════════

CREATE TABLE prm_status (
    id         INT          NOT NULL AUTO_INCREMENT,
    code       VARCHAR(50)  NOT NULL,
    type       VARCHAR(50)  NOT NULL,
    value      VARCHAR(100) NULL,
    created_at DATETIME     NULL,
    updated_at DATETIME     NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_status_code_type (code, type)
) ENGINE=InnoDB;

CREATE TABLE organization (
    id          INT          NOT NULL AUTO_INCREMENT,
    name        VARCHAR(50)  NOT NULL UNIQUE,
    description VARCHAR(250) NOT NULL,
    quota       BIGINT       NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

CREATE TABLE groupe (
    id              INT          NOT NULL AUTO_INCREMENT,
    name            VARCHAR(16)  NOT NULL,
    description     VARCHAR(32)  NOT NULL,
    quota           BIGINT       NOT NULL,
    quota_loked     BIGINT       NULL DEFAULT 0,
    quota_free      BIGINT       NULL,
    status_id       INT          NOT NULL,
    organization_id INT          NOT NULL,
    admin_id        INT          NULL,
    created_at      DATETIME     NULL,
    updated_at      DATETIME     NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_groupe_name_org (name, organization_id),
    KEY idx_groupe_org (organization_id),
    CONSTRAINT fk_groupe_status FOREIGN KEY (status_id) REFERENCES prm_status(id),
    CONSTRAINT fk_groupe_org    FOREIGN KEY (organization_id) REFERENCES organization(id)
) ENGINE=InnoDB;

CREATE TABLE budget_history (
    id                INT      NOT NULL AUTO_INCREMENT,
    groupe_id         INT      NOT NULL,
    modification_date DATE     NOT NULL,
    amount            BIGINT   NOT NULL,
    status_id         INT      NOT NULL,
    PRIMARY KEY (id),
    KEY idx_bh_groupe_date (groupe_id, modification_date),
    CONSTRAINT fk_bh_groupe FOREIGN KEY (groupe_id) REFERENCES groupe(id),
    CONSTRAINT fk_bh_status FOREIGN KEY (status_id) REFERENCES prm_status(id)
) ENGINE=InnoDB;

CREATE TABLE prm_campaign_type (
    id   INT         NOT NULL AUTO_INCREMENT,
    code VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(100) NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

CREATE TABLE campaign_type_permission (
    id            BIGINT  NOT NULL AUTO_INCREMENT,
    enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    groupe_id     INT     NOT NULL,
    campaign_type INT     NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_ctp_groupe_type (groupe_id, campaign_type),
    KEY idx_ctp_groupe (groupe_id),
    CONSTRAINT fk_ctp_groupe FOREIGN KEY (groupe_id)     REFERENCES groupe(id),
    CONSTRAINT fk_ctp_type   FOREIGN KEY (campaign_type) REFERENCES prm_campaign_type(id)
) ENGINE=InnoDB;

CREATE TABLE campagne (
    id                          BIGINT       NOT NULL AUTO_INCREMENT,
    libelle                     VARCHAR(100) NULL,
    description                 VARCHAR(255) NULL,
    date_debut                  DATETIME(6)  NULL,
    date_fin                    DATETIME(6)  NULL,
    dure_validite               INT          NULL,
    cost                        BIGINT       NULL,
    budget_used                 BIGINT       NULL,
    nbr_page                    INT          NULL,
    count_contact               BIGINT       NULL,
    deactivated_by_group        BOOLEAN      NOT NULL DEFAULT FALSE,
    hash                        VARCHAR(255) NOT NULL,
    me_only                     BOOLEAN      NULL,
    last_updated_status_at      DATETIME(6)  NULL,
    created_at                  DATETIME     NULL,
    updated_at                  DATETIME     NULL,
    status_id                   INT          NOT NULL,
    campaign_type_permission_id BIGINT       NULL,
    PRIMARY KEY (id),
    KEY idx_camp_date_debut (date_debut),
    KEY idx_camp_ctp (campaign_type_permission_id),
    CONSTRAINT fk_camp_status FOREIGN KEY (status_id) REFERENCES prm_status(id),
    CONSTRAINT fk_camp_ctp    FOREIGN KEY (campaign_type_permission_id)
                              REFERENCES campaign_type_permission(id)
) ENGINE=InnoDB;


-- ════════════════════════════════════════════════════════════════
-- PARTIE 2 — SEEDS : données de test
-- ════════════════════════════════════════════════════════════════

-- ─── Statuts : le pipeline se base sur id=1 (recharge) et id=2 (conso) ──
INSERT INTO prm_status (id, code, type, value) VALUES
  (1, 'RECHARGE',     'BUDGET',   'Rechargement de quota'),
  (2, 'CONSOMMATION', 'BUDGET',   'Consommation de quota'),
  (3, 'DRAFT',        'CAMPAIGN', 'Brouillon'),
  (4, 'SENT',         'CAMPAIGN', 'Envoyée'),
  (5, 'SCHEDULED',    'CAMPAIGN', 'Planifiée');

-- ─── Organisation unique ──────────────────────────────────────────
INSERT INTO organization (id, name, description, quota) VALUES
  (1, 'TritUX Demo', 'Organisation de test pour le pipeline ML', 5000000);

-- ─── 3 groupes : profils différents pour stresser le modèle ────────
-- grp 1 (Commercial)  : fort volume, consommation régulière
-- grp 2 (Marketing)   : volume moyen, très saisonnier (Ramadan/Aïd)
-- grp 3 (RH)          : petit volume, pics ponctuels (rentrée, Réveillon)
INSERT INTO groupe (id, name, description, quota, quota_loked, quota_free,
                    status_id, organization_id, created_at, updated_at) VALUES
  (1, 'Commercial',  'Relation client',             1500000, 200000, 1300000, 1, 1, '2024-01-01 00:00:00', '2026-03-31 00:00:00'),
  (2, 'Marketing',   'Campagnes promo saisonniere', 2000000, 150000, 1850000, 1, 1, '2024-01-01 00:00:00', '2026-03-31 00:00:00'),
  (3, 'RH',          'Communication interne',        500000,  50000,  450000, 1, 1, '2024-01-01 00:00:00', '2026-03-31 00:00:00');

-- ─── Types de campagne ────────────────────────────────────────────
INSERT INTO prm_campaign_type (id, code, name) VALUES
  (1, 'PROMO',   'Promotion / commerciale'),
  (2, 'NOTIF',   'Notification / transactionnelle');

-- ─── Permissions : chaque groupe peut envoyer les 2 types ─────────
INSERT INTO campaign_type_permission (id, enabled, groupe_id, campaign_type) VALUES
  (1, TRUE, 1, 1),  -- Commercial / PROMO
  (2, TRUE, 1, 2),  -- Commercial / NOTIF
  (3, TRUE, 2, 1),  -- Marketing / PROMO
  (4, TRUE, 2, 2),  -- Marketing / NOTIF
  (5, TRUE, 3, 1),  -- RH / PROMO
  (6, TRUE, 3, 2);  -- RH / NOTIF


-- ════════════════════════════════════════════════════════════════
-- PARTIE 3 — Génération volumétrique via CTE récursives
-- ════════════════════════════════════════════════════════════════
-- Augmente la limite de récursion (défaut = 1000, on veut ~820 jours)
SET @@cte_max_recursion_depth = 2000;

-- ─── 3.1 Recharges : 1 recharge mensuelle par groupe ──────────────
-- Montants ajustés à la taille du groupe. Format RECHARGE = montant positif.
INSERT INTO budget_history (groupe_id, modification_date, amount, status_id)
WITH RECURSIVE mois AS (
    SELECT DATE('2024-01-05') AS d
    UNION ALL
    SELECT d + INTERVAL 1 MONTH FROM mois WHERE d < DATE('2026-03-31')
)
SELECT g.id, m.d,
       -- Groupe 1 : 80k/mois, Groupe 2 : 100k/mois (×1.5 mars/avril Ramadan), Groupe 3 : 25k/mois
       CASE g.id
           WHEN 1 THEN 80000
           WHEN 2 THEN CASE WHEN MONTH(m.d) IN (3, 4) THEN 150000 ELSE 100000 END
           WHEN 3 THEN CASE WHEN MONTH(m.d) IN (9, 12) THEN 40000 ELSE 25000 END
       END AS amount,
       1 AS status_id
FROM mois m
CROSS JOIN groupe g;


-- ─── 3.2 Consommations quotidiennes ───────────────────────────────
-- Stratégie : 1 ligne par jour et par groupe avec montant dépendant
-- du profil, du mois et de la saisonnalité. Total ≈ 820 × 3 = 2460 lignes.
-- Format CONSOMMATION = montant NÉGATIF (conforme au pipeline : amount.cumsum()).
INSERT INTO budget_history (groupe_id, modification_date, amount, status_id)
WITH RECURSIVE jours AS (
    SELECT DATE('2024-01-01') AS d
    UNION ALL
    SELECT d + INTERVAL 1 DAY FROM jours WHERE d < DATE('2026-03-31')
)
SELECT g.id, j.d,
       CASE g.id
           -- Commercial : ~2500/j constant + léger pic week-ends
           WHEN 1 THEN - (2000 + FLOOR(RAND(j.d+g.id) * 1000)
                                + IF(DAYOFWEEK(j.d) IN (1, 7), 500, 0))
           -- Marketing : très saisonnier (×2.5 en Ramadan/avril, ×1.8 décembre)
           WHEN 2 THEN - (1500 + FLOOR(RAND(j.d+g.id) * 800))
                       * CASE
                             WHEN MONTH(j.d) IN (3, 4) THEN 2.5   -- Ramadan + Aïd
                             WHEN MONTH(j.d) = 6 AND DAY(j.d) BETWEEN 14 AND 18 THEN 2.0  -- Aïd al-Adha 2024
                             WHEN MONTH(j.d) = 12 AND DAY(j.d) >= 20 THEN 1.8  -- Réveillon
                             WHEN MONTH(j.d) = 9 AND DAY(j.d) <= 7  THEN 1.3  -- Rentrée
                             ELSE 1.0
                         END
           -- RH : petit volume, gros pics rentrée (sept) + Réveillon + Fête nationale (20 mars)
           WHEN 3 THEN - (400 + FLOOR(RAND(j.d+g.id) * 200))
                       * CASE
                             WHEN MONTH(j.d) = 9 AND DAY(j.d) <= 10 THEN 3.0
                             WHEN MONTH(j.d) = 12 AND DAY(j.d) >= 24 THEN 2.5
                             WHEN MONTH(j.d) = 3 AND DAY(j.d) BETWEEN 19 AND 22 THEN 2.0
                             ELSE 1.0
                         END
       END AS amount,
       2 AS status_id
FROM jours j
CROSS JOIN groupe g;


-- ─── 3.3 Campagnes : ~60 campagnes réparties avec pics saisonniers ──
-- 1 campagne toutes les 2 semaines pour grp 1 (les plus réguliers)
-- 1 campagne tous les 10 jours pour grp 2 en période saison (mars-avril, déc), sinon mensuelle
-- 1 campagne par mois pour grp 3, +pics à la rentrée
INSERT INTO campagne (libelle, description, date_debut, date_fin, dure_validite,
                      cost, budget_used, nbr_page, count_contact,
                      deactivated_by_group, hash, me_only,
                      last_updated_status_at, created_at, updated_at,
                      status_id, campaign_type_permission_id)
WITH RECURSIVE jours AS (
    SELECT DATE('2024-01-10') AS d
    UNION ALL
    SELECT d + INTERVAL 1 DAY FROM jours WHERE d < DATE('2026-03-15')
),
candidats AS (
    -- Groupe 1 : toutes les 14 jours, volume stable
    SELECT 1 AS gid, 1 AS ctp_id, d,
           CONCAT('Promo Commercial ', DATE_FORMAT(d, '%Y-%m-%d')) AS libelle,
           3000 + FLOOR(RAND(UNIX_TIMESTAMP(d)+1) * 1000) AS contacts
    FROM jours WHERE DATEDIFF(d, '2024-01-10') % 14 = 0
    UNION ALL
    -- Groupe 2 : tous les 10 j en mars-avril ou décembre, sinon 1er du mois
    SELECT 2, 3, d,
           CONCAT('Campagne Marketing ', DATE_FORMAT(d, '%Y-%m-%d')),
           (5000 + FLOOR(RAND(UNIX_TIMESTAMP(d)+2) * 2000))
           * CASE WHEN MONTH(d) IN (3, 4, 12) THEN 2 ELSE 1 END
    FROM jours
    WHERE (MONTH(d) IN (3, 4, 12) AND DAY(d) % 10 = 1)
       OR (MONTH(d) NOT IN (3, 4, 12) AND DAY(d) = 1)
    UNION ALL
    -- Groupe 3 : mensuel + rentrée + Réveillon
    SELECT 3, 5, d,
           CONCAT('Notif RH ', DATE_FORMAT(d, '%Y-%m-%d')),
           1000 + FLOOR(RAND(UNIX_TIMESTAMP(d)+3) * 500)
    FROM jours
    WHERE (DAY(d) = 5)
       OR (MONTH(d) = 9 AND DAY(d) = 1)
       OR (MONTH(d) = 12 AND DAY(d) = 28)
)
SELECT libelle,
       CONCAT('Description de ', libelle),
       TIMESTAMP(d, '09:00:00'),
       TIMESTAMP(d + INTERVAL 1 DAY, '18:00:00'),
       7,
       FLOOR(contacts * 15) AS cost,
       FLOOR(contacts * 12) AS budget_used,
       1,
       contacts,
       FALSE,
       MD5(CONCAT(libelle, d)),
       FALSE,
       TIMESTAMP(d, '09:00:00'),
       TIMESTAMP(d, '08:00:00'),
       TIMESTAMP(d, '09:00:00'),
       4,             -- status SENT
       ctp_id
FROM candidats
ORDER BY d;


-- ════════════════════════════════════════════════════════════════
-- PARTIE 4 — Vue de compatibilité pipeline
-- ════════════════════════════════════════════════════════════════
-- Le pipeline Python attend campagne.groupe_id (colonne aplatie).
-- Cette vue fait le JOIN Campagne → CampaignTypePermission → Groupe.
CREATE OR REPLACE VIEW v_campagne_groupe AS
SELECT c.id               AS id,
       ctp.groupe_id      AS groupe_id,
       c.libelle          AS libelle,
       c.date_debut       AS date_debut,
       c.date_fin         AS date_fin,
       c.count_contact    AS count_contact,
       c.budget_used      AS budget_used,
       c.cost             AS cost,
       c.status_id        AS status_id
FROM campagne c
LEFT JOIN campaign_type_permission ctp
       ON ctp.id = c.campaign_type_permission_id;


-- ════════════════════════════════════════════════════════════════
-- PARTIE 5 — Vérifications rapides après exécution
-- ════════════════════════════════════════════════════════════════
-- Exécute ces SELECTs pour vérifier la cohérence :
--
--   SELECT COUNT(*) FROM budget_history;                         -- attendu ~2540 lignes
--   SELECT groupe_id, status_id, COUNT(*), SUM(amount)
--     FROM budget_history GROUP BY groupe_id, status_id;
--   SELECT groupe_id, COUNT(*) FROM v_campagne_groupe
--     GROUP BY groupe_id;
--
-- Invariant métier quota_total = Σ recharges (doit correspondre aux
-- montants dans groupe.quota_free pour chaque groupe).
