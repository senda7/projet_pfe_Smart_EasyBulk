-- ════════════════════════════════════════════════════════════════
-- Schéma v2 — Smart SMS Predictor / EasyBulk
-- Base : MySQL 8 (XAMPP) — DB nommée `budget`
--
-- DIFFÉRENCES avec schema.sql v1 :
--   • Nom de la base : easybulk_ml → budget
--   • +1 table : predictions  (résultats du ML predict.py)
--   • +1 table : notifications (cloche d'alerte UI)
--
-- Comment l'utiliser :
--   1. Démarre XAMPP, ouvre phpMyAdmin (http://localhost/phpmyadmin)
--   2. Importe ce fichier (ou copie-colle dans l'onglet SQL)
--   3. Vérifie que la base `budget` est créée avec ses ~9 tables
-- ════════════════════════════════════════════════════════════════

CREATE DATABASE IF NOT EXISTS budget
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE budget;

-- Pour relances idempotentes pendant les tests
SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS notifications;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS campagne;
DROP TABLE IF EXISTS campaign_type_permission;
DROP TABLE IF EXISTS prm_campaign_type;
DROP TABLE IF EXISTS budget_history;
DROP TABLE IF EXISTS groupe;
DROP TABLE IF EXISTS organization;
DROP TABLE IF EXISTS prm_status;
SET FOREIGN_KEY_CHECKS = 1;


-- ════════════════════════════════════════════════════════════════
-- PARTIE 1 — DDL (9 tables)
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
    name            VARCHAR(50)  NOT NULL,
    description     VARCHAR(250) NULL,
    quota           BIGINT       NOT NULL,
    quota_loked     BIGINT       NULL DEFAULT 0,
    quota_free      BIGINT       NULL,
    status_id       INT          NOT NULL,
    organization_id INT          NOT NULL,
    admin_id        INT          NULL,
    -- Champs ajoutés v2 : utiles à l'UI
    entete_alpha    JSON         NULL,        -- ex: ["Solde","85811","Promo"]
    type_campagne   JSON         NULL,        -- ex: ["classique","transactionnelle"]
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
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
-- NOUVELLES TABLES v2
-- ════════════════════════════════════════════════════════════════

-- Stocke les résultats du pipeline ML (predict.py).
-- Une ligne = 1 prédiction d'1 horizon pour 1 groupe.
-- On garde l'historique pour comparer dans le temps.
CREATE TABLE predictions (
    id                 BIGINT       NOT NULL AUTO_INCREMENT,
    groupe_id          INT          NOT NULL,
    horizon_jours      INT          NOT NULL,            -- 7, 14, ou 30
    conso_prevue       DECIMAL(12,2) NOT NULL,           -- moyenne des 200 arbres
    conso_prudente     DECIMAL(12,2) NOT NULL,           -- moyenne + 0.84 × std
    niveau_risque      ENUM('SAFE','DANGER','CRITIQUE') NOT NULL,
    a_min              DECIMAL(12,2) NULL,               -- montant min à recharger
    a_reco             DECIMAL(12,2) NULL,               -- montant recommandé
    jours_avant_zero   INT          NULL,                -- estimation
    quota_libre_snapshot BIGINT     NOT NULL,            -- état du quota au moment du calcul
    created_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_pred_groupe_horizon (groupe_id, horizon_jours, created_at DESC),
    CONSTRAINT fk_pred_groupe FOREIGN KEY (groupe_id) REFERENCES groupe(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Cloche de notifications dans l'UI.
-- Une notification est créée automatiquement quand un groupe passe en DANGER ou CRITIQUE.
CREATE TABLE notifications (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    organization_id INT          NOT NULL,
    groupe_id       INT          NULL,                   -- NULL = notif globale (pas liée à 1 groupe)
    type            ENUM('INFO','WARNING','CRITICAL') NOT NULL,
    message         VARCHAR(500) NOT NULL,
    is_read         BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_notif_org_unread (organization_id, is_read, created_at DESC),
    CONSTRAINT fk_notif_org    FOREIGN KEY (organization_id) REFERENCES organization(id),
    CONSTRAINT fk_notif_groupe FOREIGN KEY (groupe_id)       REFERENCES groupe(id) ON DELETE CASCADE
) ENGINE=InnoDB;


-- ════════════════════════════════════════════════════════════════
-- PARTIE 2 — SEEDS : données de test
-- ════════════════════════════════════════════════════════════════

-- ─── Statuts ─────────────────────────────────────────────────────
INSERT INTO prm_status (id, code, type, value) VALUES
  (1, 'RECHARGE',     'BUDGET',   'Rechargement de quota'),
  (2, 'CONSOMMATION', 'BUDGET',   'Consommation de quota'),
  (3, 'DRAFT',        'CAMPAIGN', 'Brouillon'),
  (4, 'SENT',         'CAMPAIGN', 'Envoyée'),
  (5, 'SCHEDULED',    'CAMPAIGN', 'Planifiée');

-- ─── Organisation ────────────────────────────────────────────────
INSERT INTO organization (id, name, description, quota) VALUES
  (1, 'TritUX Demo', 'Organisation de test pour le pipeline ML', 5000000);

-- ─── 3 groupes initiaux ──────────────────────────────────────────
INSERT INTO groupe (id, name, description, quota, quota_loked, quota_free,
                    status_id, organization_id, entete_alpha, type_campagne, is_active,
                    created_at, updated_at) VALUES
  (1, 'Commercial',  'Relation client',
      1500000, 200000, 1300000, 1, 1,
      JSON_ARRAY('Solde','85811'), JSON_ARRAY('classique','transactionnelle'),
      TRUE, '2024-01-01 00:00:00', '2026-03-31 00:00:00'),
  (2, 'Marketing',   'Campagnes promo saisonniere',
      2000000, 150000, 1850000, 1, 1,
      JSON_ARRAY('Promo','MKT'), JSON_ARRAY('classique','transactionnelle'),
      TRUE, '2024-01-01 00:00:00', '2026-03-31 00:00:00'),
  (3, 'RH',          'Communication interne',
      500000,  50000,  450000, 1, 1,
      JSON_ARRAY('RH-Info'), JSON_ARRAY('classique'),
      TRUE, '2024-01-01 00:00:00', '2026-03-31 00:00:00');

-- ─── Types de campagne ───────────────────────────────────────────
INSERT INTO prm_campaign_type (id, code, name) VALUES
  (1, 'PROMO',   'Promotion / commerciale'),
  (2, 'NOTIF',   'Notification / transactionnelle');

-- ─── Permissions ─────────────────────────────────────────────────
INSERT INTO campaign_type_permission (id, enabled, groupe_id, campaign_type) VALUES
  (1, TRUE, 1, 1),
  (2, TRUE, 1, 2),
  (3, TRUE, 2, 1),
  (4, TRUE, 2, 2),
  (5, TRUE, 3, 1),
  (6, TRUE, 3, 2);


-- ════════════════════════════════════════════════════════════════
-- PARTIE 3 — Génération volumétrique (recharges + conso + campagnes)
-- (identique à v1, juste recopiée pour avoir une base autonome)
-- ════════════════════════════════════════════════════════════════
SET @@cte_max_recursion_depth = 2000;

-- 3.1 Recharges mensuelles
INSERT INTO budget_history (groupe_id, modification_date, amount, status_id)
WITH RECURSIVE mois AS (
    SELECT DATE('2024-01-05') AS d
    UNION ALL
    SELECT d + INTERVAL 1 MONTH FROM mois WHERE d < DATE('2026-03-31')
)
SELECT g.id, m.d,
       CASE g.id
           WHEN 1 THEN 80000
           WHEN 2 THEN CASE WHEN MONTH(m.d) IN (3, 4) THEN 150000 ELSE 100000 END
           WHEN 3 THEN CASE WHEN MONTH(m.d) IN (9, 12) THEN 40000 ELSE 25000 END
       END AS amount,
       1 AS status_id
FROM mois m
CROSS JOIN groupe g;

-- 3.2 Consommations quotidiennes
INSERT INTO budget_history (groupe_id, modification_date, amount, status_id)
WITH RECURSIVE jours AS (
    SELECT DATE('2024-01-01') AS d
    UNION ALL
    SELECT d + INTERVAL 1 DAY FROM jours WHERE d < DATE('2026-03-31')
)
SELECT g.id, j.d,
       CASE g.id
           WHEN 1 THEN - (2000 + FLOOR(RAND(j.d+g.id) * 1000)
                                + IF(DAYOFWEEK(j.d) IN (1, 7), 500, 0))
           WHEN 2 THEN - (1500 + FLOOR(RAND(j.d+g.id) * 800))
                       * CASE
                             WHEN MONTH(j.d) IN (3, 4) THEN 2.5
                             WHEN MONTH(j.d) = 6 AND DAY(j.d) BETWEEN 14 AND 18 THEN 2.0
                             WHEN MONTH(j.d) = 12 AND DAY(j.d) >= 20 THEN 1.8
                             WHEN MONTH(j.d) = 9 AND DAY(j.d) <= 7  THEN 1.3
                             ELSE 1.0
                         END
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


-- ════════════════════════════════════════════════════════════════
-- PARTIE 4 — Vue compatibilité pipeline ML
-- ════════════════════════════════════════════════════════════════
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
-- PARTIE 5 — Vérifications post-import
-- ════════════════════════════════════════════════════════════════
-- SELECT COUNT(*) FROM groupe;             -- attendu 3
-- SELECT COUNT(*) FROM budget_history;     -- attendu ~2540 (28 mois × 3 groupes recharges + 820 jours × 3 conso)
-- SELECT COUNT(*) FROM predictions;        -- attendu 0 (vide jusqu'à ce que predict.py l'alimente)
-- SELECT COUNT(*) FROM notifications;      -- attendu 0
