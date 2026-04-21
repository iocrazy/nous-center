-- backend/migrations/2026-04-21-api-gateway.sql
-- API gateway · plan 2026-04-21-api-gateway.md
--
-- Order of operations:
--   1. ALTER existing tables (additive, backfill-safe)
--   2. CREATE new tables
--
-- All changes are additive. Existing rows unchanged. Pre-existing
-- InstanceApiKey 1:1 bindings continue to work; new keys use the grants
-- table. Safe to run on a live DB without downtime.

-- ============================================================
-- 1. Extend ServiceInstance with api-gateway metadata
-- ============================================================

ALTER TABLE service_instances
  ADD COLUMN IF NOT EXISTS category  VARCHAR(20),  -- "llm" | "tts" | "vl" | "app"
  ADD COLUMN IF NOT EXISTS meter_dim VARCHAR(20);  -- "tokens" | "chars" | "duration" | "calls"

-- ============================================================
-- 2. Make InstanceApiKey.instance_id nullable (M:N transition)
-- ============================================================

ALTER TABLE instance_api_keys
  ALTER COLUMN instance_id DROP NOT NULL;

-- ============================================================
-- 3. api_key_grants — M:N binding
-- ============================================================

CREATE TABLE IF NOT EXISTS api_key_grants (
    id            BIGINT PRIMARY KEY,                        -- Snowflake
    api_key_id    BIGINT NOT NULL REFERENCES instance_api_keys(id) ON DELETE CASCADE,
    instance_id   BIGINT NOT NULL REFERENCES service_instances(id) ON DELETE CASCADE,
    status        VARCHAR(20) NOT NULL DEFAULT 'active',     -- active | paused | retired
    activated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    paused_at     TIMESTAMPTZ,
    retired_at    TIMESTAMPTZ,
    CONSTRAINT uq_grant_key_instance UNIQUE (api_key_id, instance_id)
);

CREATE INDEX IF NOT EXISTS ix_grant_active ON api_key_grants (api_key_id, status);

-- ============================================================
-- 4. resource_packs — quota pools per grant
-- ============================================================

CREATE TABLE IF NOT EXISTS resource_packs (
    id            BIGINT PRIMARY KEY,                        -- Snowflake
    grant_id      BIGINT NOT NULL REFERENCES api_key_grants(id) ON DELETE CASCADE,
    name          VARCHAR(100) NOT NULL,
    total_units   BIGINT NOT NULL,
    used_units    BIGINT NOT NULL DEFAULT 0,
    expires_at    TIMESTAMPTZ,                               -- null = never
    purchased_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    source        VARCHAR(20) NOT NULL DEFAULT 'purchased',  -- 'purchased' | 'free_trial'
    CONSTRAINT ck_pack_units_nonneg CHECK (used_units >= 0 AND total_units >= 0)
);

CREATE INDEX IF NOT EXISTS ix_pack_grant_active ON resource_packs (grant_id, expires_at);

-- ============================================================
-- 5. alert_rules — percent-of-pack threshold alerts
-- ============================================================

CREATE TABLE IF NOT EXISTS alert_rules (
    id                BIGINT PRIMARY KEY,                    -- Snowflake
    grant_id          BIGINT NOT NULL REFERENCES api_key_grants(id) ON DELETE CASCADE,
    threshold_percent INT NOT NULL,
    pack_id           BIGINT REFERENCES resource_packs(id) ON DELETE CASCADE,
    enabled           BOOLEAN NOT NULL DEFAULT TRUE,
    last_notified_at  TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_alert_threshold_range CHECK (threshold_percent BETWEEN 1 AND 100)
);

CREATE INDEX IF NOT EXISTS ix_alert_grant_enabled ON alert_rules (grant_id, enabled);
