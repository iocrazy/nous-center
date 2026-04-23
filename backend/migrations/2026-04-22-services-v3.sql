-- backend/migrations/2026-04-22-services-v3.sql
-- IA Rebuild v3 · plan docs/designs/2026-04-22-ia-rebuild-v3.md
--
-- Single transaction, idempotent (every step uses IF NOT EXISTS / IF EXISTS).
-- Re-running this file on a migrated DB is a no-op.
--
-- Order of operations:
--   1. ServiceInstance: ADD COLUMNs (workflow_id, snapshot, exposed_*, hash, schema_version, version)
--   2. Workflow:        ADD COLUMNs (auto_generated, generated_for_service_id) + FK
--   3. service_instances.name normalize (so the new CHECK constraint can be validated)
--   4. Backfill workflow_apps -> service_instances
--   5. Backfill instance_api_keys (with non-null instance_id) -> api_key_grants
--   6. NULL out instance_api_keys.instance_id (pre-rename of the FK target column)
--   7. RENAME api_key_grants.instance_id -> service_id (column + FK + unique)
--   8. service_instances: ADD CONSTRAINT name UNIQUE + name regex CHECK
--   9. DROP TABLE workflow_apps
--  10. CREATE INDEX idx_service_snapshot_hash (non-unique, dedup hint)

BEGIN;

-- ============================================================
-- 1. ServiceInstance v3 columns
-- ============================================================

ALTER TABLE service_instances
  ADD COLUMN IF NOT EXISTS workflow_id              BIGINT,
  ADD COLUMN IF NOT EXISTS workflow_snapshot        JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS exposed_inputs           JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS exposed_outputs          JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS snapshot_hash            TEXT,
  ADD COLUMN IF NOT EXISTS snapshot_schema_version  INT  NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS version                  INT  NOT NULL DEFAULT 1;

-- ============================================================
-- 2. Workflow: auto_generated + back-link to its service
-- ============================================================

ALTER TABLE workflows
  ADD COLUMN IF NOT EXISTS auto_generated            BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS generated_for_service_id  BIGINT;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'fk_workflows_generated_for_service'
  ) THEN
    ALTER TABLE workflows
      ADD CONSTRAINT fk_workflows_generated_for_service
      FOREIGN KEY (generated_for_service_id)
      REFERENCES service_instances(id)
      ON DELETE SET NULL;
  END IF;
END $$;

-- ============================================================
-- 3. Normalize service_instances.name to satisfy the new regex
-- ============================================================
-- The check is `^[a-z][a-z0-9-]{1,62}$`. Existing rows may have
-- spaces, mixed case, etc. (the old code allowed VARCHAR(100) free
-- form). We normalize in place; if normalization collides with another
-- row, we suffix with the row id to keep uniqueness.

UPDATE service_instances AS si
SET name = norm
FROM (
  SELECT
    id,
    -- 3a. lower + replace non-[a-z0-9-] with '-'; collapse multiple dashes
    regexp_replace(
      regexp_replace(lower(name), '[^a-z0-9-]', '-', 'g'),
      '-+', '-', 'g'
    ) AS step1
  FROM service_instances
) sub
LEFT JOIN LATERAL (
  -- 3b. ensure starts with [a-z]; if not, prefix 'svc-'
  SELECT CASE
    WHEN sub.step1 ~ '^[a-z]' THEN sub.step1
    ELSE 'svc-' || sub.step1
  END AS step2
) s2 ON TRUE
LEFT JOIN LATERAL (
  -- 3c. trim leading/trailing dashes; clamp length to 63
  SELECT substring(
    regexp_replace(s2.step2, '^-+|-+$', '', 'g')
    FROM 1 FOR 63
  ) AS norm
) s3 ON TRUE
WHERE si.id = sub.id
  AND si.name IS DISTINCT FROM s3.norm;

-- 3d. de-duplicate any names that just collided after normalization
UPDATE service_instances AS si
SET name = substring(si.name FROM 1 FOR 50) || '-' || si.id::text
WHERE EXISTS (
  SELECT 1 FROM service_instances other
  WHERE other.name = si.name AND other.id <> si.id
);

-- ============================================================
-- 4. Backfill: workflow_apps -> service_instances
-- ============================================================
-- Each WorkflowApp becomes a ServiceInstance row with
-- source_type='workflow'. We re-use the workflow_app.id as the
-- service_instance.id so existing api_key_grants pointing at the
-- workflow_app id still match (none currently exist, but this keeps
-- the migration safe under concurrent create-grants).

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'workflow_apps' AND table_schema = current_schema()
  ) THEN
    INSERT INTO service_instances (
      id, source_type, source_id, name, type, status,
      category, meter_dim,
      workflow_id, workflow_snapshot, exposed_inputs, exposed_outputs,
      snapshot_schema_version, version,
      params_override,
      created_at, updated_at
    )
    SELECT
      wa.id,
      'workflow',
      wa.workflow_id,
      -- normalize name same way as step 3
      substring(
        regexp_replace(
          regexp_replace(
            regexp_replace(lower(wa.name), '[^a-z0-9-]', '-', 'g'),
            '-+', '-', 'g'
          ),
          '^-+|-+$', '', 'g'
        )
        FROM 1 FOR 63
      ),
      'inference',
      CASE WHEN wa.active THEN 'active' ELSE 'paused' END,
      'app',
      'calls',
      wa.workflow_id,
      COALESCE(wa.workflow_snapshot::jsonb, '{}'::jsonb),
      COALESCE(wa.exposed_inputs::jsonb,    '[]'::jsonb),
      COALESCE(wa.exposed_outputs::jsonb,   '[]'::jsonb),
      1, 1,
      '{}'::json,
      wa.created_at, wa.updated_at
    FROM workflow_apps wa
    WHERE NOT EXISTS (
      SELECT 1 FROM service_instances si WHERE si.id = wa.id
    );
  END IF;
END $$;

-- ============================================================
-- 5. Backfill: instance_api_keys.instance_id -> api_key_grants
-- ============================================================
-- Legacy 1:1 keys (instance_id NOT NULL) become an explicit grant.
-- Stable id = abs(hashtextextended(api_key||'|'||instance, 0)) so
-- re-runs do not create duplicates.
--
-- Wrapped in a guard: on a re-run the column has already been renamed to
-- `service_id` (step 7), so PG would reject the INSERT at parse time.
-- We branch on the actual schema state so the second run is a no-op.

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'api_key_grants' AND column_name = 'instance_id'
  ) THEN
    EXECUTE $sql$
      INSERT INTO api_key_grants (
        id, api_key_id, instance_id, status, activated_at
      )
      SELECT
        abs(hashtextextended(iak.id::text || '|' || iak.instance_id::text, 0)),
        iak.id,
        iak.instance_id,
        'active',
        COALESCE(iak.created_at, now())
      FROM instance_api_keys iak
      WHERE iak.instance_id IS NOT NULL
        AND NOT EXISTS (
          SELECT 1 FROM api_key_grants g
          WHERE g.api_key_id = iak.id AND g.instance_id = iak.instance_id
        )
    $sql$;
  END IF;
END $$;

-- ============================================================
-- 6. Null out the legacy 1:1 column on instance_api_keys
-- ============================================================
-- (Grants now own the binding. Done before column rename so we don't
-- have to chase a deprecated column name.)

UPDATE instance_api_keys SET instance_id = NULL WHERE instance_id IS NOT NULL;

-- ============================================================
-- 7. Rename api_key_grants.instance_id -> service_id
-- ============================================================
-- PG keeps the FK and column-default attached automatically.

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'api_key_grants' AND column_name = 'instance_id'
  ) THEN
    ALTER TABLE api_key_grants RENAME COLUMN instance_id TO service_id;
  END IF;
END $$;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'uq_grant_key_instance'
  ) THEN
    ALTER TABLE api_key_grants RENAME CONSTRAINT uq_grant_key_instance TO uq_grant_key_service;
  END IF;
END $$;

-- The default index name for a column rename is auto-managed by PG; we
-- keep the explicit api-gateway index name (ix_grant_active) untouched
-- (it indexes (api_key_id, status), not the renamed column).

-- ============================================================
-- 8. service_instances name uniqueness + format
-- ============================================================

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'uq_service_instances_name'
  ) THEN
    ALTER TABLE service_instances
      ADD CONSTRAINT uq_service_instances_name UNIQUE (name);
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'ck_service_instances_name_fmt'
  ) THEN
    ALTER TABLE service_instances
      ADD CONSTRAINT ck_service_instances_name_fmt
      CHECK (name ~ '^[a-z][a-z0-9-]{1,62}$');
  END IF;
END $$;

-- ============================================================
-- 9. Drop workflow_apps (now living as service_instances rows)
-- ============================================================

DROP TABLE IF EXISTS workflow_apps;

-- ============================================================
-- 10. snapshot_hash dedup index (non-unique on purpose)
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_service_snapshot_hash
  ON service_instances (snapshot_hash);

COMMIT;
