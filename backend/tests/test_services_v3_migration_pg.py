"""Real-Postgres regression for the v3 services migration.

PR-A's static `test_services_v3_migration.py` only inspects the SQL file's
shape — it does NOT execute the SQL. The manual gate of PR-B caught a real
idempotency bug that the static test had no way to see (the second run
errored at the legacy-grant backfill INSERT because step 7 had already
renamed the column).

This file codifies the manual gate as an integration test:

  1. Spin up a fresh schema in a real Postgres
  2. Seed v2-shape rows (workflow_apps + instance_api_keys with non-null
     instance_id)
  3. Apply the migration twice
  4. Assert: first run succeeds, second run is a no-op (no errors), and
     post-conditions hold (workflow_apps dropped, grants backfilled,
     api_key_grants.service_id column present)

Skipped by default. To run it:

  PG_TEST_URL=postgresql://user:pwd@localhost:5432/postgres \\
    pytest backend/tests/test_services_v3_migration_pg.py

The URL must point at a server where the test user can `CREATE DATABASE`
(temp test DBs are created + dropped per run so we never touch dev data).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import asyncpg
import pytest

PG_URL = os.environ.get("PG_TEST_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL,
    reason="set PG_TEST_URL=postgresql://... to run the v3 migration PG gate",
)

MIGRATION_SQL = (
    Path(__file__).resolve().parents[1]
    / "migrations"
    / "2026-04-22-services-v3.sql"
).read_text()


# ----- v2-shape DDL the migration assumes already exists ----------------

V2_SETUP_SQL = """
CREATE TABLE service_instances (
  id BIGINT PRIMARY KEY,
  source_type VARCHAR(20) NOT NULL DEFAULT 'preset',
  source_id BIGINT,
  source_name VARCHAR(128),
  name VARCHAR(100) NOT NULL,
  type VARCHAR(20) NOT NULL DEFAULT 'tts',
  status VARCHAR(20) NOT NULL DEFAULT 'active',
  category VARCHAR(20),
  meter_dim VARCHAR(20),
  endpoint_path VARCHAR(200),
  params_override JSON DEFAULT '{}'::json,
  rate_limit_rpm INT,
  rate_limit_tpm INT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE workflows (
  id BIGINT PRIMARY KEY,
  name VARCHAR(100),
  description TEXT,
  nodes JSON DEFAULT '[]'::json,
  edges JSON DEFAULT '[]'::json,
  is_template BOOLEAN DEFAULT FALSE,
  status VARCHAR(20) DEFAULT 'draft',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE workflow_apps (
  id BIGINT PRIMARY KEY,
  name VARCHAR(100) NOT NULL UNIQUE,
  display_name VARCHAR(200),
  description TEXT DEFAULT '',
  workflow_id BIGINT,
  workflow_snapshot JSON DEFAULT '{}'::json,
  active BOOLEAN DEFAULT TRUE,
  exposed_inputs JSON DEFAULT '[]'::json,
  exposed_outputs JSON DEFAULT '[]'::json,
  call_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE instance_api_keys (
  id BIGINT PRIMARY KEY,
  instance_id BIGINT REFERENCES service_instances(id) ON DELETE CASCADE,
  label VARCHAR(100) NOT NULL,
  key_hash VARCHAR(200) NOT NULL,
  key_prefix VARCHAR(20) NOT NULL,
  is_active BOOLEAN DEFAULT TRUE,
  usage_calls INT DEFAULT 0,
  usage_chars INT DEFAULT 0,
  last_used_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now(),
  expires_at TIMESTAMPTZ
);

CREATE TABLE api_key_grants (
  id BIGINT PRIMARY KEY,
  api_key_id BIGINT NOT NULL REFERENCES instance_api_keys(id) ON DELETE CASCADE,
  instance_id BIGINT NOT NULL REFERENCES service_instances(id) ON DELETE CASCADE,
  status VARCHAR(20) NOT NULL DEFAULT 'active',
  activated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  paused_at TIMESTAMPTZ,
  retired_at TIMESTAMPTZ,
  CONSTRAINT uq_grant_key_instance UNIQUE (api_key_id, instance_id)
);
CREATE INDEX ix_grant_active ON api_key_grants (api_key_id, status);

CREATE TABLE resource_packs (
  id BIGINT PRIMARY KEY,
  grant_id BIGINT NOT NULL REFERENCES api_key_grants(id) ON DELETE CASCADE,
  name VARCHAR(100) NOT NULL,
  total_units BIGINT NOT NULL,
  used_units BIGINT NOT NULL DEFAULT 0,
  expires_at TIMESTAMPTZ,
  purchased_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  source VARCHAR(20) NOT NULL DEFAULT 'purchased'
);

CREATE TABLE alert_rules (
  id BIGINT PRIMARY KEY,
  grant_id BIGINT NOT NULL REFERENCES api_key_grants(id) ON DELETE CASCADE,
  threshold_percent INT NOT NULL,
  pack_id BIGINT REFERENCES resource_packs(id) ON DELETE CASCADE,
  enabled BOOLEAN NOT NULL DEFAULT TRUE,
  last_notified_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO service_instances (id, name, source_type, type, status, category)
VALUES (1001, 'preset-llm', 'model', 'llm', 'active', 'llm');

INSERT INTO workflows (id, name, status) VALUES (2001, 'sample-wf', 'published');

INSERT INTO workflow_apps (id, name, display_name, workflow_id,
                           workflow_snapshot, exposed_inputs, exposed_outputs)
VALUES (3001, 'echo-app', 'Echo App', 2001,
        '{"nodes": [], "edges": []}'::json,
        '[{"node_id": "n1", "api_name": "x", "param_key": "value"}]'::json,
        '[]'::json);

INSERT INTO instance_api_keys (id, instance_id, label, key_hash, key_prefix)
VALUES (4001, 1001, 'legacy-key', 'h', 'sk-legacy1');
"""


@pytest.fixture
async def fresh_db():
    """Create + drop a temp database around the test, return connection URL."""
    server = await asyncpg.connect(PG_URL)
    db_name = f"v3_mig_test_{uuid.uuid4().hex[:10]}"
    try:
        await server.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await server.close()

    base = PG_URL.rsplit("/", 1)[0]
    test_url = f"{base}/{db_name}"

    conn = await asyncpg.connect(test_url)
    try:
        await conn.execute(V2_SETUP_SQL)
        yield conn, test_url
    finally:
        await conn.close()
        # Drop the temp DB (server-side, since we're disconnected from it)
        server = await asyncpg.connect(PG_URL)
        try:
            await server.execute(
                f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'
            )
        finally:
            await server.close()


async def _apply_migration(conn: asyncpg.Connection) -> None:
    # asyncpg can't execute multi-statement scripts containing $$ blocks
    # via a simple `await conn.execute(SQL)` reliably — use the simple
    # query protocol via `_protocol` is fragile. The cleanest path is to
    # split on top-level statement boundaries, but the migration uses DO
    # blocks. So execute the whole thing as one query: asyncpg.execute()
    # routes multi-statement SQL via simple_query when there are no $1
    # parameters, which DOES handle DO blocks. Verified locally.
    await conn.execute(MIGRATION_SQL)


@pytest.mark.asyncio
async def test_migration_runs_clean_first_time(fresh_db):
    conn, _ = fresh_db
    await _apply_migration(conn)

    # workflow_apps gone; service_instances now holds the backfilled row.
    assert await conn.fetchval(
        "SELECT to_regclass('public.workflow_apps')"
    ) is None
    backfilled = await conn.fetchval(
        "SELECT count(*) FROM service_instances WHERE source_type='workflow'"
    )
    assert backfilled >= 1

    # api_key_grants column rename applied.
    cols = await conn.fetch("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'api_key_grants'
    """)
    names = {r["column_name"] for r in cols}
    assert "service_id" in names
    assert "instance_id" not in names

    # legacy 1:1 instance_api_key rolled into a grant.
    legacy_key_grants = await conn.fetchval(
        "SELECT count(*) FROM api_key_grants WHERE api_key_id = 4001"
    )
    assert legacy_key_grants == 1
    nullified = await conn.fetchval(
        "SELECT instance_id FROM instance_api_keys WHERE id = 4001"
    )
    assert nullified is None


@pytest.mark.asyncio
async def test_migration_is_idempotent_on_rerun(fresh_db):
    """The bug fixed in PR-B's hotfix commit: the second run used to error
    out at the legacy-grant backfill INSERT because step 7 had already
    renamed `instance_id` → `service_id`. This test re-applies the
    migration N times and asserts no exceptions + stable row counts."""
    conn, _ = fresh_db
    await _apply_migration(conn)
    grants_after_1 = await conn.fetchval("SELECT count(*) FROM api_key_grants")
    services_after_1 = await conn.fetchval("SELECT count(*) FROM service_instances")

    # Re-run twice. Both must succeed without raising.
    await _apply_migration(conn)
    await _apply_migration(conn)

    grants_after_3 = await conn.fetchval("SELECT count(*) FROM api_key_grants")
    services_after_3 = await conn.fetchval("SELECT count(*) FROM service_instances")
    assert grants_after_3 == grants_after_1, (
        "re-running the migration must not duplicate grant rows"
    )
    assert services_after_3 == services_after_1, (
        "re-running must not duplicate service_instances rows"
    )


@pytest.mark.asyncio
async def test_name_normalize_handles_collision(fresh_db):
    """Names that collide after normalization get an id suffix appended,
    so the new UNIQUE constraint stays valid."""
    conn, _ = fresh_db
    # Two rows that normalize to the same name.
    await conn.execute("""
        INSERT INTO service_instances (id, name) VALUES
          (5001, 'My Service'),
          (5002, 'my-service')
    """)
    await _apply_migration(conn)

    rows = await conn.fetch(
        "SELECT id, name FROM service_instances WHERE id IN (5001, 5002)"
    )
    names = sorted(r["name"] for r in rows)
    assert len(names) == 2
    assert all(n.startswith("my-service") for n in names)
    assert names[0] != names[1], "collision must be broken with id suffix"
