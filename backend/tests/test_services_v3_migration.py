"""Static checks on the v3 services migration SQL.

The full Legacy→Grant + WorkflowApp→Service backfill must be replayed
against a real Postgres (manual gate per plan). These tests guard the
*structure* of the migration so accidental edits don't drop a critical
section, and they cover the idempotent guards: each backfill / rename /
constraint addition must be wrapped so a re-run is a no-op.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "migrations"
    / "2026-04-22-services-v3.sql"
)


@pytest.fixture(scope="module")
def sql() -> str:
    assert MIG_PATH.exists(), f"missing migration file at {MIG_PATH}"
    return MIG_PATH.read_text()


def test_single_transaction(sql: str):
    assert "BEGIN;" in sql
    assert "COMMIT;" in sql
    # No nested BEGIN — a re-entrant transaction would crash on PG.
    assert sql.count("BEGIN;") == 1


def test_service_instances_v3_columns(sql: str):
    for col in (
        "workflow_id",
        "workflow_snapshot",
        "exposed_inputs",
        "exposed_outputs",
        "snapshot_hash",
        "snapshot_schema_version",
        "version",
    ):
        assert f"ADD COLUMN IF NOT EXISTS {col}" in sql, f"missing ADD COLUMN for {col}"


def test_workflow_v3_columns(sql: str):
    assert "ADD COLUMN IF NOT EXISTS auto_generated" in sql
    assert "ADD COLUMN IF NOT EXISTS generated_for_service_id" in sql
    assert "fk_workflows_generated_for_service" in sql
    assert "ON DELETE SET NULL" in sql


def test_grant_column_rename_is_guarded(sql: str):
    # column rename
    assert "ALTER TABLE api_key_grants RENAME COLUMN instance_id TO service_id" in sql
    # constraint rename
    assert "RENAME CONSTRAINT uq_grant_key_instance TO uq_grant_key_service" in sql
    # both wrapped in IF EXISTS-style guards
    assert sql.count("DO $$") >= 2


def test_legacy_grant_backfill(sql: str):
    # NULL out instance_id only after the backfill runs.
    backfill_idx = sql.index("INSERT INTO api_key_grants")
    null_idx = sql.index("UPDATE instance_api_keys SET instance_id = NULL")
    rename_idx = sql.index("RENAME COLUMN instance_id TO service_id")
    assert backfill_idx < null_idx < rename_idx, (
        "ordering must be backfill → null → rename so we don't lose data"
    )
    # Stable id derivation (so re-runs collide on the dedup WHERE NOT EXISTS).
    assert "hashtextextended" in sql
    assert "WHERE iak.instance_id IS NOT NULL" in sql
    assert "NOT EXISTS" in sql


def test_workflow_apps_backfill_then_drop(sql: str):
    backfill_idx = sql.index("INSERT INTO service_instances")
    drop_idx = sql.index("DROP TABLE IF EXISTS workflow_apps")
    assert backfill_idx < drop_idx, "must backfill before dropping the source"
    assert "FROM workflow_apps wa" in sql
    assert "WHERE NOT EXISTS" in sql  # idempotent insert


def test_name_constraints_added_after_backfill(sql: str):
    backfill_idx = sql.index("INSERT INTO service_instances")
    constraint_idx = sql.index("ck_service_instances_name_fmt")
    unique_idx = sql.index("uq_service_instances_name")
    assert backfill_idx < constraint_idx, (
        "regex CHECK must be added AFTER backfill or backfilled rows fail"
    )
    assert backfill_idx < unique_idx
    # The actual regex pattern.
    assert "^[a-z][a-z0-9-]{1,62}$" in sql


def test_snapshot_hash_index_is_non_unique(sql: str):
    # The plan calls for non-unique because a single snapshot may back
    # multiple services (dedup hint, not dedup enforcement). We look at
    # the actual CREATE INDEX statement, not the surrounding comments.
    create_line = next(
        line for line in sql.splitlines()
        if "idx_service_snapshot_hash" in line
        and line.lstrip().upper().startswith("CREATE")
    )
    assert "UNIQUE" not in create_line.upper(), (
        "snapshot_hash index must be non-unique to allow dedup hints"
    )
