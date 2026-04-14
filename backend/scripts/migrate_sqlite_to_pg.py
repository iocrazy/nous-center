"""Migrate legacy SQLite data (backend/data/archive/nous.db) into the
PostgreSQL nous_center database.

Usage:
    .venv/bin/python scripts/migrate_sqlite_to_pg.py

Handles:
- BOOLEAN 0/1 -> True/False
- TIMESTAMP ISO strings -> datetime objects
- JSON dict/list -> json.dumps
- FK ordering via SET session_replication_role = replica
- Idempotent via ON CONFLICT (id) DO NOTHING
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import create_async_engine

SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "archive" / "nous.db"
SRC_URL = f"sqlite+aiosqlite:///{SQLITE_PATH}"
DST_URL = "postgresql+asyncpg://nous_heygo:Heygo01!@localhost:5432/nous_center"

TABLES = ["workflows", "service_instances", "instance_api_keys", "voice_presets",
          "voice_preset_groups", "model_metadata", "execution_tasks", "tts_usage",
          "llm_usage", "workflow_apps", "tasks"]


def parse_dt(s):
    if not isinstance(s, str):
        return s
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s


async def main():
    if not SQLITE_PATH.exists():
        print(f"SQLite archive not found: {SQLITE_PATH}")
        return
    src = create_async_engine(SRC_URL)
    dst = create_async_engine(DST_URL)

    async with dst.connect() as dconn:
        def reflect(sync_conn):
            meta = MetaData()
            meta.reflect(bind=sync_conn)
            return {t: {c.name: str(c.type) for c in tbl.columns}
                    for t, tbl in meta.tables.items()}
        schema = await dconn.run_sync(reflect)

    for t in TABLES:
        if t not in schema:
            print(f"{t}: not in PG schema, skip")
            continue
        async with src.connect() as sconn:
            try:
                rows = (await sconn.execute(text(f"SELECT * FROM {t}"))).mappings().all()
            except Exception as e:
                print(f"{t}: sqlite read failed ({e}), skip")
                continue
        if not rows:
            print(f"{t}: 0 rows")
            continue
        types = schema[t]
        cols = list(rows[0].keys())
        placeholders = ", ".join(f":{c}" for c in cols)
        sql = (f"INSERT INTO {t} ({', '.join(cols)}) VALUES ({placeholders}) "
               "ON CONFLICT (id) DO NOTHING")
        async with dst.connect() as dconn, dconn.begin():
            await dconn.execute(text("SET session_replication_role = replica"))
            inserted = 0
            for row in rows:
                d = dict(row)
                for k, v in list(d.items()):
                    ct = types.get(k, "").upper()
                    if "BOOL" in ct and isinstance(v, int):
                        d[k] = bool(v)
                    elif ("TIMESTAMP" in ct or "DATETIME" in ct) and isinstance(v, str):
                        d[k] = parse_dt(v)
                    elif isinstance(v, (dict, list)):
                        d[k] = json.dumps(v)
                await dconn.execute(text(sql), d)
                inserted += 1
            await dconn.execute(text("SET session_replication_role = DEFAULT"))
            cnt = (await dconn.execute(text(f"SELECT COUNT(*) FROM {t}"))).scalar()
            print(f"{t}: attempted {inserted}, PG total {cnt}")

    await src.dispose()
    await dst.dispose()


if __name__ == "__main__":
    asyncio.run(main())
