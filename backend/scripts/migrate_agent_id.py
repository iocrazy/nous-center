"""Add agent_id column to response_sessions and llm_usage.

Idempotent (checks if column exists before ALTER). Run against dev SQLite
or production PG. For SQLite in CI/tests, Base.metadata.create_all handles it.
"""

import asyncio

from sqlalchemy import inspect, text

from src.models.database import create_engine


async def _add_column_if_missing(conn, table: str, column: str, coltype: str):
    def _check(sync_conn):
        insp = inspect(sync_conn)
        cols = [c["name"] for c in insp.get_columns(table)]
        return column in cols

    exists = await conn.run_sync(_check)
    if exists:
        print(f"  {table}.{column} already exists — skipping")
        return
    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))
    await conn.execute(
        text(f"CREATE INDEX IF NOT EXISTS ix_{table}_{column} ON {table}({column})")
    )
    print(f"  {table}.{column} added")


async def main():
    engine = create_engine()
    async with engine.begin() as conn:
        print("Adding agent_id columns...")
        await _add_column_if_missing(conn, "response_sessions", "agent_id", "VARCHAR(128)")
        await _add_column_if_missing(conn, "llm_usage", "agent_id", "VARCHAR(128)")
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
