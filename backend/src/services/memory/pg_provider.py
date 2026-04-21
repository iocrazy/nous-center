"""PGMemoryProvider — reference implementation.

Uses PG FTS (GIN) for content search. Works on SQLite via LIKE fallback
(detected by dialect) — for dev/test convenience.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from sqlalchemy import desc, select, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.memory import MemoryEntryModel
from src.services.memory.base import (
    MemoryEntry,
    MemoryProvider,
    MemoryProviderClientError,
    MemoryProviderInternalError,
    StoredMemoryEntry,
)

logger = logging.getLogger(__name__)

MAX_ENTRY_BYTES = 10 * 1024
MAX_BATCH_SIZE = 100


def _to_stored_entry(row: MemoryEntryModel) -> StoredMemoryEntry:
    return StoredMemoryEntry(
        id=row.id,
        instance_id=row.instance_id,
        category=row.category,
        content=row.content,
        context_key=row.context_key,
        created_at=row.created_at.isoformat(),
    )


class PGMemoryProvider(MemoryProvider):
    name = "pg"

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._sf = session_factory

    async def initialize(self) -> None:
        """fail-fast: ensure table exists (migration applied)."""
        async with self._sf() as s:
            try:
                await s.execute(text("SELECT 1 FROM memory_entries LIMIT 1"))
            except ProgrammingError as e:
                raise RuntimeError(
                    "memory_entries table not found — run wave1_memory.sql migration"
                ) from e

    async def shutdown(self) -> None:
        pass  # session factory managed externally

    async def add_entries(
        self,
        *,
        instance_id: int,
        api_key_id: int | None,
        entries: list[MemoryEntry],
        context_key: str | None = None,
    ) -> list[int]:
        if not entries:
            return []

        if len(entries) > MAX_BATCH_SIZE:
            raise MemoryProviderClientError(
                f"entries exceeds max batch size {MAX_BATCH_SIZE}"
            )

        for i, e in enumerate(entries):
            if len(e.get("content", "").encode()) > MAX_ENTRY_BYTES:
                raise MemoryProviderClientError(
                    f"entries[{i}].content exceeds {MAX_ENTRY_BYTES} bytes"
                )

        try:
            async with self._sf() as s:
                # Per-entry context_key takes precedence over the batch-level
                # parameter (mirrors _FakeMemoryProvider contract in W-T2.2).
                rows = [
                    MemoryEntryModel(
                        instance_id=instance_id,
                        api_key_id=api_key_id,
                        category=e["category"],
                        content=e["content"],
                        context_key=e.get("context_key") if e.get("context_key") is not None else context_key,
                    )
                    for e in entries
                ]
                s.add_all(rows)
                await s.flush()
                new_ids = [r.id for r in rows]
                await s.commit()
                return new_ids
        except (DBAPIError, asyncio.TimeoutError) as exc:
            raise MemoryProviderInternalError(str(exc)) from exc

    async def prefetch(
        self,
        *,
        instance_id: int,
        query: str,
        limit: int = 10,
        context_key: str | None = None,
    ) -> list[StoredMemoryEntry]:
        try:
            async with self._sf() as s:
                stmt = select(MemoryEntryModel).where(
                    MemoryEntryModel.instance_id == instance_id
                )
                if context_key:
                    stmt = stmt.where(MemoryEntryModel.context_key == context_key)
                if query:
                    dialect = s.bind.dialect.name if s.bind else "sqlite"
                    if dialect == "postgresql":
                        stmt = stmt.where(
                            text("to_tsvector('simple', content) @@ plainto_tsquery(:q)")
                        ).params(q=query)
                    else:
                        stmt = stmt.where(MemoryEntryModel.content.contains(query))
                stmt = stmt.order_by(desc(MemoryEntryModel.created_at)).limit(limit)
                rows = (await s.execute(stmt)).scalars().all()
                return [_to_stored_entry(r) for r in rows]
        except (DBAPIError, asyncio.TimeoutError) as exc:
            logger.warning("PGMemoryProvider.prefetch failed: %s; returning empty", exc)
            return []

    async def system_prompt_block(self, *, instance_id: int) -> str:
        return (
            "You have access to long-term memory for this user "
            "(managed by the platform)."
        )
