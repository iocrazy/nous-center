"""PGMemoryProvider — reference implementation.

Uses PG full-text search for content search(全局只用 PostgreSQL)。
注意:models/memory.py 里目前**没有** GIN 索引,to_tsvector 是逐行现算的顺序扫描 ——
数据量上去要补一个 `to_tsvector('simple', content)` 的 GIN 索引(需 alembic 迁移)。
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
        api_key_id=row.api_key_id,
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
        owner_key_id: int,
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
                # legacy rip:按 API key 切作用域;instance_id 不再填(列已降 nullable)。
                rows = [
                    MemoryEntryModel(
                        api_key_id=owner_key_id,
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
        owner_key_id: int,
        query: str,
        limit: int = 10,
        context_key: str | None = None,
    ) -> list[StoredMemoryEntry]:
        # round6:provider 是契约边界,别只靠 route 约束 —— 兜底 clamp 防直接调用方传大值。
        limit = max(1, min(int(limit), 100))
        try:
            async with self._sf() as s:
                stmt = select(MemoryEntryModel).where(
                    MemoryEntryModel.api_key_id == owner_key_id
                )
                if context_key:
                    stmt = stmt.where(MemoryEntryModel.context_key == context_key)
                if query:
                    # 全局只用 PostgreSQL,不再有 sqlite 的 LIKE 兜底分支。
                    # 两侧配置必须一致:vector 用 'simple'(不切词干,对中英混排才正确),
                    # query 若不指定就落 default_text_search_config —— PG 默认是 english,
                    # 会把 "concise" 切成 "concis",与 vector 里的 'concise' 永远对不上
                    # → 搜索静默返回空。测试以前跑 sqlite 的 LIKE 兜底,这条 PG 路径
                    # 从没被执行过(2026-09-02 换 PG 后首次暴露)。
                    stmt = stmt.where(
                        text("to_tsvector('simple', content) "
                             "@@ plainto_tsquery('simple', :q)")
                    ).params(q=query)
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
