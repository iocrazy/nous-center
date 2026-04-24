"""Process-level runtime metrics endpoint.

`GET /api/v1/observability/runtime` 返回 in-memory 累计指标 — gzip 压缩比、
context engine compaction 开销、context cache 命中率。Re-zeros on
process restart（不持久化），m04 dashboard 用这个画"运行时观测"卡片。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps_admin import require_admin
from src.services.runtime_metrics import RuntimeSnapshot, snapshot

router = APIRouter(prefix="/api/v1/observability", tags=["observability"])


@router.get(
    "/runtime",
    response_model=None,  # TypedDict — pydantic v2 schema gen 不友好，直接返 dict
    dependencies=[Depends(require_admin)],
)
async def runtime_metrics() -> RuntimeSnapshot:
    return snapshot()
