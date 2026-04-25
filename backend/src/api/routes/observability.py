"""Process-level runtime metrics endpoint.

`GET /api/v1/observability/runtime` 返回 in-memory 累计指标 — gzip 压缩比、
context engine compaction 开销、context cache 命中率。Re-zeros on
process restart（不持久化），m04 dashboard 用这个画"运行时观测"卡片。

`GET /api/v1/observability/vllm` 返回所有活跃 vLLM 实例的 KV cache + 调度
状态（从每个实例的 :PORT/metrics 抓 Prometheus 文本解析）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.deps_admin import require_admin
from src.services.runtime_metrics import RuntimeSnapshot, snapshot
from src.services.vllm_metrics import snapshot_all as vllm_snapshot_all

router = APIRouter(prefix="/api/v1/observability", tags=["observability"])


@router.get(
    "/runtime",
    response_model=None,  # TypedDict — pydantic v2 schema gen 不友好，直接返 dict
    dependencies=[Depends(require_admin)],
)
async def runtime_metrics() -> RuntimeSnapshot:
    return snapshot()


@router.get("/vllm", dependencies=[Depends(require_admin)])
async def vllm_metrics(request: Request):
    """Scrape /metrics from every active vLLM instance.

    Returns one entry per loaded model. Each entry has ``config`` (block_size,
    gpu_memory_utilization, enable_prefix_caching, num_gpu_blocks, ...) and
    ``stats`` (kv_cache_usage_perc, running, waiting, prefix hit rate, ...).
    Unhealthy instances return ``healthy=false`` with an ``error`` string.
    """
    mgr = getattr(request.app.state, "model_manager", None)
    if mgr is None:
        return {"instances": []}

    targets: list[tuple[str, int]] = []
    for mid, entry in getattr(mgr, "_models", {}).items():
        adapter = entry.adapter
        if not getattr(adapter, "is_loaded", False):
            continue
        port = getattr(adapter, "_port", None) or getattr(adapter, "port", None)
        if port:
            targets.append((mid, int(port)))

    return {"instances": await vllm_snapshot_all(targets)}
