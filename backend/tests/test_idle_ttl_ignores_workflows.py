"""spec 2026-09-05 §7:非常驻模型即使被已发布工作流"依赖",只要没有活跃引用就按 TTL 回收;
请求期的 proxy_ref 仍能挡住回收。直接驱动 ModelManager.check_idle_models,不起任何子进程。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.services.model_manager import ModelManager


def _mgr_with(entry_last_used: float, refs: set[str]) -> ModelManager:
    mgr = ModelManager.__new__(ModelManager)          # 绕过 __init__(它会拉 GPU 探测)
    mgr._models = {
        "qwen3_6_35b_a3b_fp8": SimpleNamespace(
            spec=SimpleNamespace(resident=False, ttl_seconds=300),
            last_used=entry_last_used,
            adapter=SimpleNamespace(is_loaded=True),
        )
    }
    mgr._references = {"qwen3_6_35b_a3b_fp8": set(refs)}
    mgr.unload_model = AsyncMock(return_value=True)
    return mgr


@pytest.mark.asyncio
async def test_idle_non_resident_model_is_reclaimed_without_refs(monkeypatch):
    import time
    monkeypatch.setattr(time, "monotonic", lambda: 10_000.0)
    mgr = _mgr_with(entry_last_used=10_000.0 - 301, refs=set())
    await mgr.check_idle_models()
    mgr.unload_model.assert_awaited_once_with("qwen3_6_35b_a3b_fp8")


@pytest.mark.asyncio
async def test_proxy_ref_still_blocks_reclaim(monkeypatch):
    import time
    monkeypatch.setattr(time, "monotonic", lambda: 10_000.0)
    mgr = _mgr_with(entry_last_used=10_000.0 - 301, refs={"proxy-deadbeef"})
    await mgr.check_idle_models()
    mgr.unload_model.assert_not_awaited()
