"""全局加载串行门(2026-07-06 生产事故:_load_wf_deps + preload_residents 并发
往同卡 spawn 两个 vLLM → engine core init 竞争,embedding 加载失败)。
load_model 的真正 adapter.load() 必须串行,一次只加载一个模型。"""
import asyncio
from unittest.mock import MagicMock

import pytest

from src.runner.fake_adapter import FakeAdapter
from src.services.inference.registry import ModelSpec
from src.services.model_manager import ModelManager


_STATE = {"active": 0, "max": 0}


class _TrackingAdapter(FakeAdapter):
    """记录 load() 的并发峰值,证明全局串行门(峰值应恒为 1)。"""
    async def load(self, device: str) -> None:
        _STATE["active"] += 1
        _STATE["max"] = max(_STATE["max"], _STATE["active"])
        await asyncio.sleep(0.02)  # 模拟耗时 CUDA init
        _STATE["active"] -= 1
        self._model = object()  # 非 None → is_loaded True


def _spec(mid):
    # gpu 显式设,绕开 allocator;vram_mb=0 避免预留路径。
    return ModelSpec(id=mid, model_type="llm", adapter_class="fake",
                     paths={"main": f"/fake/{mid}"}, vram_mb=0, gpu=1)


def _mm(specs):
    reg = MagicMock()
    reg.get = lambda mid: next((s for s in specs if s.id == mid), None)
    reg.add_from_scan = MagicMock(return_value=None)
    return ModelManager(registry=reg, allocator=MagicMock())


@pytest.mark.asyncio
async def test_concurrent_loads_are_serialized():
    _STATE["active"] = 0
    _STATE["max"] = 0
    specs = [_spec("a"), _spec("b"), _spec("c")]
    mm = _mm(specs)

    await asyncio.gather(*[
        mm.load_model(s.id, adapter_factory=lambda _s: _TrackingAdapter(paths={"main": "/x"}))
        for s in specs
    ])
    # 三个并发加载,但同一时刻只允许一个真正 load → max 并发 == 1
    assert _STATE["max"] == 1, f"loads not serialized, peak concurrency={_STATE['max']}"
    assert mm.is_loaded("a") and mm.is_loaded("b") and mm.is_loaded("c")
