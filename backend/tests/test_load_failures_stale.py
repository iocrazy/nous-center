"""H2:stale _load_failures 不该挡住已加载的健康模型(虚假 503)。
并发 load:A 成功、B 失败写 failure → 后续调用应返回 A 加载的 adapter,不抛。"""
import pytest
from unittest.mock import MagicMock

from src.runner.fake_adapter import FakeAdapter
from src.services.inference.registry import ModelSpec
from src.services.model_manager import LoadedModel, ModelManager


def _mm():
    reg = MagicMock()
    reg.get = MagicMock(return_value=None)
    reg.add_from_scan = MagicMock(return_value=None)
    return ModelManager(registry=reg, allocator=MagicMock())


def _loaded_entry(mid):
    ad = FakeAdapter(paths={"main": "/x"})
    ad._model = object()  # is_loaded True
    spec = ModelSpec(id=mid, model_type="llm", adapter_class="fake", paths={"main": "/x"}, vram_mb=0)
    return LoadedModel(spec=spec, adapter=ad, gpu_index=0, gpu_indices=[0])


@pytest.mark.asyncio
async def test_get_loaded_adapter_ignores_stale_failure_when_loaded():
    mm = _mm()
    mm._models["m"] = _loaded_entry("m")
    mm._load_failures["m"] = "stale(并发另一个尝试失败写的)"
    ad = await mm.get_loaded_adapter("m")
    assert ad.is_loaded
    assert "m" not in mm._load_failures  # stale 已清


@pytest.mark.asyncio
async def test_get_or_load_ignores_stale_failure_when_loaded():
    mm = _mm()
    mm._models["m"] = _loaded_entry("m")
    mm._load_failures["m"] = "stale"
    ad = await mm.get_or_load("m")
    assert ad.is_loaded
    assert "m" not in mm._load_failures


@pytest.mark.asyncio
async def test_load_failure_still_raises_when_not_loaded():
    """真失败(模型没加载)时仍抛 —— 不能因为改动把真失败也吞了。"""
    from src.services.model_manager import ModelLoadError
    mm = _mm()
    mm._load_failures["m"] = "real failure"
    with pytest.raises(ModelLoadError):
        await mm.get_loaded_adapter("m")
