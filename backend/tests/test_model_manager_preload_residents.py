"""Lane H: ModelManager.preload_residents 测试 —— preload_order 升序 + fail-soft。"""
import pytest
from unittest.mock import MagicMock

from src.services.inference.registry import ModelSpec
from src.services.model_manager import ModelManager


def _spec(model_id, *, resident=True, preload_order=None, model_type="image"):
    return ModelSpec(
        id=model_id, model_type=model_type, adapter_class="fake",
        paths={"main": f"/fake/{model_id}"}, vram_mb=1024,
        resident=resident, preload_order=preload_order,
    )


def _make_manager(specs):
    registry = MagicMock()
    registry.specs = specs
    registry.get = lambda mid: next((s for s in specs if s.id == mid), None)
    registry.add_from_scan = MagicMock(return_value=None)
    allocator = MagicMock()
    return ModelManager(registry=registry, allocator=allocator)


@pytest.mark.asyncio
async def test_preload_residents_orders_by_preload_order():
    """resident 模型按 preload_order 升序 load；preload_order=None 排最后。"""
    specs = [
        _spec("late", preload_order=30),
        _spec("none-a"),                       # preload_order None
        _spec("early", preload_order=10),
        _spec("mid", preload_order=20),
        _spec("none-b"),                       # preload_order None
    ]
    mm = _make_manager(specs)
    load_order: list[str] = []

    async def _fake_load(model_id, **kw):
        load_order.append(model_id)

    mm.load_model = _fake_load
    await mm.preload_residents()
    # 有序的在前（10/20/30），None 的在最后（保持 registry FIFO）
    assert load_order[:3] == ["early", "mid", "late"]
    assert set(load_order[3:]) == {"none-a", "none-b"}


@pytest.mark.asyncio
async def test_preload_residents_skips_non_resident():
    """resident:false 的模型不 preload。"""
    specs = [
        _spec("res", preload_order=10, resident=True),
        _spec("transient", preload_order=5, resident=False),
    ]
    mm = _make_manager(specs)
    loaded: list[str] = []

    async def _fake_load(model_id, **kw):
        loaded.append(model_id)

    mm.load_model = _fake_load
    await mm.preload_residents()
    assert loaded == ["res"]


@pytest.mark.asyncio
async def test_preload_residents_fail_soft_does_not_block():
    """单个模型 load 失败 → 写 _load_failures + 继续下一个，不向上抛。"""
    specs = [
        _spec("ok-1", preload_order=10),
        _spec("boom", preload_order=20),
        _spec("ok-2", preload_order=30),
    ]
    mm = _make_manager(specs)
    loaded: list[str] = []

    async def _fake_load(model_id, **kw):
        if model_id == "boom":
            raise RuntimeError("CUDA out of memory (simulated)")
        loaded.append(model_id)

    mm.load_model = _fake_load
    # 不抛异常 —— fail-soft
    await mm.preload_residents()
    # boom 失败但 ok-1 / ok-2 仍 load 了
    assert loaded == ["ok-1", "ok-2"]
    # 失败记录进 _load_failures
    assert "boom" in mm._load_failures
    assert "out of memory" in mm._load_failures["boom"].lower()


@pytest.mark.asyncio
async def test_preload_residents_invokes_on_loaded_callback():
    """每个成功 load 的模型触发 on_loaded(model_id) 回调（main.py 用它 invalidate cache + 推 ws）。"""
    specs = [_spec("a", preload_order=10), _spec("b", preload_order=20)]
    mm = _make_manager(specs)

    async def _fake_load(model_id, **kw):
        pass

    mm.load_model = _fake_load
    notified: list[str] = []

    async def _on_loaded(model_id):
        notified.append(model_id)

    await mm.preload_residents(on_loaded=_on_loaded)
    assert notified == ["a", "b"]


@pytest.mark.asyncio
async def test_preload_residents_callback_failure_is_swallowed():
    """on_loaded 回调本身抛异常 → 不影响后续 preload（回调是 best-effort）。"""
    specs = [_spec("a", preload_order=10), _spec("b", preload_order=20)]
    mm = _make_manager(specs)

    async def _fake_load(model_id, **kw):
        pass

    mm.load_model = _fake_load

    async def _bad_callback(model_id):
        raise RuntimeError("ws broadcast failed")

    # 不抛 —— 回调失败被吞
    await mm.preload_residents(on_loaded=_bad_callback)
