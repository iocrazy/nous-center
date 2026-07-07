"""Lane D: ModelManager.get_or_load —— OOM evict + 重试一次（spec §4.3）。

不碰真 GPU：用一个 stub adapter + stub registry/allocator，OOM 用一个类名含
'OutOfMemoryError' 的异常模拟（runner venv 测试里 torch 是 MagicMock，不能
import torch.cuda.OutOfMemoryError）。
"""
import pytest

from src.errors import ModelLoadError
from src.services.inference.base import (
    InferenceAdapter, InferenceResult, MediaModality, UsageMeter,
)
from src.services.inference.registry import ModelRegistry, ModelSpec
from src.services.model_manager import ModelManager


class _CudaOutOfMemoryError(RuntimeError):
    """类名含 OutOfMemoryError —— get_or_load 据类名判定 OOM 路径。"""


class _StubAdapter(InferenceAdapter):
    modality = MediaModality.IMAGE
    estimated_vram_mb = 0

    def __init__(self, paths, device="cpu", *, oom_loads=0, **params):
        super().__init__(paths, device, **params)
        self._oom_loads = oom_loads
        self._load_calls = 0

    async def load(self, device):
        self._load_calls += 1
        if self._load_calls <= self._oom_loads:
            raise _CudaOutOfMemoryError("CUDA out of memory")
        self.device = device
        self._model = object()

    async def infer(self, req):
        return InferenceResult(
            media_type="image/png", data=b"x", metadata={},
            usage=UsageMeter(latency_ms=1, image_count=1),
        )


class _StubRegistry(ModelRegistry):
    """不读 yaml —— 直接塞 spec。"""
    def __init__(self, specs):
        self._config_path = ""
        self._specs = {s.id: s for s in specs}


class _StubAllocator:
    def get_best_gpu(self, vram_mb, *, reserve=True):
        return 0

    def release_reservation(self, gpu_index, vram_mb):
        pass


def _spec(model_id, *, resident=False):
    return ModelSpec(
        id=model_id, model_type="image",
        adapter_class="tests.test_model_manager_get_or_load._StubAdapter",
        paths={"main": f"/fake/{model_id}"}, vram_mb=1024, resident=resident,
    )


def _mm(specs):
    """构造 ModelManager，注入 stub registry + allocator。"""
    reg = _StubRegistry(specs)
    mm = ModelManager(registry=reg, allocator=_StubAllocator())
    return mm


@pytest.mark.asyncio
async def test_get_or_load_fast_path_when_already_loaded():
    """已加载 → get_or_load 直接返回，不重新 load。"""
    mm = _mm([_spec("m1")])
    a = _StubAdapter(paths={"main": "/fake/m1"})
    await mm.load_model("m1", adapter_factory=lambda spec: a)
    again = await mm.get_or_load("m1")
    assert again is a
    assert a._load_calls == 1  # 没有第二次 load


@pytest.mark.asyncio
async def test_get_or_load_lazy_loads_on_first_call():
    """未加载 → get_or_load 触发一次 load_model。"""
    mm = _mm([_spec("m1")])
    holder = {}

    def factory(spec):
        holder["a"] = _StubAdapter(paths=spec.paths)
        return holder["a"]

    adapter = await mm.get_or_load("m1", adapter_factory=factory)
    assert adapter is holder["a"]
    assert adapter.is_loaded


@pytest.mark.asyncio
async def test_get_or_load_oom_evicts_then_retries_once():
    """第一次 load OOM → evict 同 GPU LRU → 重试一次成功。"""
    mm = _mm([_spec("victim"), _spec("m2")])
    # 先放一个可被 evict 的 victim（非 resident、无引用）
    victim = _StubAdapter(paths={"main": "/fake/victim"})
    await mm.load_model("victim", adapter_factory=lambda s: victim)
    assert "victim" in mm.loaded_model_ids

    # m2 的 adapter：第 1 次 load OOM，第 2 次成功
    m2_adapter = _StubAdapter(paths={"main": "/fake/m2"}, oom_loads=1)
    adapter = await mm.get_or_load("m2", adapter_factory=lambda s: m2_adapter)

    assert adapter is m2_adapter
    assert adapter.is_loaded
    assert m2_adapter._load_calls == 2          # OOM 一次 + 重试成功一次
    assert "victim" not in mm.loaded_model_ids  # LRU 被 evict
    assert "m2" not in mm._load_failures        # 成功 → 不留失败记录


@pytest.mark.asyncio
async def test_oom_evicts_auto_selected_card_not_global_lru():
    """round3 #2:自动分配(spec.gpu=None)OOM 时,驱逐 load_model 实际落的那张卡,
    而不是退成 evict_lru(None) 驱全局 LRU(可能驱了另一张没满的卡)。"""
    class _AllocTo2:
        def get_best_gpu(self, vram_mb, *, reserve=True):
            return 2  # 自动分配落 cuda:2

        def release_reservation(self, gpu_index, vram_mb):
            pass

    reg = _StubRegistry([
        ModelSpec(id="v0", model_type="image",
                  adapter_class="tests.test_model_manager_get_or_load._StubAdapter",
                  paths={"main": "/fake/v0"}, vram_mb=1024, gpu=0),
        ModelSpec(id="v2", model_type="image",
                  adapter_class="tests.test_model_manager_get_or_load._StubAdapter",
                  paths={"main": "/fake/v2"}, vram_mb=1024, gpu=2),
        _spec("newm"),  # gpu=None → 自动分配
    ])
    mm = ModelManager(registry=reg, allocator=_AllocTo2())
    # v0 落 cuda:0(全局最老 LRU),v2 落 cuda:2
    await mm.load_model("v0", adapter_factory=lambda s: _StubAdapter(paths=s.paths))
    await mm.load_model("v2", adapter_factory=lambda s: _StubAdapter(paths=s.paths))

    # newm 自动落 cuda:2、首次 OOM → 应驱逐 cuda:2 上的 v2(同卡),不动 cuda:0 的 v0。
    newm = _StubAdapter(paths={"main": "/fake/newm"}, oom_loads=1)
    await mm.get_or_load("newm", adapter_factory=lambda s: newm)

    assert "v2" not in mm.loaded_model_ids   # 与 OOM 同卡 → 被驱逐
    assert "v0" in mm.loaded_model_ids        # 另一张卡 → 不受影响(老 bug 会误驱它)
    assert "newm" in mm.loaded_model_ids
    assert mm._last_attempt_gpu["newm"] == 2


@pytest.mark.asyncio
async def test_get_or_load_second_oom_records_load_failure():
    """evict 后重试仍 OOM → 落 _load_failures + raise ModelLoadError。"""
    mm = _mm([_spec("m3")])
    # oom_loads=5：怎么试都 OOM
    m3_adapter = _StubAdapter(paths={"main": "/fake/m3"}, oom_loads=5)
    with pytest.raises(ModelLoadError):
        await mm.get_or_load("m3", adapter_factory=lambda s: m3_adapter)
    assert "m3" in mm._load_failures
    assert "OOM" in mm._load_failures["m3"] or "out of memory" in mm._load_failures["m3"].lower()
    # 二次 OOM → load 被调了 2 次（首次 + evict 后重试），不无限重试
    assert m3_adapter._load_calls == 2


@pytest.mark.asyncio
async def test_get_or_load_prior_failure_raises_without_retry():
    """已有 _load_failures 记录 → get_or_load 直接 raise，不重试。"""
    mm = _mm([_spec("m4")])
    mm._load_failures["m4"] = "previous OOM"
    with pytest.raises(ModelLoadError):
        await mm.get_or_load("m4")


class _TrackingAllocator:
    """记账 allocator:统计 get_best_gpu 的在途预留次数 vs release 次数。"""
    def __init__(self):
        self.reserved = 0
        self.releases = 0

    def get_best_gpu(self, vram_mb, *, reserve=True):
        if reserve:
            self.reserved += 1
        return 0

    def release_reservation(self, gpu_index, vram_mb):
        self.releases += 1


@pytest.mark.asyncio
async def test_reservation_released_when_adapter_build_raises():
    """回归:adapter 构建(load 锁之前)抛异常时,GPU 在途预留必须释放。

    否则 _pending[gpu] 永不清 → allocator 永远认为那张卡少 spec.vram_mb 可用
    (幻影预留),后续选卡失真。审查发现的真 bug。
    """
    alloc = _TrackingAllocator()
    reg = _StubRegistry([_spec("m1")])
    mm = ModelManager(registry=reg, allocator=alloc)

    def boom(spec):
        raise RuntimeError("adapter build failed")

    with pytest.raises(RuntimeError, match="adapter build failed"):
        await mm.load_model("m1", adapter_factory=boom)

    assert alloc.reserved == 1, "应已登记一次在途预留"
    assert alloc.releases == 1, "adapter 构建抛异常后在途预留泄漏了(未 release)"


@pytest.mark.asyncio
async def test_unload_model_returns_false_when_in_use():
    """in-use 硬守卫跳过卸载时,unload_model 必须回报 False(不是 None)。"""
    mm = _mm([_spec("m1")])
    a = _StubAdapter(paths={"main": "/fake/m1"})
    await mm.load_model("m1", adapter_factory=lambda s: a)
    mm._in_use["m1"] = 1  # 正在 infer
    ok = await mm.unload_model("m1", force=True)
    assert ok is False, "in-use 跳过卸载应回报 False"
    assert mm.is_loaded("m1"), "in-use 模型不该被卸"


@pytest.mark.asyncio
async def test_unload_model_returns_true_on_success():
    mm = _mm([_spec("m1")])
    a = _StubAdapter(paths={"main": "/fake/m1"})
    await mm.load_model("m1", adapter_factory=lambda s: a)
    ok = await mm.unload_model("m1", force=True)
    assert ok is True
    assert not mm.is_loaded("m1")


@pytest.mark.asyncio
async def test_evict_lru_returns_none_when_unload_skipped_by_in_use(monkeypatch):
    """竞态:候选选中后、unload 前 m1 变 in-use → unload 被跳过、显存没腾出。

    evict_lru 不该谎报 evicted(否则 OOM-evict-retry 以为腾了空转)。用 stash_model
    调用点注入「此刻变 in-use」的竞态。
    """
    mm = _mm([_spec("m1")])
    a = _StubAdapter(paths={"main": "/fake/m1"})
    await mm.load_model("m1", adapter_factory=lambda s: a)

    async def fake_stash(mid):
        mm._in_use[mid] = 1  # 竞态:选完候选后变 in-use
        return False

    monkeypatch.setattr(mm, "stash_model", fake_stash)
    evicted = await mm.evict_lru()
    assert evicted is None, "unload 被 in-use 跳过时 evict_lru 不该返回 model_id"
    assert mm.is_loaded("m1")
