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
    def get_best_gpu(self, vram_mb):
        return 0


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
