"""RAM stash(spec 2026-06-12):组件释放默认挪 CPU 留池,命中秒回;水位/LRU/回退。

bookkeeping 单测(mock build + 假模块带 .to);真权重搬运/出图一致性由真机验
(torchao fp8 .to 往返 bit 一致已 spike 验证 2026-06-12)。
"""
from __future__ import annotations

from types import SimpleNamespace

import psutil
import pytest

import src.services.inference.image_modular as IM
import src.services.model_manager as MM
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import InferenceAdapter as _IA
from src.services.inference.component_spec import ComponentSpec
from src.services.model_manager import ModelManager


class _FakeModule:
    """带 .to(device) 的假模块(stash/restore 路径要求)。"""

    def __init__(self):
        self.device_history: list = []

    def to(self, device):
        self.device_history.append(str(device))
        return self


def _comps(unet_dev="cuda:1"):
    return {
        "diffusion_models": ComponentSpec(
            kind="diffusion_models", file="/m/X-bf16.safe",
            device=unet_dev, dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/clipY.safe", device=unet_dev, dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/vaeZ.safe", device=unet_dev, dtype="bfloat16"),
    }


@pytest.fixture
def l1(monkeypatch):
    from src.services.inference.registry import ModelRegistry

    class _Reg(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    mm = ModelManager(registry=_Reg(), allocator=GPUAllocator())
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)
    monkeypatch.setattr(MM, "_modular_repo_from_components", lambda resolved: "/fake/repo")
    monkeypatch.setattr(MM, "_is_standalone_single_file", lambda spec: True)
    monkeypatch.setattr(MM, "_is_comfy_single_file_unet", lambda spec: False)
    # 文件大小估算不读磁盘(假路径)
    monkeypatch.setattr(ModelManager, "_component_bytes", staticmethod(lambda f: 1024))

    calls = {"transformer": [], "text_encoder": [], "vae": []}

    def _mk(role):
        def _fn(spec, repo, device):
            mod = _FakeModule()
            calls[role].append({"file": spec.file, "device": device, "module": mod})
            return mod
        return _fn

    monkeypatch.setattr(IM, "build_bridged_transformer", _mk("transformer"))
    monkeypatch.setattr(IM, "build_bridged_text_encoder", _mk("text_encoder"))
    monkeypatch.setattr(IM, "build_bridged_vae", _mk("vae"))

    class _FakeBackend(_IA):
        def __init__(self, **kw):
            self._kw = kw
            self._model = None

        async def load(self, dev):
            self._model = object()

        async def infer(self, req):  # pragma: no cover
            raise NotImplementedError

        def _ensure_pipe(self):
            return None

        def unload(self):
            self._model = None

    monkeypatch.setattr(IM, "ModularImageBackend", _FakeBackend)
    # RAM 充足(默认水位 24G,给 100G available)
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=100 * 10**9))
    return mm, calls


def _comp_by_file(mm, name):
    for c in mm._components.values():
        if c["key"][0] == name:
            return c
    return None


@pytest.mark.asyncio
async def test_release_stashes_instead_of_destroy(l1):
    """卸 combo:独占组件不出池 → stashed=True + module.to('cpu')。"""
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    mid = next(iter(mm._models))
    await mm.unload_model(mid, force=True)

    comp = _comp_by_file(mm, "/m/X-bf16.safe")
    assert comp is not None, "stash 后应留池"
    assert comp["stashed"] is True
    assert comp["module"].device_history[-1] == "cpu"


@pytest.mark.asyncio
async def test_rehit_restores_without_rebuild(l1):
    """同 combo 再来:stashed 组件 .to(身份卡) 搬回,build 不重跑。"""
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    mid = next(iter(mm._models))
    await mm.unload_model(mid, force=True)
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")

    assert len(calls["transformer"]) == 1, "restore 命中不应重 build"
    comp = _comp_by_file(mm, "/m/X-bf16.safe")
    assert comp["stashed"] is False
    assert comp["module"].device_history[-1] == "cuda:1"


@pytest.mark.asyncio
async def test_low_ram_falls_back_to_destroy(l1, monkeypatch):
    """RAM 低于水位 → 不 stash,出池销毁(旧行为)。"""
    mm, calls = l1
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=1 * 10**9))
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    mid = next(iter(mm._models))
    await mm.unload_model(mid, force=True)

    assert _comp_by_file(mm, "/m/X-bf16.safe") is None, "水位不足应销毁出池"


@pytest.mark.asyncio
async def test_trim_lru_destroys_oldest_stashed(l1, monkeypatch):
    """stash 后 RAM 跌破水位 → _trim_stash_lru 按最旧销毁。"""
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    mid = next(iter(mm._models))
    await mm.unload_model(mid, force=True)
    assert _comp_by_file(mm, "/m/X-bf16.safe")["stashed"] is True

    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=1 * 10**9))
    mm._trim_stash_lru()
    assert not any(c.get("stashed") for c in mm._components.values()), "水位不足应清空 stash 池"


@pytest.mark.asyncio
async def test_module_without_to_falls_back_to_destroy(l1, monkeypatch):
    """模块无 .to(异常)→ stash 失败回退销毁,不挡释放主流程。"""
    mm, calls = l1

    def _mk_plain(role):
        def _fn(spec, repo, device):
            return object()  # 无 .to
        return _fn
    monkeypatch.setattr(IM, "build_bridged_transformer", _mk_plain("transformer"))
    monkeypatch.setattr(IM, "build_bridged_text_encoder", _mk_plain("text_encoder"))
    monkeypatch.setattr(IM, "build_bridged_vae", _mk_plain("vae"))

    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    mid = next(iter(mm._models))
    await mm.unload_model(mid, force=True)
    assert _comp_by_file(mm, "/m/X-bf16.safe") is None


@pytest.mark.asyncio
async def test_snapshot_reports_stashed_flag(l1):
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    mid = next(iter(mm._models))
    snap_before = mm.loaded_components_snapshot()
    assert all(item["stashed"] is False for item in snap_before)
    await mm.unload_model(mid, force=True)
    snap_after = mm.loaded_components_snapshot()
    assert snap_after and all(item["stashed"] is True for item in snap_after)


def test_pinned_stash_budget_and_whitelist(monkeypatch):
    """PR-4 纯逻辑:预算记账 + 仅常规 Tensor 可 pin(无 GPU 环境直接空清单,零回归)。"""
    from src.services.inference import pinned_stash as PS

    # 无 cuda(CI)→ pin 返回空,restore 走 .to() 等价路径不崩
    class _M:
        def parameters(self):
            return iter([])

        def buffers(self):
            return iter([])
    assert PS.pin_module_inplace(_M()) == []
    PS.unpin([])  # 空清单安全
    assert PS.total_pinned_bytes() == 0


def test_external_pinned_ledger_accounts_and_releases():
    """spec ram-pinned-linkage PR-1:流式预 pin 经 register_external 入账,
    total_pinned_bytes() 计入,release_external 出账;重复释放 / None 安全。"""
    from src.services.inference import pinned_stash as PS

    base = PS.total_pinned_bytes()
    h1 = PS.register_external(10 * 1024**3)  # 10G
    h2 = PS.register_external(5 * 1024**3)   # 5G
    assert PS.total_pinned_bytes() == base + 15 * 1024**3
    PS.release_external(h1)
    assert PS.total_pinned_bytes() == base + 5 * 1024**3
    PS.release_external(h1)   # 重复释放 no-op
    PS.release_external(None)  # None 安全
    assert PS.total_pinned_bytes() == base + 5 * 1024**3
    PS.release_external(h2)
    assert PS.total_pinned_bytes() == base


def test_external_pin_counts_against_stash_budget(monkeypatch):
    """流式预 pin 入账后,stash 原地 pin 的预算按真实总量(stash + 外部)统一卡 ——
    外部已占满预算时,后续 pin_module_inplace 直接跳过(无 GPU 环境本就空清单,
    这里验证记账口径:total_pinned_bytes 含外部)。"""
    from src.services.inference import pinned_stash as PS

    monkeypatch.setenv("NOUS_STASH_PIN_BUDGET_GB", "8")
    base = PS.total_pinned_bytes()
    assert base == 0  # 干净起点(前序测试已出账)
    h = PS.register_external(8 * 1024**3)  # 占满 8G 预算
    try:
        # 预算口径已含外部占用 → 真实总量 == 预算上限
        assert PS.total_pinned_bytes() >= PS._pin_budget_bytes()
    finally:
        PS.release_external(h)
    assert PS.total_pinned_bytes() == 0


@pytest.mark.asyncio
async def test_stash_records_pin_regs_and_restore_consumes(l1, monkeypatch):
    """stash 写 pin_regs(CI 无 GPU 为空列表),restore 后消费掉(pop)。"""
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    mid = next(iter(mm._models))
    await mm.unload_model(mid, force=True)
    comp = _comp_by_file(mm, "/m/X-bf16.safe")
    assert "pin_regs" in comp and comp["pin_regs"] == []
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    comp = _comp_by_file(mm, "/m/X-bf16.safe")
    assert "pin_regs" not in comp and comp["stashed"] is False
