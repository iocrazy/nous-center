"""组件级 L1 缓存 + 跨 combo 复用 + 引用计数(spec 2026-06-02)。

核心:同一组件(file|load_device|dtype|loras = 一个 id)被多个 combo 共享时只 build 一份、
跨 combo 复用;refcount 保证卸一个 combo 不误伤别 combo 在用的共享组件。

这里测 **bookkeeping 正确性**(build 次数 / refs / 出池),mock 掉真 build_bridged_*(返回
假模块)+ ModularImageBackend(无真 pipe)。共享组件 offload/device 真行为(§3 风险)由真模型
smoke 验(CLAUDE.md:CI mock torch 测不了引擎)。
"""
from __future__ import annotations

import pytest

import src.services.inference.image_modular as IM
import src.services.model_manager as MM
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import InferenceAdapter as _IA
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


def _comps(unet_dev="cuda:1", clip_file="/m/clipY.safe"):
    """A 用 clipY,B 用 clipW(只 clip 不同)→ 验 transformer/vae 复用、clip 各建。

    整模型单卡(clip/vae 跟 unet 同卡):L1 池化只在「同卡 + 全 offload=none」的可池化路径
    生效(逐组件跨卡/offload 会给模块挂 pipe-specific hook,跨 combo 共享不安全 → 不入池)。
    所以这些 bookkeeping 测试用同卡 comps,跑在可池化路径上。"""
    return {
        "diffusion_models": ComponentSpec(
            kind="diffusion_models", file="/m/X-bf16.safe",
            device=unet_dev, dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=clip_file, device=unet_dev, dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/vaeZ.safe", device=unet_dev, dtype="bfloat16"),
    }


@pytest.fixture
def l1(monkeypatch):
    """mock 真 build seam:每个组件强制走单文件桥接路径,build_bridged_* 返回唯一假模块。"""
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)  # 跳过 VRAM 守卫
    monkeypatch.setattr(MM, "_modular_repo_from_components", lambda resolved: "/fake/repo")
    # 关键:走单文件桥接路径(L1 缓存就在这条路径上)
    monkeypatch.setattr(MM, "_is_standalone_single_file", lambda spec: True)
    monkeypatch.setattr(MM, "_is_comfy_single_file_unet", lambda spec: False)

    # build_bridged_* 记录调用 + 返回唯一对象(便于断言「复用 = 同一对象」)。
    calls = {"transformer": [], "text_encoder": [], "vae": []}

    def _mk(role):
        def _fn(spec, repo, device):
            mod = object()
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
    return mm, calls


def _comp_by_file(mm, name):
    for c in mm._components.values():
        if c["key"][0] == name:
            return c
    return None


@pytest.mark.asyncio
async def test_same_combo_builds_each_component_once(l1):
    """同 combo 跑两次:第二次 combo L2 命中,组件一个都不重 build。"""
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    assert len(calls["transformer"]) == 1
    assert len(calls["text_encoder"]) == 1
    assert len(calls["vae"]) == 1


@pytest.mark.asyncio
async def test_cross_combo_reuse_shared_components(l1):
    """用户原话场景:A(X+clipY+Z)、B(X+clipW+Z)→ X、Z 复用不重 build,只 clip 各建。"""
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipY.safe"), "Flux2KleinPipeline")
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipW.safe"), "Flux2KleinPipeline")

    # transformer(X-bf16)、vae(vaeZ)各只 build 一次 —— 跨 combo 复用
    assert len(calls["transformer"]) == 1, "X-bf16 应复用,不重 build"
    assert len(calls["vae"]) == 1, "vaeZ 应复用,不重 build"
    # clip:clipY、clipW 各建一次
    assert len(calls["text_encoder"]) == 2
    assert {c["file"] for c in calls["text_encoder"]} == {"/m/clipY.safe", "/m/clipW.safe"}

    # 共享组件 refs 含两个 combo
    x = _comp_by_file(mm, "/m/X-bf16.safe")
    z = _comp_by_file(mm, "/m/vaeZ.safe")
    assert len(x["refs"]) == 2 and len(z["refs"]) == 2
    # clipY / clipW 各被一个 combo 引用
    assert all(len(c["refs"]) == 1 for c in mm._components.values() if c["role"] == "text_encoder")


@pytest.mark.asyncio
async def test_reuse_returns_same_module_object(l1):
    """L1 命中复用 = 把同一个已加载模块对象喂给第二个 combo 的 adapter(非重建)。"""
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipY.safe"), "Flux2KleinPipeline")
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipW.safe"), "Flux2KleinPipeline")
    # transformer build 只发生一次 → 那个 module 对象就是池里的,被两个 adapter 共用
    x = _comp_by_file(mm, "/m/X-bf16.safe")
    assert x["module"] is calls["transformer"][0]["module"]


@pytest.mark.asyncio
async def test_unload_one_combo_keeps_shared_frees_owned(l1):
    """卸 A:共享的 X/Z 被 B 用着保留;A 独占的 clipY 出池。卸 B:全清。"""
    mm, _calls = l1
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipY.safe"), "Flux2KleinPipeline")
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipW.safe"), "Flux2KleinPipeline")
    mids = list(mm._models)
    assert len(mids) == 2
    a_id, b_id = mids

    await mm.unload_model(a_id, force=True)
    # X、Z 仍在(B 引用),clipY 出池
    assert _comp_by_file(mm, "/m/X-bf16.safe") is not None
    assert _comp_by_file(mm, "/m/vaeZ.safe") is not None
    assert _comp_by_file(mm, "/m/clipY.safe") is None, "A 独占 clipY 应出池"
    assert _comp_by_file(mm, "/m/clipW.safe") is not None
    # 共享组件 refs 只剩 B
    assert mm._components and all(c["refs"] == {b_id} for c in mm._components.values())

    await mm.unload_model(b_id, force=True)
    assert mm._components == {}, "两个 combo 全卸 → L1 池空"


@pytest.mark.asyncio
async def test_resident_component_not_freed_on_unload(l1):
    """resident=True 的组件即使 refs 空也不出池(常驻钉死,spec §3 表)。"""
    mm, _calls = l1
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipY.safe"), "Flux2KleinPipeline")
    z = _comp_by_file(mm, "/m/vaeZ.safe")
    z["resident"] = True  # 模拟常驻 pin(PR-2 经端点设)
    mid = next(iter(mm._models))
    await mm.unload_model(mid, force=True)
    # 非常驻组件出池,vaeZ 常驻保留(refs 空也不释放)
    assert _comp_by_file(mm, "/m/vaeZ.safe") is not None
    assert _comp_by_file(mm, "/m/X-bf16.safe") is None


@pytest.mark.asyncio
async def test_offload_components_not_pooled(l1):
    """offload=cpu:组件带 cpu_offload hook,两 combo 共享会冲突(§3)→ 不进 L1 池,各自 build。"""
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipY.safe"), "Flux2KleinPipeline", offload="cpu")
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipW.safe"), "Flux2KleinPipeline", offload="cpu")
    # 不进池
    assert mm._components == {}
    # transformer / vae 各 build 两次(每 combo 自己一份)
    assert len(calls["transformer"]) == 2
    assert len(calls["vae"]) == 2


@pytest.mark.asyncio
async def test_different_load_device_not_shared(l1):
    """同文件落不同卡(offload=none → load_device=compute 卡 = unet device)→ 不同 L1 key,
    各建各的。验 _l1_component_key 用真实 load_device 而非 spec.device。

    整模型单卡 _comps 里 vae 跟 unet 同卡,两 combo unet 落不同卡(cuda:1 vs cuda:2)→
    vae 也落不同卡 → 不共享。"""
    mm, calls = l1
    await mm.get_or_load_image_adapter(_comps(unet_dev="cuda:1"), "Flux2KleinPipeline")
    await mm.get_or_load_image_adapter(_comps(unet_dev="cuda:2"), "Flux2KleinPipeline")
    # 两 combo 的 vaeZ 落到不同 compute 卡(cuda:1 vs cuda:2)→ 不共享
    assert len(calls["vae"]) == 2
    devs = {c["device"] for c in calls["vae"]}
    assert devs == {"cuda:1", "cuda:2"}


@pytest.mark.asyncio
async def test_partial_build_failure_releases_components(l1, monkeypatch):
    """combo 装配失败(adapter 建好但 _ensure_pipe 抛非 OOM)→ 已 L1 存的组件回收,不泄漏。"""
    mm, _calls = l1

    class _FailBackend(_IA):
        def __init__(self, **kw):
            self._model = None

        async def load(self, dev):
            self._model = object()

        async def infer(self, req):  # pragma: no cover
            raise NotImplementedError

        def _ensure_pipe(self):
            raise RuntimeError("synthetic non-OOM build fail")

        def unload(self):
            self._model = None

    monkeypatch.setattr(IM, "ModularImageBackend", _FailBackend)
    with pytest.raises(RuntimeError):
        await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    # combo 没入 _models,组件也不该泄漏在池里
    assert mm._models == {}
    assert mm._components == {}


@pytest.mark.asyncio
async def test_get_status_exposes_components(l1):
    """get_status 露组件池 —— 真机 smoke 验「A、B 共用 X」就看 refs 是否含两个 combo。"""
    mm, _calls = l1
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipY.safe"), "Flux2KleinPipeline")
    await mm.get_or_load_image_adapter(_comps(clip_file="/m/clipW.safe"), "Flux2KleinPipeline")
    st = mm.get_status()
    comps = {c["file"]: c for c in st["components"]}
    assert comps["X-bf16.safe"]["role"] == "transformer"
    assert len(comps["X-bf16.safe"]["refs"]) == 2  # 跨 combo 共享
    assert comps["X-bf16.safe"]["resident"] is False
