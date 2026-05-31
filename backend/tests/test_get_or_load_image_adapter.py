"""get_or_load_image_adapter(PR-4 起 modular-only)—— auto 解析 + combo 缓存 + 四态事件。

PR-4 删了 legacy 组件 L1 缓存(get_or_load_component / DiffusersImageBackend);本文件改测
图像 adapter 路径(_get_or_load_modular_adapter),mock 掉真 build seam(ModularImageBackend /
repo 推导)。实际出图由真模型 smoke 验。PR-A 起 modular 死代码退役;class 名 ModularImageBackend
是历史包袱(实际是标准 diffusers Flux2KleinPipeline 引擎)。
"""
from __future__ import annotations

import pytest

import src.services.inference.image_modular as IM
import src.services.model_manager as MM
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


def _comps(unet_dev="cuda:1"):
    return {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file="/m/u.safe", device=unet_dev, dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }


@pytest.fixture
def stubbed(monkeypatch):
    """mock modular build seam:repo 推导 + ModularImageBackend(无真 pipe)。"""
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)  # 跳过 VRAM 守卫
    monkeypatch.setattr(MM, "_modular_repo_from_components", lambda resolved: "/fake/repo")
    # PR-2:adapter builder 用 _is_standalone_single_file 判每个组件是否需桥接;stub→False
    # 让这些 wiring 测试走 HF-layout 路径(不对 fake 文件触发 build_bridged_*)。
    monkeypatch.setattr(MM, "_is_standalone_single_file", lambda spec: False)
    monkeypatch.setattr(MM, "_is_comfy_single_file_unet", lambda spec: False)
    # PR-A:_import_modular 已删 —— monkeypatch 不再需要(_FakeBackend 也绕开 _ensure_pipe 真路径)。

    builds = []
    fail = {"on": False}

    # PR-D4(2026-05-28):image adapter 改入 `_models[derived_id]` 统一字典,
    # `LoadedModel.adapter` pydantic 校验需要真 `InferenceAdapter` 子类。
    # _FakeBackend 继承 InferenceAdapter 满足 isinstance 检查,but stub 出 abstract
    # 方法走假实现。
    from src.services.inference.base import InferenceAdapter as _IA

    class _FakeBackend(_IA):
        def __init__(self, **kw):
            builds.append(kw)
            self._model = None

        async def load(self, dev):
            self._model = object()  # is_loaded → True(after load)

        async def infer(self, req):  # pragma: no cover — fake 不跑 infer
            raise NotImplementedError

        def _ensure_pipe(self):
            if fail["on"]:
                raise RuntimeError("synthetic build fail")

    monkeypatch.setattr(IM, "ModularImageBackend", _FakeBackend)
    return mm, builds, fail


@pytest.mark.asyncio
async def test_same_combo_cache_hit(stubbed):
    mm, builds, _ = stubbed
    a1 = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    a2 = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    assert a1 is a2
    assert len(builds) == 1  # 第二次走 combo 缓存,不重建


@pytest.mark.asyncio
async def test_loaded_models_snapshot_carries_source_files(stubbed):
    """loaded_models_snapshot()(runner→主进程 Pong 上报用)要带 source_files,
    主进程才能把 runner 里的 combo-hash adapter 映射回引擎库卡片。Bug 3 修复基础。"""
    mm, _builds, _ = stubbed
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    snap = mm.loaded_models_snapshot()
    assert len(snap) == 1
    e = snap[0]
    assert e["model_type"] == "image"
    assert e["pipeline_class"] == "Flux2KleinPipeline"
    # 源组件文件(unet/clip/vae)随快照过进程边界 —— 用于映射回引擎卡
    assert set(e["source_files"]) == {"/m/u.safe", "/m/c.safe", "/m/v.safe"}
    assert e["last_used_ago_sec"] >= 0
    assert isinstance(e["gpu_index"], int)


@pytest.mark.asyncio
async def test_stick_cleared_on_unload(stubbed, monkeypatch):
    """#210 回归修复:unload/evict 后清 _image_stick —— 否则 stale stick 把同工作流粘回
    已卸载/可能已满的卡 + 跳 VRAM 守卫 → 反复 OOM。清后再跑可按 get_best_gpu 重新解析。"""
    mm, builds, _ = stubbed
    cards = iter([1, 2])
    monkeypatch.setattr(mm._allocator, "get_best_gpu", lambda v: next(cards))

    await mm.get_or_load_image_adapter(_comps(unet_dev="auto"), "Flux2KleinPipeline")
    assert builds[0]["device"] == "cuda:1"
    assert mm._image_stick and mm._image_stick_keys  # stick 已登记

    mid = next(iter(mm._models))
    await mm.unload_model(mid, force=True)
    assert mm._image_stick == {} and mm._image_stick_keys == {}  # 清空

    # 再跑:stick 没了 → 不强制粘 cuda:1,按 get_best_gpu=2 落 cuda:2 重建
    await mm.get_or_load_image_adapter(_comps(unet_dev="auto"), "Flux2KleinPipeline")
    assert builds[1]["device"] == "cuda:2"


@pytest.mark.asyncio
async def test_in_use_adapter_not_unloaded_even_force(stubbed):
    """in-use 守卫:正在 infer 的 adapter,unload 即使 force=True 也拒绝(否则 mid-CUDA 卸载 segfault)。
    释放后才可卸载。修「释放 image 按钮 + 正在跑 → runner segfault」。"""
    mm, _builds, _ = stubbed
    adapter = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    mid = next(iter(mm._models))

    mm.mark_adapter_in_use(adapter)
    assert mid in mm._in_use
    await mm.unload_model(mid, force=True)
    assert mid in mm._models  # in-use → 拒绝卸载

    mm.release_adapter(adapter)
    assert mid not in mm._in_use
    await mm.unload_model(mid, force=True)
    assert mid not in mm._models  # 释放后可卸载


@pytest.mark.asyncio
async def test_evict_lru_skips_in_use(stubbed):
    """evict_lru 不驱逐正在 infer 的 adapter(否则正用的被踢 → segfault / 重载)。"""
    mm, _builds, _ = stubbed
    adapter = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    mid = next(iter(mm._models))
    mm.mark_adapter_in_use(adapter)
    assert await mm.evict_lru() is None  # 唯一候选在 in-use → 不驱逐
    assert mid in mm._models


@pytest.mark.asyncio
async def test_auto_device_resolved(stubbed, monkeypatch):
    mm, builds, _ = stubbed
    monkeypatch.setattr(mm._allocator, "get_best_gpu", lambda vram: 2)
    await mm.get_or_load_image_adapter(_comps(unet_dev="auto"), "Flux2KleinPipeline")
    assert builds[0]["device"] == "cuda:2"  # auto → cuda:2


@pytest.mark.asyncio
async def test_auto_device_flip_reuses_adapter(stubbed, monkeypatch):
    """#199 根治:同工作流、同组件,device='auto' 在两次 Run 间即使 get_best_gpu 返回不同卡
    (allocator 模拟首张卡被占后第二次落下一张),粘性放置把第二次解析回首次的卡 → combo_key
    稳定 → cache hit、不重建。builds==1。"""
    mm, builds, _ = stubbed
    cards = iter([1, 2])  # get_best_gpu:第一次 1,第二次本想给 2(被粘性覆盖回 1)
    monkeypatch.setattr(mm._allocator, "get_best_gpu", lambda vram: next(cards))

    a1 = await mm.get_or_load_image_adapter(_comps(unet_dev="auto"), "Flux2KleinPipeline")
    a2 = await mm.get_or_load_image_adapter(_comps(unet_dev="auto"), "Flux2KleinPipeline")

    assert a1 is a2, "同工作流二次 Run 应复用同一 adapter(粘性放置防翻卡)"
    assert len(builds) == 1, f"应只 build 一次,实际 {len(builds)} 次(粘性失效→翻卡 miss)"
    assert builds[0]["device"] == "cuda:1"  # 粘回首次的卡


@pytest.mark.asyncio
async def test_emits_component_events(stubbed):
    from src.services.inference.component_spec import component_state_key
    mm, _builds, _ = stubbed
    events = []

    async def _on_event(key, state, error):
        events.append((key, state, error))

    comps = _comps()
    await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline", on_event=_on_event)
    unet_dev = mm._resolve_component_device(comps["diffusion_models"]).device
    keys = {component_state_key(comps[k].model_copy(update={"device": unet_dev})) for k in ("diffusion_models", "clip", "vae")}
    loaded = {k for (k, s, _e) in events if s == "loaded"}
    loading = {k for (k, s, _e) in events if s == "loading"}
    assert keys <= loaded and keys <= loading


@pytest.mark.asyncio
async def test_emits_failed_on_build_error(stubbed):
    mm, _builds, fail = stubbed
    fail["on"] = True
    events = []

    async def _on_event(key, state, error):
        events.append((key, state, error))

    with pytest.raises(RuntimeError):
        await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline", on_event=_on_event)
    assert any(s == "failed" for (_k, s, _e) in events)
