"""PR-1 T5: 整模型单卡统一 + LLM 卡显存前置保护。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


@pytest.fixture
def mm():
    return ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())


def _comps(dev="cuda:1"):
    return {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file="/m/u.safe", device=dev, dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }


@pytest.mark.asyncio
async def test_insufficient_vram_raises_clear_error(mm, monkeypatch, tmp_path):
    # 逐卡守卫(2026-06-04):unet 落 cuda:1,该卡空闲严重不足(LLM 占着)→ 装载前清晰报错。
    # 真文件让逐卡守卫能估出该卡需求(守卫读 file bytes 分卡求和)。
    u = tmp_path / "u.safe"
    u.write_bytes(b"\0" * 40_000_000)  # 40MB transformer → 约需 ~50MB
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(u), device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(u), device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(u), device="cuda:2", dtype="bfloat16"),
    }
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: 1 if dev == "cuda:1" else 90000)
    with pytest.raises(RuntimeError, match="显存不足|cuda:1"):
        await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline")


@pytest.mark.asyncio
async def test_guard_skips_pooled_components(mm, monkeypatch, tmp_path):
    """组件已在 runner L1 池 → 守卫不计入该卡(combo 装配复用、不需新显存)。
    修用户报告:节点四态显「已加载」但 combo cache miss 时,守卫按全新载入估满尺寸 → 误拦显存不足。"""
    u = tmp_path / "u.safe"
    u.write_bytes(b"\0" * 40_000_000)  # 40MB → 约需 ~50MB
    spec = ComponentSpec(kind="diffusion_models", file=str(u), device="cuda:1", dtype="bfloat16", adapter_arch="flux2")
    comps = {
        "diffusion_models": spec,
        "clip": ComponentSpec(kind="clip", file=str(u), device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(u), device="cuda:2", dtype="bfloat16"),
    }
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: 1 if dev == "cuda:1" else 90000)
    # 未入池 + 无可驱逐(_models 空)→ 该卡空闲不足、腾不出 → 拦
    with pytest.raises(RuntimeError, match="显存不足"):
        await mm._guard_image_vram_per_card(comps)
    # transformer 放进 L1 池(模拟已加载)→ 该卡不再计入 → 不拦
    key = mm._l1_component_key(spec, "cuda:1")
    mm._components[key] = {
        "module": object(), "role": "transformer", "key": key,
        "refs": set(), "resident": False, "device": "cuda:1",
    }
    await mm._guard_image_vram_per_card(comps)  # 不应抛


@pytest.mark.asyncio
async def test_guard_evicts_lru_to_fit(mm, monkeypatch, tmp_path):
    """先腾后载(spec 2026-06-07):卡空闲不足但有可驱逐的空闲 adapter → 守卫 evict 腾够后放行,不报错。"""
    u = tmp_path / "u.safe"
    u.write_bytes(b"\0" * 40_000_000)  # ~50MB need
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(u), device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(u), device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(u), device="cuda:2", dtype="bfloat16"),
    }
    # cuda:1 一开始满(10MB),evict 一次后腾到充足(90000MB);别的卡始终充足。
    state = {"freed": False}
    evicted = []

    def _free(dev):
        if dev == "cuda:1":
            return 90000 if state["freed"] else 10
        return 90000
    monkeypatch.setattr(mm, "_free_vram_mb", _free)

    async def _fake_evict(gpu_index=None):
        if gpu_index == 1 and not state["freed"]:
            state["freed"] = True
            evicted.append(gpu_index)
            return "image:old:1"
        return None
    monkeypatch.setattr(mm, "evict_lru", _fake_evict)

    await mm._guard_image_vram_per_card(comps)  # evict 后放行,不抛
    assert evicted == [1]  # 确实驱逐了一次 cuda:1 的 LRU


@pytest.mark.asyncio
async def test_guard_raises_when_nothing_evictable(mm, monkeypatch, tmp_path):
    """卡满且无可驱逐(全 resident/in-use/被 vLLM 占)→ evict 返回 None → 仍清晰报错。"""
    u = tmp_path / "u.safe"
    u.write_bytes(b"\0" * 40_000_000)
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(u), device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(u), device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(u), device="cuda:2", dtype="bfloat16"),
    }
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: 10 if dev == "cuda:1" else 90000)

    async def _no_evict(gpu_index=None):
        return None
    monkeypatch.setattr(mm, "evict_lru", _no_evict)

    with pytest.raises(RuntimeError, match="已尝试驱逐空闲模型仍不足|显存不足"):
        await mm._guard_image_vram_per_card(comps)


def test_evictable_mb_on_card(mm):
    """该卡可驱逐 adapter 显存之和:非常驻/未引用/未在用 才算(spec 2026-06-07)。"""
    from types import SimpleNamespace
    mm._models = {
        "a": SimpleNamespace(gpu_index=1, spec=SimpleNamespace(resident=False, vram_mb=18000)),
        "b": SimpleNamespace(gpu_index=1, spec=SimpleNamespace(resident=True, vram_mb=9000)),   # 常驻不算
        "c": SimpleNamespace(gpu_index=2, spec=SimpleNamespace(resident=False, vram_mb=5000)),  # 别的卡
    }
    assert mm._evictable_mb_on_card(1) == 18000
    mm._references = {"a": {"combo-x"}}  # a 被引用 → 不算
    assert mm._evictable_mb_on_card(1) == 0
    mm._references = {}
    mm._in_use = {"a"}  # a 在 infer → 不算
    assert mm._evictable_mb_on_card(1) == 0


def test_resolve_auto_card_prefers_raw_free(mm, monkeypatch):
    """auto 选卡:有真空闲就直接用 allocator 的挑选(零回归)。"""
    monkeypatch.setattr(mm._allocator, "get_best_gpu", lambda need: 2)
    assert mm._resolve_auto_card(40000) == 2


def test_resolve_auto_card_falls_back_to_evictable(mm, monkeypatch):
    """没卡有真空闲,但某卡腾掉空闲 adapter 后装得下 → 选那张卡(主动找能腾的卡注入)。"""
    from types import SimpleNamespace
    monkeypatch.setattr(mm._allocator, "get_best_gpu", lambda need: -1)  # 无真空闲
    mm._models = {
        "a": SimpleNamespace(gpu_index=0, spec=SimpleNamespace(resident=False, vram_mb=0)),
        "b": SimpleNamespace(gpu_index=2, spec=SimpleNamespace(resident=False, vram_mb=0)),
    }
    monkeypatch.setattr(mm, "_card_effective_free_mb",
                        lambda i: {0: 20000, 2: 60000}.get(i))
    assert mm._resolve_auto_card(40000) == 2   # cuda:0 腾完才 20G 不够,cuda:2 腾完 60G → 选 2


def test_resolve_auto_card_cpu_when_nothing_fits(mm, monkeypatch):
    """真空闲没有、可驱逐也腾不够 → -1(调用方退 CPU)。"""
    monkeypatch.setattr(mm._allocator, "get_best_gpu", lambda need: -1)
    mm._models = {}
    assert mm._resolve_auto_card(40000) == -1


@pytest.mark.asyncio
async def test_guard_no_evict_when_fits(mm, monkeypatch, tmp_path):
    """够装 → 不调 evict(零回归)。"""
    u = tmp_path / "u.safe"
    u.write_bytes(b"\0" * 40_000_000)
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(u), device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(u), device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(u), device="cuda:2", dtype="bfloat16"),
    }
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: 90000)  # 都充足
    called = []

    async def _evict(gpu_index=None):
        called.append(gpu_index)
        return None
    monkeypatch.setattr(mm, "evict_lru", _evict)

    await mm._guard_image_vram_per_card(comps)
    assert called == []  # 够装,不该驱逐


@pytest.mark.asyncio
async def test_guard_skipped_when_free_unknown(mm, monkeypatch):
    # 无 GPU / 查询失败 → free=None → 跳过保护(不阻塞)。modular 装配 stub 让流程走通。
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)

    async def _fake_modular(resolved, combo_key, pc, target, emit, offload="none", comp_devices=None, comp_offloads=None):
        return object()

    monkeypatch.setattr(mm, "_get_or_load_modular_adapter", _fake_modular)
    adapter = await mm.get_or_load_image_adapter(_comps("cuda:1"), "Flux2KleinPipeline")
    assert adapter is not None


@pytest.mark.asyncio
async def test_guard_skipped_when_combo_already_loaded(mm, monkeypatch):
    """combo 已加载(self._models 有该 model_id)→ 跳守卫。显式选卡 re-run 时组件已在该卡上,
    守卫却按「全新装载」从 file bytes 估需求 → free 因 combo 已占而偏低 → 旧版误判「卡被自己
    占满」拦死合法复用(本 session 真机踩)。已加载 = 纯 cache hit、零新显存 → 跳守卫安全。"""
    # 卡空闲极低 —— 若守卫真跑(估出需求)会 raise;这里只验它被**跳过**(根本不调)。
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: 1)
    # 固定 model_id 并预置进 _models = 模拟 combo 已加载。
    monkeypatch.setattr(mm, "_derive_image_model_id", lambda combo_key: "already-loaded-combo")
    mm._models["already-loaded-combo"] = object()
    guard_calls: list = []

    async def _spy_guard(resolved):  # 守卫现为 async,mock 也得 async(否则被 await 时报错)
        guard_calls.append(True)
    monkeypatch.setattr(mm, "_guard_image_vram_per_card", _spy_guard)

    async def _fake_modular(resolved, combo_key, pc, target, emit, offload="none", comp_devices=None, comp_offloads=None):
        return "stub-adapter"

    monkeypatch.setattr(mm, "_get_or_load_modular_adapter", _fake_modular)
    adapter = await mm.get_or_load_image_adapter(_comps("cuda:1"), "Flux2KleinPipeline")
    assert adapter == "stub-adapter"
    assert guard_calls == []  # combo 已加载 → 守卫被跳过


def test_estimate_vram_fp8_halves_transformer_and_clip(mm, tmp_path):
    """fp8 weight-only:transformer + clip 按 file bytes 的一半估(vae 不量化全计),
    所以 fp8 估算 ≈ bf16 估算 - (transformer+clip)/2 的余量倍数。"""
    t = tmp_path / "t.safe"
    t.write_bytes(b"\0" * 8_000_000)   # 8MB transformer
    c = tmp_path / "c.safe"
    c.write_bytes(b"\0" * 4_000_000)   # 4MB clip
    v = tmp_path / "v.safe"
    v.write_bytes(b"\0" * 1_000_000)   # 1MB vae

    def comps(dt):
        return {
            "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(t), device="cuda:1", dtype=dt, adapter_arch="flux2"),
            "clip": ComponentSpec(kind="clip", file=str(c), device="cuda:1", dtype=dt),
            "vae":  ComponentSpec(kind="vae",  file=str(v), device="cuda:1", dtype="bfloat16"),  # vae 永不 fp8
        }

    bf16 = mm._estimate_image_vram_mb(comps("bfloat16"))
    fp8 = mm._estimate_image_vram_mb(comps("fp8_e4m3"))
    # bf16: (8+4+1)MB*1.3 ; fp8: (4+2+1)MB*1.3 —— transformer/clip 减半
    assert bf16 is not None and fp8 is not None
    assert fp8 < bf16
    assert fp8 == int((4_000_000 + 2_000_000 + 1_000_000) / (1024 * 1024) * 1.3)


@pytest.mark.asyncio
async def test_explicit_per_component_cards_honored(mm, monkeypatch):
    # 逐组件选卡(2026-06-04):三组件显式不同卡(unet cuda:1, clip cuda:0, vae cuda:2)
    # → **各落各的卡**(不再统一到 unet 卡)。
    seen = {}
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)

    async def _fake_modular(resolved, combo_key, pc, target, emit, offload="none", comp_devices=None, comp_offloads=None):
        seen["resolved"] = resolved
        seen["comp_devices"] = comp_devices
        return object()

    monkeypatch.setattr(mm, "_get_or_load_modular_adapter", _fake_modular)
    await mm.get_or_load_image_adapter(_comps("cuda:1"), "Flux2KleinPipeline")
    assert {s.device for s in seen["resolved"].values()} == {"cuda:1", "cuda:0", "cuda:2"}
    assert seen["comp_devices"] == {
        "transformer": "cuda:1", "text_encoder": "cuda:0", "vae": "cuda:2"}


@pytest.mark.asyncio
async def test_auto_clip_vae_follow_unet_card(mm, monkeypatch):
    # 逐组件选卡零回归:clip/vae device=auto → 跟随 transformer 解析出的卡(整模型单卡)。
    seen = {}
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)

    async def _fake_modular(resolved, combo_key, pc, target, emit, offload="none", comp_devices=None, comp_offloads=None):
        seen["comp_devices"] = comp_devices
        return object()

    monkeypatch.setattr(mm, "_get_or_load_modular_adapter", _fake_modular)
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file="/m/u.safe", device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="auto", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="auto", dtype="bfloat16"),
    }
    await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline")
    assert seen["comp_devices"] == {
        "transformer": "cuda:1", "text_encoder": "cuda:1", "vae": "cuda:1"}


# --- PR-2: 单文件装配辅助(架构参考整模型 + 单文件检测)---

def test_reference_repo_for_arch_matches_class(tmp_path, monkeypatch):
    """PR-B 后:flux2 优先返回仓内 bundle(几 MB);未知架构 fallback 扫 LOCAL_MODELS_PATH。"""
    from src.services import model_manager as mm_mod
    base = tmp_path / "image" / "diffusers"
    (base / "ERNIE-Image").mkdir(parents=True)
    (base / "ERNIE-Image" / "model_index.json").write_text('{"_class_name": "ErnieImagePipeline"}')
    settings = MagicMock()
    settings.LOCAL_MODELS_PATH = str(tmp_path)
    monkeypatch.setattr("src.config.get_settings", lambda: settings)
    # flux2 → 优先 bundle(无需 LOCAL_MODELS_PATH/diffusers/Flux2-klein-9B);
    assert mm_mod._reference_repo_for_arch("flux2").endswith("configs/image_arch/flux2")
    # ernie 未 bundle → fallback 扫 LOCAL_MODELS_PATH。
    assert mm_mod._reference_repo_for_arch("ernie").endswith("ERNIE-Image")
    assert mm_mod._reference_repo_for_arch("nope") is None


def test_is_standalone_single_file(tmp_path):
    from src.services.model_manager import _is_standalone_single_file
    sf = tmp_path / "diffusion_models" / "flux" / "x.safetensors"
    sf.parent.mkdir(parents=True)
    sf.write_text("x")
    assert _is_standalone_single_file(
        ComponentSpec(kind="diffusion_models", file=str(sf), device="cuda:0", dtype="bfloat16"))
    hf = tmp_path / "diffusers" / "M" / "transformer" / "y.safetensors"
    hf.parent.mkdir(parents=True)
    hf.write_text("y")
    (hf.parent / "config.json").write_text("{}")
    assert not _is_standalone_single_file(
        ComponentSpec(kind="diffusion_models", file=str(hf), device="cuda:0", dtype="bfloat16"))


# --- 整模型同卡 footprint 选卡(2026-06-08 真机 OOM 根因)---


def test_colocated_auto_footprint_sums_auto_offload_none(mm, tmp_path):
    """transformer auto 选卡的 footprint = 会同卡常驻(auto + offload=none)组件 need 之和,
    不是 transformer 单件。clip/vae auto 会强制跟 transformer 卡 → 必须计入。"""
    t = tmp_path / "t.safe"
    t.write_bytes(b"\0" * 20_000_000)
    c = tmp_path / "c.safe"
    c.write_bytes(b"\0" * 12_000_000)
    v = tmp_path / "v.safe"
    v.write_bytes(b"\0" * 8_000_000)
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(t), device="auto", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(c), device="auto", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(v), device="auto", dtype="bfloat16"),
    }
    whole = sum(mm._component_need_mb(comps[k]) for k in ("diffusion_models", "clip", "vae"))
    assert mm._colocated_auto_footprint_mb(comps) == whole
    # transformer 单件远小于整模型 → 正是误派小卡的根因。
    assert mm._component_need_mb(comps["diffusion_models"]) < whole


def test_colocated_footprint_excludes_offloaded_and_explicit(mm, tmp_path):
    """offload!=none 的组件 forward 时才上卡、不常驻 → 不计入；显式选别的卡的组件也不跟随 → 不计入。"""
    t = tmp_path / "t.safe"
    t.write_bytes(b"\0" * 20_000_000)
    c = tmp_path / "c.safe"
    c.write_bytes(b"\0" * 12_000_000)
    v = tmp_path / "v.safe"
    v.write_bytes(b"\0" * 8_000_000)
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(t), device="auto", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(c), device="auto", dtype="bfloat16", offload="cpu"),  # offload → 不常驻
        "vae":  ComponentSpec(kind="vae",  file=str(v), device="cuda:0", dtype="bfloat16"),  # 显式别的卡 → 不跟随
    }
    # 只有 transformer(auto+offload none)计入。
    assert mm._colocated_auto_footprint_mb(comps) == mm._component_need_mb(comps["diffusion_models"])


@pytest.mark.asyncio
async def test_auto_transformer_card_uses_whole_model_footprint(mm, monkeypatch, tmp_path):
    """真机根因复现:transformer device=auto 时,选卡的 need 必须是整模型 footprint,
    不是 transformer 单件。否则 transformer 单件估值让小卡看着够 → get_best_gpu 选小卡 →
    clip/vae 强制跟卡 → 整模型压小卡 OOM(2026-06-08 Flux2 派 3090 OOM)。"""
    t = tmp_path / "t.safe"
    t.write_bytes(b"\0" * 20_000_000)
    c = tmp_path / "c.safe"
    c.write_bytes(b"\0" * 12_000_000)
    v = tmp_path / "v.safe"
    v.write_bytes(b"\0" * 8_000_000)
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(t), device="auto", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(c), device="auto", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(v), device="auto", dtype="bfloat16"),
    }
    seen_need: list[int] = []
    monkeypatch.setattr(mm, "_resolve_auto_card", lambda need: seen_need.append(need) or 1)
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)  # 跳守卫

    async def _fake_modular(resolved, combo_key, pc, target, emit, offload="none", comp_devices=None, comp_offloads=None):
        return object()
    monkeypatch.setattr(mm, "_get_or_load_modular_adapter", _fake_modular)

    await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline")

    whole = sum(mm._component_need_mb(comps[k]) for k in ("diffusion_models", "clip", "vae"))
    trans_only = mm._component_need_mb(comps["diffusion_models"])
    # transformer(dict 首位)的 auto 选卡按整模型 footprint,非单件。
    assert seen_need, "_resolve_auto_card 应被调用"
    assert seen_need[0] == whole, f"transformer 选卡 need 应为整模型 footprint {whole},实际 {seen_need[0]}"
    assert seen_need[0] != trans_only, "不能再用 transformer 单件 need 选卡(根因)"


# --- 分片求和口径统一(2026-06-08 真机:vram_mb 只算第 1 片 → 退 CPU hang)---


def test_component_bytes_sums_shards(tmp_path):
    """_component_bytes:单文件返回字节;分片返回同组件所有 sibling 分片之和(非只第 1 片)。"""
    from src.services.model_manager import ModelManager
    single = tmp_path / "x.safetensors"
    single.write_bytes(b"\0" * 1000)
    assert ModelManager._component_bytes(str(single)) == 1000
    for i in (1, 2, 3):
        (tmp_path / f"model-0000{i}-of-00003.safetensors").write_bytes(b"\0" * 1000)
    first = tmp_path / "model-00001-of-00003.safetensors"
    assert ModelManager._component_bytes(str(first)) == 3000  # 求和,非 1000(第 1 片)


def test_estimate_vram_sums_shards(mm, tmp_path):
    """分片整模型的 vram 估算(= 记录的 vram_mb = 可驱逐空间)必须求和所有分片。
    只算第 1 片 → 可驱逐空间低估 → effective free 不足 → auto 退 CPU(2026-06-08 真机根因)。"""
    for i in (1, 2, 3):
        (tmp_path / f"t-0000{i}-of-00003.safetensors").write_bytes(b"\0" * 10_000_000)  # 3×10MB=30MB
    c = tmp_path / "c.safe"
    c.write_bytes(b"\0" * 6_000_000)
    v = tmp_path / "v.safe"
    v.write_bytes(b"\0" * 1_000_000)
    resolved = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(tmp_path / "t-00001-of-00003.safetensors"), device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(c), device="cuda:1", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(v), device="cuda:1", dtype="bfloat16"),
    }
    est = mm._estimate_image_vram_mb(resolved)
    # 全分片:(30+6+1)MB ×1.3
    assert est == int((30_000_000 + 6_000_000 + 1_000_000) / (1024 * 1024) * 1.3)
    # 不能是只第 1 片 (10+6+1)MB ×1.3
    assert est != int((10_000_000 + 6_000_000 + 1_000_000) / (1024 * 1024) * 1.3)


@pytest.mark.asyncio
async def test_guard_need_sums_shards(mm, monkeypatch, tmp_path):
    """守卫 per-card need 也分片求和:3×15MB 分片 transformer 落 cuda:1,该卡空闲只够 1 片不够整体
    → 无可驱逐 → 报错(若只算第 1 片会误判够装、放过 → 后续 OOM)。"""
    for i in (1, 2, 3):
        (tmp_path / f"t-0000{i}-of-00003.safetensors").write_bytes(b"\0" * 15_000_000)  # 共 45MB→~55MB
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(tmp_path / "t-00001-of-00003.safetensors"), device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(tmp_path / "t-00001-of-00003.safetensors"), device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(tmp_path / "t-00001-of-00003.safetensors"), device="cuda:2", dtype="bfloat16"),
    }
    # cuda:1 空闲 ~25MB:够 1 片(~19MB)但不够整 transformer(~55MB)→ 须拦。
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: 25 if dev == "cuda:1" else 90000)

    async def _no_evict(gpu_index=None):
        return None
    monkeypatch.setattr(mm, "evict_lru", _no_evict)
    with pytest.raises(RuntimeError, match="显存不足|cuda:1"):
        await mm._guard_image_vram_per_card(comps)
