"""lowvram PR-2(spec 2026-06-12):auto 选卡整模型无卡可容 → 自动降级流式分块。

bookkeeping 单测(mock 选卡/估算);真机行为由 PR-1 e2e(3090 出图 22.8G)守。
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


def _comps(dtype="bfloat16", offload="none"):
    return {
        "diffusion_models": ComponentSpec(
            kind="diffusion_models", file="/m/big-dit.safe", device="auto",
            dtype=dtype, adapter_arch="ideogram4", offload=offload),
        "clip": ComponentSpec(kind="clip", file="/m/te.safe", device="auto", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/vae.safe", device="auto", dtype="bfloat16"),
    }


@pytest.fixture
def mm(monkeypatch):
    class _Reg(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    m = ModelManager(registry=_Reg(), allocator=GPUAllocator())
    monkeypatch.setattr(m, "_free_vram_mb", lambda dev: None)
    monkeypatch.setattr(MM, "_modular_repo_from_components", lambda resolved: "/fake/repo")
    monkeypatch.setattr(MM, "_is_standalone_single_file", lambda spec: True)
    monkeypatch.setattr(MM, "_is_comfy_single_file_unet", lambda spec: False)
    # 组件 need 固定小值(TE/VAE 各 1G);整模型 footprint 固定大值(装不下)
    monkeypatch.setattr(m, "_component_need_mb", lambda spec: 1024)
    monkeypatch.setattr(m, "_colocated_auto_footprint_mb", lambda comps: 60000)
    # 选卡:need > 30000(整模型)无卡;小 need(stream 口径)→ cuda:2
    monkeypatch.setattr(m, "_resolve_auto_card", lambda need: -1 if need > 30000 else 2)

    calls = {"transformer": [], "text_encoder": [], "vae": []}

    def _mk(role):
        def _fn(spec, repo, device):
            calls[role].append({"file": spec.file, "device": device, "offload": spec.offload})
            return object()
        return _fn

    monkeypatch.setattr(IM, "build_bridged_transformer", _mk("transformer"))
    monkeypatch.setattr(IM, "build_bridged_text_encoder", _mk("text_encoder"))
    monkeypatch.setattr(IM, "build_bridged_vae", _mk("vae"))

    class _FakeBackend(_IA):
        last_kw = None

        def __init__(self, **kw):
            _FakeBackend.last_kw = kw
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
    return m, calls, _FakeBackend


@pytest.mark.asyncio
async def test_auto_downgrades_to_stream_when_nothing_fits(mm):
    """整模型无卡可容 + stream 口径装得下 → transformer 自动 offload=stream 落该卡(cuda:2)。
    单文件桥接组件 stream 时**建在 CPU**(_apply_stream_offload 再挂 group offloading 流式到 cuda:2);
    onload/compute 卡仍 cuda:2(传给 ModularImageBackend 的 device)。2026-06-13 _load_device_for
    修(stream→cpu build)后,build 设备从 cuda:2 变 cpu(否则单文件双 DiT bf16 直建 GPU 37G 在小卡 OOM)。"""
    m, calls, be = mm
    await m.get_or_load_image_adapter(_comps(), "Ideogram4Pipeline")
    tr = calls["transformer"][0]
    assert tr["device"] == "cpu"             # stream 桥接组件建在 CPU
    assert tr["offload"] == "stream"
    assert be.last_kw["device"] == "cuda:2"  # onload/compute 卡仍 cuda:2(stream 流式上卡)


@pytest.mark.asyncio
async def test_auto_no_downgrade_for_fp8(mm):
    """fp8 在线量化不可流式(引擎 fail-loud)→ 不降级,退 CPU(旧行为)。"""
    m, calls, be = mm
    await m.get_or_load_image_adapter(_comps(dtype="fp8_e4m3"), "Ideogram4Pipeline")
    tr = calls["transformer"][0]
    assert tr["device"] == "cpu" and tr["offload"] == "none"


@pytest.mark.asyncio
async def test_auto_respects_explicit_offload(mm):
    """用户显式选了 offload(cpu)→ 尊重,不改写成 stream。"""
    m, calls, be = mm
    await m.get_or_load_image_adapter(_comps(offload="cpu"), "Ideogram4Pipeline")
    tr = calls["transformer"][0]
    assert tr["offload"] == "cpu" and tr["device"] == "cpu"


@pytest.mark.asyncio
async def test_auto_zero_regression_when_fits(mm, monkeypatch):
    """整模型装得下 → 不降级(零回归)。"""
    m, calls, be = mm
    monkeypatch.setattr(m, "_resolve_auto_card", lambda need: 1)
    await m.get_or_load_image_adapter(_comps(), "Ideogram4Pipeline")
    tr = calls["transformer"][0]
    assert tr["device"] == "cuda:1" and tr["offload"] == "none"


def test_repo_total_mb_counts_all_repo_weights(tmp_path):
    """PR-3:HF-layout 整模型 footprint 按 repo 总权重(含三件套之外的 unconditional_transformer
    ——Ideogram-4 漏算 18.6G 让 auto 误判装得下,贴边挤卡,真机逮到)。"""
    (tmp_path / "model_index.json").write_text("{}")
    for sub, mb in (("transformer", 4), ("unconditional_transformer", 4), ("text_encoder", 2), ("vae", 1)):
        d = tmp_path / sub
        d.mkdir()
        (d / "w.safetensors").write_bytes(b"\0" * (mb * 1024 * 1024))
    got = ModelManager._repo_total_mb(str(tmp_path / "transformer" / "w.safetensors"))
    assert got == 11, f"应计 repo 全部权重(11MB),得 {got}"
    # 非 repo(无 model_index)→ 0(回退逐件旧逻辑)
    assert ModelManager._repo_total_mb("/m/loose/transformer/w.safetensors") == 0


# ── PR-2(spec ram-pinned-linkage):流式挂载 RAM 门禁 + 降级梯子 ──

from types import SimpleNamespace

import src.services.inference.pinned_stash as PS


def test_stream_pin_estimate_repo_vs_single_file(mm, tmp_path):
    """预 pin 估算:HF-layout repo → repo 总权重;单文件 → DiT 文件 + uncond 文件。"""
    m, _, _ = mm
    # repo 路:model_index + transformer 权重
    (tmp_path / "model_index.json").write_text("{}")
    d = tmp_path / "transformer"
    d.mkdir()
    (d / "w.safetensors").write_bytes(b"\0" * (8 * 1024 * 1024))
    repo_spec = ComponentSpec(kind="diffusion_models",
                              file=str(tmp_path / "transformer" / "w.safetensors"),
                              device="cuda:2", dtype="bfloat16")
    assert m._stream_pin_estimate_mb(repo_spec) == 8

    # 单文件路:DiT 5MB + uncond 3MB = 8MB
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"\0" * (5 * 1024 * 1024))
    unc = tmp_path / "unc.safetensors"
    unc.write_bytes(b"\0" * (3 * 1024 * 1024))
    sf_spec = ComponentSpec(kind="diffusion_models", file=str(dit), device="cuda:2",
                            dtype="bfloat16", unconditional_file=str(unc))
    assert m._stream_pin_estimate_mb(sf_spec) == 8


def test_should_stream_low_ram_false_when_ample(mm, monkeypatch):
    """host RAM 充裕 + pinned 远低于预算 → 不降级(全量预 pin,最快)。"""
    m, _, _ = mm
    monkeypatch.setattr(m, "_stream_pin_estimate_mb", lambda s: 37000)
    monkeypatch.setattr(m, "_stash_ram_reserve_bytes", lambda: 24 * 1024**3)
    monkeypatch.setattr("psutil.virtual_memory",
                        lambda: SimpleNamespace(available=100 * 1024**3))
    monkeypatch.setattr(PS, "total_pinned_bytes", lambda: 0)
    monkeypatch.setattr(PS, "pin_budget_bytes", lambda: 64 * 1024**3)
    assert m._should_stream_low_ram(_comps()["diffusion_models"]) is False


def test_should_stream_low_ram_true_when_ram_tight(mm, monkeypatch):
    """腾 stash 后 host RAM 仍不足水位 → 降级 low_cpu_mem_usage。"""
    m, _, _ = mm
    monkeypatch.setattr(m, "_stream_pin_estimate_mb", lambda s: 37000)
    monkeypatch.setattr(m, "_stash_ram_reserve_bytes", lambda: 24 * 1024**3)
    monkeypatch.setattr(m, "_trim_stash_lru", lambda extra_need=0: None)  # 腾不出
    monkeypatch.setattr("psutil.virtual_memory",
                        lambda: SimpleNamespace(available=40 * 1024**3))  # 40-37<24
    monkeypatch.setattr(PS, "total_pinned_bytes", lambda: 0)
    monkeypatch.setattr(PS, "pin_budget_bytes", lambda: 64 * 1024**3)
    assert m._should_stream_low_ram(_comps()["diffusion_models"]) is True


def test_should_stream_low_ram_true_when_over_pin_budget(mm, monkeypatch):
    """host RAM 够,但已 pin + 预 pin 超 pinned 预算 → 降级(pinned 不可换页,不能叠爆)。"""
    m, _, _ = mm
    monkeypatch.setattr(m, "_stream_pin_estimate_mb", lambda s: 37000)
    monkeypatch.setattr(m, "_stash_ram_reserve_bytes", lambda: 24 * 1024**3)
    monkeypatch.setattr("psutil.virtual_memory",
                        lambda: SimpleNamespace(available=100 * 1024**3))
    monkeypatch.setattr(PS, "total_pinned_bytes", lambda: 40 * 1024**3)  # 40+37>64
    monkeypatch.setattr(PS, "pin_budget_bytes", lambda: 64 * 1024**3)
    assert m._should_stream_low_ram(_comps()["diffusion_models"]) is True


def test_should_stream_low_ram_estimate_failure_keeps_prepin(mm, monkeypatch):
    """估算/psutil 抛 → False(保持全量预 pin 最快路径,零回归)。"""
    m, _, _ = mm

    def _boom(s):
        raise RuntimeError("nope")
    monkeypatch.setattr(m, "_stream_pin_estimate_mb", _boom)
    assert m._should_stream_low_ram(_comps()["diffusion_models"]) is False


def test_trim_stash_lru_extra_need_evicts_for_incoming_pin(mm, monkeypatch):
    """_trim_stash_lru(extra_need) 为即将 pin 的权重腾地方:available-extra_need<reserve
    时按最旧 stashed 销毁(直至无 stashed)。"""
    m, _, _ = mm
    m._components = {
        ("/m/a.safe", "cuda:2", "bf16"): {
            "stashed": True, "stashed_at": 1.0, "module": object(),
            "role": "vae", "key": ("/m/a.safe",)},
        ("/m/b.safe", "cuda:2", "bf16"): {
            "stashed": True, "stashed_at": 2.0, "module": object(),
            "role": "text_encoder", "key": ("/m/b.safe",)},
    }
    monkeypatch.setattr(m, "_stash_ram_reserve_bytes", lambda: 24 * 1024**3)
    # available 恒等于 reserve → extra_need>0 时恒不足 → 把 stashed 全清(然后 stashed 空,返回)
    monkeypatch.setattr("psutil.virtual_memory",
                        lambda: SimpleNamespace(available=24 * 1024**3))
    m._trim_stash_lru(extra_need=5 * 1024**3)
    assert not any(c.get("stashed") for c in m._components.values())


@pytest.mark.asyncio
async def test_gate_passes_stream_low_ram_to_backend(mm, monkeypatch):
    """门禁判定 low_ram=True → 透传给 ModularImageBackend(stream_low_ram=True),
    不进 combo_key(同 combo 两种 pin 策略出图相同)。"""
    m, calls, be = mm
    monkeypatch.setattr(m, "_should_stream_low_ram", lambda dm: True)
    await m.get_or_load_image_adapter(_comps(), "Ideogram4Pipeline")
    assert be.last_kw["stream_low_ram"] is True


@pytest.mark.asyncio
async def test_gate_no_stream_no_low_ram_check(mm, monkeypatch):
    """非流式(装得下,offload=none)→ 不跑门禁,stream_low_ram=False(零回归)。"""
    m, calls, be = mm
    monkeypatch.setattr(m, "_resolve_auto_card", lambda need: 1)  # 装得下,不降级 stream
    called = {"gate": False}
    monkeypatch.setattr(m, "_should_stream_low_ram",
                        lambda dm: called.__setitem__("gate", True) or True)
    await m.get_or_load_image_adapter(_comps(), "Ideogram4Pipeline")
    assert be.last_kw["stream_low_ram"] is False
    assert called["gate"] is False, "非流式不应触发门禁"
