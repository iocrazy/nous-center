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
    """整模型无卡可容 + stream 口径装得下 → transformer 自动 offload=stream 落该卡。"""
    m, calls, be = mm
    await m.get_or_load_image_adapter(_comps(), "Ideogram4Pipeline")
    tr = calls["transformer"][0]
    assert tr["device"] == "cuda:2"
    assert tr["offload"] == "stream"


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
