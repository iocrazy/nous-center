"""V1' Lane C / P3.3 — EncodePrompt + KSampler + VAEDecode component-node
executors. Exercised with mocked adapter + helpers so the test stays
fast and CPU-only."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


PKG_DIR = Path(__file__).parents[1] / "nodes" / "flux2-components"


def _load_executors():
    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))
    spec = importlib.util.spec_from_file_location(
        "flux2_components_executor_sampling_test", PKG_DIR / "executor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stub_we(monkeypatch):
    """Patch workflow_executor._model_manager + the three helpers so the
    sampling nodes are exercised in isolation."""
    from src.services import workflow_executor as we
    from src.services.inference import image_diffusers as image_mod

    adapter = MagicMock()
    adapter.pipe = MagicMock(name="pipe")
    adapter.device = "cuda:0"
    adapter.set_active_loras = MagicMock()

    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    monkeypatch.setattr(we, "_model_manager", mgr)

    monkeypatch.setattr(
        image_mod, "encode_prompt",
        MagicMock(return_value={"prompt_embeds": "EMBEDS", "text_ids": "TEXT_IDS"}),
    )
    monkeypatch.setattr(image_mod, "sample", MagicMock(return_value="LATENT_TENSOR"))

    pil_image = MagicMock()
    pil_image.width = 1024
    pil_image.height = 1024
    pil_image.save = MagicMock(side_effect=lambda buf, format="PNG": buf.write(b"PNG"))
    monkeypatch.setattr(image_mod, "vae_decode", MagicMock(return_value=pil_image))

    from src.services import image_output_storage
    monkeypatch.setattr(
        image_output_storage, "write_image",
        MagicMock(return_value={"url": "/files/images/abc.png?token=xyz",
                                "uuid": "abc",
                                "expires": 9999999999}),
    )
    return {"adapter": adapter, "mgr": mgr}


@pytest.mark.asyncio
async def test_encode_prompt_acquires_adapter_and_emits_conditioning(stub_we):
    mod = _load_executors()
    out = await mod.exec_encode_prompt(
        {"text": "a cat"},
        {"clip": {"_type": "flux2_clip", "model_id": "flux2-klein-9b"}},
    )
    stub_we["mgr"].get_loaded_adapter.assert_awaited_once_with("flux2-klein-9b")
    assert out["conditioning"]["_type"] == "flux2_conditioning"
    assert out["conditioning"]["model_id"] == "flux2-klein-9b"
    assert out["conditioning"]["prompt_embeds"] == "EMBEDS"


@pytest.mark.asyncio
async def test_encode_prompt_rejects_non_clip_input(stub_we):
    mod = _load_executors()
    with pytest.raises(RuntimeError, match="CLIP 端口未连接"):
        await mod.exec_encode_prompt({"text": "x"}, {"clip": {"_type": "flux2_model"}})


@pytest.mark.asyncio
async def test_encode_prompt_requires_text(stub_we):
    mod = _load_executors()
    with pytest.raises(RuntimeError, match="缺少 text"):
        await mod.exec_encode_prompt(
            {},
            {"clip": {"_type": "flux2_clip", "model_id": "x"}},
        )


@pytest.mark.asyncio
async def test_ksampler_applies_loras_and_returns_latent(stub_we, monkeypatch):
    # torch.Generator chain is hit when seed is set — patch torch so the
    # test stays CPU-only.
    import torch
    gen = MagicMock()
    gen.manual_seed = MagicMock(return_value=gen)
    monkeypatch.setattr(torch, "Generator", MagicMock(return_value=gen))

    mod = _load_executors()
    out = await mod.exec_ksampler(
        {"width": 768, "height": 768, "steps": 20, "cfg_scale": 4.5, "seed": 42},
        {
            "model": {"_type": "flux2_model", "model_id": "flux2-klein-9b",
                      "loras": [{"name": "lo", "strength": 0.7}]},
            "conditioning": {"_type": "flux2_conditioning", "model_id": "flux2-klein-9b",
                             "prompt_embeds": "E", "text_ids": "T"},
        },
    )
    adapter = stub_we["adapter"]
    # LoRAs flowed through
    adapter.set_active_loras.assert_called_once()
    spec_arg = adapter.set_active_loras.call_args.args[0]
    assert spec_arg[0].name == "lo" and spec_arg[0].strength == 0.7

    assert out["latent"]["_type"] == "flux2_latent"
    assert out["latent"]["tensor"] == "LATENT_TENSOR"
    assert out["latent"]["model_id"] == "flux2-klein-9b"


@pytest.mark.asyncio
async def test_ksampler_rejects_cross_model_conditioning(stub_we):
    mod = _load_executors()
    with pytest.raises(RuntimeError, match="model_id 不一致"):
        await mod.exec_ksampler(
            {},
            {
                "model": {"_type": "flux2_model", "model_id": "flux2-klein-9b", "loras": []},
                "conditioning": {"_type": "flux2_conditioning", "model_id": "ernie-image",
                                 "prompt_embeds": "E"},
            },
        )


@pytest.mark.asyncio
async def test_ksampler_drops_blank_lora_entries(stub_we, monkeypatch):
    import torch
    monkeypatch.setattr(torch, "Generator", MagicMock())
    mod = _load_executors()
    await mod.exec_ksampler(
        {"width": 512, "height": 512, "steps": 1, "cfg_scale": 1.0},
        {
            "model": {"_type": "flux2_model", "model_id": "m", "loras": [
                {"name": "", "strength": 1.0},   # blank — should be dropped
                {"name": "real", "strength": 0.5},
            ]},
            "conditioning": {"_type": "flux2_conditioning", "model_id": "m",
                             "prompt_embeds": "E"},
        },
    )
    specs = stub_we["adapter"].set_active_loras.call_args.args[0]
    assert [s.name for s in specs] == ["real"]


@pytest.mark.asyncio
async def test_vae_decode_writes_image_and_returns_signed_url(stub_we):
    mod = _load_executors()
    out = await mod.exec_vae_decode(
        {"url_ttl_seconds": "1800"},
        {
            "vae": {"_type": "flux2_vae", "model_id": "m"},
            "latent": {"_type": "flux2_latent", "model_id": "m", "tensor": "T"},
        },
    )
    assert out["image_url"].startswith("/files/images/")
    assert out["media_type"] == "image/png"
    assert out["width"] == 1024 and out["height"] == 1024
    assert out["image_uuid"] == "abc"


@pytest.mark.asyncio
async def test_vae_decode_rejects_cross_model_pair(stub_we):
    mod = _load_executors()
    with pytest.raises(RuntimeError, match="model_id 不一致"):
        await mod.exec_vae_decode(
            {},
            {
                "vae": {"_type": "flux2_vae", "model_id": "m1"},
                "latent": {"_type": "flux2_latent", "model_id": "m2", "tensor": "T"},
            },
        )


def test_yaml_declares_eight_total_nodes_after_p3_3():
    import yaml
    cfg = yaml.safe_load((PKG_DIR / "node.yaml").read_text())
    assert set(cfg["nodes"]) == {
        "flux2_load_checkpoint", "flux2_load_diffusion_model",
        "flux2_load_clip", "flux2_load_vae", "flux2_load_lora",
        "flux2_encode_prompt", "flux2_ksampler", "flux2_vae_decode",
    }


def test_executors_dict_includes_all_eight():
    mod = _load_executors()
    assert set(mod.EXECUTORS) == {
        "flux2_load_checkpoint", "flux2_load_diffusion_model",
        "flux2_load_clip", "flux2_load_vae", "flux2_load_lora",
        "flux2_encode_prompt", "flux2_ksampler", "flux2_vae_decode",
    }
