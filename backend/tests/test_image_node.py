"""Tests for image_generate / image_output nodes.

V1' Lane D P5: image_generate now composes the Lane C helpers
(encode_prompt + sample + vae_decode) instead of calling adapter.infer.
The fixture mocks the three helpers + write_image so the node is
exercised end-to-end without touching real diffusers / GPU.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def fake_image_adapter(monkeypatch):
    """Mock the helper chain so we can assert what image_generate calls.

    Returns a dict the tests use to read back:
      adapter / mgr / encode_calls / sample_calls / vae_decode_calls
    """
    from src.services import workflow_executor as we
    from src.services.inference import image_diffusers as image_mod
    from src.services import image_output_storage

    encode_calls: list = []
    sample_calls: list = []
    vae_decode_calls: list = []

    def _encode(pipe, prompt, **kw):
        encode_calls.append({"pipe": pipe, "prompt": prompt, "kw": kw})
        return {"prompt_embeds": "EMBEDS", "text_ids": "TEXT_IDS"}

    def _sample(pipe, conditioning, **kw):
        sample_calls.append({"pipe": pipe, "conditioning": conditioning, "kw": kw})
        return "LATENTS"

    pil_image = MagicMock()
    pil_image.save = MagicMock(side_effect=lambda buf, format="PNG": buf.write(b"\x89PNGFAKE"))

    def _vae_decode(pipe, latents):
        vae_decode_calls.append({"pipe": pipe, "latents": latents})
        return pil_image

    monkeypatch.setattr(image_mod, "encode_prompt", _encode)
    monkeypatch.setattr(image_mod, "sample", _sample)
    monkeypatch.setattr(image_mod, "vae_decode", _vae_decode)

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.device = "cuda:0"
    adapter.pipe = MagicMock(name="pipe")
    adapter.set_active_loras = MagicMock()

    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    monkeypatch.setattr(we, "_model_manager", mgr)

    monkeypatch.setattr(
        image_output_storage, "write_image",
        MagicMock(return_value={
            "url": "/files/images/2026-05-12/fake.png?token=t&expires=1",
            "uuid": "fake",
            "expires": 1778640000,
        }),
    )

    # torch.Generator chain — patch so the test doesn't need CUDA.
    import torch
    gen = MagicMock()
    gen.manual_seed = MagicMock(return_value=gen)
    monkeypatch.setattr(torch, "Generator", MagicMock(return_value=gen))

    return {
        "adapter": adapter,
        "mgr": mgr,
        "encode_calls": encode_calls,
        "sample_calls": sample_calls,
        "vae_decode_calls": vae_decode_calls,
    }


async def test_image_generate_composes_helpers_in_correct_order(fake_image_adapter):
    """Lane D P5 — image_generate composes encode_prompt → sample → vae_decode."""
    from src.services.nodes.image import ImageGenerateNode

    node = ImageGenerateNode()
    out = await node.invoke(
        data={
            "model_key": "flux2-klein-9b",
            "negative_prompt": "blurry",   # accepted in data but not used by helpers yet
            "width": 768,
            "height": 768,
            "steps": 30,
            "seed": 1234,
            "cfg_scale": 5.5,
            "loras": [{"name": "anime-v2", "strength": 0.7}],
        },
        inputs={"prompt": "a cat in space"},
    )

    # 1. Adapter acquired by model_key
    fake_image_adapter["mgr"].get_loaded_adapter.assert_awaited_once_with("flux2-klein-9b")

    # 2. LoRAs applied through set_active_loras
    adapter = fake_image_adapter["adapter"]
    adapter.set_active_loras.assert_called_once()
    loras = adapter.set_active_loras.call_args.args[0]
    assert len(loras) == 1 and loras[0].name == "anime-v2" and loras[0].strength == 0.7

    # 3. encode_prompt called with the prompt text
    assert len(fake_image_adapter["encode_calls"]) == 1
    assert fake_image_adapter["encode_calls"][0]["prompt"] == "a cat in space"

    # 4. sample called with the conditioning bundle + sampling params
    assert len(fake_image_adapter["sample_calls"]) == 1
    s = fake_image_adapter["sample_calls"][0]
    assert s["conditioning"]["prompt_embeds"] == "EMBEDS"
    assert s["kw"]["width"] == 768
    assert s["kw"]["height"] == 768
    assert s["kw"]["num_inference_steps"] == 30
    assert s["kw"]["guidance_scale"] == 5.5

    # 5. vae_decode called with sample's output
    assert len(fake_image_adapter["vae_decode_calls"]) == 1
    assert fake_image_adapter["vae_decode_calls"][0]["latents"] == "LATENTS"

    # 6. Output schema preserved byte-for-byte vs the old adapter.infer route
    assert out["image_url"].startswith("/files/images/")
    assert "image" not in out
    assert out["media_type"] == "image/png"
    assert out["width"] == 768
    assert out["height"] == 768
    assert out["steps"] == 30
    assert out["seed"] == 1234
    assert out["cfg_scale"] == 5.5
    assert out["loras"] == [{"name": "anime-v2", "strength": 0.7}]
    assert isinstance(out["duration_ms"], int)


async def test_image_generate_falls_back_to_text_input(fake_image_adapter):
    """Wired downstream of a text_input node, prompt arrives via inputs.text."""
    from src.services.nodes.image import ImageGenerateNode

    out = await ImageGenerateNode().invoke(
        data={"model_key": "flux2-klein-9b"},
        inputs={"text": "via text edge"},
    )
    assert fake_image_adapter["encode_calls"][0]["prompt"] == "via text edge"
    assert out["image_url"]


async def test_image_generate_uses_data_prompt_when_no_input(fake_image_adapter):
    from src.services.nodes.image import ImageGenerateNode

    out = await ImageGenerateNode().invoke(
        data={"model_key": "flux2-klein-9b", "prompt": "from data"},
        inputs={},
    )
    assert fake_image_adapter["encode_calls"][0]["prompt"] == "from data"
    assert out["image_url"]


async def test_image_generate_generates_random_seed_when_unset(fake_image_adapter):
    """Omit seed → secrets.randbelow draws one, echoed back in metadata."""
    from src.services.nodes.image import ImageGenerateNode

    out = await ImageGenerateNode().invoke(
        data={"model_key": "flux2-klein-9b"},
        inputs={"prompt": "p"},
    )
    assert isinstance(out["seed"], int)
    assert 0 <= out["seed"] < 2**63


async def test_image_generate_missing_prompt_raises():
    from src.services import workflow_executor as we
    from src.services.nodes.image import ImageGenerateNode

    node = ImageGenerateNode()
    with pytest.raises(we.ExecutionError, match="prompt"):
        await node.invoke(data={"model_key": "x"}, inputs={})


async def test_image_generate_missing_model_key_raises():
    from src.services import workflow_executor as we
    from src.services.nodes.image import ImageGenerateNode

    node = ImageGenerateNode()
    with pytest.raises(we.ExecutionError, match="model_key"):
        await node.invoke(data={}, inputs={"prompt": "hi"})


async def test_image_generate_missing_model_manager_raises(monkeypatch):
    from src.services import workflow_executor as we
    from src.services.nodes.image import ImageGenerateNode

    monkeypatch.setattr(we, "_model_manager", None)
    node = ImageGenerateNode()
    with pytest.raises(we.ExecutionError, match="ModelManager"):
        await node.invoke(data={"model_key": "x"}, inputs={"prompt": "hi"})


async def test_image_generate_default_field_values(fake_image_adapter):
    """Omit advanced fields → defaults flow through to the helpers + metadata."""
    from src.services.nodes.image import ImageGenerateNode

    out = await ImageGenerateNode().invoke(
        data={"model_key": "flux2-klein-9b"},
        inputs={"prompt": "default size"},
    )
    s = fake_image_adapter["sample_calls"][0]
    assert s["kw"]["width"] == 1024
    assert s["kw"]["height"] == 1024
    assert s["kw"]["num_inference_steps"] == 25
    assert s["kw"]["guidance_scale"] == 7.0

    assert out["width"] == 1024
    assert out["height"] == 1024
    assert out["steps"] == 25
    assert out["cfg_scale"] == 7.0
    assert out["loras"] == []


async def test_image_generate_skips_lora_entries_without_name(fake_image_adapter):
    """Defensive: a UI dropdown left blank arrives as {'name': '', 'strength': 1}.
    set_active_loras would crash at lora_paths lookup with a blank name;
    _coerce_loras filters those out before they reach the adapter."""
    from src.services.nodes.image import ImageGenerateNode

    out = await ImageGenerateNode().invoke(
        data={
            "model_key": "flux2-klein-9b",
            "loras": [
                {"name": "valid", "strength": 0.5},
                {"name": "", "strength": 1.0},
                {"strength": 1.0},  # name missing
            ],
        },
        inputs={"prompt": "p"},
    )
    adapter = fake_image_adapter["adapter"]
    specs = adapter.set_active_loras.call_args.args[0]
    assert [s.name for s in specs] == ["valid"]
    assert out["loras"] == [{"name": "valid", "strength": 0.5}]


async def test_image_output_passes_through_image_envelope():
    from src.services.nodes.image import ImageOutputNode

    node = ImageOutputNode()
    out = await node.invoke(
        data={},
        inputs={
            "image_url": "/files/images/2026-05-04/abcd.png?token=t&expires=1",
            "media_type": "image/png",
            "width": 1024,
            "height": 1024,
            "stray": "ignored",
        },
    )
    assert out == {
        "image_url": "/files/images/2026-05-04/abcd.png?token=t&expires=1",
        "media_type": "image/png",
        "width": 1024,
        "height": 1024,
    }


async def test_image_output_defaults_when_input_missing():
    from src.services.nodes.image import ImageOutputNode

    out = await ImageOutputNode().invoke(data={}, inputs={})
    assert out["image_url"] is None
    assert "image" not in out
    assert out["media_type"] == "image/png"
    assert out["width"] is None


def test_image_nodes_registered():
    """@register decorators must fire on import; otherwise the dispatcher
    falls through to the legacy plugin path and raises 'Unknown node type'."""
    from src.services.nodes.registry import get_node_class

    assert get_node_class("image_generate") is not None
    assert get_node_class("image_output") is not None
