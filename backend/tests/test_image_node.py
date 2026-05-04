"""Tests for image_generate / image_output nodes (PR-4 backend half)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def fake_image_adapter(monkeypatch):
    """Install a fake v2 IMAGE adapter on workflow_executor._model_manager.

    The fake captures the ImageRequest pydantic instance so tests can
    assert field passthrough, and returns a tiny PNG-marker payload so
    the node's base64 round-trip can be verified.
    """
    from src.services import workflow_executor as we
    from src.services.inference.base import InferenceResult, UsageMeter

    captured: dict = {}

    async def _infer(req):
        captured["req"] = req
        return InferenceResult(
            media_type="image/png",
            data=b"\x89PNGFAKE",
            metadata={
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "seed": req.seed,
                "loras": [{"name": s.name, "strength": s.strength} for s in req.loras],
            },
            usage=UsageMeter(image_count=1, latency_ms=42),
        )

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.infer = _infer

    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    monkeypatch.setattr(we, "_model_manager", mgr)
    return captured


async def test_image_generate_dispatches_to_adapter_with_typed_request(fake_image_adapter):
    from src.services.nodes.image import ImageGenerateNode

    node = ImageGenerateNode()
    out = await node.invoke(
        data={
            "model_key": "flux2-klein-9b",
            "negative_prompt": "blurry",
            "width": 768,
            "height": 768,
            "steps": 30,
            "seed": 1234,
            "cfg_scale": 5.5,
            "loras": [{"name": "anime-v2", "strength": 0.7}],
        },
        inputs={"prompt": "a cat in space"},
    )

    req = fake_image_adapter["req"]
    assert req.prompt == "a cat in space"
    assert req.negative_prompt == "blurry"
    assert req.width == 768
    assert req.height == 768
    assert req.steps == 30
    assert req.seed == 1234
    assert req.cfg_scale == 5.5
    assert len(req.loras) == 1
    assert req.loras[0].name == "anime-v2"
    assert req.loras[0].strength == 0.7

    # Signed URL is the only render path now (p2-polish-3 dropped base64).
    assert out["image_url"].startswith("/files/images/")
    assert "image" not in out
    assert out["media_type"] == "image/png"
    assert out["width"] == 768
    assert out["seed"] == 1234
    assert out["loras"] == [{"name": "anime-v2", "strength": 0.7}]
    assert out["duration_ms"] == 42


async def test_image_generate_falls_back_to_text_input(fake_image_adapter):
    """Wired downstream of a text_input node, prompt arrives via inputs.text."""
    from src.services.nodes.image import ImageGenerateNode

    node = ImageGenerateNode()
    out = await node.invoke(
        data={"model_key": "flux2-klein-9b"},
        inputs={"text": "via text edge"},
    )
    assert fake_image_adapter["req"].prompt == "via text edge"
    assert out["image_url"]


async def test_image_generate_uses_data_prompt_when_no_input(fake_image_adapter):
    from src.services.nodes.image import ImageGenerateNode

    node = ImageGenerateNode()
    out = await node.invoke(
        data={"model_key": "flux2-klein-9b", "prompt": "from data"},
        inputs={},
    )
    assert fake_image_adapter["req"].prompt == "from data"
    assert out["image_url"]


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
    """Omit advanced fields → schema defaults from ImageRequest."""
    from src.services.nodes.image import ImageGenerateNode

    node = ImageGenerateNode()
    await node.invoke(
        data={"model_key": "flux2-klein-9b"},
        inputs={"prompt": "default size"},
    )
    req = fake_image_adapter["req"]
    assert req.width == 1024
    assert req.height == 1024
    assert req.steps == 25
    assert req.seed is None
    assert req.cfg_scale == 7.0
    assert req.loras == []


async def test_image_generate_skips_lora_entries_without_name(fake_image_adapter):
    """Defensive: a UI dropdown left blank arrives as {'name': '', 'strength': 1}.
    The adapter would crash at lora_paths lookup; coerce filters first."""
    from src.services.nodes.image import ImageGenerateNode

    node = ImageGenerateNode()
    await node.invoke(
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
    req = fake_image_adapter["req"]
    assert [s.name for s in req.loras] == ["valid"]


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
