"""PR-6: L2 image output cache — deterministic key + LRU + serve/re-sign."""
from __future__ import annotations

from src.runner import protocol as P
from src.services.inference.base import ImageRequest, LoRASpec
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_l2_cache import ImageOutputCache, image_l2_key, serve_image_l2


def _node(model_key=None, deterministic=True):
    return P.RunNode(task_id=1, node_id="g", node_type="image", model_key=model_key,
                     inputs={}, is_deterministic=deterministic)


def _req_components(seed=42, prompt="a cat"):
    comps = {
        "unet": ComponentSpec(kind="unet", file="/m/u.safe", device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16", clip_arch="flux2"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }
    return ImageRequest(request_id="r", prompt=prompt, seed=seed, steps=9, width=512, height=512, components=comps)


def test_key_stable_and_sensitive():
    k1 = image_l2_key(_node(), _req_components(seed=42))
    assert k1 == image_l2_key(_node(), _req_components(seed=42))
    assert k1 != image_l2_key(_node(), _req_components(seed=43))
    assert k1 != image_l2_key(_node(), _req_components(prompt="a dog"))


def test_key_legacy_model_key_path():
    req = ImageRequest(request_id="r", prompt="x", seed=1, loras=[LoRASpec(name="s", strength=0.8)])
    assert image_l2_key(_node(model_key="flux2-klein-9b"), req) != image_l2_key(_node(model_key="other"), req)


def test_lru_evicts_oldest():
    c = ImageOutputCache(maxsize=2)
    c.put("a", {"image_uuid": "a"})
    c.put("b", {"image_uuid": "b"})
    c.get("a")
    c.put("c", {"image_uuid": "c"})
    assert c.get("a") is not None and c.get("c") is not None and c.get("b") is None


def test_serve_miss_when_png_gone(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    entry = {"image_uuid": "ghost", "date": "2026-05-20", "ext": "png", "meta": {}, "width": 512, "height": 512}
    assert serve_image_l2(entry, ttl=3600) is None


def test_serve_hit_resigns(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "sek")
    d = tmp_path / "2026-05-20"
    d.mkdir()
    (d / "real.png").write_bytes(b"\x89PNG")
    entry = {"image_uuid": "real", "date": "2026-05-20", "ext": "png", "meta": {"seed": 42}, "width": 512, "height": 512}
    out = serve_image_l2(entry, ttl=3600)
    assert out is not None and out["cached"] is True
    assert "/files/images/2026-05-20/real.png?token=" in out["image_url"]
    assert out["image_uuid"] == "real" and out["width"] == 512
