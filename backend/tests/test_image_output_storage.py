"""Tests for image_output_storage + /files/images signed-URL route."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def storage_tmp(tmp_path, monkeypatch):
    """Redirect storage to a tmp dir AND restore on teardown."""
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path / "outputs"))
    yield tmp_path / "outputs"


@pytest.fixture
def with_signing_secret(monkeypatch):
    """Force ADMIN_SESSION_SECRET so the signing path runs."""
    from src.config import get_settings
    from src.services import image_output_storage as svc

    settings = get_settings()
    monkeypatch.setattr(settings, "ADMIN_SESSION_SECRET", "test-secret-bytes")
    # The module reads via get_settings() each call, so no further patching
    # needed — but cache is per-process; just toggle directly.
    return svc


# ----- write_image -----


def test_write_image_writes_under_dated_subdir(storage_tmp):
    from src.services.image_output_storage import write_image

    rec = write_image(b"\x89PNGFAKE", ext="png")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    expected_dir = storage_tmp / today
    assert rec["path"].parent == expected_dir
    assert rec["path"].suffix == ".png"
    assert rec["path"].read_bytes() == b"\x89PNGFAKE"
    assert rec["uuid"] == rec["path"].stem
    assert len(rec["uuid"]) == 32  # uuid4 hex


def test_write_image_no_secret_yields_null_url(storage_tmp):
    from src.services.image_output_storage import write_image

    rec = write_image(b"x", ext="png")
    # conftest forces ADMIN_SESSION_SECRET="" → no signing
    assert rec["url"] is None
    assert rec["expires"] is None


def test_write_image_with_secret_signs_url(storage_tmp, with_signing_secret):
    from src.services.image_output_storage import verify_token, write_image

    rec = write_image(b"x", ext="png", ttl_seconds=600)
    assert rec["url"] is not None
    assert rec["url"].startswith("/files/images/")
    assert "token=" in rec["url"] and "expires=" in rec["url"]
    assert rec["expires"] is not None
    # Round-trip verify
    token = rec["url"].split("token=")[1].split("&")[0]
    assert verify_token(rec["uuid"], rec["expires"], token)


def test_write_image_ttl_floor_is_60s(storage_tmp, with_signing_secret):
    from src.services.image_output_storage import write_image

    rec = write_image(b"x", ext="png", ttl_seconds=10)
    # max(60, 10) → expires at least 60 seconds out
    assert rec["expires"] - int(time.time()) >= 60


# ----- HMAC verify -----


def test_verify_token_rejects_tampered_uuid(storage_tmp, with_signing_secret):
    from src.services.image_output_storage import _sign, verify_token

    expires = int(time.time()) + 3600
    legit = _sign("aaaa", expires)
    assert verify_token("aaaa", expires, legit) is True
    assert verify_token("bbbb", expires, legit) is False


def test_verify_token_rejects_tampered_expires(storage_tmp, with_signing_secret):
    from src.services.image_output_storage import _sign, verify_token

    expires = int(time.time()) + 3600
    legit = _sign("aaaa", expires)
    assert verify_token("aaaa", expires + 1, legit) is False


def test_verify_token_rejects_expired(storage_tmp, with_signing_secret):
    from src.services.image_output_storage import _sign, verify_token

    expires = int(time.time()) - 1
    legit = _sign("aaaa", expires)
    assert verify_token("aaaa", expires, legit) is False


def test_verify_token_rejects_when_no_secret(storage_tmp):
    from src.services.image_output_storage import verify_token
    # No secret → can't sign, can't verify
    assert verify_token("aaaa", int(time.time()) + 3600, "anything") is False


# ----- /files/images route -----


@pytest.mark.asyncio
async def test_image_route_serves_valid_signed_url(storage_tmp, with_signing_secret, client):
    from src.services.image_output_storage import write_image

    rec = write_image(b"\x89PNG_BYTES", ext="png", ttl_seconds=600)
    resp = await client.get(rec["url"])
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == b"\x89PNG_BYTES"


@pytest.mark.asyncio
async def test_image_route_403_on_expired_token(storage_tmp, with_signing_secret, client):
    from src.services.image_output_storage import _sign, write_image

    rec = write_image(b"x", ext="png", ttl_seconds=600)
    expired = int(time.time()) - 10
    bad_token = _sign(rec["uuid"], expired)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = f"/files/images/{today}/{rec['uuid']}.png?token={bad_token}&expires={expired}"
    resp = await client.get(url)
    assert resp.status_code == 403
    body = resp.json()
    assert body["error"]["code"] == "url_expired"
    assert "fix" in body["error"]


@pytest.mark.asyncio
async def test_image_route_403_on_tampered_uuid(storage_tmp, with_signing_secret, client):
    from src.services.image_output_storage import write_image

    rec = write_image(b"x", ext="png", ttl_seconds=600)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    token = rec["url"].split("token=")[1].split("&")[0]
    expires = rec["expires"]
    other_uuid = "0" * 32
    url = f"/files/images/{today}/{other_uuid}.png?token={token}&expires={expires}"
    resp = await client.get(url)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_image_route_403_when_token_missing(client):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    resp = await client.get(f"/files/images/{today}/{'a' * 32}.png")
    assert resp.status_code == 403
    body = resp.json()
    assert body["error"]["code"] == "missing_signature"


@pytest.mark.asyncio
async def test_image_route_403_when_no_signing_secret(client):
    """Conftest forces secret = "". Even a 'valid'-looking URL must reject
    so a leaked URL from a one-off run with a transient secret can't
    survive a deploy that rotated keys."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    expires = int(time.time()) + 3600
    url = f"/files/images/{today}/{'a' * 32}.png?token=deadbeef&expires={expires}"
    resp = await client.get(url)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_image_route_400_on_bad_filename(client):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    resp = await client.get(f"/files/images/{today}/no-extension")
    assert resp.status_code == 403  # _UrlInvalidError → 403 per Stripe shape
    resp = await client.get(f"/files/images/{today}/file.exe?token=x&expires=1")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_image_route_400_on_bad_date_segment(client):
    resp = await client.get(f"/files/images/not-a-date/{'a' * 32}.png?token=x&expires=1")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_image_route_404_when_signed_but_file_missing(
    storage_tmp, with_signing_secret, client
):
    from src.services.image_output_storage import write_image

    rec = write_image(b"x", ext="png", ttl_seconds=600)
    # Delete the file but keep the (still valid) signed URL
    os.unlink(rec["path"])
    resp = await client.get(rec["url"])
    assert resp.status_code == 404


# ----- node integration: URL flows through envelope -----


@pytest.mark.asyncio
async def test_image_generate_emits_url_when_secret_configured(
    storage_tmp, with_signing_secret
):
    from unittest.mock import AsyncMock, MagicMock
    from src.services import workflow_executor as we
    from src.services.inference.base import InferenceResult, UsageMeter
    from src.services.nodes.image import ImageGenerateNode

    async def _infer(req):
        return InferenceResult(
            media_type="image/png",
            data=b"\x89PNG_REAL",
            metadata={"width": req.width, "height": req.height,
                      "steps": req.steps, "seed": req.seed, "loras": []},
            usage=UsageMeter(image_count=1, latency_ms=12),
        )

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.infer = _infer
    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    we._model_manager = mgr

    out = await ImageGenerateNode().invoke(
        data={"model_key": "flux2-klein-9b"},
        inputs={"prompt": "a cat in space"},
    )
    assert out["image_url"] is not None
    assert out["image_url"].startswith("/files/images/")
    assert out["image_uuid"]
    # In secret mode, base64 inline is omitted
    assert "image" not in out


@pytest.mark.asyncio
async def test_image_generate_falls_back_to_base64_in_dev_mode(storage_tmp):
    """No ADMIN_SESSION_SECRET → image_url is None, image (base64) is set."""
    from unittest.mock import AsyncMock, MagicMock
    from src.services import workflow_executor as we
    from src.services.inference.base import InferenceResult, UsageMeter
    from src.services.nodes.image import ImageGenerateNode

    async def _infer(req):
        return InferenceResult(
            media_type="image/png",
            data=b"\x89PNG_DEV",
            metadata={"width": req.width, "height": req.height,
                      "steps": req.steps, "seed": req.seed, "loras": []},
            usage=UsageMeter(image_count=1, latency_ms=1),
        )

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.infer = _infer
    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    we._model_manager = mgr

    out = await ImageGenerateNode().invoke(
        data={"model_key": "flux2-klein-9b"},
        inputs={"prompt": "hi"},
    )
    assert out["image_url"] is None
    import base64 as _b64
    assert _b64.b64decode(out["image"]) == b"\x89PNG_DEV"


def test_outputs_root_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path / "custom"))
    from src.services.image_output_storage import _outputs_root
    assert _outputs_root() == tmp_path / "custom"


def test_outputs_root_default_under_home(monkeypatch):
    monkeypatch.delenv("NOUS_IMAGE_OUTPUTS", raising=False)
    from src.services.image_output_storage import _outputs_root
    root = _outputs_root()
    assert root == Path.home() / ".gstack" / "outputs" / "images"
