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


def test_write_image_writes_under_dated_subdir(storage_tmp, with_signing_secret):
    from src.services.image_output_storage import write_image

    rec = write_image(b"\x89PNGFAKE", ext="png")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    expected_dir = storage_tmp / today
    assert rec["path"].parent == expected_dir
    assert rec["path"].suffix == ".png"
    assert rec["path"].read_bytes() == b"\x89PNGFAKE"
    assert rec["uuid"] == rec["path"].stem
    assert len(rec["uuid"]) == 32  # uuid4 hex


def test_write_image_no_secret_yields_null_url(storage_tmp, monkeypatch):
    """Conftest sets ADMIN_SESSION_SECRET (since p2-polish-3 the node
    rejects empty); explicitly clear here to exercise the no-signing path."""
    from src.config import get_settings
    from src.services.image_output_storage import write_image

    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "")
    rec = write_image(b"x", ext="png")
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


def test_verify_token_rejects_when_no_secret(storage_tmp, monkeypatch):
    from src.config import get_settings
    from src.services.image_output_storage import verify_token

    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "")
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
async def test_image_route_403_when_no_signing_secret(client, monkeypatch):
    """Even a 'valid'-looking URL must reject so a leaked URL from a one-off
    run with a transient secret can't survive a deploy that rotated keys."""
    from src.config import get_settings

    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "")
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


def _install_fake_image_helpers(monkeypatch):
    """V1' Lane D P5 — image_generate composes encode_prompt + sample +
    vae_decode now. Stub all three so the node hits the real write_image
    path with a real-looking PIL image."""
    from unittest.mock import AsyncMock, MagicMock
    from src.services import workflow_executor as we
    from src.services.inference import image_diffusers as image_mod

    pil_image = MagicMock()
    pil_image.save = MagicMock(
        side_effect=lambda buf, format="PNG": buf.write(b"\x89PNG_REAL"),
    )

    monkeypatch.setattr(image_mod, "encode_prompt",
                        lambda *a, **kw: {"prompt_embeds": "E", "text_ids": "T"})
    monkeypatch.setattr(image_mod, "sample", lambda *a, **kw: "LATENTS")
    monkeypatch.setattr(image_mod, "vae_decode", lambda *a, **kw: pil_image)

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.device = "cuda:0"
    adapter.pipe = MagicMock()
    adapter.set_active_loras = MagicMock()

    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    monkeypatch.setattr(we, "_model_manager", mgr)

    import torch
    gen = MagicMock()
    gen.manual_seed = MagicMock(return_value=gen)
    monkeypatch.setattr(torch, "Generator", MagicMock(return_value=gen))


@pytest.mark.asyncio
async def test_image_generate_emits_url_when_secret_configured(
    storage_tmp, with_signing_secret, monkeypatch,
):
    from src.services.nodes.image import ImageGenerateNode

    _install_fake_image_helpers(monkeypatch)

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
async def test_image_generate_raises_when_no_signing_secret(storage_tmp, monkeypatch):
    """Dev-mode base64 fallback was removed in p2-polish-3. With no
    ADMIN_SESSION_SECRET write_image returns url=None and the node raises
    ExecutionError pointing the operator at the missing config."""
    from src.config import get_settings
    from src.services import workflow_executor as we
    from src.services.nodes.image import ImageGenerateNode

    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "")
    _install_fake_image_helpers(monkeypatch)

    with pytest.raises(we.ExecutionError, match="ADMIN_SESSION_SECRET"):
        await ImageGenerateNode().invoke(
            data={"model_key": "flux2-klein-9b"},
            inputs={"prompt": "hi"},
        )


def test_outputs_root_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path / "custom"))
    from src.services.image_output_storage import _outputs_root
    assert _outputs_root() == tmp_path / "custom"


def test_outputs_root_default_under_home(monkeypatch):
    monkeypatch.delenv("NOUS_IMAGE_OUTPUTS", raising=False)
    from src.services.image_output_storage import _outputs_root
    root = _outputs_root()
    assert root == Path.home() / ".gstack" / "outputs" / "images"


# ----- orphan reaper -----


def test_reap_orphans_deletes_only_old_files(storage_tmp):
    """Files older than cutoff get unlinked; fresh files survive."""
    import os as _os
    from src.services.image_output_storage import reap_orphans

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bucket = storage_tmp / today
    bucket.mkdir(parents=True)
    fresh = bucket / "fresh.png"
    old = bucket / "old.png"
    fresh.write_bytes(b"\x89PNG-fresh")
    old.write_bytes(b"\x89PNG-old")
    # Backdate `old` 2 days
    two_days = time.time() - 2 * 24 * 3600
    _os.utime(old, (two_days, two_days))

    summary = reap_orphans(older_than_seconds=24 * 3600)
    assert summary["scanned"] == 2
    assert summary["deleted"] == 1
    assert fresh.exists()
    assert not old.exists()


def test_reap_orphans_prunes_empty_date_dirs(storage_tmp):
    """After deletion the empty date subdir gets rmdir'd."""
    import os as _os
    from src.services.image_output_storage import reap_orphans

    bucket = storage_tmp / "2026-04-01"
    bucket.mkdir(parents=True)
    f = bucket / "lonely.png"
    f.write_bytes(b"x")
    _os.utime(f, (time.time() - 7 * 24 * 3600, time.time() - 7 * 24 * 3600))

    summary = reap_orphans(older_than_seconds=24 * 3600)
    assert summary["deleted"] == 1
    assert summary["dirs_pruned"] == 1
    assert not bucket.exists()


def test_reap_orphans_skips_non_image_files(storage_tmp):
    """Stray .md / .log files alongside don't get touched even if old."""
    import os as _os
    from src.services.image_output_storage import reap_orphans

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bucket = storage_tmp / today
    bucket.mkdir(parents=True)
    stray = bucket / "readme.md"
    stray.write_bytes(b"keep me")
    _os.utime(stray, (time.time() - 365 * 24 * 3600, time.time() - 365 * 24 * 3600))

    reap_orphans(older_than_seconds=24 * 3600)
    assert stray.exists()


def test_reap_orphans_no_root_dir_returns_zero(monkeypatch, tmp_path):
    """Outputs root never created yet → no error, summary all zeros."""
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path / "missing"))
    from src.services.image_output_storage import reap_orphans

    summary = reap_orphans(older_than_seconds=3600)
    assert summary == {"scanned": 0, "deleted": 0, "dirs_pruned": 0, "errors": 0}


def test_reap_orphans_floor_60s(storage_tmp):
    """older_than_seconds < 60 still keeps files newer than 60s — never
    accidentally nuke a brand-new file by passing a 0 cutoff."""
    from src.services.image_output_storage import reap_orphans

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bucket = storage_tmp / today
    bucket.mkdir(parents=True)
    fresh = bucket / "just-written.png"
    fresh.write_bytes(b"x")
    # mtime = now (newer than the 60s floor)

    summary = reap_orphans(older_than_seconds=0)  # operator passed 0
    assert summary["deleted"] == 0
    assert fresh.exists()
