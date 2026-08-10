"""Tests for write_media alias + ffmpeg first-frame thumbnail extraction."""
from __future__ import annotations

import asyncio

import pytest

from src.services.comfy import thumbnail
from src.services.image_output_storage import write_media
from src.services.comfy.thumbnail import extract_first_frame


def test_write_media_mp4_roundtrip(tmp_path, monkeypatch):
    # _outputs_root() honors $NOUS_IMAGE_OUTPUTS as its override env var
    # (verified in src/services/image_output_storage.py:_outputs_root).
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    out = write_media(b"\x00fakemp4", ext="mp4", ttl_seconds=60)
    assert out["path"].read_bytes() == b"\x00fakemp4"
    assert out["path"].suffix == ".mp4"
    assert out["ext"] == "mp4"
    # conftest sets ADMIN_SESSION_SECRET globally, so a signed URL is issued.
    assert out["url"].endswith(".mp4") or "mp4" in out["url"]


@pytest.mark.asyncio
async def test_extract_first_frame_missing_ffmpeg_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # 空 PATH → ffmpeg 不存在
    assert await extract_first_frame(tmp_path / "x.mp4") is None


@pytest.mark.asyncio
async def test_extract_first_frame_permission_error_returns_none(tmp_path, monkeypatch):
    """OSError (not just FileNotFoundError) must degrade to None — a present
    but non-executable ffmpeg binary raises PermissionError, which is a
    subclass of OSError, not FileNotFoundError."""

    async def _raise_permission_error(*args, **kwargs):
        raise PermissionError("ffmpeg not executable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _raise_permission_error)
    assert await extract_first_frame(tmp_path / "x.mp4") is None


@pytest.mark.asyncio
async def test_extract_first_frame_timeout_kills_process(tmp_path, monkeypatch):
    """A hung ffmpeg (e.g. decoding a corrupt/huge input) must not block the
    caller forever — extract_first_frame degrades to None once its internal
    timeout fires, and kills the child rather than leaking it."""
    monkeypatch.setattr(thumbnail, "_TIMEOUT_SECONDS", 0.05)
    killed = False

    class _HangingProc:
        async def wait(self):
            nonlocal killed
            if not killed:
                await asyncio.sleep(10)  # never resolves within the test's patience
            return 0

        def kill(self):
            nonlocal killed
            killed = True

    async def _fake_create_subprocess_exec(*args, **kwargs):
        return _HangingProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    result = await extract_first_frame(tmp_path / "hung.mp4")
    assert result is None
    assert killed is True
