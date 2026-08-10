"""Tests for write_media alias + ffmpeg first-frame thumbnail extraction."""
from __future__ import annotations

import pytest

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
