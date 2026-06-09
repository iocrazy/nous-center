"""latent_storage 落盘 + 读回(spec 2026-06-08 路 B,PR-B1)。latent 一律落盘传路径引用,
不进 msgpack;写回 bit-exact 是同空间真 latent 接力的前提。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def _tmp_outputs(tmp_path, monkeypatch):
    # NOUS_IMAGE_OUTPUTS 指向 .../images → latent_storage 落到同级 latents/。
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path / "images"))
    return tmp_path


def test_write_latent_roundtrip_bit_exact(_tmp_outputs):
    from src.services.latent_storage import write_latent
    blob = os.urandom(4096)
    rec = write_latent(blob)
    assert set(rec) >= {"uuid", "path", "date"}
    p = Path(rec["path"])
    assert p.exists() and p.suffix == ".safetensors"
    assert "latents" in p.parts  # 落到 latents/ 而非 images/
    assert p.read_bytes() == blob  # bit-exact


def test_write_latent_unique_paths(_tmp_outputs):
    from src.services.latent_storage import write_latent
    a = write_latent(b"aaaa")
    b = write_latent(b"bbbb")
    assert a["path"] != b["path"]
    assert Path(a["path"]).read_bytes() == b"aaaa"
    assert Path(b["path"]).read_bytes() == b"bbbb"


def test_reap_orphans_handles_missing_root(_tmp_outputs):
    from src.services.latent_storage import reap_orphans
    # 根目录还没建(没写过)→ 不崩,返 0。
    s = reap_orphans(older_than_seconds=60)
    assert s["deleted"] == 0
