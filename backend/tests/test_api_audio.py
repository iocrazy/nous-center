# Uses shared `client` fixture from conftest.py
from unittest.mock import patch
import struct


async def test_upload_audio(client, tmp_path):
    with patch("src.api.routes.audio._get_upload_dir", return_value=tmp_path):
        data_size = 100
        wav = bytearray()
        wav.extend(b"RIFF")
        wav.extend(struct.pack("<I", 36 + data_size))
        wav.extend(b"WAVEfmt ")
        wav.extend(struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16))
        wav.extend(b"data")
        wav.extend(struct.pack("<I", data_size))
        wav.extend(b"\x00" * data_size)

        resp = await client.post(
            "/api/v1/audio/upload",
            files={"file": ("test.wav", bytes(wav), "audio/wav")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["path"].endswith(".wav")


async def test_upload_rejects_non_audio(client):
    resp = await client.post(
        "/api/v1/audio/upload",
        files={"file": ("test.txt", b"not audio", "text/plain")},
    )
    assert resp.status_code == 422
