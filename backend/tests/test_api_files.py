"""End-to-end Files API boundary tests (Step 5)."""

from __future__ import annotations

import io

import pytest


def _png_bytes(size: int = 100) -> bytes:
    # Tiny valid-ish payload — we don't need a real PNG, server doesn't decode.
    return b"\x89PNG\r\n\x1a\n" + b"x" * (size - 8)


@pytest.fixture
def file_client(db_client, sample_api_key):
    """db_client + auth header preloaded."""
    db_client.headers["Authorization"] = f"Bearer {sample_api_key}"
    return db_client


async def _upload(client, *, name="cat.png", purpose="vision", size=100, content_type="image/png"):
    files = {"file": (name, io.BytesIO(_png_bytes(size)), content_type)}
    return await client.post("/v1/files", files=files, data={"purpose": purpose})


async def test_upload_basic(file_client):
    r = await _upload(file_client)
    assert r.status_code == 200
    body = r.json()
    assert body["id"].startswith("file-")
    assert body["bytes"] == 100
    assert body["purpose"] == "vision"


async def test_upload_invalid_purpose(file_client):
    r = await _upload(file_client, purpose="not_a_purpose")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_purpose"


async def test_upload_idempotent_returns_same_id(file_client):
    r1 = await _upload(file_client)
    r2 = await _upload(file_client)
    assert r1.json()["id"] == r2.json()["id"]


async def test_upload_too_large(file_client, monkeypatch):
    # Slash the cap so we don't have to send 50MB in a unit test
    from src.api.routes import files as files_route
    monkeypatch.setattr(files_route, "MAX_FILE_BYTES", 256)
    r = await _upload(file_client, size=512)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "file_too_large"


async def test_get_metadata_and_download_roundtrip(file_client):
    payload = _png_bytes(200)
    r = await file_client.post(
        "/v1/files",
        files={"file": ("c.png", io.BytesIO(payload), "image/png")},
        data={"purpose": "vision"},
    )
    fid = r.json()["id"]

    meta = await file_client.get(f"/v1/files/{fid}")
    assert meta.status_code == 200
    assert meta.json()["bytes"] == 200

    blob = await file_client.get(f"/v1/files/{fid}/content")
    assert blob.status_code == 200
    assert blob.content == payload


async def test_get_unknown_file_404(file_client):
    r = await file_client.get("/v1/files/file-nonexistent")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "file_not_found"


async def test_other_instance_blocked_403(db_client, sample_api_key, sample_instance, db_session):
    """A file uploaded by instance A must not be GETtable by instance B."""
    import bcrypt
    import secrets as _sec
    from src.models.instance_api_key import InstanceApiKey
    from src.models.service_instance import ServiceInstance

    # Upload as instance A
    db_client.headers["Authorization"] = f"Bearer {sample_api_key}"
    r = await _upload(db_client)
    fid = r.json()["id"]

    # Create instance B + key
    inst_b = ServiceInstance(
        source_type="model", source_name="other", name="b",
        type="llm", status="active",
    )
    db_session.add(inst_b)
    await db_session.commit()
    await db_session.refresh(inst_b)
    raw_b = f"sk-other-{_sec.token_hex(8)}"
    db_session.add(InstanceApiKey(
        instance_id=inst_b.id, label="b",
        key_hash=bcrypt.hashpw(raw_b.encode(), bcrypt.gensalt()).decode(),
        key_prefix=raw_b[:10], is_active=True,
    ))
    await db_session.commit()

    db_client.headers["Authorization"] = f"Bearer {raw_b}"
    cross = await db_client.get(f"/v1/files/{fid}")
    assert cross.status_code == 403
    assert cross.json()["error"]["code"] == "file_wrong_instance"


async def test_list_pagination_with_after_cursor(file_client):
    ids = []
    for i in range(5):
        r = await _upload(file_client, name=f"f{i}.png", size=100 + i)
        ids.append(r.json()["id"])

    page1 = await file_client.get("/v1/files?limit=2")
    p1 = page1.json()
    assert len(p1["data"]) == 2
    assert p1["has_more"] is True

    page2 = await file_client.get(f"/v1/files?limit=2&after={p1['last_id']}")
    p2 = page2.json()
    assert len(p2["data"]) == 2
    # No overlap between pages
    p1_ids = {f["id"] for f in p1["data"]}
    p2_ids = {f["id"] for f in p2["data"]}
    assert not (p1_ids & p2_ids)


async def test_list_filter_by_purpose(file_client):
    await _upload(file_client, name="a.png", purpose="vision")
    await _upload(file_client, name="b.txt", purpose="user_data", content_type="text/plain")
    r = await file_client.get("/v1/files?purpose=vision")
    purposes = {f["purpose"] for f in r.json()["data"]}
    assert purposes == {"vision"}


async def test_after_cursor_unknown_400(file_client):
    r = await file_client.get("/v1/files?after=file-doesnotexist")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


async def test_delete_then_get_404(file_client):
    r = await _upload(file_client)
    fid = r.json()["id"]
    d = await file_client.delete(f"/v1/files/{fid}")
    assert d.status_code == 200
    assert d.json()["deleted"] is True
    g = await file_client.get(f"/v1/files/{fid}")
    assert g.status_code == 404


async def test_unauthenticated_rejected(db_client):
    r = await db_client.get("/v1/files")
    # FastAPI's required Header dep can return 400/401/422 depending on version.
    assert r.status_code in (400, 401, 422)
