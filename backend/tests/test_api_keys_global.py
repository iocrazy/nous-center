"""m10 全局 API Key 路由测试 — 创建 / list / 详情 / reset / 一键多授权。

只跑 SQLite，绕开真实 vLLM。Anthropic 兼容端点的 happy-path 实测留
test_anthropic_compat.py 跑（mock model_manager）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.api_gateway import ApiKeyGrant
from src.models.service_instance import ServiceInstance


@pytest.fixture
async def two_services(db_session):
    a = ServiceInstance(
        name="svc-a", type="inference", status="active",
        source_type="model", source_name="qwen", category="llm", meter_dim="tokens",
    )
    b = ServiceInstance(
        name="svc-b", type="inference", status="active",
        source_type="model", source_name="claude", category="llm", meter_dim="tokens",
    )
    db_session.add_all([a, b])
    await db_session.commit()
    await db_session.refresh(a)
    await db_session.refresh(b)
    return a, b


@pytest.mark.asyncio
async def test_create_key_returns_plaintext_and_grants(db_client, two_services):
    a, b = two_services
    r = await db_client.post(
        "/api/v1/keys",
        json={
            "label": "my-key", "note": "for mediahub",
            "service_ids": [a.id, b.id],
        },
    )
    assert r.status_code == 201, r.text
    data = r.json()
    # secret 字段是创建时一次性返回的明文。
    assert data["secret"].startswith("sk-")
    assert data["secret_plaintext"] == data["secret"]
    assert data["label"] == "my-key"
    assert data["note"] == "for mediahub"
    assert data["grant_count"] == 2
    assert data["active_grant_count"] == 2
    svc_ids = sorted(g["service_id"] for g in data["grants"])
    assert svc_ids == sorted([a.id, b.id])


@pytest.mark.asyncio
async def test_create_key_rejects_unknown_service(db_client, two_services):
    a, _ = two_services
    r = await db_client.post(
        "/api/v1/keys",
        json={"label": "bad", "service_ids": [a.id, 99999]},
    )
    assert r.status_code == 404
    assert "99999" in r.text


@pytest.mark.asyncio
async def test_list_keys_includes_grant_counts_and_array(db_client, two_services):
    a, b = two_services
    await db_client.post("/api/v1/keys", json={"label": "k1", "service_ids": [a.id]})
    await db_client.post(
        "/api/v1/keys", json={"label": "k2", "service_ids": [a.id, b.id]},
    )
    r = await db_client.get("/api/v1/keys")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    by_label = {row["label"]: row for row in rows}
    assert by_label["k1"]["grant_count"] == 1
    assert by_label["k2"]["grant_count"] == 2
    # m10 列表页要靠 grants 数组渲染"授权服务"徽章 — list 必须填上。
    assert {g["service_name"] for g in by_label["k1"]["grants"]} == {"svc-a"}
    assert {g["service_name"] for g in by_label["k2"]["grants"]} == {"svc-a", "svc-b"}


@pytest.mark.asyncio
async def test_get_key_includes_grant_summaries(db_client, two_services):
    a, b = two_services
    created = (await db_client.post(
        "/api/v1/keys", json={"label": "detail", "service_ids": [a.id, b.id]},
    )).json()
    r = await db_client.get(f"/api/v1/keys/{created['id']}")
    assert r.status_code == 200
    d = r.json()
    assert d["grant_count"] == 2
    names = {g["service_name"] for g in d["grants"]}
    assert names == {"svc-a", "svc-b"}


@pytest.mark.asyncio
async def test_reset_key_changes_secret_and_prefix(db_client, two_services):
    a, _ = two_services
    created = (await db_client.post(
        "/api/v1/keys", json={"label": "rot", "service_ids": [a.id]},
    )).json()
    old_secret = created["secret"]
    old_prefix = created["key_prefix"]

    r = await db_client.post(f"/api/v1/keys/{created['id']}/reset")
    assert r.status_code == 200
    new = r.json()
    assert new["secret"] != old_secret
    assert new["secret_plaintext"] == new["secret"]
    # prefix 也跟着轮换（前 10 位）。
    assert new["key_prefix"] != old_prefix
    # grants 不变。
    assert new["grant_count"] == 1


@pytest.mark.asyncio
async def test_patch_key_label_and_disable(db_client, two_services):
    a, _ = two_services
    created = (await db_client.post(
        "/api/v1/keys", json={"label": "p1", "service_ids": [a.id]},
    )).json()
    expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    r = await db_client.patch(
        f"/api/v1/keys/{created['id']}",
        json={"label": "renamed", "is_active": False, "expires_at": expires},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["label"] == "renamed"
    assert d["is_active"] is False
    assert d["expires_at"].startswith(expires[:19])


@pytest.mark.asyncio
async def test_delete_key_cascades_grants(db_client, two_services, db_session):
    a, _ = two_services
    created = (await db_client.post(
        "/api/v1/keys", json={"label": "doomed", "service_ids": [a.id]},
    )).json()
    r = await db_client.delete(f"/api/v1/keys/{created['id']}")
    assert r.status_code == 204
    # grants 跟着 cascade 删干净。
    from sqlalchemy import select
    rows = (await db_session.execute(
        select(ApiKeyGrant).where(ApiKeyGrant.api_key_id == created["id"])
    )).all()
    assert rows == []


@pytest.mark.asyncio
async def test_service_grants_endpoint_for_m03_tab(db_client, two_services):
    a, b = two_services
    await db_client.post(
        "/api/v1/keys", json={"label": "k1", "service_ids": [a.id]},
    )
    await db_client.post(
        "/api/v1/keys", json={"label": "k2", "service_ids": [a.id, b.id]},
    )
    r = await db_client.get(f"/api/v1/services/{a.id}/grants")
    assert r.status_code == 200, r.text
    rows = r.json()
    assert len(rows) == 2
    labels = {row["api_key_label"] for row in rows}
    assert labels == {"k1", "k2"}
    for row in rows:
        assert row["pack_total"] == 0
        assert row["pack_used"] == 0
        assert row["grant_status"] == "active"


@pytest.mark.asyncio
async def test_service_grants_404_when_service_missing(db_client):
    r = await db_client.get("/api/v1/services/99999/grants")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_key_shows_orphan_grants_with_fallback_name(
    db_client, db_session, two_services,
):
    """v3 IA 重构清理过 service_instances，旧 grant 行可能指向已删的
    service_id。LEFT JOIN 让这些孤儿 grant 仍出现在详情，service_name
    显示为"(已删除)"，UI 才能解除。"""
    a, _ = two_services
    created = (await db_client.post(
        "/api/v1/keys",
        json={"label": "k-orphan", "service_ids": [a.id]},
    )).json()
    # 直接物理删 service，把 grant 变成孤儿（cascade 用模型默认 ondelete）
    await db_session.delete(await db_session.get(ServiceInstance, a.id))
    await db_session.commit()

    r = await db_client.get(f"/api/v1/keys/{created['id']}")
    assert r.status_code == 200
    d = r.json()
    # grant_count 仍统计到（旧 INNER JOIN 会导致 grants:[] 但 count=1）
    assert d["grant_count"] == 1
    # 孤儿 grant 用兜底 name 露出来
    assert len(d["grants"]) == 1
    assert d["grants"][0]["service_name"] == "(已删除)"
    assert d["grants"][0]["service_category"] is None
