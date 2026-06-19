"""GET /v1/models 按 key 授权返回服务(发现==能调,对齐 Doubao 式 provider)。

修前 bug:/v1/models 列全量注册表(模型 id)、无视 key 授权 → 客户端发现到的名字调用即 404。
修后:返回该 key active-grant 的全部服务(各类目),id=服务名(与 /v1/chat 等一致)。
"""
from __future__ import annotations

import bcrypt
import pytest

from src.models.api_gateway import ApiKeyGrant
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance


def _hash(t: str) -> str:
    return bcrypt.hashpw(t.encode(), bcrypt.gensalt()).decode()


async def _svc(db_session, name, category):
    s = ServiceInstance(source_type="workflow", source_name="x", name=name,
                        type="inference", status="active", category=category)
    db_session.add(s)
    await db_session.commit()
    await db_session.refresh(s)
    return s


async def _key(db_session, prefix):
    raw = prefix + "abcdef"
    k = InstanceApiKey(instance_id=None, label="t", key_hash=_hash(raw),
                       key_prefix=raw[:10], is_active=True)
    db_session.add(k)
    await db_session.commit()
    await db_session.refresh(k)
    return raw, k


@pytest.mark.asyncio
async def test_models_lists_only_granted_services(db_client, db_session):
    llm = await _svc(db_session, "qwen-chat", "llm")
    img = await _svc(db_session, "ideogram-img", "image")
    await _svc(db_session, "other-svc", "llm")  # 不授权这个
    raw, key = await _key(db_session, "sk-mdl12345")
    db_session.add_all([
        ApiKeyGrant(api_key_id=key.id, service_id=llm.id, status="active"),
        ApiKeyGrant(api_key_id=key.id, service_id=img.id, status="active"),
    ])
    await db_session.commit()

    r = await db_client.get("/v1/models", headers={"Authorization": f"Bearer {raw}"})
    assert r.status_code == 200, r.text
    body = r.json()
    ids = {m["id"]: m["type"] for m in body["data"]}
    assert ids == {"qwen-chat": "llm", "ideogram-img": "image"}  # 只授权的两个,服务名+类目
    assert "other-svc" not in ids  # 未授权的不出现


@pytest.mark.asyncio
async def test_models_type_filter(db_client, db_session):
    llm = await _svc(db_session, "c1", "llm")
    img = await _svc(db_session, "i1", "image")
    raw, key = await _key(db_session, "sk-flt12345")
    db_session.add_all([
        ApiKeyGrant(api_key_id=key.id, service_id=llm.id, status="active"),
        ApiKeyGrant(api_key_id=key.id, service_id=img.id, status="active"),
    ])
    await db_session.commit()
    r = await db_client.get("/v1/models?type=llm", headers={"Authorization": f"Bearer {raw}"})
    assert [m["id"] for m in r.json()["data"]] == ["c1"]


@pytest.mark.asyncio
async def test_models_excludes_paused_grant(db_client, db_session):
    llm = await _svc(db_session, "p1", "llm")
    raw, key = await _key(db_session, "sk-pau12345")
    db_session.add(ApiKeyGrant(api_key_id=key.id, service_id=llm.id, status="paused"))
    await db_session.commit()
    r = await db_client.get("/v1/models", headers={"Authorization": f"Bearer {raw}"})
    assert r.json()["data"] == []  # paused 不算可调


@pytest.mark.asyncio
async def test_get_one_model_grant_scoped(db_client, db_session):
    llm = await _svc(db_session, "g1", "llm")
    raw, key = await _key(db_session, "sk-one12345")
    db_session.add(ApiKeyGrant(api_key_id=key.id, service_id=llm.id, status="active"))
    await db_session.commit()
    h = {"Authorization": f"Bearer {raw}"}
    ok = await db_client.get("/v1/models/g1", headers=h)
    assert ok.status_code == 200 and ok.json()["id"] == "g1" and ok.json()["type"] == "llm"
    miss = await db_client.get("/v1/models/not-granted", headers=h)
    assert miss.status_code == 404
