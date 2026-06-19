"""GET /api/v1/keys/matrix — 对外出口控制台聚合(服务×key×grant + 今日调用)。"""
from __future__ import annotations

from datetime import datetime, timezone

import bcrypt
import pytest

from src.models.api_gateway import ApiKeyGrant
from src.models.instance_api_key import InstanceApiKey
from src.models.llm_usage import LLMUsage
from src.models.service_instance import ServiceInstance


def _hash(t: str) -> str:
    return bcrypt.hashpw(t.encode(), bcrypt.gensalt()).decode()


@pytest.mark.asyncio
async def test_matrix_shape_and_backing(db_client, db_session):
    llm = ServiceInstance(source_type="model", source_name="qwen3_6_35b_a3b_fp8",
                          name="qwen3-6-35b", type="inference", status="active", category="llm")
    img = ServiceInstance(source_type="workflow", source_name=None, name="ideogram4",
                          type="inference", status="active", category="image", workflow_id=999)
    db_session.add_all([llm, img])
    await db_session.commit()
    await db_session.refresh(llm)
    await db_session.refresh(img)

    key = InstanceApiKey(instance_id=None, label="k1", key_hash=_hash("sk-mx12345abc"),
                         key_prefix="sk-mx12345", is_active=True)
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)
    db_session.add(ApiKeyGrant(api_key_id=key.id, service_id=llm.id, status="active"))
    # 今日 2 条 llm 调用,归到 service=llm / key=key
    now = datetime.now(timezone.utc)
    db_session.add_all([
        LLMUsage(instance_id=llm.id, api_key_id=key.id, model="m", total_tokens=10, created_at=now),
        LLMUsage(instance_id=llm.id, api_key_id=key.id, model="m", total_tokens=10, created_at=now),
    ])
    await db_session.commit()

    r = await db_client.get("/api/v1/keys/matrix")
    assert r.status_code == 200, r.text
    body = r.json()

    svc = {s["name"]: s for s in body["services"]}
    assert svc["qwen3-6-35b"]["category"] == "llm"
    assert svc["qwen3-6-35b"]["backing"] == "qwen3_6_35b_a3b_fp8"   # model→source_name
    assert svc["qwen3-6-35b"]["today_calls"] == 2
    assert svc["ideogram4"]["backing"] == "wf:999"                  # workflow→wf:{id}
    assert svc["ideogram4"]["today_calls"] == 0

    k = {x["label"]: x for x in body["keys"]}
    assert k["k1"]["key_prefix"] == "sk-mx12345" and k["k1"]["today_calls"] == 2

    # grant 格:key↔llm active,带 grant_id(前端 revoke 用)
    g = body["grants"]
    assert any(x["key_id"] == str(key.id) and x["service_id"] == str(llm.id)
               and x["status"] == "active" and x["id"] for x in g)
    # ids 都是字符串(避免 JS bigint 精度丢失)
    assert all(isinstance(s["id"], str) for s in body["services"])
