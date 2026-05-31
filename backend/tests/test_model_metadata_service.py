"""round9 PR-B · model_metadata upsert 鲁棒性。

覆盖两个回归:
- BUG2:refresh_metadata 旧实现「先 delete+commit 再 fetch」,fetch 网络失败返 None
  → 把好的元数据永久删了。现在 fetch 失败保留旧行。
- BUG3:fetch_and_store 旧实现恒 add(无 select-before-insert),对同一 engine_key
  二次调用撞 unique(engine_key) → IntegrityError 500。现在是 upsert。
"""

from __future__ import annotations

import pytest

from src.models.model_metadata import ModelMetadata
from src.services import model_metadata_service as svc


def _fake_meta(org: str) -> dict:
    return {
        "organization": org,
        "model_size_bytes": 123,
        "frameworks": None,
        "libraries": None,
        "license": "apache-2.0",
        "languages": None,
        "tags": None,
        "tensor_types": None,
        "description": None,
    }


async def _count(session) -> int:
    from sqlalchemy import func, select
    return (await session.execute(select(func.count()).select_from(ModelMetadata))).scalar_one()


@pytest.mark.asyncio
async def test_fetch_and_store_second_call_updates_not_500(db_session, monkeypatch):
    """同 engine_key 二次 fetch_and_store → update 同一行,不撞 unique。"""
    cfg = {"modelscope_id": "org/m"}
    monkeypatch.setattr(svc, "_fetch_modelscope", _amock(_fake_meta("first")))
    row1 = await svc.fetch_and_store(db_session, "eng-x", cfg)
    assert row1 is not None and row1.organization == "first"

    monkeypatch.setattr(svc, "_fetch_modelscope", _amock(_fake_meta("second")))
    row2 = await svc.fetch_and_store(db_session, "eng-x", cfg)  # 旧实现这里 IntegrityError 500
    assert row2 is not None and row2.organization == "second"
    assert await _count(db_session) == 1  # 没新增,是 upsert


@pytest.mark.asyncio
async def test_refresh_keeps_old_row_when_fetch_fails(db_session, monkeypatch):
    """fetch 返回 None(网络抖动)→ refresh 不清旧行(BUG2)。"""
    cfg = {"modelscope_id": "org/m"}
    monkeypatch.setattr(svc, "load_model_configs", lambda: {"eng-y": cfg})

    monkeypatch.setattr(svc, "_fetch_modelscope", _amock(_fake_meta("good")))
    await svc.fetch_and_store(db_session, "eng-y", cfg)
    assert await _count(db_session) == 1

    # 现在远端挂了:两个 fetcher 都返 None
    monkeypatch.setattr(svc, "_fetch_modelscope", _amock(None))
    monkeypatch.setattr(svc, "_fetch_huggingface", _amock(None))
    result = await svc.refresh_metadata(db_session, "eng-y")
    assert result is None  # 没拿到新数据
    assert await _count(db_session) == 1  # 但旧行还在(旧实现这里会是 0)

    from sqlalchemy import select
    kept = (await db_session.execute(
        select(ModelMetadata).where(ModelMetadata.engine_key == "eng-y")
    )).scalar_one()
    assert kept.organization == "good"


def _amock(return_value):
    async def _fn(*a, **k):
        return return_value
    return _fn
