"""Arc 3(spec 2026-07-20-moss-asr §8):record_api_call_task 单测。

锁三件事:① completed 正常落库(字段契约);② failed 落库(error/无 result);
③ 写库异常被吞(只 warning,绝不抛到主调用路)。独立 session 用 monkeypatch 指向 tmp
sqlite(仿 test_inference_usage_sqlite 的 get_session_factory patch 姿势)。
"""
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.database import Base
from src.models.execution_task import ExecutionTask


async def _make_sf(tmp_path, monkeypatch):
    from src.services import api_call_tasks

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'act.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(api_call_tasks, "get_session_factory", lambda: sf)
    return api_call_tasks, sf


@pytest.mark.asyncio
async def test_record_api_call_task_writes_completed(tmp_path, monkeypatch):
    api_call_tasks, sf = await _make_sf(tmp_path, monkeypatch)

    await api_call_tasks.record_api_call_task(
        service_name="moss-asr",
        api_key_id=42,
        status="completed",
        duration_ms=1234,
        input_json={"model": "moss-asr", "timestamps": True,
                    "context": None, "filename": "a.wav"},
        result={"text": "你好", "segments_count": 2,
                "speakers": ["S01", "S02"], "audio_seconds": 6},
    )

    async with sf() as s:
        rows = (await s.execute(select(ExecutionTask))).scalars().all()
    assert len(rows) == 1
    t = rows[0]
    assert t.workflow_id is None
    assert t.workflow_name == "moss-asr"
    assert t.api_key_id == 42
    assert t.status == "completed"
    assert t.duration_ms == 1234
    assert t.result["segments_count"] == 2
    assert t.result["audio_seconds"] == 6
    assert t.input_json["filename"] == "a.wav"
    assert t.error is None


@pytest.mark.asyncio
async def test_record_api_call_task_writes_failed(tmp_path, monkeypatch):
    api_call_tasks, sf = await _make_sf(tmp_path, monkeypatch)

    await api_call_tasks.record_api_call_task(
        service_name="moss-asr",
        api_key_id=None,
        status="failed",
        duration_ms=88,
        input_json={"model": "moss-asr", "timestamps": False,
                    "context": None, "filename": "b.wav"},
        error="MOSS ASR 引擎不可用",
    )

    async with sf() as s:
        rows = (await s.execute(select(ExecutionTask))).scalars().all()
    assert len(rows) == 1
    t = rows[0]
    assert t.status == "failed"
    assert t.api_key_id is None
    assert t.error == "MOSS ASR 引擎不可用"
    assert t.result is None


@pytest.mark.asyncio
async def test_record_api_call_task_swallows_errors(monkeypatch, caplog):
    """写库失败(session factory 抛)→ 只 warning,绝不抛到主调用路。"""
    from src.services import api_call_tasks

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(api_call_tasks, "get_session_factory", _boom)

    # 不应抛 —— 主路(转写)不能因任务记账挂掉。
    await api_call_tasks.record_api_call_task(
        service_name="x", api_key_id=None, status="completed",
    )
    assert "record_api_call_task failed" in caplog.text
