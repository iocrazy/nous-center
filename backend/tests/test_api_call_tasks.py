"""Arc 3 / PR-9(spec 2026-07-20-moss-asr §8):api_call_tasks 两段式单测。

锁:① create 写 running(返回 task_id)→ finalize 翻 completed(两阶段状态);
② 失败流 finalize 翻 failed(error/无 result);③ create 失败返回 None、finalize 对 None 短路;
④ 孤儿 running 清理(status=running∧workflow_id NULL∧kind=asr → failed);
⑤ running 态 serialize 靠 input_json.kind 派生 type=asr。
独立 session 用 monkeypatch 指向 tmp sqlite(仿 test_inference_usage_sqlite 的
get_session_factory patch 姿势)。
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
async def test_two_phase_running_then_completed(tmp_path, monkeypatch):
    """create → running(可见);finalize(completed)→ 终态 + result 契约。"""
    api_call_tasks, sf = await _make_sf(tmp_path, monkeypatch)

    task_id = await api_call_tasks.create_api_call_task(
        service_name="moss-asr",
        api_key_id=42,
        input_meta={"model": "moss-asr", "timestamps": True, "context": None,
                    "filename": "a.wav", "kind": "asr", "audio_seconds": 6},
    )
    assert isinstance(task_id, int)

    # 阶段一:running 已落库,result 未落。
    async with sf() as s:
        t = await s.get(ExecutionTask, task_id)
        assert t.status == "running"
        assert t.result is None
        assert t.workflow_id is None
        assert t.workflow_name == "moss-asr"
        assert t.api_key_id == 42
        assert t.input_json["kind"] == "asr"

    await api_call_tasks.finalize_api_call_task(
        task_id,
        status="completed",
        duration_ms=1234,
        result={"text": "你好", "segments_count": 2,
                "speakers": ["S01", "S02"], "audio_seconds": 6},
    )

    # 阶段二:completed + 字段契约。
    async with sf() as s:
        rows = (await s.execute(select(ExecutionTask))).scalars().all()
    assert len(rows) == 1  # 同一行翻状态,不新建
    t = rows[0]
    assert t.status == "completed"
    assert t.duration_ms == 1234
    assert t.result["segments_count"] == 2
    assert t.result["audio_seconds"] == 6
    assert t.error is None


@pytest.mark.asyncio
async def test_two_phase_running_then_failed(tmp_path, monkeypatch):
    """create → running;finalize(failed)→ error 落库、无 result。"""
    api_call_tasks, sf = await _make_sf(tmp_path, monkeypatch)

    task_id = await api_call_tasks.create_api_call_task(
        service_name="moss-asr",
        api_key_id=None,
        input_meta={"model": "moss-asr", "kind": "asr", "audio_seconds": 3},
    )
    assert isinstance(task_id, int)

    await api_call_tasks.finalize_api_call_task(
        task_id,
        status="failed",
        duration_ms=88,
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
async def test_create_failure_returns_none_and_finalize_short_circuits(monkeypatch, caplog):
    """create 写库失败(session factory 抛)→ 返回 None、只 warning;finalize(None)直接短路。"""
    from src.services import api_call_tasks

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(api_call_tasks, "get_session_factory", _boom)

    # 不应抛 —— 主路(转写)不能因任务记账挂掉。
    task_id = await api_call_tasks.create_api_call_task(
        service_name="x", api_key_id=None, input_meta={"kind": "asr"},
    )
    assert task_id is None
    assert "create_api_call_task failed" in caplog.text

    # finalize(None)短路:即便 factory 仍会抛,也不触碰它、不抛。
    await api_call_tasks.finalize_api_call_task(
        None, status="failed", duration_ms=1, error="x",
    )


@pytest.mark.asyncio
async def test_fail_orphaned_running_asr_tasks(tmp_path, monkeypatch):
    """启动清理:running∧workflow_id NULL∧kind=asr → failed;其余 running 不动。"""
    api_call_tasks, sf = await _make_sf(tmp_path, monkeypatch)

    async with sf() as s:
        orphan = ExecutionTask(
            workflow_id=None, workflow_name="moss-asr", status="running",
            input_json={"kind": "asr", "model": "moss-asr"},
        )
        # 工作流家族的 running(有 workflow_id)—— 不该被清。
        wf_running = ExecutionTask(
            workflow_id=123, workflow_name="img-wf", status="running",
            input_json={"prompt": "x"},
        )
        # 非 asr 的直连 running(理论上没有,防御):kind 不匹配 → 不清。
        other = ExecutionTask(
            workflow_id=None, workflow_name="other", status="running",
            input_json={"kind": "chat"},
        )
        s.add_all([orphan, wf_running, other])
        await s.commit()
        orphan_id, wf_id, other_id = orphan.id, wf_running.id, other.id

    n = await api_call_tasks.fail_orphaned_running_asr_tasks()
    assert n == 1

    async with sf() as s:
        assert (await s.get(ExecutionTask, orphan_id)).status == "failed"
        assert (await s.get(ExecutionTask, orphan_id)).error == "backend restarted mid-call"
        assert (await s.get(ExecutionTask, wf_id)).status == "running"
        assert (await s.get(ExecutionTask, other_id)).status == "running"


def test_running_asr_serialize_derives_type():
    """running 态(result=None)靠 input_json.kind=asr 派生 type=asr + audio_seconds。"""
    from src.services.execution_task_serialize import _task_to_dict

    t = ExecutionTask(
        workflow_id=None, workflow_name="moss-asr", status="running",
        input_json={"kind": "asr", "audio_seconds": 12, "model": "moss-asr"},
        result=None,
    )
    d = _task_to_dict(t)
    assert d["type"] == "asr"
    assert d["audio_seconds"] == 12
    assert d["segments_count"] is None  # running 时段数未知
