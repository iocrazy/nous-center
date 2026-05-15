"""Lane S: 后台 workflow 执行入口 run_workflow_task。"""
import pytest


@pytest.mark.asyncio
async def test_run_workflow_task_marks_completed(db_session, monkeypatch):
    """跑完一个 inline-only workflow → task.status=completed + result + duration_ms。"""
    from src.models.execution_task import ExecutionTask
    from src.services import workflow_runner

    # workflow_runner 自己开 session（来自 create_session_factory()）；
    # 测试里把那个 factory 替成 db_session 同库的 factory，使读写同一数据库。
    from sqlalchemy.ext.asyncio import async_sessionmaker
    bind = db_session.bind
    test_factory = async_sessionmaker(bind, expire_on_commit=False)
    monkeypatch.setattr(workflow_runner, "create_session_factory",
                        lambda: test_factory)

    task = ExecutionTask(workflow_name="t", status="queued", nodes_total=1)
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    wf = {"nodes": [{"id": "t1", "type": "text_input", "data": {"text": "hi"}}],
          "edges": []}
    await workflow_runner.run_workflow_task(
        task.id, wf, runner_client=None, channel_id=None
    )

    await db_session.refresh(task)
    assert task.status == "completed"
    assert task.result is not None
    assert task.duration_ms is not None
    assert task.nodes_done == 1


@pytest.mark.asyncio
async def test_run_workflow_task_marks_failed_on_error(db_session, monkeypatch):
    """workflow 抛 ExecutionError → task.status=failed + error 落表，不抛出。"""
    from src.models.execution_task import ExecutionTask
    from src.services import workflow_runner

    from sqlalchemy.ext.asyncio import async_sessionmaker
    bind = db_session.bind
    test_factory = async_sessionmaker(bind, expire_on_commit=False)
    monkeypatch.setattr(workflow_runner, "create_session_factory",
                        lambda: test_factory)

    task = ExecutionTask(workflow_name="t", status="queued", nodes_total=1)
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # 空 workflow → _topological_sort 抛 ExecutionError("工作流为空")
    await workflow_runner.run_workflow_task(
        task.id, {"nodes": [], "edges": []},
        runner_client=None, channel_id=None,
    )

    await db_session.refresh(task)
    assert task.status == "failed"
    assert task.error
