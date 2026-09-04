"""启动对账 helper(E-D:N+1 → 单 DISTINCT 查询)。孤儿 published→draft、有关联登记引用;
以及上次进程遗留的在飞 execution_task → failed。"""
from unittest.mock import MagicMock

import pytest

from src.models.execution_task import ExecutionTask
from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow
from src.services.startup_reconcile import (
    reconcile_orphan_inflight_tasks,
    reconcile_orphan_published_workflows,
)


@pytest.mark.asyncio
async def test_orphan_published_reverts_to_draft_and_linked_registers_deps(db_session):
    # 有关联服务的 wf
    wf_linked = Workflow(name="linked", status="published", nodes=[], edges=[])
    # 无关联服务的 wf(孤儿)
    wf_orphan = Workflow(name="orphan", status="published", nodes=[], edges=[])
    db_session.add_all([wf_linked, wf_orphan])
    await db_session.commit()
    await db_session.refresh(wf_linked)
    await db_session.refresh(wf_orphan)
    # 只给 linked 建服务
    svc = ServiceInstance(source_type="workflow", name="svc", type="inference",
                          category="app", meter_dim="calls", workflow_id=wf_linked.id)
    db_session.add(svc)
    await db_session.commit()

    mm = MagicMock()
    mm.get_model_dependencies = MagicMock(return_value=[{"key": "qwen3_8b", "type": "llm"}])
    mm.add_reference = MagicMock()

    orphan = await reconcile_orphan_published_workflows(db_session, mm)

    assert orphan == 1
    await db_session.refresh(wf_orphan)
    await db_session.refresh(wf_linked)
    assert wf_orphan.status == "draft"       # 孤儿退回
    assert wf_linked.status == "published"   # 有关联的保持
    mm.add_reference.assert_called_once_with("qwen3_8b", str(wf_linked.id))
    # 登记引用只挡 idle/LRU 卸载,绝不触发加载(2026-09-03 删 _load_wf_deps 预热)。
    mm.load_model.assert_not_called()


@pytest.mark.asyncio
async def test_no_published_is_noop(db_session):
    orphan = await reconcile_orphan_published_workflows(db_session, MagicMock())
    assert orphan == 0


@pytest.mark.asyncio
async def test_orphan_inflight_tasks_converge_to_failed(db_session):
    """queued/running(不管新旧)全收敛成 failed;终态行一个字都不动。"""
    running_a = ExecutionTask(workflow_name="wf-a", status="running", nodes_total=3)
    running_b = ExecutionTask(workflow_name="wf-b", status="running", nodes_total=1,
                              current_node="node-7")
    queued = ExecutionTask(workflow_name="wf-c", status="queued")
    done = ExecutionTask(workflow_name="wf-d", status="completed",
                         result={"ok": True}, duration_ms=1234)
    cancelled = ExecutionTask(workflow_name="wf-e", status="cancelled",
                              cancel_reason="client cancel")
    db_session.add_all([running_a, running_b, queued, done, cancelled])
    await db_session.commit()

    n = await reconcile_orphan_inflight_tasks(db_session)

    assert n == 3
    for task in (running_a, running_b, queued):
        await db_session.refresh(task)
        assert task.status == "failed"
        assert "重启" in (task.error or "")
        assert task.finished_at is not None      # = API 的 completed_at
        assert task.duration_ms is not None
        assert task.current_node is None
    await db_session.refresh(done)
    await db_session.refresh(cancelled)
    assert done.status == "completed"            # 终态不动
    assert done.error is None
    assert done.duration_ms == 1234              # 已有耗时不被覆盖
    assert cancelled.status == "cancelled"
    assert cancelled.error is None


@pytest.mark.asyncio
async def test_orphan_inflight_noop_when_all_terminal(db_session):
    """全是终态 → 返回 0(调用方据此不打日志)。"""
    db_session.add(ExecutionTask(workflow_name="wf", status="completed"))
    await db_session.commit()
    assert await reconcile_orphan_inflight_tasks(db_session) == 0


@pytest.mark.asyncio
async def test_orphan_inflight_noop_on_empty_table(db_session):
    assert await reconcile_orphan_inflight_tasks(db_session) == 0
