"""启动对账 helper(E-D:N+1 → 单 DISTINCT 查询)。孤儿 published→draft、有关联登记引用。"""
from unittest.mock import MagicMock

import pytest

from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow
from src.services.startup_reconcile import reconcile_orphan_published_workflows


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

    orphan, deps = await reconcile_orphan_published_workflows(db_session, mm)

    assert orphan == 1
    await db_session.refresh(wf_orphan)
    await db_session.refresh(wf_linked)
    assert wf_orphan.status == "draft"       # 孤儿退回
    assert wf_linked.status == "published"   # 有关联的保持
    assert deps == [{"key": "qwen3_8b", "wf_id": wf_linked.id}]
    mm.add_reference.assert_called_once_with("qwen3_8b", str(wf_linked.id))


@pytest.mark.asyncio
async def test_no_published_is_noop(db_session):
    orphan, deps = await reconcile_orphan_published_workflows(db_session, MagicMock())
    assert orphan == 0 and deps == []
