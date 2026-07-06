"""启动期一次性对账 —— 从 main.py lifespan 抽出以便单测。"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow


async def reconcile_orphan_published_workflows(session: AsyncSession, model_mgr) -> tuple[int, list[dict]]:
    """published 但已无关联服务的 workflow → 退回 draft(清存量孤儿:历史「删服务」
    没回退 workflow status);有关联的重登记模型引用(防常驻模型被 idle/LRU 卸)。
    返回 (orphan_count, wf_model_deps)。

    E-D(性能二轮):有无关联服务的判定用**一次** DISTINCT workflow_id 查询取集合,
    替代每 wf 一次 svc_exists 查询(N+1,阻塞启动 readiness)。P+1 查询 → 2 查询。
    """
    published = (await session.execute(
        select(Workflow).where(Workflow.status == "published")
    )).scalars().all()
    linked_wf_ids = set((await session.execute(
        select(ServiceInstance.workflow_id)
        .where(ServiceInstance.workflow_id.isnot(None))
        .distinct()
    )).scalars().all())

    orphan = 0
    deps_out: list[dict] = []
    for wf in published:
        if wf.id not in linked_wf_ids:
            wf.status = "draft"
            orphan += 1
            continue
        for dep in model_mgr.get_model_dependencies({"nodes": wf.nodes, "edges": wf.edges}):
            model_mgr.add_reference(dep["key"], str(wf.id))
            deps_out.append({"key": dep["key"], "wf_id": wf.id})
    if orphan:
        await session.commit()
    return orphan, deps_out
