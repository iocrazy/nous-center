"""启动期一次性对账 —— 从 main.py lifespan 抽出以便单测。"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.execution_task import ExecutionTask
from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow


async def reconcile_orphan_published_workflows(session: AsyncSession, model_mgr) -> int:
    """published 但已无关联服务的 workflow → 退回 draft(清存量孤儿:历史「删服务」
    没回退 workflow status);有关联的重登记模型引用(防常驻模型被 idle/LRU 卸)。
    返回 orphan_count。

    **登记引用 ≠ 加载模型**:`add_reference` 只是挡 idle checker / LRU 驱逐,绝不触发
    load。历史上 lifespan 拿本函数返回的 deps 列表在后台 `load_model` 全量预热
    (`_load_wf_deps`),不看 resident 标记 —— 用户没开「自动加载」的模型开机照样被
    拉满显存。2026-09-03 删掉那条预热,返回值只剩 orphan_count。工作流执行时
    runner 走 `get_or_load` 按需加载。

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
    for wf in published:
        if wf.id not in linked_wf_ids:
            wf.status = "draft"
            orphan += 1
            continue
        for dep in model_mgr.get_model_dependencies({"nodes": wf.nodes, "edges": wf.edges}):
            model_mgr.add_reference(dep["key"], str(wf.id))
    if orphan:
        await session.commit()
    return orphan


# 上次进程遗留的在飞任务收敛后写进 `error` 的文案 —— 前端/API 直接展示给用户。
ORPHAN_INFLIGHT_ERROR = "后端重启,任务中断(未完成),请重新提交"

# 非终态集合。终态是 completed / failed / cancelled(见 models/execution_task.py 的
# status 注释);API 层 `prediction_service._STATUS_MAP` 把 queued→starting、
# running→processing,所以「前端一直显示 processing 的孤儿」就是这两个状态的行。
_INFLIGHT_STATUSES = ("queued", "running")


async def reconcile_orphan_inflight_tasks(session: AsyncSession) -> int:
    """启动时把所有非终态 `execution_tasks` 收敛为 failed。返回收敛条数。

    **为什么不看年龄、不做任何存活判断**:所有执行都是本进程内的 asyncio task
    (`workflow_runner.run_workflow_task`,由 /run、predictions 端点
    `asyncio.create_task` 拉起),没有跨进程的执行器。进程刚起来时不可能有任何任务
    真的在飞 —— 上一个进程一死,它们的 asyncio task 就随之消失,而 DB 行永远停在
    running/queued(线上表现:前端永远 processing 的孤儿,用户只能手动 cancel 再重发)。
    所以启动瞬间凡是非终态的行,**定义上**都是孤儿,一律收敛,不需要 age 阈值
    (阈值反而会漏掉「刚提交二十秒就重启」的那一条,正是 2026-09-03 线上遇到的例子)。

    **不碰 ComfyUI sidecar**:这里不发 `/interrupt`。sidecar 是独立 systemd 单元、
    队列是它自己的事,而且后端重启时它多半也刚重启(或压根没收到过这些任务)。
    代价是:若 sidecar 的 `/queue` 里确实还留着对应渲染,它会自己跑完、产物没人收,
    白烧一次 GPU。取舍是「宁可浪费一次渲染,也不误杀」—— `/interrupt` 只能打断
    sidecar **当前**那个渲染(串行语义,见 routes/predictions.py::cancel_prediction
    的 C1 注释),启动期我们无从知道它当前跑的是不是我们的孤儿,盲发有一定概率打断
    的是另一个刚提交的新渲染。
    """
    tasks = (await session.execute(
        select(ExecutionTask).where(ExecutionTask.status.in_(_INFLIGHT_STATUSES))
    )).scalars().all()
    if not tasks:
        return 0

    now = datetime.now(timezone.utc)
    for task in tasks:
        task.status = "failed"
        task.error = ORPHAN_INFLIGHT_ERROR
        task.current_node = None
        # 收尾字段对齐 workflow_runner 的终态路径:duration_ms 已有就不覆盖(执行侧
        # 写过的耗时更准),没有就按「开始/入队 → 现在」补一个。finished_at 就是 API
        # 里的 `completed_at`(prediction_service.task_to_prediction 的映射)。
        task.finished_at = now
        if task.duration_ms is None:
            started = task.started_at or task.queued_at or task.created_at
            if started is not None:
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                task.duration_ms = max(0, int((now - started).total_seconds() * 1000))
    await session.commit()

    # 广播 best-effort:`ws_manager` 是 ws_hub 的模块级单例,lifespan 阶段一定可用,
    # 且 `broadcast_task_update` 在没有订阅者时直接 return —— 启动这一刻通常还没有
    # 浏览器连上,所以实际多半是 no-op;真有早连上的客户端就能立刻看到状态翻掉。
    # `_broadcast_task_status` 自身吞异常,不会影响启动。
    from src.services.workflow_runner import _broadcast_task_status
    for task in tasks:
        await _broadcast_task_status(task, event="updated")
    return len(tasks)
