"""Arc 3(spec 2026-07-20-moss-asr-sglang-serving §8):把 OpenAI 兼容**直连** API 调用
落成一条 ExecutionTask,让任务中心对直连推理有感知(现状:只有工作流家族建 task,
直连推理零感知)。

PR-9:one-shot(完成才落 completed/failed)升级为**两段式** —— 调用开始即写 running,
让任务中心在长音频(30s~2min)同步转写期间就显示「正在转写」,不再空窗到完成。

- `create_api_call_task` 进入推理前写 status=running 的 task + 广播 created,返回 task_id;
  写失败回 None(主路继续,后续 finalize 对 None 短路)。
- `finalize_api_call_task` 推理结束翻 completed/failed + 广播 updated;task_id 为 None 直接返回。
- 独立 session(仿 usage_service.record_llm_usage 的 get_session_factory 模式,不借请求
  session,避免与主路事务耦合)。
- 全程 try/except 兜底:写 task 失败只 logger.warning 降级,**绝不抛到主调用路**
  (转写响应不能因任务记账挂掉)。

当前只接 ASR 转写端点(spec §8:当前痛点);chat/图像后续按需一行接入;embeddings
这类高频批量默认不接(任务中心会刷屏)。
"""
import logging

from src.models.database import get_session_factory
from src.models.execution_task import ExecutionTask

logger = logging.getLogger(__name__)


async def create_api_call_task(
    *,
    service_name: str,
    api_key_id: int | None,
    input_meta: dict | None = None,
) -> int | None:
    """写一条 status=running 的直连 API 调用 ExecutionTask,广播 created,返回 task_id。

    字段契约(spec §8):
    - `workflow_id=None` —— 非工作流家族,不归属任何 workflow。
    - `workflow_name=service_name` —— 服务名(任务卡标题)。
    - `api_key_id` —— 归属调用方 key(admin/Playground 直连为 None)。
    - `input_json=input_meta` —— 回显入参 + `kind`/`audio_seconds` 派生钩子。running 态
      result 还没落,execution_task_serialize 靠 input_json.kind="asr" 派生 task_type=asr
      + audio_seconds(否则 running 卡片没类型徽标/时长)。
    - `result=None` —— 结束时由 finalize 落。

    写失败绝不影响主调用路 —— 整段 try/except,只 logger.warning,返回 None
    (后续 finalize 对 None 短路,主路照常返回转写结果)。
    """
    try:
        sf = get_session_factory()
        async with sf() as session:
            task = ExecutionTask(
                workflow_id=None,
                workflow_name=service_name or "",
                api_key_id=api_key_id,
                status="running",
                nodes_total=0,
                nodes_done=0,
                input_json=input_meta,
                result=None,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            task_id = task.id
            # 即时可见:广播 `/ws/tasks` created(否则前端只能等 60s polling fallback)。
            # 复用 workflow_runner 的序列化+广播口径,保持 WS payload 与 REST 一致。
            await _broadcast(task, "created")
        return task_id
    except Exception as e:  # noqa: BLE001 —— 任务记录是旁路,绝不能拖垮主调用路
        logger.warning(
            "create_api_call_task failed (service=%s): %s", service_name, e,
        )
        return None


async def finalize_api_call_task(
    task_id: int | None,
    *,
    status: str,
    result: dict | None = None,
    error: str | None = None,
    duration_ms: int | None,
) -> None:
    """把 create 建的 running task 翻到终态(completed/failed),广播 updated。

    - `task_id` 为 None(create 曾失败)→ 直接返回,主路不受影响。
    - `result`:completed 时同现契约 `{text 预览, segments_count, speakers, audio_seconds}`;
      failed 时通常 None,`error` 带简短原因。
    - 显式刷 updated_at(除 ORM onupdate 外再兜一手,保证任务卡按更新时间排序正确)。

    写失败绝不影响主调用路 —— 整段 try/except,只 logger.warning。
    """
    if task_id is None:
        return
    try:
        from datetime import datetime, timezone

        sf = get_session_factory()
        async with sf() as session:
            task = await session.get(ExecutionTask, task_id)
            if task is None:
                logger.warning("finalize_api_call_task: task %s 不存在", task_id)
                return
            task.status = status
            task.result = result
            task.error = error
            task.duration_ms = duration_ms
            task.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(task)
            await _broadcast(task, "updated")
    except Exception as e:  # noqa: BLE001 —— 任务记录是旁路,绝不能拖垮主调用路
        logger.warning(
            "finalize_api_call_task failed (task=%s status=%s): %s",
            task_id, status, e,
        )


async def fail_orphaned_running_asr_tasks() -> int:
    """backend 崩溃/重启会留永久 running 的直连 ASR 任务 —— 同步转写没有后台 worker
    复活它们,进程一没了这行就永远卡在 running。启动时(app lifespan)一次性把
    `status=running 且 workflow_id IS NULL 且 input_json.kind="asr"` 的行批量置 failed。

    workflow_id/kind 两道过滤只圈直连 ASR task,不误伤工作流家族的 running task
    (那些由 workflow_runner 管理,重启后另有处理)。input_json.kind 判据在 Python 侧过滤
    (JSON key 查询各库语法不一,候选量极小,拉出来筛最稳)。防御式:失败只 warning、返回 0。
    返回置 failed 的行数(供启动日志/测试)。
    """
    try:
        from sqlalchemy import select

        sf = get_session_factory()
        async with sf() as session:
            rows = (
                await session.execute(
                    select(ExecutionTask).where(
                        ExecutionTask.status == "running",
                        ExecutionTask.workflow_id.is_(None),
                    )
                )
            ).scalars().all()
            n = 0
            for t in rows:
                ij = t.input_json
                if isinstance(ij, dict) and ij.get("kind") == "asr":
                    t.status = "failed"
                    t.error = "backend restarted mid-call"
                    n += 1
            if n:
                await session.commit()
            return n
    except Exception as e:  # noqa: BLE001 —— 清理是旁路,绝不阻断启动
        logger.warning("fail_orphaned_running_asr_tasks failed: %s", e)
        return 0


async def _broadcast(task: ExecutionTask, event: str) -> None:
    """广播 task 变更到全局 `/ws/tasks`(event=created/updated);失败静默降级。

    沿用 routes/execution_tasks.py + workflow_runner 的既有事件名(created/updated),
    不发明新事件;复用 execution_task_serialize._task_to_dict,保持 WS payload 与 REST 一致。
    """
    try:
        from src.services.execution_task_serialize import _task_to_dict
        from src.services.ws_hub import ws_manager
        await ws_manager.broadcast_task_update(event, _task_to_dict(task))
    except Exception as e:  # noqa: BLE001
        logger.warning("api_call_task broadcast(%s) failed: %s", event, e)
