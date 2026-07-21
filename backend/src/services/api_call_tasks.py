"""Arc 3(spec 2026-07-20-moss-asr-sglang-serving §8):把 OpenAI 兼容**直连** API 调用
落成一条 ExecutionTask,让任务中心对直连推理有感知(现状:只有工作流家族建 task,
直连推理零感知)。

- 一次性写 completed/failed 的 task —— 同步调用无进度可言,不建 running 中间态,
  也不在请求关键路径上加往返。
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


async def record_api_call_task(
    *,
    service_name: str,
    api_key_id: int | None,
    status: str,
    duration_ms: int | None = None,
    input_json: dict | None = None,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    """写一条直连 API 调用的 ExecutionTask(completed/failed)。

    字段契约(spec §8):
    - `workflow_id=None` —— 非工作流家族,不归属任何 workflow。
    - `workflow_name=service_name` —— 服务名(任务卡标题)。
    - `api_key_id` —— 归属调用方 key(admin/Playground 直连为 None)。
    - `duration_ms` —— 调用耗时。
    - `input_json={model, timestamps, context?, filename}` —— 回显入参。
    - `result={text 预览(截断), segments_count, speakers, audio_seconds}` —— 前端据此
      派生 task_type=asr(execution_task_serialize._detect_asr_meta 认 audio_seconds 判据)
      + 卡片摘要展示时长/段数/预览。failed 时 result 通常为 None,error 带简短原因。

    写失败绝不影响主调用路 —— 整段 try/except,只 logger.warning。
    """
    try:
        sf = get_session_factory()
        async with sf() as session:
            task = ExecutionTask(
                workflow_id=None,
                workflow_name=service_name or "",
                api_key_id=api_key_id,
                status=status,
                nodes_total=0,
                nodes_done=0,
                duration_ms=duration_ms,
                input_json=input_json,
                result=result,
                error=error,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
        # 即时可见:广播 `/ws/tasks` created(否则前端只能等 60s polling fallback)。
        # 复用 workflow_runner 的序列化+广播口径,保持 WS payload 与 REST 一致。
        await _broadcast_created(task)
    except Exception as e:  # noqa: BLE001 —— 任务记录是旁路,绝不能拖垮主调用路
        logger.warning(
            "record_api_call_task failed (service=%s status=%s): %s",
            service_name, status, e,
        )


async def _broadcast_created(task: ExecutionTask) -> None:
    """广播新建 task 到全局 `/ws/tasks`(event=created);失败静默降级。"""
    try:
        from src.services.execution_task_serialize import _task_to_dict
        from src.services.ws_hub import ws_manager
        await ws_manager.broadcast_task_update("created", _task_to_dict(task))
    except Exception as e:  # noqa: BLE001
        logger.warning("record_api_call_task broadcast failed: %s", e)
