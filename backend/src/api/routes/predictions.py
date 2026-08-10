"""统一「预测(prediction)」端点(服务层 API spec 2026-06-03,PR-2)。

`POST /api/v1/services/{name}/predictions` —— 一个端点,`Prefer` 头切同步/异步,跑已发布的
**workflow** 服务(对齐 Cog predictions + ComfyUI prompt)。修「工作流不可参数化」(旧 /run 丢 inputs)
+「M:N key 调不了工作流」(旧只 openai_compat 走 model,workflow 501)。

- 默认同步:阻塞至终态。
- `Prefer: respond-async` → 202 + status:processing,客户端轮询 `GET /predictions/{id}`。
- `Prefer: wait=N` → 阻塞至多 N 秒,超时返当前态转轮询。

model(LLM)服务仍走 `/v1/chat/completions`(chat 形,不塞进通用 predictions);preset(TTS)暂留
`/v1/instances/{id}/synthesize`(后续折叠)。本 PR 删旧 `/v1/instances/{id}/run`(被 predictions 取代)。
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import enforce_instance_rate_limit, verify_bearer_token_any
from src.models.database import get_async_session
from src.models.execution_task import ExecutionTask
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.model_resolver import ModelNotFound, resolve_target_service
from src.services.prediction_service import (
    apply_inputs_to_snapshot,
    snapshot_to_executor_form,
    task_to_prediction,
)
from src.services.service_schema import build_service_io_schema, validate_service_input

# /v1/* = 对外 bearer-authed 端点(AdminSessionGate 只拦 /api/*,这里用各自的 bearer 校验)。
# 放 /api/v1 会被 admin cookie 门拦死 bearer 客户端(真机 smoke 逮到)。
router = APIRouter(prefix="/v1", tags=["predictions"])


async def _auth_predictions(
    request: Request,
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[ServiceInstance | None, InstanceApiKey | None]:
    """Bearer-token auth,带 admin-session 旁路(Task 10:Playground 异步运行态)。

    镜像 `apps.py::_auth_apps_run`:外部 Bearer 客户端走完整 key 校验(优先);Playground
    对 `comfy_template` 服务走 respond-async 提交/轮询/取消时用的是 admin session cookie,
    不是 Bearer key —— 原先三端点的 `Authorization: Header(...)` 是必填,FastAPI 会在
    header 校验阶段就以「Field required」拒绝,同 apps.py 改前的坑。返回 `(None, None)`
    让下游(`_resolve_service` / get_prediction / cancel_prediction)按 admin 路径跳过
    grant/限流/IDOR owner 校验(单管理员部署里 admin 隐式对所有 prediction 有权限)。
    """
    if authorization:
        return await verify_bearer_token_any(authorization, session)
    from src.api.admin_session import request_is_authed  # noqa: PLC0415
    if request_is_authed(request):
        return None, None
    raise HTTPException(401, detail="Missing API key or admin session")

# 同步默认上限(秒):无 Prefer 时阻塞,但封顶避免无限挂(长任务用 respond-async)。
_SYNC_CAP_SECONDS = 600.0


class PredictionRequest(BaseModel):
    input: dict[str, Any] | None = None
    # PR-3:webhook 回调(对齐 Cog)。webhook=完成/开始时 POST Prediction 的 URL;
    # webhook_events_filter=["start","completed",...] 过滤,省略=全发。
    webhook: str | None = None
    webhook_events_filter: list[str] | None = None


def _parse_prefer(prefer: str | None) -> tuple[bool, float | None]:
    """Prefer 头 → (async_mode, wait_seconds)。respond-async → 异步;wait=N → 阻塞 N 秒;否则同步。"""
    p = (prefer or "").lower().replace(" ", "")
    if "respond-async" in p:
        return True, None
    if "wait=" in p:
        try:
            return False, float(p.split("wait=", 1)[1].split(",")[0])
        except (ValueError, IndexError):
            return False, None
    return False, None


async def _resolve_service(
    session, auth, name: str,
) -> tuple[ServiceInstance, InstanceApiKey | None]:
    """bearer key(或 admin session)→ 目标服务(URL 的 {name}),校验 active + 加载 deferred 列。

    admin 路径(`api_key is None`,Task 10 旁路)跳过 grant/限流 —— 与 `apps.py`
    的 `admin_run` 分支一致:单管理员部署里 admin 隐式对所有服务有权限。
    """
    instance, api_key = auth
    if api_key is None:  # admin session 旁路:直接按 name 查活跃服务,不涉及任何 key
        stmt = select(ServiceInstance).where(ServiceInstance.name == name)
        instance = (await session.execute(stmt)).scalar_one_or_none()
        if instance is None:
            raise HTTPException(404, detail="service not found")
        if instance.status != "active":
            raise HTTPException(403, detail="service is inactive")
    elif instance is None:  # M:N key:按 URL name 解析授权 + 限流(verify 没做)
        try:
            instance = await resolve_target_service(session, api_key=api_key, requested_model=name)
        except ModelNotFound as e:
            raise HTTPException(404, detail=str(e)) from e
        if instance.name != name:
            raise HTTPException(403, detail="API key not authorized for this service")
        if instance.status != "active":
            raise HTTPException(403, detail="service is inactive")
        await enforce_instance_rate_limit(instance)
    else:  # legacy 1:1 key:verify_bearer_token_any 已解析 + 限流过,只补 name 校验
        if instance.name != name:
            raise HTTPException(403, detail="API key not authorized for this service")
        if instance.status != "active":
            raise HTTPException(403, detail="service is inactive")
    await session.refresh(
        instance, attribute_names=["workflow_snapshot", "exposed_inputs", "exposed_outputs"])
    return instance, api_key


@router.get("/services/{name}/schema")
async def get_service_schema(
    name: str,
    session: AsyncSession = Depends(get_async_session),
):
    """per-service I/O JSON-Schema(input/output)—— 机器可发现的调用契约(服务层 API spec PR-1)。

    **公开端点**(第三方集成先拿契约,无 auth),按 service `name` 查。在 /v1(非 /api/v1)避开 admin cookie 门。
    从 exposed_inputs/outputs + 各节点 node.yaml widget 生成(对齐 Cog 声明即 schema + ComfyUI object_info)。
    """
    from sqlalchemy.orm import undefer  # noqa: PLC0415

    from src.services.service_schema import build_service_io_schema  # noqa: PLC0415
    stmt = (
        select(ServiceInstance)
        .options(
            undefer(ServiceInstance.workflow_snapshot),
            undefer(ServiceInstance.exposed_inputs),
            undefer(ServiceInstance.exposed_outputs),
        )
        .where(ServiceInstance.name == name)
    )
    svc = (await session.execute(stmt)).scalar_one_or_none()
    if svc is None:
        raise HTTPException(404, detail="service not found")
    schema = build_service_io_schema(
        svc.exposed_inputs, svc.exposed_outputs, svc.workflow_snapshot)
    return {
        "service": name,
        "category": svc.category,
        "source_type": svc.source_type,
        "input_schema": schema["input_schema"],
        "output_schema": schema["output_schema"],
    }


@router.post("/services/{name}/predictions")
async def create_prediction(
    name: str,
    body: PredictionRequest,
    request: Request,
    response: Response,
    prefer: str | None = Header(default=None),
    auth: tuple[ServiceInstance | None, InstanceApiKey | None] = Depends(_auth_predictions),
    session: AsyncSession = Depends(get_async_session),
):
    """跑一个已发布 workflow 服务,返回 Prediction 对象(Cog 形)。

    Task 7:`comfy_template`(ComfyUI workflow bridge,见 comfy_templates.py)也走这条统一入口
    ——design spec 2026-08-10 §「提交」明确画的就是这条路径。它的 workflow_snapshot 是固定桥
    快照(list 形,单 `comfyui_workflow` 节点 + `video_output` 节点),下面的
    apply_inputs_to_snapshot/snapshot_to_executor_form 对 list 形 snapshot 已经是透传+按
    node_id 注入,天然兼容,不需要专门分支。
    """
    instance, api_key = await _resolve_service(session, auth, name)
    if instance.source_type == "model":
        raise HTTPException(400, detail="model(LLM)服务请用 /v1/chat/completions")
    if instance.source_type not in ("workflow", "comfy_template"):
        raise HTTPException(400, detail=f"source_type {instance.source_type!r} 暂不支持 predictions")

    snapshot = instance.workflow_snapshot or {}
    if not (snapshot.get("nodes")):
        raise HTTPException(400, detail="service workflow has no nodes")

    # 类型校验(PR-1):按 per-service input schema 校验请求 input。
    inputs = body.input or {}
    schema = build_service_io_schema(
        instance.exposed_inputs, instance.exposed_outputs, snapshot)
    errors = validate_service_input(schema["input_schema"], inputs)
    if errors:
        raise HTTPException(422, detail={"message": "input validation failed", "errors": errors})

    # 注入 inputs 到快照副本(PR-2 补:旧 /run 丢弃 inputs)→ 再转 executor 吃的编辑形
    # (发布存 api-shape dict-of-nodes,executor 要 list,旧 /run 没转直接崩,无消费者没暴露)。
    patched = apply_inputs_to_snapshot(snapshot, instance.exposed_inputs, inputs)
    patched = snapshot_to_executor_form(patched)

    task = ExecutionTask(
        workflow_id=instance.source_id,
        # 归属:by-id 端点据此校验 owner(IDOR 防护)。admin session 旁路(Task 10)无
        # key 可归属 —— NULL,与 apps.py 的 admin_run 一致(admin 隐式对所有 prediction 有权限,
        # get/cancel_prediction 对 api_key is None 的调用方跳过 owner 校验)。
        api_key_id=api_key.id if api_key is not None else None,
        workflow_name=instance.name,
        status="queued",
        nodes_total=len(patched.get("nodes") or []),
        input_json=inputs,
        webhook_url=body.webhook,
        webhook_events=body.webhook_events_filter,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    if api_key is not None:
        # 用量计数(原子自增)。完整配额消费收敛到 PR-5。admin 路径无 key 可计,跳过
        # (镜像 apps.py::execute_service 的 admin_run 分支:admin 不耗配额)。
        await session.execute(
            update(InstanceApiKey).where(InstanceApiKey.id == api_key.id)
            .values(usage_calls=InstanceApiKey.usage_calls + 1))
        await session.commit()

    from src.services.workflow_runner import run_workflow_task  # noqa: PLC0415
    runner_client = getattr(request.app.state, "runner_client", None)
    runner_clients = getattr(request.app.state, "runner_clients", None)
    exec_coro = run_workflow_task(
        task.id, patched, runner_client=runner_client,
        runner_clients=runner_clients, channel_id=str(instance.id))
    exec_task = asyncio.create_task(exec_coro)

    async_mode, wait_seconds = _parse_prefer(prefer)
    if async_mode:
        response.status_code = 202
    else:
        # 同步 / wait=N:shield 防超时取消执行;超时则返当前态(任务后台继续)。
        timeout = wait_seconds if wait_seconds is not None else _SYNC_CAP_SECONDS
        try:
            await asyncio.wait_for(asyncio.shield(exec_task), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        await session.refresh(task)

    return task_to_prediction(task, service=instance.name, input_values=inputs)


@router.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: int,
    auth: tuple[ServiceInstance | None, InstanceApiKey | None] = Depends(_auth_predictions),
    session: AsyncSession = Depends(get_async_session),
):
    """轮询一个 prediction 的状态/结果。"""
    _, api_key = auth
    task = (await session.execute(
        select(ExecutionTask).where(ExecutionTask.id == prediction_id))).scalar_one_or_none()
    # IDOR 防护:by-id 只允许创建它的 key 访问;归属不符对 api-key 调用方一律 404(不泄漏
    # 存在性)。admin session 旁路(`api_key is None`,Task 10)跳过 —— admin 对所有
    # prediction 隐式有权限,与 `/api/v1/tasks/{id}`(admin-only,无 owner 校验)对齐。
    if task is None or (api_key is not None and task.api_key_id != api_key.id):
        raise HTTPException(404, detail="prediction not found")
    return task_to_prediction(task, service=task.workflow_name)


@router.get("/predictions/{prediction_id}/stream")
async def stream_prediction(
    prediction_id: int,
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
):
    """SSE 流:每次状态/进度变化推一条 `data: <prediction>`,终态后结束(对齐 Cog SSE + ComfyUI ws)。

    每轮用独立 session 读最新已提交态(后台执行写的是另一个 session)。404 不存在则发一条 error 即结束。
    """
    import json  # noqa: PLC0415

    from fastapi.responses import StreamingResponse  # noqa: PLC0415

    from src.models.database import get_session_factory  # noqa: PLC0415

    _, api_key = auth

    async def _gen():
        sf = get_session_factory()
        last_sig = None
        while True:
            async with sf() as s:
                task = await s.get(ExecutionTask, prediction_id)
            # IDOR 防护:归属不符 / NULL 对 api-key 调用方一律当"不存在"(同 get_prediction)。
            if task is None or task.api_key_id != api_key.id:
                yield f"event: error\ndata: {json.dumps({'error': 'prediction not found'})}\n\n"
                return
            pred = task_to_prediction(task, service=task.workflow_name)
            sig = (pred["status"], task.nodes_done)
            if sig != last_sig:
                yield f"data: {json.dumps(pred, ensure_ascii=False)}\n\n"
                last_sig = sig
            if pred["status"] in ("succeeded", "failed", "canceled"):
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.post("/predictions/{prediction_id}/cancel")
async def cancel_prediction(
    prediction_id: int,
    auth: tuple[ServiceInstance | None, InstanceApiKey | None] = Depends(_auth_predictions),
    session: AsyncSession = Depends(get_async_session),
):
    """取消一个进行中的 prediction(runner 在节点边界检查 status=cancelled 中止)。

    comfy_template 服务的桥节点(Task 6)不经 runner_clients 广播那套 abort —— 它是串行
    跑在这个进程里、正堵在 `ComfyClient.wait()` 上,唯一能真正打断当前渲染的是转发
    ComfyUI 的 `/interrupt`(串行语义下"中断当前"即"中断本任务",局限见 spec §12)。
    """
    _, api_key = auth
    task = (await session.execute(
        select(ExecutionTask).where(ExecutionTask.id == prediction_id))).scalar_one_or_none()
    # IDOR 防护:非 owner 一律 404,防跨租户取消(DoS)。admin session 旁路
    # (`api_key is None`,Task 10)跳过 —— admin 可取消任意 prediction,同 get_prediction。
    if task is None or (api_key is not None and task.api_key_id != api_key.id):
        raise HTTPException(404, detail="prediction not found")
    if task.status in ("completed", "failed", "cancelled"):
        return task_to_prediction(task, service=task.workflow_name)
    if task.status == "running":
        svc = (await session.execute(
            select(ServiceInstance).where(ServiceInstance.name == task.workflow_name)
        )).scalar_one_or_none()
        if svc is not None and svc.source_type == "comfy_template":
            from src.api.routes.comfy_templates import get_client  # noqa: PLC0415
            try:
                await get_client().interrupt()
            except Exception as e:  # noqa: BLE001 — sidecar 掉线不该挡住 DB 落 cancelled
                import logging  # noqa: PLC0415
                logging.getLogger(__name__).warning(
                    "cancel_prediction: comfy interrupt failed for task %s: %s",
                    prediction_id, e)
    # TOCTOU 守护(镜像 workflow_runner.py run_workflow_task 反方向的 cancel-race guard):
    # 上面 `await interrupt()`(生产里最长 5s 真实 HTTP 往返)期间,后台渲染完全可能已经
    # 跑完并把 completed+result 落库 —— 这里若无条件 UPDATE status='cancelled',会把一个
    # 已经成功的终态硬翻回 cancelled 且丢 output(task_to_prediction 只在 succeeded 时给
    # output),终态错报。条件 UPDATE 只在仍是 queued/running 时才真的落 cancelled,否则
    # 0 行生效,下面 refresh 拿到的就是渲染赢下竞态后的真实终态。
    await session.execute(
        update(ExecutionTask).where(
            ExecutionTask.id == prediction_id,
            ExecutionTask.status.in_(("queued", "running")),
        ).values(status="cancelled", cancel_reason="client cancel"))
    await session.commit()
    await session.refresh(task)
    return task_to_prediction(task, service=task.workflow_name)
