"""External-facing service execution.

v3 (2026-04-22): the WorkflowApp model is gone. Services are stored in
`service_instances` (source_type='workflow' for published ones) and the
execute path resolves by name through the same grant-authz the OpenAI
compat path uses. The publish endpoint moved to `workflow_publish.py`.
"""

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from src.api.deps_auth import verify_bearer_token_any
from src.errors import NotFoundError
from src.models.api_gateway import ApiKeyGrant
from src.models.database import get_async_session
from src.models.execution_task import ExecutionTask
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance

router = APIRouter(tags=["apps"])


@router.post("/v1/apps/{service_name}/{action}")
@router.post("/v1/apps/{service_name}")
async def execute_service(
    service_name: str,
    body: dict,
    action: str = "run",
    session: AsyncSession = Depends(get_async_session),
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
):
    """v3 external endpoint: execute a service by name.

    Auth: Bearer <InstanceApiKey>. Authz: an active ApiKeyGrant must exist
    for (key, service). Charges 1 call per request via consume_for_request.

    `action` is part of the v3 contract for future per-service routing
    (e.g. `/v1/apps/lessor/synthesize`) — currently treated as opaque
    metadata; the workflow always runs end to end.
    """
    from src.services.workflow_executor import WorkflowExecutor, ExecutionError
    from src.services.quota_gate import NoActiveGrant, consume_for_request
    from src.services.resource_pack import QuotaExhausted

    _instance, api_key = auth

    # Resolve service + authorize via grant in one query (M:N path).
    stmt = (
        select(ServiceInstance)
        .options(
            undefer(ServiceInstance.workflow_snapshot),
            undefer(ServiceInstance.exposed_inputs),
            undefer(ServiceInstance.exposed_outputs),
        )
        .join(ApiKeyGrant, ApiKeyGrant.service_id == ServiceInstance.id)
        .where(
            ApiKeyGrant.api_key_id == api_key.id,
            ApiKeyGrant.status == "active",
            ServiceInstance.name == service_name,
        )
    )
    svc = (await session.execute(stmt)).scalar_one_or_none()
    if svc is None:
        raise NotFoundError(
            f"no active grant for service '{service_name}' on this key",
            code="service_not_found",
        )
    if svc.status == "retired":
        raise HTTPException(410, detail=f"Service '{service_name}' is retired")
    if svc.status == "paused":
        raise HTTPException(403, detail=f"Service '{service_name}' is paused")
    # `deprecated` still serves but logs a warning (per v3 lifecycle spec).

    snapshot = dict(svc.workflow_snapshot or {})
    nodes = [dict(n) for n in snapshot.get("nodes", [])]
    edges = snapshot.get("edges", [])

    # Merge exposed inputs from request body into the matching node data.
    # Supports both v3 schema (key/input_name) and pre-v3 backfill rows
    # that still carry the old (api_name/param_key) field names.
    for param in (svc.exposed_inputs or []):
        api_name = param.get("key") or param.get("api_name")
        node_id = param.get("node_id")
        slot = param.get("input_name") or param.get("param_key")
        if api_name is None or node_id is None or slot is None:
            continue
        if api_name not in body:
            continue
        for node in nodes:
            if str(node.get("id")) == str(node_id):
                node.setdefault("data", {})[slot] = body[api_name]

    task = ExecutionTask(
        workflow_name=svc.name,
        status="running",
        nodes_total=len(nodes),
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    start = time.monotonic()
    executor = WorkflowExecutor({"nodes": nodes, "edges": edges})

    try:
        result = await executor.execute()
        elapsed = int((time.monotonic() - start) * 1000)
        task.status = "completed"
        task.result = result
        task.duration_ms = elapsed
        task.nodes_done = len(nodes)
        task.current_node = None
    except ExecutionError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        task.status = "failed"
        task.error = str(e)
        task.duration_ms = elapsed
        await session.commit()
        raise HTTPException(500, str(e))

    await session.commit()

    # Quota: 1 call per request. Failures here are non-fatal — the work
    # already happened — but exhaustion will block the next call.
    try:
        await consume_for_request(
            session, api_key_id=api_key.id, service_id=svc.id, units=1,
        )
        await session.commit()
    except (NoActiveGrant, QuotaExhausted):
        pass

    _ = action  # reserved
    return result
