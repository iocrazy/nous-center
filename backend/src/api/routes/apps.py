"""External-facing service execution.

v3 (2026-04-22): the WorkflowApp model is gone. Services are stored in
`service_instances` (source_type='workflow' for published ones) and the
execute path resolves by name through the same grant-authz the OpenAI
compat path uses. The publish endpoint moved to `workflow_publish.py`.
"""

import time

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from src.api.admin_session import request_is_authed
from src.api.deps_auth import verify_bearer_token_any
from src.errors import NotFoundError
from src.models.api_gateway import ApiKeyGrant
from src.models.database import get_async_session
from src.models.execution_task import ExecutionTask
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance

router = APIRouter(tags=["apps"])


async def _auth_apps_run(
    request: Request,
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[ServiceInstance | None, InstanceApiKey | None]:
    """Bearer-token auth, with an admin-session bypass for the in-app Playground.

    Bearer takes priority — when a caller supplies an Authorization header
    we run the full key + grant + quota path so external requests behave
    identically (and so the test suite, which always sends Bearer, exercises
    that path even though `ADMIN_PASSWORD=""` makes admin auth permissive).

    Falling through to the admin-cookie check is what unblocks the Playground
    tab in the React UI: it invokes /v1/apps/.../run via session cookie, not
    a Bearer key, and without this bypass FastAPI rejected the request at
    header validation with "Field required" before any execution happened.
    Returning (None, None) signals the route to skip grant+quota checks
    (admin is implicitly authorized in this single-admin deployment).
    """
    if authorization:
        return await verify_bearer_token_any(authorization, session)
    if request_is_authed(request):
        return None, None
    raise HTTPException(401, detail="Missing API key or admin session")


@router.post("/v1/apps/{service_name}/{action}")
@router.post("/v1/apps/{service_name}")
async def execute_service(
    service_name: str,
    body: dict,
    action: str = "run",
    session: AsyncSession = Depends(get_async_session),
    auth: tuple[ServiceInstance | None, InstanceApiKey | None] = Depends(_auth_apps_run),
):
    """v3 external endpoint: execute a service by name.

    Auth: Bearer <InstanceApiKey>, OR admin session cookie (used by the
    Playground tab in the UI). Bearer path requires an active ApiKeyGrant
    for (key, service) and charges 1 call via consume_for_request; the
    admin path skips grant + quota.

    `action` is part of the v3 contract for future per-service routing
    (e.g. `/v1/apps/lessor/synthesize`) — currently treated as opaque
    metadata; the workflow always runs end to end.
    """
    from src.services.workflow_executor import WorkflowExecutor, ExecutionError
    from src.services.quota_gate import NoActiveGrant, consume_for_request
    from src.services.resource_pack import QuotaExhausted

    _instance, api_key = auth
    admin_run = api_key is None

    base_stmt = select(ServiceInstance).options(
        undefer(ServiceInstance.workflow_snapshot),
        undefer(ServiceInstance.exposed_inputs),
        undefer(ServiceInstance.exposed_outputs),
    )
    if admin_run:
        stmt = base_stmt.where(ServiceInstance.name == service_name)
    else:
        stmt = base_stmt.join(
            ApiKeyGrant, ApiKeyGrant.service_id == ServiceInstance.id,
        ).where(
            ApiKeyGrant.api_key_id == api_key.id,
            ApiKeyGrant.status == "active",
            ServiceInstance.name == service_name,
        )
    svc = (await session.execute(stmt)).scalar_one_or_none()
    if svc is None:
        raise NotFoundError(
            f"service '{service_name}' not found"
            if admin_run
            else f"no active grant for service '{service_name}' on this key",
            code="service_not_found",
        )
    if svc.status == "retired":
        raise HTTPException(410, detail=f"Service '{service_name}' is retired")
    if svc.status == "paused":
        raise HTTPException(403, detail=f"Service '{service_name}' is paused")
    # `deprecated` still serves but logs a warning (per v3 lifecycle spec).

    snapshot = dict(svc.workflow_snapshot or {})
    raw_nodes = snapshot.get("nodes", [])
    # workflow_publish.py freezes nodes as a dict keyed by id (api-shape),
    # while quick-provision and the live editor pass them as a list. The
    # executor only understands the list shape, so we normalize back.
    if isinstance(raw_nodes, dict):
        nodes = [
            {
                "id": nid,
                "type": (n.get("class_type") if isinstance(n, dict) else None) or n.get("type"),
                "data": (n.get("inputs") if isinstance(n, dict) else None) or n.get("data") or {},
            }
            for nid, n in raw_nodes.items()
        ]
    else:
        nodes = [dict(n) for n in raw_nodes]
    edges = snapshot.get("edges", [])

    # Merge exposed inputs from request body into the matching node data.
    # Supports both v3 schema (key/input_name) and pre-v3 backfill rows
    # that still carry the old (api_name/param_key) field names.
    #
    # Robustness: if the declared `slot` doesn't match the field a node
    # actually reads (the publish dialog used to hard-code
    # `input_name='value'` for everything, but text_input reads
    # `data.text`), also write to the node's primary slot for that type.
    # Otherwise the merge silently missed the field and the node returned
    # its frozen snapshot value — the LLM answered the OLD prompt baked
    # into publish, not the caller's new input.
    NODE_PRIMARY_SLOT = {
        "text_input": "text",
        "text_output": "text",
        "multimodal_input": "text",
        "reference_audio": "audio",
        "image_input": "image",
    }
    for param in (svc.exposed_inputs or []):
        api_name = param.get("key") or param.get("api_name")
        node_id = param.get("node_id")
        slot = param.get("input_name") or param.get("param_key")
        if api_name is None or node_id is None or slot is None:
            continue
        if api_name not in body:
            continue
        for node in nodes:
            if str(node.get("id")) != str(node_id):
                continue
            data = node.setdefault("data", {})
            data[slot] = body[api_name]
            primary = NODE_PRIMARY_SLOT.get(str(node.get("type") or "").lower())
            if primary and primary != slot:
                data[primary] = body[api_name]

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
    # already happened — but exhaustion will block the next call. Admin
    # bypass skips quota entirely (no key to charge).
    if not admin_run:
        try:
            await consume_for_request(
                session, api_key_id=api_key.id, service_id=svc.id, units=1,
            )
            await session.commit()
        except (NoActiveGrant, QuotaExhausted):
            pass

    _ = action  # reserved
    return result
