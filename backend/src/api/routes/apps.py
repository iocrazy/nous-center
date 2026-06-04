"""External-facing service execution.

v3 (2026-04-22): the WorkflowApp model is gone. Services are stored in
`service_instances` (source_type='workflow' for published ones) and the
execute path resolves by name through the same grant-authz the OpenAI
compat path uses. The publish endpoint moved to `workflow_publish.py`.
"""


from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from src.api.admin_session import request_is_authed
from src.api.deps_auth import verify_bearer_token_any
from src.errors import NotFoundError
from src.models.api_gateway import ApiKeyGrant
from src.models.database import get_async_session
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
    request: Request,
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
    from src.services.workflow_service_runner import run_published_workflow

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

    # Flat body keys map directly onto exposed input keys (the v3 app
    # contract). All execution wiring — node normalize, input merge,
    # runner_clients injection, task record, quota — lives in the shared
    # core so this external path can't drift from workflows.py (see #339).
    _ = action  # reserved for future per-service routing
    return await run_published_workflow(request, session, svc, body, api_key)
