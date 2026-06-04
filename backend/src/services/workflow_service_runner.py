"""Shared execution core for published workflow services.

Both `/v1/apps/{name}/run` (apps.py) and `/v1/images/generations`
(openai_compat.py) freeze a published workflow into a `ServiceInstance`,
merge caller inputs into the exposed nodes, run the `WorkflowExecutor`
with `runner_clients` injected, and charge 1 call against the
(key, service) grant.

Keeping that core in one place stops the two entrypoints from drifting:
the #339 regression was exactly apps.py forgetting the `runner_clients`
injection that `workflows.py` already had, so any GPU-node workflow
(image/tts/seedvr2) crashed only on the external path. New entrypoints
must go through here, not re-implement the executor wiring.
"""

import time

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.execution_task import ExecutionTask
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance

# Flat input key → the node's primary data slot. The publish dialog used
# to hard-code input_name='value' for everything, but e.g. text_input
# reads data.text — so we also write the type's primary slot, otherwise
# the merge silently misses and the node returns its frozen snapshot
# value (the LLM answers the OLD baked-in prompt, not the caller's).
NODE_PRIMARY_SLOT = {
    "text_input": "text",
    "text_output": "text",
    "multimodal_input": "text",
    "reference_audio": "audio",
    "image_input": "image",
}


def _normalize_nodes(snapshot: dict) -> list:
    """workflow_publish.py freezes nodes as a dict keyed by id (api-shape),
    while the live editor passes a list. The executor only understands the
    list shape, so normalize back to a list of {id,type,data}."""
    raw_nodes = snapshot.get("nodes", [])
    if isinstance(raw_nodes, dict):
        return [
            {
                "id": nid,
                "type": (n.get("class_type") if isinstance(n, dict) else None) or n.get("type"),
                "data": (n.get("inputs") if isinstance(n, dict) else None) or n.get("data") or {},
            }
            for nid, n in raw_nodes.items()
        ]
    return [dict(n) for n in raw_nodes]


def _merge_inputs(nodes: list, exposed_inputs: list | None, inputs: dict) -> None:
    """Merge caller `inputs` (flat {key: value}) into the matching exposed
    nodes' data. Supports v3 (key/input_name) + pre-v3 (api_name/param_key)
    field names. Writes both the declared slot and the node's primary slot."""
    for param in (exposed_inputs or []):
        api_name = param.get("key") or param.get("api_name")
        node_id = param.get("node_id")
        slot = param.get("input_name") or param.get("param_key")
        if api_name is None or node_id is None or slot is None:
            continue
        if api_name not in inputs:
            continue
        for node in nodes:
            if str(node.get("id")) != str(node_id):
                continue
            data = node.setdefault("data", {})
            data[slot] = inputs[api_name]
            primary = NODE_PRIMARY_SLOT.get(str(node.get("type") or "").lower())
            if primary and primary != slot:
                data[primary] = inputs[api_name]


async def run_published_workflow(
    request: Request,
    session: AsyncSession,
    svc: ServiceInstance,
    inputs: dict,
    api_key: InstanceApiKey | None,
) -> dict:
    """Execute a published workflow service end to end.

    Returns the executor result ({"outputs": {node_id: {...}}}). Charges
    1 call against the (api_key, svc) grant unless `api_key` is None (admin
    path skips quota). The caller is responsible for resolving `svc` with
    workflow_snapshot / exposed_inputs / exposed_outputs undeferred.

    Raises HTTPException(410/403) for retired/paused services and
    HTTPException(500) on execution failure.
    """
    from src.services.quota_gate import NoActiveGrant, consume_for_request
    from src.services.resource_pack import QuotaExhausted
    from src.services.workflow_executor import ExecutionError, WorkflowExecutor

    if svc.status == "retired":
        raise HTTPException(410, detail=f"Service '{svc.name}' is retired")
    if svc.status == "paused":
        raise HTTPException(403, detail=f"Service '{svc.name}' is paused")
    # `deprecated` still serves (per v3 lifecycle spec).

    snapshot = dict(svc.workflow_snapshot or {})
    nodes = _normalize_nodes(snapshot)
    edges = snapshot.get("edges", [])
    _merge_inputs(nodes, svc.exposed_inputs, inputs)

    task = ExecutionTask(
        workflow_name=svc.name,
        status="running",
        nodes_total=len(nodes),
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    start = time.monotonic()
    # Lane K: inject runner_clients (group_id → RunnerClient) so GPU nodes
    # (image/tts/seedvr2) reach their runner subprocess. Omitting this is
    # the #339 regression — keep it identical to workflows.py.
    runner_client = getattr(request.app.state, "runner_client", None)
    runner_clients = getattr(request.app.state, "runner_clients", None)
    executor = WorkflowExecutor(
        {"nodes": nodes, "edges": edges},
        runner_client=runner_client,
        runner_clients=runner_clients,
    )
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

    # Quota: 1 call per request. Failure here is non-fatal — the work
    # already happened — but exhaustion blocks the next call. Admin path
    # (api_key is None) skips quota entirely (no key to charge).
    if api_key is not None:
        try:
            await consume_for_request(
                session, api_key_id=api_key.id, service_id=svc.id, units=1,
            )
            await session.commit()
        except (NoActiveGrant, QuotaExhausted):
            pass

    return result
