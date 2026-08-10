"""模板即服务(ComfyUI workflow bridge Task 3)—— comfy_templates 表 + 注册/映射 API。

每个上传的 ComfyUI API-format workflow JSON 注册成一个模板,同时建一个
`ServiceInstance`(source_type="comfy_template")作为对外服务入口。服务的
`workflow_snapshot` 不是 ComfyUI JSON 本身,而是固定的桥快照(单
`comfyui_workflow` 节点 + `video_output` 节点,见 spec)——真正的节点类型由
Task 6 提供,这里只是把数据形状定下来。`exposed_inputs` 的映射把 nous 侧
`node_id="bridge"` 与 ComfyUI 侧 `comfy_node_id`/`comfy_input` 并存,前者供
`apply_inputs_to_snapshot` 用,后者供重新上传时的 stale 校验用。
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from src.api.deps_admin import require_admin
from src.api.response_cache import invalidate
from src.models.comfy_template import ComfyTemplate
from src.models.database import get_async_session
from src.models.service_instance import ServiceInstance
from src.services.comfy.client import ComfyClient
from src.services.workflow_snapshot import NAME_RE

router = APIRouter(prefix="/api/v1/comfy-templates", tags=["comfy-templates"])
health_router = APIRouter(prefix="/api/v1/comfy", tags=["comfy-health"])

# Module-level cache for object_info: (timestamp, data) tuple or None
_object_info_cache: tuple[float, dict] | None = None


def get_client() -> ComfyClient:
    """Factory function to create a ComfyClient instance.
    Tests monkeypatch this to inject a mock client."""
    return ComfyClient()


def _validate_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise ValueError(
            "service name must match ^[a-z][a-z0-9-]{1,62}$ "
            "(start with a-z, then a-z/0-9/-, total 2-63 chars)",
        )
    return name


def _bridge_snapshot(template_id: int) -> dict:
    return {
        "nodes": [
            {"id": "bridge", "type": "comfyui_workflow", "data": {"template_id": template_id}},
            {"id": "out", "type": "video_output", "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "bridge", "sourceHandle": "outputs",
             "target": "out", "targetHandle": "outputs"},
        ],
    }


# ---------- Pydantic shapes ----------


class CreateTemplateBody(BaseModel):
    name: str
    workflow: dict[str, Any]

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _validate_name(v)


class ExposedParamMapping(BaseModel):
    key: str
    label: str = ""
    type: str = "string"
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list | None = None
    required: bool = True
    random: bool = False
    comfy_node_id: str
    comfy_input: str


class MappingBody(BaseModel):
    exposed_params: list[ExposedParamMapping]


class ReuploadBody(BaseModel):
    workflow: dict[str, Any]


# ---------- Helpers ----------


async def _get_template_and_service(
    session: AsyncSession, template_id: int,
) -> tuple[ComfyTemplate, ServiceInstance]:
    tpl = await session.get(ComfyTemplate, template_id)
    if tpl is None:
        raise HTTPException(404, detail="template not found")
    stmt = (
        select(ServiceInstance)
        .options(undefer(ServiceInstance.exposed_inputs))
        .where(ServiceInstance.source_type == "comfy_template")
        .where(ServiceInstance.source_id == template_id)
    )
    svc = (await session.execute(stmt)).scalar_one_or_none()
    if svc is None:
        raise HTTPException(404, detail="service not found for template")
    return tpl, svc


def _mapping_to_exposed_input(m: ExposedParamMapping) -> dict:
    """展开成 exposed_inputs 存储项:node_id/input_name 固定指向桥节点(供
    apply_inputs_to_snapshot 用),comfy_node_id/comfy_input 保留原 ComfyUI
    定位(供重新上传时的 stale 校验用)。"""
    return {
        "key": m.key,
        "label": m.label,
        "node_id": "bridge",
        "input_name": m.key,
        "type": m.type,
        "default": m.default,
        "min": m.min,
        "max": m.max,
        "step": m.step,
        "options": m.options,
        "required": m.required,
        "random": m.random,
        "comfy_node_id": m.comfy_node_id,
        "comfy_input": m.comfy_input,
    }


def _exposed_input_to_param(item: dict) -> dict:
    return {
        "key": item.get("key"),
        "label": item.get("label", ""),
        "type": item.get("type", "string"),
        "default": item.get("default"),
        "min": item.get("min"),
        "max": item.get("max"),
        "step": item.get("step"),
        "options": item.get("options"),
        "required": item.get("required", True),
        "random": item.get("random", False),
        "comfy_node_id": item.get("comfy_node_id"),
        "comfy_input": item.get("comfy_input"),
    }


def _stale_keys(exposed_inputs: list[dict], workflow: dict) -> list[str]:
    stale = []
    for item in exposed_inputs:
        node_id = item.get("comfy_node_id")
        input_name = item.get("comfy_input")
        node = workflow.get(node_id)
        if node is None or input_name not in (node.get("inputs") or {}):
            stale.append(item.get("key"))
    return stale


# ---------- Routes ----------


@router.post("", status_code=201, dependencies=[Depends(require_admin)])
async def create_template(
    body: CreateTemplateBody,
    session: AsyncSession = Depends(get_async_session),
):
    existing = await session.scalar(
        select(ComfyTemplate).where(ComfyTemplate.name == body.name)
    )
    if existing is not None:
        raise HTTPException(409, detail=f"template name '{body.name}' already exists")
    existing_svc = await session.scalar(
        select(ServiceInstance).where(ServiceInstance.name == body.name)
    )
    if existing_svc is not None:
        raise HTTPException(409, detail=f"service name '{body.name}' already exists")

    tpl = ComfyTemplate(name=body.name, workflow_json=body.workflow)
    session.add(tpl)
    await session.flush()

    svc = ServiceInstance(
        name=body.name,
        type="workflow",
        status="active",
        source_type="comfy_template",
        source_id=tpl.id,
        category="app",
        meter_dim="calls",
        workflow_snapshot=_bridge_snapshot(tpl.id),
        exposed_inputs=[],
    )
    session.add(svc)
    # round4-style TOCTOU guard (see services.py quick_provision): the precheck
    # above is not atomic with the insert — a concurrent same-name create can
    # still race past it and hit the UNIQUE constraint at commit time. Convert
    # that into a 409 instead of letting IntegrityError bubble to a 500.
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, detail=f"template/service name '{body.name}' already exists")

    invalidate("services")
    return {
        "id": str(tpl.id),
        "name": tpl.name,
        "service_name": svc.name,
        "node_count": len(body.workflow),
    }


@router.get("", dependencies=[Depends(require_admin)])
async def list_templates(session: AsyncSession = Depends(get_async_session)):
    tpls = (await session.execute(select(ComfyTemplate))).scalars().all()
    out = []
    for tpl in tpls:
        stmt = (
            select(ServiceInstance)
            .options(undefer(ServiceInstance.exposed_inputs))
            .where(ServiceInstance.source_type == "comfy_template")
            .where(ServiceInstance.source_id == tpl.id)
        )
        svc = (await session.execute(stmt)).scalar_one_or_none()
        out.append({
            "id": str(tpl.id),
            "name": tpl.name,
            "service_name": svc.name if svc else None,
            "node_count": len(tpl.workflow_json or {}),
            "exposed_count": len(svc.exposed_inputs or []) if svc else 0,
        })
    return out


@router.get("/{template_id}", dependencies=[Depends(require_admin)])
async def get_template(
    template_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    tpl, svc = await _get_template_and_service(session, template_id)
    return {
        "id": str(tpl.id),
        "name": tpl.name,
        "service_name": svc.name,
        "workflow_json": tpl.workflow_json,
        "exposed_params": [_exposed_input_to_param(i) for i in (svc.exposed_inputs or [])],
    }


@router.put("/{template_id}/mapping", dependencies=[Depends(require_admin)])
async def update_mapping(
    template_id: int,
    body: MappingBody,
    session: AsyncSession = Depends(get_async_session),
):
    _tpl, svc = await _get_template_and_service(session, template_id)
    svc.exposed_inputs = [_mapping_to_exposed_input(m) for m in body.exposed_params]
    await session.commit()
    invalidate("services")
    return {"ok": True}


@router.put("/{template_id}", dependencies=[Depends(require_admin)])
async def reupload_template(
    template_id: int,
    body: ReuploadBody,
    session: AsyncSession = Depends(get_async_session),
):
    tpl, svc = await _get_template_and_service(session, template_id)
    tpl.workflow_json = body.workflow
    stale = _stale_keys(svc.exposed_inputs or [], body.workflow)
    await session.commit()
    invalidate("services")
    return {"stale_keys": stale}


@router.delete("/{template_id}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_template(
    template_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    tpl, svc = await _get_template_and_service(session, template_id)
    await session.delete(svc)
    await session.delete(tpl)
    await session.commit()
    invalidate("services")


# ---------- ComfyUI Health & Object Info Proxy Routes ----------


@health_router.get("/health")
async def comfy_health():
    """Proxy ComfyUI health status + resolved configuration."""
    client = get_client()
    health_result = await client.health()
    # Merge with configuration info
    base_url = os.getenv("NOUS_COMFY_URL", "http://127.0.0.1:8188")
    timeout_s = int(os.getenv("NOUS_COMFY_TIMEOUT", "14400"))
    health_result["base_url"] = base_url
    health_result["timeout_s"] = timeout_s
    return health_result


@health_router.get("/object-info")
async def comfy_object_info():
    """Proxy ComfyUI object_info with 60s TTL cache."""
    global _object_info_cache
    now = time.monotonic()

    # Check cache validity (60s TTL)
    if _object_info_cache is not None:
        cached_time, cached_data = _object_info_cache
        if now - cached_time < 60:
            return cached_data

    # Cache miss or expired: fetch from client
    client = get_client()
    try:
        data = await client.object_info()
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(
            502,
            detail=f"ComfyUI sidecar 不可达或返回无效响应:{str(e)[:100]}"
        )
    _object_info_cache = (now, data)
    return data
