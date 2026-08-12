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

import asyncio
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
from src.services.comfy.client import ComfyError
from src.services.comfy.client import get_comfy_client as get_client
from src.services.workflow_snapshot import NAME_RE

router = APIRouter(prefix="/api/v1/comfy-templates", tags=["comfy-templates"])
health_router = APIRouter(prefix="/api/v1/comfy", tags=["comfy-health"])

# Module-level cache for object_info: (timestamp, data) tuple or None
_object_info_cache: tuple[float, dict] | None = None


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
        # I3 fix:min/max/step/options/random 嵌套进 `constraints`,不再是顶层平铺键——
        # 这是前端 SchemaDrivenForm.classifyField / isRandomizable(见 SchemaDrivenForm.tsx:
        # 34-61)以及 service_schema.py::_input_property 实际读取的形状(都读
        # `param.constraints.{min,max,step,enum,random}`,不读顶层)。旧的平铺写法两边都
        # 读不到——Playground 数字字段永远渲不成 slider,随机种子按钮也出不来。
        "constraints": _numeric_constraints(m),
        "required": m.required,
        "comfy_node_id": m.comfy_node_id,
        "comfy_input": m.comfy_input,
    }


def _numeric_constraints(m: ExposedParamMapping) -> dict[str, Any]:
    c: dict[str, Any] = {}
    if m.min is not None:
        c["min"] = m.min
    if m.max is not None:
        c["max"] = m.max
    if m.step is not None:
        c["step"] = m.step
    if m.options:
        c["enum"] = m.options
    if m.random:
        c["random"] = m.random
    return c


def _exposed_input_to_param(item: dict) -> dict:
    """exposed_inputs 存储项 → 编辑器吃的平铺 `ComfyExposedParam` 形(min/max/step/
    options/random 从 `constraints` 里读出来摊平)。兼容 I3 修复前写入的旧行——那些行
    没有 `constraints` 键,直接退回顶层同名字段(迁移期防止已保存映射的 min/max 消失)。
    """
    c = item.get("constraints")
    if not isinstance(c, dict):
        c = {}
    return {
        "key": item.get("key"),
        "label": item.get("label", ""),
        "type": item.get("type", "string"),
        "default": item.get("default"),
        "min": c.get("min", item.get("min")),
        "max": c.get("max", item.get("max")),
        "step": c.get("step", item.get("step")),
        "options": c.get("enum", item.get("options")),
        "required": item.get("required", True),
        "random": c.get("random", item.get("random", False)),
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
        # I1 fix:不种 exposed_outputs,`/v1/services/{name}/schema` 的 output_schema
        # 就是空 {}——SchemaDrivenOutput 在 Playground 里没有任何 declared output 可渲染,
        # 只能整坨 dump 原始 JSON,video player 出不来。桥快照(_bridge_snapshot)固定的
        # 终端节点是 id="out" 的 video_output,它把桥节点输出原样透传(见 comfy_bridge.py
        # VideoOutputNode),桥节点的返回形状固定是 {items,video_url,thumbnails,seed}——
        # 这里种的 node_id/input_name 必须精确对上那个形状,SchemaDrivenOutput.pluck()
        # 才能按 (node_id, input_name) 取到 video_url。
        exposed_outputs=[{
            "key": "video_url", "node_id": "out", "input_name": "video_url",
            "type": "video", "label": "视频",
        }],
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


def _devices_from_stats(stats: dict) -> list[dict]:
    """system_stats() 的 devices 数组 → 前端要的精简形状(原始字节,前端自己格式化)。"""
    out = []
    for d in stats.get("devices") or []:
        vram_total = d.get("vram_total", 0) or 0
        vram_free = d.get("vram_free", 0) or 0
        out.append({
            "name": d.get("name", ""),
            "type": d.get("type", ""),
            "index": d.get("index"),
            "vram_total": vram_total,
            "vram_free": vram_free,
            "vram_used": vram_total - vram_free,
            "torch_vram_total": d.get("torch_vram_total", 0) or 0,
        })
    return out


@health_router.get("/health")
async def comfy_health():
    """Proxy ComfyUI health status + resolved configuration + per-device VRAM."""
    client = get_client()
    health_result = await client.health()
    # Merge with configuration info
    base_url = os.getenv("NOUS_COMFY_URL", "http://127.0.0.1:8188")
    timeout_s = int(os.getenv("NOUS_COMFY_TIMEOUT", "14400"))
    health_result["base_url"] = base_url
    health_result["timeout_s"] = timeout_s
    health_result["devices"] = (
        _devices_from_stats(await client.system_stats()) if health_result.get("online") else []
    )
    return health_result


def _used_of(devices: list[dict]) -> int:
    return sum(d.get("vram_used", 0) or 0 for d in devices)


# 释放后的轮询节奏(测试 monkeypatch 成 0 免得空等 6s)。
_FREE_POLL_ROUNDS = 12
_FREE_POLL_INTERVAL = 0.5


@health_router.post("/free", dependencies=[Depends(require_admin)])
async def comfy_free():
    """释放 ComfyUI 侧缓存模型占用的显存(POST /free,unload_models+free_memory)。

    ComfyUI 的 /free **是异步的**:server.py 只 `prompt_queue.set_flag(...)`,真正的
    `unload_all_models()` 要等 main.py 的 worker 被 `not_empty.notify()` 唤醒后才跑
    (实测 64.3G → 43.8G 用了几秒)。所以调完立刻拍 system_stats 必然还是旧值,按钮
    看着像没生效。这里轮询到显存真降下来(或 6s 超时)再返回快照,`settled` 告诉前端
    这份数字是否已经落定。
    """
    client = get_client()
    try:
        before = _used_of(_devices_from_stats(await client.system_stats()))
        await client.free()
    except ComfyError as e:
        raise HTTPException(e.status_code, detail=e.message)
    except httpx.HTTPError as e:
        raise HTTPException(502, detail=f"ComfyUI sidecar 不可达:{str(e)[:100]}")

    devices: list[dict] = []
    settled = False
    for _ in range(_FREE_POLL_ROUNDS):  # 默认 12 × 0.5s = 6s 上限
        await asyncio.sleep(_FREE_POLL_INTERVAL)
        devices = _devices_from_stats(await client.system_stats())
        # 归还 >1GB 才算落定 —— 显存读数本身有百 MB 级抖动(同卡其它进程)。
        if before - _used_of(devices) > 1_000_000_000:
            settled = True
            break
    return {"ok": True, "settled": settled, "freed_bytes": max(0, before - _used_of(devices)),
            "devices": devices}


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
