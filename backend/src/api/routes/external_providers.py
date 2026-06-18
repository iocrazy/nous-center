"""外部 CLI 生成 provider 内网管理端点(admin-gated)。

供前端面板 + 你另一平台诊断:列 provider / 探活 / 触发登录。
**生成不走这里** —— 生成统一经 service + /v1/images/generations(spec §3.5),保证对外接口单一。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps_admin import require_admin
from src.services.external_providers import get_registry
from src.services.external_providers.base import ProviderError

router = APIRouter(prefix="/api/v1/external-providers", tags=["external-providers"])


@router.get("", dependencies=[Depends(require_admin)])
async def list_providers():
    """列出已配置 provider + 实时探活状态。"""
    registry = get_registry()
    out = []
    for name, governed in registry.items():
        try:
            status = await governed.probe_status()
            out.append(status.model_dump())
        except ProviderError as exc:
            out.append({"name": name, "available": False, "message": exc.message})
        except Exception as exc:  # noqa: BLE001 — 探活不该 500 整个列表
            out.append({"name": name, "available": False, "message": f"探活异常:{exc}"})
    return {"providers": out}


@router.get("/{name}", dependencies=[Depends(require_admin)])
async def provider_status(name: str):
    governed = get_registry().get(name)
    if governed is None:
        raise HTTPException(status_code=404, detail=f"未知 provider:{name}")
    try:
        return (await governed.probe_status()).model_dump()
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/{name}/login", dependencies=[Depends(require_admin)])
async def provider_login(name: str):
    governed = get_registry().get(name)
    if governed is None:
        raise HTTPException(status_code=404, detail=f"未知 provider:{name}")
    try:
        return await governed.login_start()
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
