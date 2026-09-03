"""服务级「开机启动」—— 开机预加载哪些模型的唯一显式来源。

背景:2026-09-03 之前 lifespan 有一条隐藏的 `_load_wf_deps`,把**所有** published
工作流引用到的模型开机全 `load_model`,完全不看 resident 标记 —— 用户在 UI 里没开
「自动加载」的模型照样开机占满显存。那条路已删(见 startup_reconcile.py 的
docstring)。本模块是它的替代品:**显式、服务级、默认关**。

开机加载策略(全仓唯一一句话版本):
    开机只加载 ① `resident: true` 的模型 ② `autostart=true` 服务引用到的模型;
    其余一律等工作流真正执行时 runner 走 `get_or_load` 按需加载。

只预加载 **registry engine ref**(`kind == "engine"`,即 `engine_key` 能在
ModelRegistry 里查到的模型)。图像类服务引用的是**组件文件**(diffusion/clip/vae
单文件,`kind == "component"`),它们不走 `load_model`(要 dtype/device/lora 才能定
combo),这里既不列进预览也不加载 —— 预览承诺什么、开机就加载什么,不多不少。
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.orm import undefer

from src.models.service_instance import ServiceInstance
from src.services.service_models import extract_service_models

logger = logging.getLogger(__name__)


def service_engine_keys(svc: ServiceInstance) -> list[str]:
    """本服务引用到的 registry 模型 key(去重,保持首次出现顺序)。

    与 `routes/services.py::_service_model_refs` 同源:
      - `source_type == "model"` 的服务直接绑一个引擎,key 就是 `source_name`;
      - 其余从冻结的 `workflow_snapshot` 抽 refs,只取 `kind == "engine"` 的。
    """
    if svc.source_type == "model" and svc.source_name:
        return [svc.source_name]
    keys: list[str] = []
    for ref in extract_service_models(svc.workflow_snapshot):
        if ref.get("kind") == "engine" and ref.get("engine_key"):
            key = str(ref["engine_key"])
            if key not in keys:
                keys.append(key)
    return keys


def preload_model_infos(svc: ServiceInstance, registry) -> list[dict[str, Any]]:
    """开机会为本服务加载的模型清单 —— `{name, vram_gb?, gpu?}`。

    registry 里查不到的 key 仍然列出(name 而已,无 vram/gpu),开机时会 fail-soft
    地 warning 掉:宁可预览诚实地说「有这么个引用」,也不要静默吞掉。
    """
    infos: list[dict[str, Any]] = []
    for key in service_engine_keys(svc):
        spec = registry.get(key) if registry is not None else None
        info: dict[str, Any] = {"name": key}
        # 显式 isinstance 而不是 truthy —— registry 可能是 yaml 加载的、scanner 合成的,
        # 测试里还可能是 MagicMock;拿到怪东西时宁可只报 name,也不要塞进响应炸 500。
        vram_mb = getattr(spec, "vram_mb", None)
        if isinstance(vram_mb, (int, float)) and not isinstance(vram_mb, bool) and vram_mb > 0:
            info["vram_gb"] = round(vram_mb / 1024, 1)
        gpu = getattr(spec, "gpu", None)
        if isinstance(gpu, int) and not isinstance(gpu, bool):
            info["gpu"] = gpu
        elif isinstance(gpu, list) and all(isinstance(g, int) for g in gpu):
            info["gpu"] = gpu
        infos.append(info)
    return infos


async def collect_autostart_models(session, registry) -> tuple[list[str], list[str]]:
    """扫 `autostart=true` 且 status=active 的服务 → (服务名列表, 去重后的模型 key)。

    status 过滤是有意的:retired/paused 的服务不该在开机时占显存;重新 active 后
    下次开机自然生效。`workflow_snapshot` 是 deferred 列,必须 undefer,否则
    async 路径上触发 lazy load 会 MissingGreenlet。
    """
    rows = (await session.execute(
        select(ServiceInstance)
        .options(undefer(ServiceInstance.workflow_snapshot))
        .where(ServiceInstance.autostart.is_(True), ServiceInstance.status == "active")
        .order_by(ServiceInstance.created_at.asc())
    )).scalars().all()

    names: list[str] = []
    keys: list[str] = []
    for svc in rows:
        names.append(svc.name)
        for key in service_engine_keys(svc):
            if key not in keys:
                keys.append(key)
    return names, keys


async def preload_autostart_services(
    session_factory,
    model_mgr,
    on_loaded: Callable[[str], Awaitable[None]] | None = None,
) -> list[str]:
    """开机预加载 autostart 服务引用的模型。返回真正加载成功的 key。

    **必须在 `preload_residents` 之后、同一个后台任务里顺序 await**,不要另起一个
    并发 task —— 2026-07-06 生产事故就是两个独立 task 同一瞬间往同一张卡 spawn 两个
    vLLM,engine core init 竞争把 embedding 加载搞失败(见 model_manager 的
    `_global_load_lock` 注释)。串行门虽然还在,但少一个并发源就少一分意外。

    跳过:已加载的(is_loaded)、resident 的(preload_residents 已经管了)。
    Fail-soft:单个模型加载失败只 warning,不阻塞其余模型、更不阻塞启动。
    """
    registry = getattr(model_mgr, "_registry", None)
    try:
        async with session_factory() as session:
            names, keys = await collect_autostart_models(session, registry)
    except Exception:  # noqa: BLE001 — 查不到就当没有,绝不阻塞启动
        logger.exception("autostart: failed to collect autostart services")
        return []

    if not names:
        return []
    logger.info("Autostart services: %s → preloading models %s", names, keys)

    loaded: list[str] = []
    for key in keys:
        spec = registry.get(key) if registry is not None else None
        if spec is not None and spec.resident:
            continue  # preload_residents 已经加载过
        if model_mgr.is_loaded(key):
            continue
        try:
            await model_mgr.load_model(key)
        except Exception as e:  # noqa: BLE001 — fail-soft,同 preload_residents
            logger.warning("Autostart preload failed for %s: %s", key, e)
            continue
        loaded.append(key)
        if on_loaded is not None:
            try:
                await on_loaded(key)
            except Exception:  # noqa: BLE001 — 回调 best-effort
                logger.warning("autostart on_loaded callback failed for %s", key, exc_info=True)
    return loaded
