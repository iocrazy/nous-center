import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.api.response_cache import cached, invalidate
from src.services.model_scanner import scan_models
from src.gpu.detector import gpu_summary
from src.models.database import get_async_session
from src.models.schemas import EngineInfo, EngineLoadResponse
from src.services.model_metadata_service import (
    get_all_metadata, sync_metadata, refresh_metadata, scan_local_models,
    _format_size,
)
from src.api.websocket import ws_manager

# 持后台 fire-and-forget task 的强引用,防止 asyncio 中途 GC 掉(round4 #8/#9 修过同类)。
# add_done_callback(discard) 完成即自动移除。
_bg_tasks: set[asyncio.Task] = set()


def _spawn_bg(coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return t

router = APIRouter(prefix="/api/v1/engines", tags=["engines"])
logger = logging.getLogger(__name__)

# In-memory loading state tracker: model_id -> {"status": "loading"|"failed", "detail": str}
_loading_states: dict[str, dict[str, str]] = {}


@router.get("/gpus")
async def list_gpus():
    """Return detected GPU information."""
    return gpu_summary()


def _get_model_manager(request: Request):
    return getattr(request.app.state, "model_manager", None)


def _is_engine_loaded(name: str, request: Request | None = None) -> bool:
    if request is not None:
        mgr = _get_model_manager(request)
        if mgr is not None:
            return mgr.is_loaded(name)
    # Fallback to old registries
    from src.workers.tts_engines import registry as tts_registry
    from src.workers.llm_engines import registry as llm_registry
    engine = tts_registry._ENGINE_INSTANCES.get(name)
    if engine is not None:
        return engine.is_loaded
    engine = llm_registry._ENGINE_INSTANCES.get(name)
    if engine is not None:
        return engine.is_loaded
    return False


def _get_loaded_gpu(name: str, request: Request | None = None) -> int | None:
    if request is not None:
        mgr = _get_model_manager(request)
        if mgr is not None and mgr.is_loaded(name):
            lm = mgr._models.get(name)
            return lm.gpu_index if lm else None
    return None


def _get_loaded_gpus(name: str, request: Request | None = None) -> list[int] | None:
    if request is not None:
        mgr = _get_model_manager(request)
        if mgr is not None and mgr.is_loaded(name):
            lm = mgr._models.get(name)
            if lm and lm.gpu_indices:
                return lm.gpu_indices
            elif lm:
                return [lm.gpu_index]
    return None


def _build_engine_info(key: str, cfg: dict, meta, local_dirs: set[str], request: Request | None = None) -> EngineInfo:
    local_path = cfg.get("local_path")
    local_exists = local_path in local_dirs if local_path else False
    loaded = _is_engine_loaded(key, request)

    # Determine status: check loading states first, then fall back to loaded/unloaded
    loading_state = _loading_states.get(key)
    if loading_state:
        status = loading_state["status"]
        status_detail = loading_state.get("detail", "")
    elif loaded:
        status = "loaded"
        status_detail = None
    else:
        status = "unloaded"
        status_detail = None

    # `gpu` can be None when the YAML leaves the slot to the detector. For
    # display, resolve via the detector so the UI shows the real assignment.
    gpu_field = cfg.get("gpu")
    if gpu_field is None:
        from src.gpu.detector import get_device_for_engine
        device = get_device_for_engine(cfg)
        try:
            gpu_field = int(device.split(":")[-1]) if device.startswith("cuda") else 0
        except (ValueError, IndexError):
            gpu_field = 0

    # Image card chip: "<n> LoRA" — always architecture-filtered count, even
    # when the model is loaded. Adapter holds ALL scanned LoRAs internally
    # so existing workflows can reference incompatible ones (the apply path
    # produces a friendly error if architecture mismatches at runtime). The
    # chip should show the *useful* count, not the raw collection size, and
    # must be consistent loaded vs unloaded — otherwise the same model
    # flips from "12 LoRA" to "0 LoRA" just by toggling resident state.
    #
    # yaml `params.accepts_lora_archs` lists architectures this model can
    # load (e.g. ['flux1', 'flux2'] for Flux2 Klein). Empty/missing →
    # "everything" (legacy behaviour, falls back to total count).
    lora_count: int | None = None
    if cfg.get("type") == "image":
        accepts = (cfg.get("params") or {}).get("accepts_lora_archs") or []
        from src.services.lora_scanner import count_loras_for_arches
        lora_count = count_loras_for_arches(accepts)

    info = EngineInfo(
        name=key,
        display_name=cfg["name"],
        type=cfg["type"],
        status=status,
        gpu=gpu_field,
        vram_gb=cfg.get("vram_gb", 0),
        resident=cfg.get("resident", False),
        local_path=local_path,
        local_exists=local_exists,
        auto_detected=cfg.get("auto_detected", False),
        has_adapter=bool(cfg.get("adapter")),
        loaded_gpu=_get_loaded_gpu(key, request) if loaded else None,
        loaded_gpus=_get_loaded_gpus(key, request) if loaded else None,
        status_detail=status_detail,
        lora_count=lora_count,
    )
    if meta:
        info.organization = meta.organization
        info.model_size = _format_size(meta.model_size_bytes)
        info.frameworks = meta.frameworks
        info.libraries = meta.libraries
        info.license = meta.license
        info.languages = meta.languages
        info.tags = meta.tags
        info.tensor_types = meta.tensor_types
        info.description = meta.description
        info.has_metadata = True
    return info


@router.get("")
@cached("engines", ttl=30)
async def list_all_engines(
    request: Request,
    type: str | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """List all engines with metadata. Optionally filter by type.

    Cached 30s in-process. Cache key includes ``?type=`` filter via the cache
    layer's query-string handling. Writes to engine state (load/unload/scan/
    resident/gpu/install_deps and the background loaders) all call
    ``invalidate("engines")`` to drop the cached body before the next read.
    """
    configs = scan_models()
    metadata = await get_all_metadata(session)
    local_dirs = scan_local_models()
    result = []
    for key, cfg in configs.items():
        if type and cfg.get("type") != type:
            continue
        # Only show models that exist locally
        local_path = cfg.get("local_path")
        if not local_path or local_path not in local_dirs:
            continue
        result.append(_build_engine_info(key, cfg, metadata.get(key), local_dirs, request))
    # 统一引擎库(spec 2026-06-02):补 by-key 超分(SeedVR2)+ 单文件组件(diffusion_models/clip/
    # vae/loras)目录条目,带 VRAM 残留状态(loaded/gpu 从 aggregate_runner_loaded 多键匹配)。
    # registry 不含它们(model_scanner skip 组件 / SeedVR2 by-key)→ 在此并入,让引擎库统一可见。
    from src.services.engine_catalog import catalog_extra_engines
    result.extend(catalog_extra_engines(request.app.state, type))
    return result


@router.post("/scan", dependencies=[Depends(require_admin)])
async def scan_models_endpoint():
    """Re-scan models directory for new models. Also drops LoRA arch cache so
    newly-added LoRAs get architecture-detected on the next /api/v1/engines.

    Response shape distinguishes **配置识别** vs **本地可用**:用户报告 toast
    显示「扫到 25」结果引擎库只有 16,差异在于 yaml 配的 9 个模型本地没下载,
    被 list_all_engines 在 `local_path not in local_dirs` 过滤。本接口把两个
    数字一起返回,前端可以拼出「识别 N · 本地可用 K · 未下载 (N-K)」消除误导。
    """
    # 显式 /scan 意图就是"现在重扫" → 先清 TTL 缓存,强制走盘一次(否则最长 30s 内看不到
    # 新模型)。
    from src.services.model_scanner import invalidate_scan_cache
    invalidate_scan_cache()
    configs = scan_models()
    local_dirs = scan_local_models()
    # configs 已包含 local_path;在 local_dirs 中即「磁盘上真有目录的模型」。
    local_available = sum(
        1 for cfg in configs.values()
        if cfg.get("local_path") and cfg["local_path"] in local_dirs
    )
    invalidate("engines")
    from src.services.lora_scanner import invalidate_cache as _inv_lora
    _inv_lora()
    return {
        "count": len(configs),
        "local_available": local_available,
        "not_local": len(configs) - local_available,
        "models": list(configs.keys()),
    }


@router.post("/reload", dependencies=[Depends(require_admin)])
async def reload_registry(request: Request):
    """热加载模型定义,不重启即认新模型。

    重读单一来源 collect_model_entries(`configs/models.d/*.yaml` 一模型一文件
    + 兼容 `models.yaml` 的 legacy `models:` list)→ 丢个新 `<id>.yaml` 进 models.d
    后调本接口即可插拔上线,无需重启后端。
    """
    mgr = _get_model_manager(request)
    if mgr is None:
        raise HTTPException(503, "ModelManager not initialized")
    new_count = mgr._registry.reload()
    invalidate("engines")
    return {"status": "reloaded", "new_models": new_count, "total": len(mgr._registry.specs)}


@router.post("/sync-metadata", dependencies=[Depends(require_admin)])
async def sync_all_metadata(session: AsyncSession = Depends(get_async_session)):
    """Fetch metadata for any engine not yet in DB."""
    metadata = await sync_metadata(session)
    invalidate("engines")
    return {"synced": len(metadata)}


@router.post("/{name}/refresh-metadata", response_model=EngineInfo, dependencies=[Depends(require_admin)])
async def refresh_engine_metadata(
    name: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Force re-fetch metadata for a specific engine."""
    configs = scan_models()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")
    meta = await refresh_metadata(session, name)
    local_dirs = scan_local_models()
    invalidate("engines")
    return _build_engine_info(name, configs[name], meta, local_dirs)


# 注:必须定义在 `/{name}/unload` **之前** —— 否则参数路由 `/{name}/unload` 抢先匹配
# `component/unload`(name='component')致 404。统一模型管理收尾 PR-1。
@router.post("/component/unload", dependencies=[Depends(require_admin)])
async def unload_component(request: Request, body: dict = Body(...)):
    """卸载**已预加载**组件(引擎库组件卡「出缓存」,统一模型管理收尾 PR-1)。

    body: `state_key`(component_state_key,file|device|dtype|loras 串)或 `name=
    'component:<kind>:<path>'`+`device`/`dtype` 自拼(同 /component/resident)。派 image runner →
    mm.unload_image_component(出 L1 + 释放显存;combo 在用则只清常驻)。没加载则 no-op。状态经下个 Pong 反映。"""
    state_key = str(body.get("state_key") or "")
    if not state_key:
        parsed = _parse_component_name(str(body.get("name") or ""))
        if parsed:
            dev = str(body.get("device") or "auto")
            dtype = str(body.get("dtype") or "bfloat16")
            state_key = f"{parsed[1]}|{dev}|{dtype}|"
    if not state_key:
        raise HTTPException(422, "需要 state_key 或 name='component:<kind>:<path>'(+device/dtype)")
    client = (getattr(request.app.state, "runner_clients", {}) or {}).get("image")
    if client is None or not getattr(client, "_connected", True):
        raise HTTPException(503, "image runner not available")
    await client.unload_component(state_key=state_key)
    invalidate("engines")
    return {"status": "accepted", "state_key": state_key}


# 注:同 component/unload —— 必须在 `/{name}/unload` 之前定义,否则参数路由抢先匹配致 404。
@router.post("/loaded-adapter/unload", dependencies=[Depends(require_admin)])
async def unload_loaded_adapter(request: Request, body: dict = Body(...)):
    """卸载**已加载的 combo adapter**(引擎库「已加载」卡的卸载按钮,统一模型管理收尾 PR-2)。

    body: `model_id`(combo 哈希 id,/loaded-adapters 列的)。按 aggregate_runner_loaded 找它所在
    runner group → 向该 runner 派 UnloadModel + 对账快照。combo = 工作流动态组装的单文件(unet+clip+vae)
    /anima/SeedVR2 等,model_id 是哈希。没加载则 no-op。状态经下个 Pong 反映。"""
    from src.services.runner_models import aggregate_runner_loaded  # noqa: PLC0415
    model_id = str(body.get("model_id") or "")
    if not model_id:
        raise HTTPException(422, "需要 model_id(/loaded-adapters 列的 combo id)")
    state = request.app.state
    sups_by_group = {
        getattr(s, "group_id", None): s
        for s in (getattr(state, "runner_supervisors", None) or [])
    }
    unloaded = False
    for e in aggregate_runner_loaded(state):
        if str(e.get("model_id") or "") != model_id:
            continue
        sup = sups_by_group.get(e.get("group_id"))
        if sup is not None and getattr(sup, "client", None) is not None:
            try:
                await sup.client.unload_model(model_id)
                unloaded = True
                if hasattr(sup, "_reconcile_loaded"):
                    await sup._reconcile_loaded()
            except Exception as ex:  # noqa: BLE001 — 单个失败不挡
                logger.warning("unload loaded-adapter %s failed: %s", model_id, ex)
        break
    invalidate("engines")
    return {"status": "accepted", "model_id": model_id, "unloaded": unloaded}


@router.post("/{name}/load", response_model=EngineLoadResponse, dependencies=[Depends(require_admin)])
async def load_engine(name: str, request: Request):
    configs = scan_models()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    cfg = configs[name]
    # Refuse to start a load that we know will crash. Auto-detected
    # diffusers (image/video) ship without an adapter — letting the
    # background task run only to ValueError("Unknown model") gives
    # the user a stale toast and a "failed" badge with no path forward.
    if not cfg.get("adapter"):
        raise HTTPException(
            422,
            detail=(
                f"'{name}' was auto-detected on disk but has no adapter "
                f"configured. Image/video diffusers loading is not "
                f"implemented yet — add an adapter entry to "
                f"backend/configs/models.yaml to enable loading."
            ),
        )

    # Reject if already loading
    if name in _loading_states and _loading_states[name]["status"] == "loading":
        return EngineLoadResponse(name=name, status="loading")

    model_mgr = request.app.state.model_manager

    # If already loaded, return immediately
    if model_mgr.is_loaded(name):
        return EngineLoadResponse(name=name, status="loaded")

    # Start background loading
    _loading_states[name] = {"status": "loading", "detail": "Starting..."}
    invalidate("engines")
    _spawn_bg(_load_in_background(name, model_mgr))

    return EngineLoadResponse(name=name, status="loading")


async def _load_in_background(name: str, model_mgr):
    import logging
    logger = logging.getLogger(__name__)
    start = time.monotonic()
    try:
        _loading_states[name] = {"status": "loading", "detail": "Loading model..."}
        await ws_manager.broadcast_model_status(name, "loading", "Loading model...")
        await model_mgr.load_model(name)
        elapsed = round(time.monotonic() - start, 2)
        # Clear loading state on success — the model is now truly loaded
        _loading_states.pop(name, None)
        # Invalidate cache so the next /engines GET reflects the new status
        # (background task changes status without going through an HTTP write).
        invalidate("engines")
        await ws_manager.broadcast_model_status(name, "loaded", f"Ready ({elapsed}s)")
        logger.info("Model %s loaded in %.2fs", name, elapsed)
    except Exception as e:
        _loading_states[name] = {"status": "failed", "detail": str(e)}
        invalidate("engines")
        await ws_manager.broadcast_model_status(name, "failed", str(e))
        logger.error("Model %s load failed: %s", name, e)


@router.post("/{name}/unload", response_model=EngineLoadResponse, dependencies=[Depends(require_admin)])
async def unload_engine(name: str, request: Request, force: bool = False):
    configs = scan_models()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    cfg = configs[name]
    if cfg.get("resident", False) and not force:
        raise HTTPException(409, detail=f"Engine {name} is resident. Use force=true to unload.")

    model_mgr = request.app.state.model_manager
    await model_mgr.unload_model(name, force=force)

    # round9 BUG4:清掉残留的 loading/failed 状态。_build_engine_info 里
    # _loading_states 优先级高于 loaded/unloaded —— load 失败写了 {"status":"failed"}
    # 后从不被清(unload 旧实现不 pop),GET /engines 会永远显示 "failed",哪怕重新
    # unload 也甩不掉。卸载即代表该 engine 回到干净 unloaded 态,这里 pop 掉。
    _loading_states.pop(name, None)

    invalidate("engines")
    return EngineLoadResponse(name=name, status="unloaded")


@router.get("/image-cache", dependencies=[Depends(require_admin)])
async def list_image_adapter_cache(request: Request):
    """诊断:列当前已加载的所有 image adapter 的 derived_id + GPU + 上次用。

    返回:`[{model_id, gpu_index, last_used_ago_sec, pipeline_class, vram_mb,
    source_files, group_id}]`。

    **关键(#198 修正)**:image adapter 真加载在 **runner 子进程**,主进程的
    `_models` 永远是空的 —— 老实现查主进程恒返 count=0(不是 cache miss,是查错进程)。
    现改读 `aggregate_runner_loaded`(runner 经 Pong 上报的快照)。用户报告「跑两次同
    工作流应复用却 OOM」时,从这接口看 list 是否真有 2+ entry(同 base 不同 combo),
    再对照 backend 日志 `image adapter HIT/MISS` 那行 combo dump 找哪个字段不稳。
    """
    from src.services.runner_models import aggregate_runner_loaded
    entries = [
        e for e in aggregate_runner_loaded(request.app.state)
        if e.get("model_type") == "image"
    ]
    return {"count": len(entries), "entries": entries}


@router.get("/loaded-adapters", dependencies=[Depends(require_admin)])
async def list_loaded_adapters(request: Request):
    """Bug 3 PR-2c:列所有 runner 子进程加载的 combo adapter 实体(image/tts),供引擎库
    「已加载」tab 渲染。

    它们是工作流动态组装的单文件 combo(unet+clip+vae,model_id 是哈希)—— 不对应
    注册卡片,所以不能靠 engines 列表的 status='loaded' 显示。前端在「已加载」tab 把这些
    作为独立实体卡渲染(源文件 basename + GPU + VRAM + pipeline_class)。

    只返 runner 上报的(group_id != 'main');主进程内模型(LLM)本就以注册卡形式出现在
    engines 列表,不在此重复。
    """
    from src.services.runner_models import aggregate_runner_loaded
    import os
    entries = []
    for e in aggregate_runner_loaded(request.app.state):
        if e.get("group_id") == "main":
            continue
        srcs = e.get("source_files") or []
        entries.append({
            "model_id": e.get("model_id"),
            "model_type": e.get("model_type"),
            "group_id": e.get("group_id"),
            "gpu_index": e.get("gpu_index"),
            "vram_mb": e.get("vram_mb"),
            "pipeline_class": e.get("pipeline_class"),
            "source_files": srcs,
            # 人读名:取 diffusion_models(首个源文件)basename;无则退回 model_id。
            "display_name": os.path.basename(srcs[0]) if srcs else e.get("model_id"),
            "last_used_ago_sec": e.get("last_used_ago_sec"),
        })
    return {"count": len(entries), "entries": entries}


@router.post("/unload-image-adapters", dependencies=[Depends(require_admin)])
async def unload_all_image_adapters(request: Request):
    """手动救援:卸载所有 image adapter,释放显存。

    **Bug 3 PR-2a 修正**:image adapter 真加载在 **runner 子进程**,主进程 `_models`
    恒空 —— 老实现遍历主进程字典 = no-op,「释放 image」按钮按了显存掉不下来。现改向
    持有该 adapter 的 runner 派 `UnloadModel`(runner 内 `mm.unload_model` + `empty_cache`,
    显存就在那张卡上),再 reconcile 快照(pipe FIFO:UnloadModel 先于随后的 Ping 被处理,
    Pong 必反映卸载后状态)。

    用户改 dtype/LoRA 多次 Run 后想主动清空(避免下一次 OOM)调这个接口。
    """
    from src.services.runner_models import aggregate_runner_loaded
    state = request.app.state
    sups_by_group = {
        getattr(s, "group_id", None): s
        for s in (getattr(state, "runner_supervisors", None) or [])
    }
    image_entries = [
        e for e in aggregate_runner_loaded(state) if e.get("model_type") == "image"
    ]
    unloaded: list[str] = []
    touched_groups: set[str] = set()
    for e in image_entries:
        gid, mid = e.get("group_id"), e.get("model_id")
        sup = sups_by_group.get(gid)
        if sup is not None and getattr(sup, "client", None) is not None:
            try:
                await sup.client.unload_model(mid)
                unloaded.append(mid)
                touched_groups.add(gid)
            except Exception as ex:  # noqa: BLE001 — 单个失败不挡其余
                logger.warning("unload image adapter %s on runner %s failed: %s", mid, gid, ex)
        elif gid == "main":
            # 主进程内的 image(极少;留兜底路径)。
            await state.model_manager.unload_model(mid, force=True)
            unloaded.append(mid)
    # 卸载后立刻对账,让响应 + UI 反映真实剩余(FIFO 保证 Pong 在 UnloadModel 之后)。
    for gid in touched_groups:
        sup = sups_by_group.get(gid)
        if sup is not None and hasattr(sup, "_reconcile_loaded"):
            await sup._reconcile_loaded()
    invalidate("engines")
    return {"unloaded": unloaded, "count": len(unloaded)}


@router.post("/seedvr2/preload", status_code=202, dependencies=[Depends(require_admin)])
async def preload_seedvr2(request: Request, body: dict = Body(...)):
    """从引擎库预热 SeedVR2 超分(默认配置,无 tiling/blockswap)。name='seedvr2:<filename>' 或 dit 文件名。
    派给 image runner(get_or_load_seedvr2_adapter);loaded 状态经下个 Pong 反映。统一引擎库 PR-3。"""
    import os  # noqa: PLC0415

    from src.config import get_settings  # noqa: PLC0415
    from src.services.inference.image_seedvr2 import DEFAULT_DIT, DEFAULT_VAE  # noqa: PLC0415
    name = str(body.get("name") or "")
    dit = name.split(":", 1)[1] if name.startswith("seedvr2:") else (name or DEFAULT_DIT)
    nas = (get_settings().NAS_MODELS_PATH or "").strip()
    model_dir = os.path.join(nas, "image", "SEEDVR2")
    client = (getattr(request.app.state, "runner_clients", {}) or {}).get("image")
    if client is None or not getattr(client, "_connected", True):
        raise HTTPException(503, "image runner not available")
    await client.preload_seedvr2(model_dir=model_dir, dit_model=dit, vae_model=DEFAULT_VAE)
    invalidate("engines")
    return {"status": "accepted", "dit": dit}


@router.post("/seedvr2/unload", dependencies=[Depends(require_admin)])
async def unload_seedvr2(request: Request, body: dict = Body(...)):
    """卸载已加载的 SeedVR2。name='seedvr2:<filename>' → 匹配 source_files 含该 dit 的 adapter(model_id
    前缀 image:SeedVR2:);name 空 → 卸所有 SeedVR2。向持有它的 runner 派 UnloadModel + 对账快照。"""
    import os  # noqa: PLC0415

    from src.services.runner_models import aggregate_runner_loaded  # noqa: PLC0415
    name = str(body.get("name") or "")
    dit = name.split(":", 1)[1] if name.startswith("seedvr2:") else name
    state = request.app.state
    sups_by_group = {
        getattr(s, "group_id", None): s
        for s in (getattr(state, "runner_supervisors", None) or [])
    }
    unloaded: list[str] = []
    touched: set[str] = set()
    for e in aggregate_runner_loaded(state):
        mid = str(e.get("model_id") or "")
        if not mid.startswith("image:SeedVR2:"):
            continue
        if dit and not any(os.path.basename(str(s)) == dit for s in (e.get("source_files") or [])):
            continue
        gid = e.get("group_id")
        sup = sups_by_group.get(gid)
        if sup is not None and getattr(sup, "client", None) is not None:
            try:
                await sup.client.unload_model(mid)
                unloaded.append(mid)
                touched.add(gid)
            except Exception as ex:  # noqa: BLE001 — 单个失败不挡其余
                logger.warning("unload seedvr2 %s on runner %s failed: %s", mid, gid, ex)
    for gid in touched:
        sup = sups_by_group.get(gid)
        if sup is not None and hasattr(sup, "_reconcile_loaded"):
            await sup._reconcile_loaded()
    invalidate("engines")
    return {"unloaded": unloaded, "count": len(unloaded)}


@router.post("/seedvr2/resident", dependencies=[Depends(require_admin)])
async def set_seedvr2_resident(request: Request, body: dict = Body(...)):
    """切已加载 SeedVR2 的常驻位(引擎库 SeedVR2 卡常驻 toggle,组件 L1 PR-2c)。name='seedvr2:<dit>'
    → 匹配 source_files 含该 dit 的 by-key 模型(id 前缀 image:SeedVR2:);name 空 → 切所有 SeedVR2。
    向持有它的 runner 派 SetModelResident。resident=True → 不被 LRU 自动驱逐。"""
    import os  # noqa: PLC0415

    from src.services.runner_models import aggregate_runner_loaded  # noqa: PLC0415
    name = str(body.get("name") or "")
    dit = name.split(":", 1)[1] if name.startswith("seedvr2:") else name
    resident = bool(body.get("resident", True))
    state = request.app.state
    sups_by_group = {
        getattr(s, "group_id", None): s
        for s in (getattr(state, "runner_supervisors", None) or [])
    }
    touched: list[str] = []
    for e in aggregate_runner_loaded(state):
        mid = str(e.get("model_id") or "")
        if not mid.startswith("image:SeedVR2:"):
            continue
        if dit and not any(os.path.basename(str(s)) == dit for s in (e.get("source_files") or [])):
            continue
        sup = sups_by_group.get(e.get("group_id"))
        if sup is not None and getattr(sup, "client", None) is not None:
            try:
                await sup.client.set_model_resident(mid, resident)
                touched.append(mid)
            except Exception as ex:  # noqa: BLE001 — 单个失败不挡其余
                logger.warning("set seedvr2 resident %s failed: %s", mid, ex)
    invalidate("engines")
    return {"status": "accepted", "model_ids": touched, "resident": resident, "count": len(touched)}


# 引擎库 name 格式:`component:<kind>:<path>`(CopyButton 用同款)。kind ∈ diffusion_models/clip/vae。
def _parse_component_name(name: str) -> tuple[str, str] | None:
    parts = name.split(":", 2)
    if len(parts) == 3 and parts[0] == "component" and parts[1] in {"diffusion_models", "clip", "vae"}:
        return parts[1], parts[2]
    return None


@router.post("/component/preload", status_code=202, dependencies=[Depends(require_admin)])
async def preload_component(request: Request, body: dict = Body(...)):
    """引擎库预加载**单个**组件(clip/vae/diffusion_models)进 image runner 的 L1 池 + 可选常驻。

    body: `name="component:<kind>:<path>"`(引擎库卡片标识)或显式 `{kind,file}`;可选
    `dtype`(默认 bfloat16)、`device`(默认 auto)、`arch`(默认 flux2,单组件 build 反推 repo 用)、
    `resident`(默认 false,true=同时钉常驻不被 LRU 驱逐)。派给 image runner;loaded/resident
    状态经下个 Pong 快照反映。组件 L1 PR-2。"""
    name = str(body.get("name") or "")
    parsed = _parse_component_name(name)
    kind = str(body.get("kind") or (parsed[0] if parsed else ""))
    file = str(body.get("file") or (parsed[1] if parsed else ""))
    if kind not in {"diffusion_models", "clip", "vae"} or not file:
        raise HTTPException(422, "需要 name='component:<kind>:<path>' 或 {kind, file}(kind ∈ diffusion_models/clip/vae)")
    # arch 单独传(clip/vae 的 ComponentSpec 不接受 adapter_arch;仅 diffusion_models 接受)。
    arch = str(body.get("arch") or "flux2")
    spec = {
        "kind": kind,
        "file": file,
        "device": str(body.get("device") or "auto"),
        "dtype": str(body.get("dtype") or "bfloat16"),
    }
    if kind == "diffusion_models":
        spec["adapter_arch"] = arch
    resident = bool(body.get("resident", False))
    client = (getattr(request.app.state, "runner_clients", {}) or {}).get("image")
    if client is None or not getattr(client, "_connected", True):
        raise HTTPException(503, "image runner not available")
    await client.preload_component(spec=spec, resident=resident, arch=arch)
    invalidate("engines")
    return {"status": "accepted", "kind": kind, "file": file, "resident": resident}


@router.post("/component/resident", dependencies=[Depends(require_admin)])
async def set_component_resident(request: Request, body: dict = Body(...)):
    """切**已加载**组件的常驻位(引擎库组件卡常驻 toggle,组件 L1 PR-2b)。

    body: `state_key`(component_state_key,file|device|dtype|loras 串)或 `name=
    'component:<kind>:<path>'`+`device`/`dtype` 自拼;`resident`(true/false)。派 image runner;
    没加载该组件则 no-op。状态经下个 Pong 快照反映。"""
    state_key = str(body.get("state_key") or "")
    if not state_key:
        # 从 name + device/dtype 拼 component_state_key(无 LoRA 的常见情形)
        parsed = _parse_component_name(str(body.get("name") or ""))
        if parsed:
            dev = str(body.get("device") or "auto")
            dtype = str(body.get("dtype") or "bfloat16")
            state_key = f"{parsed[1]}|{dev}|{dtype}|"
    if not state_key:
        raise HTTPException(422, "需要 state_key 或 name='component:<kind>:<path>'(+device/dtype)")
    resident = bool(body.get("resident", True))
    client = (getattr(request.app.state, "runner_clients", {}) or {}).get("image")
    if client is None or not getattr(client, "_connected", True):
        raise HTTPException(503, "image runner not available")
    await client.set_component_resident(state_key=state_key, resident=resident)
    invalidate("engines")
    return {"status": "accepted", "state_key": state_key, "resident": resident}


_install_states: dict[str, dict[str, str]] = {}  # engine -> {status, detail}


@router.get("/deps")
async def list_engine_deps():
    """Return install/probe status for every TTS engine in the manifest."""
    from src.services.tts_deps import list_manifest
    data = list_manifest()
    # overlay any in-flight install state
    for k, v in _install_states.items():
        if k in data:
            data[k]["install_state"] = v
    return data


@router.post("/{name}/install_deps", dependencies=[Depends(require_admin)])
async def install_engine_deps(name: str):
    """Install pip deps for a TTS engine (background). Status pushed via ws."""
    from src.services.tts_deps import get as get_dep
    if get_dep(name) is None:
        raise HTTPException(404, detail=f"No dep manifest for engine: {name}")
    state = _install_states.get(name)
    if state and state.get("status") == "installing":
        return {"name": name, "status": "installing"}
    _install_states[name] = {"status": "installing", "detail": "Starting..."}
    invalidate("engines")
    _spawn_bg(_install_in_background(name))
    return {"name": name, "status": "installing"}


async def _install_in_background(name: str):
    import logging as _lg
    from src.services.tts_deps import install
    log = _lg.getLogger(__name__)
    await ws_manager.broadcast_model_status(name, "installing", "Installing deps...")

    async def _push(line: str):
        # Throttle: only push lines that look meaningful (avoid noise)
        if any(k in line.lower() for k in ("collecting", "downloading", "installing", "successfully", "error")):
            _install_states[name] = {"status": "installing", "detail": line[:200]}
            await ws_manager.broadcast_model_status(name, "installing", line[:200])

    try:
        ok, output = await install(name, on_log=_push)
        if ok:
            _install_states[name] = {"status": "installed", "detail": "Install complete"}
            invalidate("engines")
            await ws_manager.broadcast_model_status(name, "installed", "Install complete")
            log.info("TTS deps installed for %s", name)
        else:
            tail = "\n".join(output.splitlines()[-5:])
            _install_states[name] = {"status": "install_failed", "detail": tail}
            invalidate("engines")
            await ws_manager.broadcast_model_status(name, "install_failed", tail)
            log.error("TTS dep install failed for %s: %s", name, tail)
    except Exception as e:
        _install_states[name] = {"status": "install_failed", "detail": str(e)}
        invalidate("engines")
        await ws_manager.broadcast_model_status(name, "install_failed", str(e))
        log.exception("TTS dep install crashed for %s", name)


@router.patch("/{name}/resident", dependencies=[Depends(require_admin)])
async def set_resident(name: str, request: Request, resident: bool = True,
                       session: AsyncSession = Depends(get_async_session)):
    """Toggle 常驻(随启动预加载 + 不被 TTL/LRU 自动卸)for an engine.

    两个修复:
    1. **持久到 DB 表 model_runtime_overrides**(数据加载统一 2026-06-16;不碰 git 跟踪的
       models.yaml)—— 否则 UI 设的常驻写进 yaml 是未提交本地改动,git checkout/pull/reset 冲掉。
    2. **立即作用到已加载实例**(set_model_resident,no-op if 未加载)—— 否则当前加载着的模型标常驻后
       内存 spec.resident 没变,仍被 TTL 卸,常驻只下次加载才生效(用户报告「标常驻还是被卸」)。
    """
    from src.config import load_model_configs
    from src.services import runtime_override_store

    if name not in load_model_configs():
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    await runtime_override_store.set_override(session, name, "resident", resident)

    mgr = getattr(request.app.state, "model_manager", None)
    if mgr is not None:
        mgr.set_model_resident(name, resident)

    invalidate("engines")
    return {"name": name, "resident": resident}


@router.patch("/{name}/launch-params", dependencies=[Depends(require_admin)])
async def set_launch_params(name: str, body: dict):
    """Edit the per-model `params` block in models.yaml.

    Whitelisted keys only. Changes affect the **next** load — existing running
    instances keep their current launch parameters. Caller must unload + load
    to apply.
    """
    import yaml

    allowed = {
        "enable_prefix_caching",
        "max_num_seqs",
        "max_model_len",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "quantization",
        "dtype",
    }
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        raise HTTPException(400, detail=f"no allowed keys; whitelist: {sorted(allowed)}")

    configs_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "models.yaml"
    with open(configs_path) as f:
        data = yaml.safe_load(f)

    models = data.get("models", [])
    if not isinstance(models, list):
        raise HTTPException(500, detail="models.yaml is not in list-based format")

    found = False
    for entry in models:
        if entry.get("id") != name:
            continue
        params = entry.setdefault("params", {})
        params.update(updates)
        # Remove keys explicitly set to null (lets caller "unset" overrides)
        for k, v in list(params.items()):
            if v is None:
                params.pop(k, None)
        found = True
        break
    if not found:
        raise HTTPException(404, detail=f"unknown engine: {name}")

    with open(configs_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    invalidate("engines")
    return {"name": name, "params": updates, "applied": False, "hint": "unload + load to apply"}


def _card_total_gb_for_engine(cfg: dict, loaded_gpu: int | None = None) -> float:
    """目标卡总显存(GB)。优先级:**真实落卡(已加载)> cfg.gpu 钉卡 > detector 推断** → 回退 24。

    未钉卡的引擎,detector 的 get_device_for_engine 与运行时 GPUAllocator 的实际落卡可能分歧
    (如 embedding 无 gpu 字段:detector 给 3090 24G,实际 vLLM 落 Pro6000 96G)。已加载时直接
    用真实落卡总显存,否则推荐%/absolute-cap 校验会按错卡算 → 误拒合法预算(用户报告诱因)。"""
    from src.gpu.detector import get_device_for_engine, gpu_summary
    devices = gpu_summary().get("devices", [])
    by_index = {d["index"]: d.get("vram_gb") for d in devices}

    if loaded_gpu is not None and loaded_gpu in by_index:
        return float(by_index[loaded_gpu] or 24.0)

    gpu_field = cfg.get("gpu")
    if gpu_field is None:
        try:
            dev = get_device_for_engine(cfg)
            gpu_field = int(dev.split(":")[-1]) if dev.startswith("cuda") else 0
        except (ValueError, IndexError):
            gpu_field = 0
    if isinstance(gpu_field, str) and gpu_field.startswith("cuda"):
        try:
            gpu_field = int(gpu_field.split(":")[-1])
        except (ValueError, IndexError):
            gpu_field = 0
    return float(by_index.get(gpu_field) or 24.0)


_VLLM_ADAPTER = "src.services.inference.llm_vllm.VLLMAdapter"


@router.get("/{name}/vram-budget", dependencies=[Depends(require_admin)])
async def get_vram_budget(name: str, request: Request):
    """返回该引擎当前显存预算设置 + 推荐值 + 目标卡总显存(spec 2026-06-13)。
    仅 vLLM 类(llm/embedding/vl)有意义;image/tts(非 vLLM)返回 applicable=false。"""
    from src.config import load_model_configs, load_runtime_overrides, recommend_vram_budget_gb

    cfgs = load_model_configs()
    cfg = cfgs.get(name)
    if cfg is None:
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    applicable = cfg.get("adapter") == _VLLM_ADAPTER
    card_total_gb = _card_total_gb_for_engine(cfg, _get_loaded_gpu(name, request))
    weights_gb = float(cfg.get("vram_gb") or 0)  # load_model_configs 已把 vram_mb 转成 vram_gb
    rec_gb = recommend_vram_budget_gb(cfg.get("type", "llm"), weights_gb)
    rec_percent = round(min(0.98, rec_gb / card_total_gb), 3) if card_total_gb > 0 else None

    current = (load_runtime_overrides().get(name) or {}).get("vram_budget") or {"mode": "auto"}
    return {
        "name": name,
        "applicable": applicable,
        "current": current,
        "recommended_gb": rec_gb,
        "recommended_percent": rec_percent,
        "card_total_gb": round(card_total_gb, 1),
        "yaml_gpu_memory_utilization": (cfg.get("params") or {}).get("gpu_memory_utilization"),
    }


@router.patch("/{name}/vram-budget", dependencies=[Depends(require_admin)])
async def set_vram_budget(name: str, request: Request, body: dict = Body(...),
                          session: AsyncSession = Depends(get_async_session)):
    """写每模型显存预算到 DB(model_runtime_overrides;数据加载统一 2026-06-16)。
    body: {mode, value}。mode=auto 清除走 adapter 公式;percent(0–1)/absolute(GB)。需重载生效。"""
    from src.config import load_model_configs
    from src.services import runtime_override_store

    cfg = load_model_configs().get(name)
    if cfg is None:
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    mode = str(body.get("mode") or "auto").lower()
    if mode not in ("auto", "percent", "absolute"):
        raise HTTPException(400, detail="mode must be auto|percent|absolute")

    if mode == "auto":
        await runtime_override_store.set_override(session, name, "vram_budget", {"mode": "auto"})
        invalidate("engines")
        return {"name": name, "vram_budget": {"mode": "auto"}, "applied": False,
                "hint": "需重新加载模型生效(unload + load)"}

    val = body.get("value")
    if not isinstance(val, (int, float)) or val <= 0:
        raise HTTPException(400, detail="value must be a positive number")
    if mode == "percent" and not (0 < val <= 0.98):
        raise HTTPException(400, detail="percent value must be in (0, 0.98]")
    if mode == "absolute":
        card_total_gb = _card_total_gb_for_engine(cfg, _get_loaded_gpu(name, request))
        if val > card_total_gb:
            raise HTTPException(
                400, detail=f"absolute {val}GB exceeds card total {card_total_gb:.1f}GB")

    budget = {"mode": mode, "value": float(val)}
    await runtime_override_store.set_override(session, name, "vram_budget", budget)
    invalidate("engines")
    return {"name": name, "vram_budget": budget, "applied": False,
            "hint": "需重新加载模型生效(unload + load)"}


@router.patch("/{name}/gpu", dependencies=[Depends(require_admin)])
async def set_gpu(name: str, request: Request, gpu: int = 0,
                  session: AsyncSession = Depends(get_async_session)):
    """Change GPU assignment for an engine.

    写 DB 表 model_runtime_overrides(与 resident/vram_budget 一致;数据加载统一 2026-06-16)
    —— 不再写 git 跟踪的 models.yaml(旧行为污染 git 树、且 registry 读 yaml/overlay 口径分裂
    导致落卡设置不生效)。registry 套用覆盖 → spec.gpu 驱动 vLLM 落卡。写后 reload registry 让
    spec.gpu 立即刷新,随后 unload + load 即落新卡。
    """
    from src.config import load_model_configs
    from src.services import runtime_override_store

    if name not in load_model_configs():
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    await runtime_override_store.set_override(session, name, "gpu", gpu)
    # reload registry → spec.gpu 从覆盖刷新(否则停在旧值,unload+load 仍落旧卡)。
    mgr = getattr(request.app.state, "model_manager", None)
    if mgr is not None and getattr(mgr, "_registry", None) is not None:
        mgr._registry.reload()
    invalidate("engines")
    return {"name": name, "gpu": gpu, "applied": False,
            "hint": "需重新加载模型生效(unload + load)"}


@router.get("/scheduler/status")
async def scheduler_status(request: Request):
    """Return current model manager status."""
    model_mgr = request.app.state.model_manager
    return model_mgr.get_status()
