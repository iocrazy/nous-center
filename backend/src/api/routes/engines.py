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
    """Hot-reload models.yaml without restarting. Picks up new model configs."""
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
    asyncio.create_task(_load_in_background(name, model_mgr))

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
    asyncio.create_task(_install_in_background(name))
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
async def set_resident(name: str, resident: bool = True):
    """Toggle auto-load on startup for an engine."""
    import yaml

    configs_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "models.yaml"

    with open(configs_path) as f:
        data = yaml.safe_load(f)

    # Support both old dict format and new list format
    models = data.get("models", [])
    if isinstance(models, list):
        found = False
        for entry in models:
            if entry.get("id") == name:
                entry["resident"] = resident
                found = True
                break
        if not found:
            raise HTTPException(404, detail=f"Unknown engine: {name}")
    else:
        if name not in models:
            raise HTTPException(404, detail=f"Unknown engine: {name}")
        models[name]["resident"] = resident

    with open(configs_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

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


@router.patch("/{name}/gpu", dependencies=[Depends(require_admin)])
async def set_gpu(name: str, gpu: int = 0):
    """Change GPU assignment for an engine."""
    import yaml

    configs_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "models.yaml"

    with open(configs_path) as f:
        data = yaml.safe_load(f)

    # Support both old dict format and new list format
    models = data.get("models", [])
    if isinstance(models, list):
        found = False
        for entry in models:
            if entry.get("id") == name:
                entry["gpu"] = gpu
                found = True
                break
        if not found:
            raise HTTPException(404, detail=f"Unknown engine: {name}")
    else:
        if name not in models:
            raise HTTPException(404, detail=f"Unknown engine: {name}")
        models[name]["gpu"] = gpu

    with open(configs_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    invalidate("engines")
    return {"name": name, "gpu": gpu}


@router.get("/scheduler/status")
async def scheduler_status(request: Request):
    """Return current model manager status."""
    model_mgr = request.app.state.model_manager
    return model_mgr.get_status()
