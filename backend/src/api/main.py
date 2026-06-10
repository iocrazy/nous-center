import asyncio
import logging
import os

# torch 默认 CUDA_DEVICE_ORDER=FASTEST_FIRST,把最快的卡(Pro 6000)排到 cuda:0,
# 跟 nvidia-smi(PCI 顺序)+ hardware.yaml(按 nvidia-smi 写)错位 ——
# ModelManager.get_best_gpu() 用 nvidia-smi poll 取 PCI index,喂给 torch
# 当 cuda:N 就装错卡(实测 flux2 想去 Pro 6000 → 装到 3090)。
# setdefault 在 import torch 之前固定 PCI_BUS_ID,让三个索引系统一致;
# 用户 .env 同名变量优先(setdefault 不覆盖)。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import understand, generate, tts, engines, audio, voices, openai_compat, ollama_compat, api_gateway as api_gateway_routes, settings, instances, workflows, agents, skills, monitor, node_packages, execution_tasks, apps, logs, context_cache as context_cache_routes, files as files_routes, services as services_routes, workflow_publish as workflow_publish_routes, usage as usage_routes, dashboard as dashboard_routes, api_keys as api_keys_routes, anthropic_compat, observability, loras as loras_routes, image_files as image_files_routes, models as models_routes, predictions as predictions_routes
from src.api.websocket import ws_manager
from src.api.ws_tts import handle_tts_websocket
from src.services.gpu_monitor import memory_guard_loop

logger = logging.getLogger(__name__)

# Active WebSocket connections for workflow progress updates, keyed by instance_id
_ws_connections: dict[str, list[WebSocket]] = {}


def _make_component_event_handler(registry, ws):
    """Build the sync callback RunnerClient.on_component_event uses: update the
    backend mirror + fan out a WS push. WS broadcast is async → scheduled."""
    def _handler(evt) -> None:
        # round6:registry.update 早先在 try 外,抛异常会逃进 RunnerClient demux loop 杀掉它
        # (后续 run_node 全挂 5min)。client 侧已加回调守卫,这里再各自 try 兜底纵深。
        try:
            registry.update(evt.component_key, evt.state, evt.error)
        except Exception:  # noqa: BLE001
            logger.exception("component registry.update failed (%s)", evt.component_key)
        try:
            asyncio.get_running_loop().create_task(
                ws.broadcast_component_state(evt.component_key, evt.state, evt.error))
        except RuntimeError:
            pass  # no running loop — registry still updated
    return _handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup."""
    # Ensure localhost requests bypass proxy
    import os
    no_proxy = os.environ.get("NO_PROXY", "")
    if "localhost" not in no_proxy:
        os.environ["NO_PROXY"] = f"{no_proxy},localhost,127.0.0.1" if no_proxy else "localhost,127.0.0.1"
    from src.models.database import Base, create_engine
    import src.models.voice_preset  # noqa: F401
    import src.models.tts_usage  # noqa: F401
    import src.models.service_instance  # noqa: F401
    import src.models.instance_api_key  # noqa: F401
    import src.models.model_metadata  # noqa: F401
    import src.models.workflow  # noqa: F401
    import src.models.execution_task  # noqa: F401
    import src.models.llm_usage  # noqa: F401
    import src.models.context_cache  # noqa: F401
    import src.models.response_session  # noqa: F401
    import src.models.memory  # noqa: F401
    import src.models.api_gateway  # noqa: F401
    import src.models.admin_credentials  # noqa: F401
    import src.models.log_entry  # noqa: F401  # structured logs live in main DB now

    # Retry DB connect: docker postgres 容器可能 backend 启动时还在 healthcheck 阶段
    # (race condition seen 2026-05-07). Backoff: 2s, 4s, 8s, 16s, 32s, 60s = 122s 总等
    # 超时再死。SQLite URL 永远 1 次成功,这个循环退化为零成本。
    import asyncio as _asyncio
    last_err = None
    for attempt in range(6):
        # round4 #2:engine 用 try/finally dispose —— begin() 在 postgres 还没起来时会抛,
        # 早先 dispose 在 success 之后,失败路径跳过它 → 每次重试泄漏一个连接池。
        engine = create_engine()
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                # 无 alembic,create_all 不给已存在表加列。新列用幂等 ALTER ADD COLUMN IF NOT EXISTS
                # 补(Postgres 支持)。开发期微迁移,单条失败不阻断启动。服务层 API spec PR-2:input_json。
                from sqlalchemy import text  # noqa: PLC0415
                for _ddl in (
                    "ALTER TABLE execution_tasks ADD COLUMN IF NOT EXISTS input_json JSONB",
                    "ALTER TABLE execution_tasks ADD COLUMN IF NOT EXISTS webhook_url VARCHAR(500)",
                    "ALTER TABLE execution_tasks ADD COLUMN IF NOT EXISTS webhook_events JSONB",
                    # PR-5b:files 作用域 instance_id → api_key_id。加列 + 旧列降 nullable(孤儿)+ 新键/索引。
                    "ALTER TABLE files ADD COLUMN IF NOT EXISTS api_key_id BIGINT",
                    "ALTER TABLE files ALTER COLUMN instance_id DROP NOT NULL",
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_files_apikey_sha256 ON files (api_key_id, sha256)",
                    "CREATE INDEX IF NOT EXISTS ix_files_apikey_created ON files (api_key_id, created_at)",
                    # legacy rip:memory_entries 作用域 instance_id → api_key_id。降 instance_id nullable
                    # (M:N 无单一 instance)+ 建 api_key 索引(旧 idx_mem_inst_* create_all 已建,不删)。
                    "ALTER TABLE memory_entries ALTER COLUMN instance_id DROP NOT NULL",
                    "CREATE INDEX IF NOT EXISTS idx_mem_key_created ON memory_entries (api_key_id, created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_mem_key_ctx_cat ON memory_entries (api_key_id, context_key, category)",
                    # 节点分组(ComfyUI 式可视框,不入执行图)。旧表补列,默认空 []。
                    "ALTER TABLE workflows ADD COLUMN IF NOT EXISTS groups JSONB DEFAULT '[]'::jsonb",
                ):
                    try:
                        await conn.execute(text(_ddl))
                    except Exception as _e:  # noqa: BLE001 — 微迁移 best-effort
                        logger.warning("micro-migration skipped (%s): %s", _ddl, _e)
            logger.info("Database tables ensured%s", f" (after {attempt} retries)" if attempt else "")
            break
        except Exception as e:
            last_err = e
            wait_s = min(2 ** (attempt + 1), 60)
            logger.warning(
                "DB connect attempt %d failed: %s — retrying in %ds",
                attempt + 1, type(e).__name__, wait_s,
            )
            await _asyncio.sleep(wait_s)
        finally:
            await engine.dispose()
    else:
        logger.error("DB connect failed after 6 retries; last error: %s", last_err)
        raise last_err

    # Start the structured-log writer: async queue + single batch-insert consumer
    # into the main PG DB (spec 2026-06-10 — one DB, no separate SQLite log_db).
    from src.services.log_store import log_writer
    log_writer.start()
    logger.info("Log writer started (PG-backed)")

    # Install DB log handler for application logs
    from src.services.log_collector import DbLogHandler
    import logging as _logging
    db_handler = DbLogHandler()
    db_handler.setLevel(_logging.INFO)
    # Surface application INFO logs to stdout too — otherwise operators tailing
    # uvicorn only see the access log, and slow image-load helpers (which print
    # "image: dequant fp8→bf16 done ..." progress markers) look like a black
    # hole from the terminal. Without these handlers the lines went only to
    # the DB log handler, which is queryable but invisible to humans.
    stream_handler = _logging.StreamHandler()
    stream_handler.setLevel(_logging.INFO)
    stream_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)-5s %(name)s — %(message)s")
    )
    for _logger_name in ("src", "nous"):
        _l = _logging.getLogger(_logger_name)
        _l.setLevel(_logging.INFO)
        _l.addHandler(db_handler)
        _l.addHandler(stream_handler)
    logger.info("Application log collector installed (db + stdout)")

    # Auto-sync model metadata for any new engines
    from src.models.database import get_session_factory
    from src.services.model_metadata_service import sync_metadata

    # round4 #1/#2:共享 memoized 工厂(别每处新建 engine);round4 #2(config#1):sf 在
    # try 外取 —— get_session_factory() 只返回工厂、不碰 DB,放 try 外才不会在 sync_metadata
    # 失败被吞后让下面 152 行的 `async with sf()` 撞 NameError(早先 sf 在 try 内)。
    sf = get_session_factory()

    # Wave 1 MemoryProvider: init PGMemoryProvider + expose via app.state
    from src.services.memory.pg_provider import PGMemoryProvider
    try:
        app.state.memory_provider = PGMemoryProvider(session_factory=sf)
        await app.state.memory_provider.initialize()
        logger.info("MemoryProvider initialized (pg)")
    except Exception as e:
        logger.warning("MemoryProvider init failed (non-fatal): %s", e)

    try:
        async with sf() as session:
            await sync_metadata(session)
        logger.info("Model metadata synced")
    except Exception as e:
        logger.warning("Model metadata sync failed (non-fatal): %s", e)

    # Auto-migrate voice presets to workflow templates
    from src.models.voice_preset import VoicePreset
    from src.models.workflow import Workflow as WfModel
    from sqlalchemy import select, func as sa_func

    async with sf() as session:
        wf_count = await session.scalar(
            select(sa_func.count()).select_from(WfModel).where(WfModel.is_template == True)  # noqa: E712
        )
        if wf_count == 0:
            result = await session.execute(select(VoicePreset))
            presets = result.scalars().all()
            for preset in presets:
                wf = WfModel(
                    name=preset.name,
                    description=f"从预设 '{preset.name}' 自动迁移",
                    is_template=True,
                    nodes=[
                        {"id": "n1", "type": "text_input", "data": {"text": ""}, "position": {"x": 0, "y": 0}},
                        {"id": "n2", "type": "tts_engine", "data": {
                            "engine": preset.engine,
                            **(preset.params or {}),
                        }, "position": {"x": 350, "y": 0}},
                        {"id": "n3", "type": "output", "data": {}, "position": {"x": 700, "y": 0}},
                    ],
                    edges=[
                        {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
                        {"id": "e2", "source": "n2", "sourceHandle": "audio", "target": "n3", "targetHandle": "audio"},
                    ],
                )
                session.add(wf)
            await session.commit()
            if presets:
                logger.info("Migrated %d voice presets to workflow templates", len(presets))

    # Scan node packages
    from nodes import scan_packages
    scan_packages()
    logger.info("Node packages scanned")

    # Create ModelManager
    from src.services.inference.registry import ModelRegistry
    from src.services.gpu_allocator import GPUAllocator
    from src.services.model_manager import ModelManager

    config_path = str(Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml")
    registry = ModelRegistry(config_path)
    allocator = GPUAllocator()
    model_mgr = ModelManager(registry=registry, allocator=allocator)
    app.state.model_manager = model_mgr

    # 启动扫描 + 自检:暖组件下拉索引(loader 节点用)+ 每角色计数 + 整模型完整性。
    # Fail-soft — 扫描/自检出错不阻塞启动,降级到空索引。
    try:
        from src.services.component_scanner import get_component_index, selfcheck_report
        report = selfcheck_report(force_refresh=True)  # 扫一遍 + 填缓存
        app.state.component_index = get_component_index()
        _roles = ", ".join(f"{r}={n}" for r, n in report["counts"].items())
        logger.info("模型扫描自检:%s", _roles)
        for _w in report["warnings"]:
            logger.warning("模型扫描自检:%s", _w)
    except Exception:  # noqa: BLE001 — index is non-critical at boot
        logger.exception("模型扫描自检失败;serving empty index")
        app.state.component_index = {role: [] for role in ("diffusion_models", "clip", "vae", "loras", "checkpoint")}

    # Wire ModelManager into workflow executor
    from src.services.workflow_executor import set_model_manager
    set_model_manager(model_mgr)

    # ------------------------------------------------------------------
    # Lane K: lifespan wiring —— spawn RunnerSupervisor / LLMRunner per
    # hardware.yaml group + expose via app.state for /health, /runners,
    # and workflow dispatch (spec §4.1 / §4.2).
    #
    # V1.5 lanes (#95–#106) shipped RunnerSupervisor / LLMRunner /
    # RunnerClient as standalone classes but nobody instantiated them in
    # lifespan, so app.state.runner_supervisors was unset, /runners
    # returned [], and dispatch nodes hit "runner_client is None".
    #
    # Gate: NOUS_DISABLE_RUNNER_SPAWN=1 (or unset by default in tests via
    # conftest.NOUS_DISABLE_BG_TASKS) skips the spawn block entirely so the
    # existing test suite — which would otherwise multiprocessing.spawn real
    # subprocesses + try to load real models — stays fast. Production
    # systemd unit sets NOUS_DISABLE_RUNNER_SPAWN=0 explicitly.
    # ------------------------------------------------------------------
    runner_supervisors: list = []
    runner_clients: dict[str, object] = {}
    llm_runner = None
    # Spawn runners only when explicitly enabled. Production systemd unit sets
    # NOUS_DISABLE_RUNNER_SPAWN=0 ; tests / dev default = skip (= "1") so the
    # existing fast pytest suite is unaffected, and conftest.NOUS_DISABLE_BG_TASKS
    # cannot accidentally trigger real multiprocessing.spawn of runner subprocesses.
    import os as _lane_k_os
    _runner_spawn_enabled = _lane_k_os.getenv("NOUS_DISABLE_RUNNER_SPAWN", "1") == "0"
    if _runner_spawn_enabled:
        from src.runner.supervisor import RunnerSupervisor
        from src.runner.llm_runner import LLMRunner
        from src.runner.gpu_free_probe import make_gpu_free_probe

        gpu_probe = make_gpu_free_probe()
        models_yaml_path = config_path  # 同一份 models.yaml,runner 子进程也用它

        for group in allocator.groups():
            if group.role == "llm":
                # LLMRunner —— per spec §4.1 每个 role:llm group 一个。
                # 不在 lifespan 里 spawn vLLM —— vLLM 的实际启动走现有
                # preload_residents 路径（下面 _resident_preload_task）。
                # LLMRunner 这里只持有「将来要管的 adapter 引用」+ GPU 列表,
                # 让 crash-recovery / health probe / restart 接口可用。
                #
                # 选取该 group 的代表 model_key:首个 type=llm 的 spec。无 spec
                # → 仍构造一个空壳（adapter=None,model_key=空）—— 测试 / 早期
                # 部署不带 yaml 时也要让 /health 等读 app.state.llm_runner 不崩。
                llm_specs = [s for s in registry.specs if s.model_type == "llm"]
                rep_model_key = llm_specs[0].id if llm_specs else f"llm-{group.id}"
                rep_adapter = (
                    model_mgr.get_adapter(rep_model_key) if llm_specs else None
                )
                llm_runner = LLMRunner(
                    model_key=rep_model_key,
                    adapter=rep_adapter,
                    llm_gpus=list(group.gpus),
                    gpu_free_probe=gpu_probe,
                )
                logger.info(
                    # rep_model_key 是 llm group 的「代表标识」(取 llm_specs[0].id),
                    # **不是启动加载目标** —— 它 status 一直 unloaded。实际加载由
                    # published 工作流依赖(_load_wf_deps)/ resident preload / 手动决定。
                    # 旧文案打 `model_key=%s` 易被误读成「启动加载了这个模型」(排查
                    # startup 自动加载时踩过坑,见 memory project_startup_model_load_paths)。
                    "Lane K: LLMRunner instantiated (group=%s, gpus=%s, "
                    "rep_model_key=%s [group 代表标识,非启动加载目标], adapter_present=%s)",
                    group.id, group.gpus, rep_model_key, rep_adapter is not None,
                )
            else:
                # image / tts group → fork runner 子进程 + 建 client。
                sup = RunnerSupervisor(
                    group_id=group.id,
                    gpus=list(group.gpus),
                    models_yaml_path=models_yaml_path,
                    fake_adapter=False,
                    gpu_free_probe=gpu_probe,
                )
                try:
                    await sup.start()
                except Exception:
                    logger.exception(
                        "Lane K: failed to start RunnerSupervisor for group %s — "
                        "continuing without it (fail-soft, /health will report degraded)",
                        group.id,
                    )
                    continue
                runner_supervisors.append(sup)
                # sup.client 是 supervisor._spawn 建好的 RunnerClient —— 复用即可,
                # 不再自己新建一个（每对 pipe 只能有一个 reader）。
                if sup.client is not None:
                    runner_clients[group.id] = sup.client
                logger.info(
                    "Lane K: RunnerSupervisor spawned (group=%s, gpus=%s, pid=%s)",
                    group.id, group.gpus, sup.pid,
                )

    app.state.runner_supervisors = runner_supervisors
    app.state.runner_clients = runner_clients
    app.state.llm_runner = llm_runner

    # PR-5a: component-state mirror fed by the image runner's ComponentEvents.
    from src.services.component_state import ComponentStateRegistry
    app.state.component_state_registry = ComponentStateRegistry()
    _img_client = runner_clients.get("image")
    if _img_client is not None:
        _img_client.on_component_event = _make_component_event_handler(
            app.state.component_state_registry, ws_manager)

    # Auto-detect running vLLM instances BEFORE resident auto-load
    # (so we reconnect to orphans instead of spawning duplicates)
    import os as _os
    import signal as _signal
    from src.services.inference.vllm_scanner import scan_running_vllm
    running_vllm = scan_running_vllm()
    if running_vllm:
        logger.info("Found %d running vLLM process(es)", len(running_vllm))
    reconnected: set[str] = set()
    for vllm_info in running_vllm:
        # Kill unhealthy orphans immediately
        if not vllm_info["healthy"]:
            logger.warning(
                "Killing unhealthy orphan vLLM (pid=%d, port=%d, model=%s)",
                vllm_info["pid"], vllm_info["port"], vllm_info["model_path"],
            )
            try:
                pid = vllm_info["pid"]
                # Try process group kill first (catches worker subprocesses)
                try:
                    pgid = _os.getpgid(pid)
                    _os.killpg(pgid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    _os.kill(pid, _signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning("Failed to kill orphan pid=%d: %s", vllm_info["pid"], e)
            continue

        # Reconnect healthy ones
        for spec in registry.specs:
            spec_main_path = spec.paths.get("main", "")
            if spec.model_type != "llm" or not spec_main_path:
                continue
            if vllm_info["model_path"].rstrip("/").endswith(spec_main_path.rstrip("/")):
                logger.info(
                    "Reconnecting to running vLLM for %s (pid=%s, port=%s)",
                    spec.id, vllm_info["pid"], vllm_info["port"],
                )
                try:
                    def _factory(s, port=vllm_info["port"], pid=vllm_info["pid"]):
                        from src.services.inference.llm_vllm import VLLMAdapter
                        return VLLMAdapter(paths=s.paths, vllm_port=port, adopt_pid=pid, **s.params)
                    await model_mgr.load_model(spec.id, adapter_factory=_factory)
                    reconnected.add(spec.id)
                except Exception as e:
                    logger.warning("Failed to reconnect %s: %s", spec.id, e)
                break

    # Auto-load disabled — models are loaded manually from the UI
    # Resident flag only prevents auto-unload by idle checker

    # Re-register model references for published workflows
    from sqlalchemy import select
    from src.models.workflow import Workflow
    wf_model_deps: list[dict] = []
    async with sf() as session:
        stmt = select(Workflow).where(Workflow.status == "published")
        result = await session.execute(stmt)
        for wf in result.scalars():
            deps = model_mgr.get_model_dependencies({"nodes": wf.nodes, "edges": wf.edges})
            for dep in deps:
                model_mgr.add_reference(dep["key"], str(wf.id))
                wf_model_deps.append({"key": dep["key"], "wf_id": wf.id})

    # NOUS_DISABLE_BG_TASKS=1 → skip all background tasks.
    # CRITICAL for tests: the default background set includes memory_guard_loop
    # which polls `nvidia-smi` via subprocess every 5s. When multiple test
    # processes simultaneously trigger lifespan (via TestClient), concurrent
    # nvidia-smi invocations contend with gnome-shell's GPU compositor, which
    # can crash the NVIDIA driver and log out the X session. conftest.py sets
    # this env var at the top so `uv run pytest` never spawns these loops.
    import os as _os
    _bg_tasks_disabled = _os.getenv("NOUS_DISABLE_BG_TASKS") == "1"

    cache_cleanup_task = None
    response_cleanup_task = None
    partial_worker = None
    # round4 #8/#9:常驻后台 loop 早先用裸 asyncio.create_task,既不持引用(Py3.11+
    # event loop 只持弱引用 → _load_wf_deps 这种起手无 sleep 护栏的可能被 GC 丢弃),
    # shutdown 也不 cancel(留半完成 subprocess)。收进 list 持引用 + finally 统一 cancel。
    bg_tasks: list = []

    if not _bg_tasks_disabled:
        # Load workflow dependencies in background (non-blocking)
        async def _load_wf_deps():
            for dep in wf_model_deps:
                try:
                    await model_mgr.load_model(dep["key"])
                except Exception as e:
                    logger.warning("Failed to load model %s for workflow %s: %s", dep["key"], dep["wf_id"], e)

        if wf_model_deps:
            bg_tasks.append(asyncio.create_task(_load_wf_deps()))

        # Resident models marked resident: preload in the background, ordered
        # by preload_order ascending (spec 4.2). The ~120s diffusers compose
        # must not block /health (cloudflared / systemd probes would mark the
        # backend down). preload_residents is fail-soft: a single model's
        # OOM / corrupt-weights failure records into mm._load_failures and is
        # surfaced on /health — it never blocks startup or the rest of the
        # preload sequence (spec 4.3). on_loaded flips the engines/models
        # cache + UI badge within ~1s per successful load.
        async def _on_resident_loaded(spec_id: str) -> None:
            from src.api.response_cache import invalidate as _invalidate
            _invalidate("models", "engines")
            from src.api.websocket import ws_manager as _ws
            await _ws.broadcast_model_status(spec_id, "loaded")

        # Persist the task ref so 3.11+ doesn't garbage-collect a still-running
        # background coroutine and silently drop the preload.
        app.state._resident_preload_task = asyncio.create_task(
            model_mgr.preload_residents(on_loaded=_on_resident_loaded)
        )

        # Start idle model checker background task
        async def idle_checker():
            while True:
                await asyncio.sleep(60)
                try:
                    await model_mgr.check_idle_models()
                except Exception as e:
                    logger.warning("Idle model check failed: %s", e)

        bg_tasks.append(asyncio.create_task(idle_checker()))
        bg_tasks.append(asyncio.create_task(memory_guard_loop(model_mgr, reserved_gb=4.0)))

        async def log_cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # Every hour
                try:
                    from src.services.log_store import cleanup_logs
                    from src.models.database import get_session_factory
                    # Now async on the main DB; no to_thread needed (it awaits I/O,
                    # doesn't block the loop). One short-lived session per sweep.
                    async with get_session_factory()() as session:
                        await cleanup_logs(session)
                except Exception as e:
                    logger.warning("Log cleanup failed: %s", e)

        bg_tasks.append(asyncio.create_task(log_cleanup_loop()))

        # Image output orphan reaper. PR-6's signed-URL TTL is 1h by
        # default; once a URL expires the file is unreachable but stays
        # on disk. Walk every 6h and delete files older than 24h
        # (4× the URL TTL leaves enough room for a caller who fetched
        # the URL near expiry to still pull the bytes once).
        async def image_orphan_reap_loop(interval_seconds: int = 6 * 3600):
            from src.api.routes.execution_tasks import collect_referenced_image_uuids
            from src.models.database import get_session_factory as _isf
            from src.services.image_output_storage import reap_orphans
            sf = _isf()
            while True:
                try:
                    # 图寿命=任务寿命(spec 2026-06-09 run-history):先查仍被 ExecutionTask
                    # 引用的图 uuid,只清没人引用的真 orphan(失败/已删任务残留)→ /history
                    # 画廊历史图不被误删。round4 #6:reap 同步全盘遍历,丢 to_thread 不卡 loop。
                    async with sf() as session:
                        keep = await collect_referenced_image_uuids(session)
                    await asyncio.to_thread(
                        reap_orphans, older_than_seconds=24 * 3600, keep_uuids=keep,
                    )
                except Exception:
                    logger.exception("image orphan reap error")
                try:
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break

        bg_tasks.append(asyncio.create_task(image_orphan_reap_loop()))

        async def context_cache_cleanup_loop(interval_seconds: int = 3600):
            from src.services.context_cache_service import cleanup_expired
            from src.models.database import get_session_factory as _csf
            sf = _csf()
            while True:
                try:
                    async with sf() as s:
                        n = await cleanup_expired(s)
                        if n:
                            logger.info("context cache cleanup: %d expired rows", n)
                except Exception:
                    logger.exception("context cache cleanup error")
                try:
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break

        cache_cleanup_task = asyncio.create_task(context_cache_cleanup_loop())

        # Step 4: expired-session cleanup + partial-write background worker
        async def response_cleanup_loop(interval_seconds: int = 3600):
            from src.services.responses_service import cleanup_expired_sessions
            from src.models.database import get_session_factory as _csf
            sf = _csf()
            while True:
                try:
                    async with sf() as s:
                        n = await cleanup_expired_sessions(s)
                        if n:
                            logger.info("response cleanup: %d expired sessions", n)
                except Exception:
                    logger.exception("response cleanup error")
                try:
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break

        response_cleanup_task = asyncio.create_task(response_cleanup_loop())

        from src.api.routes import responses as responses_routes
        responses_routes._set_queue(asyncio.Queue(maxsize=1000))
        partial_worker = asyncio.create_task(responses_routes.partial_write_worker())

    try:
        yield
    finally:
        # Gracefully shut down background tasks (only if they were started)
        if cache_cleanup_task is not None:
            cache_cleanup_task.cancel()
        if response_cleanup_task is not None:
            response_cleanup_task.cancel()
        for t in (cache_cleanup_task, response_cleanup_task):
            if t is None:
                continue
            try:
                await t
            except asyncio.CancelledError:
                pass
        # round4 #9:cancel + await 常驻后台 loop(idle/memory_guard/log_cleanup/
        # orphan_reap/_load_wf_deps),否则它们随 loop 关闭被硬杀,可能在
        # check_idle_models / nvidia-smi poll 中途留半完成状态。
        for t in bg_tasks:
            t.cancel()
        for t in bg_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # Flush + stop the structured-log writer (drains queued log rows).
        try:
            from src.services.log_store import log_writer
            await log_writer.stop()
        except Exception:  # noqa: BLE001
            pass
        # Drain partial-write worker
        if partial_worker is not None:
            from src.api.routes import responses as responses_routes
            if responses_routes._partial_write_queue is not None:
                await responses_routes._partial_write_queue.put(None)
                try:
                    await asyncio.wait_for(partial_worker, timeout=5.0)
                except asyncio.TimeoutError:
                    partial_worker.cancel()

        # Lane K: stop runner supervisors + LLMRunner (terminate child subprocs).
        # getattr 兜底:NOUS_DISABLE_RUNNER_SPAWN 默认 skip 时这些属性不一定有.
        for sup in getattr(app.state, "runner_supervisors", []) or []:
            try:
                await sup.stop()
            except Exception:
                logger.exception("Lane K: RunnerSupervisor stop failed (group=%s)",
                                 getattr(sup, "group_id", "?"))
        _llm = getattr(app.state, "llm_runner", None)
        if _llm is not None:
            try:
                await _llm.shutdown()
            except Exception:
                logger.exception("Lane K: LLMRunner shutdown failed")

        # vLLM 子进程是 model_manager 直接 spawn 的(VLLMAdapter._process / EngineCore),
        # **不归** runner supervisors / llm_runner 管 —— 上面 stop 完它们,vLLM 仍活着。
        # 不在此 force-unload,backend 退出后 vLLM 成 orphan 继续占显存:反复重启时每个
        # ~40G 累积,把常驻 LLM 的卡占爆 → image 出图 OOM(真机实锤:Pro6000 被 2 个
        # orphan vLLM + 当前 vLLM 占满,只剩 12G < 需 22.8G)。force-unload 走
        # adapter.unload() → killpg 整个进程组,连 EngineCore worker 一起收。
        # in-use 守卫仍强于 force(正在 infer 的不卸,避免 segfault)。
        _mm = getattr(app.state, "model_manager", None)
        if _mm is not None:
            for mid in list(getattr(_mm, "loaded_model_ids", []) or []):
                try:
                    await _mm.unload_model(mid, force=True)
                except Exception:
                    logger.exception(
                        "shutdown: force-unload %s failed (vLLM 可能 orphan)", mid)


def create_app() -> FastAPI:
    app = FastAPI(title="Nous Center", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # ETag is exposed so JS clients can read it for explicit If-None-Match
        # validation. The browser's native HTTP cache uses ETag transparently
        # regardless, but custom diagnostic / instrumentation code needs CORS
        # to expose the header.
        expose_headers=["X-Request-Id", "ETag"],
    )

    from src.api.middleware import (
        RequestLoggingMiddleware,
        AuditMiddleware,
        RequestIdMiddleware,
        AdminSessionGateMiddleware,
    )
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    # Admin session gate sits before logging so 401s still get a request id but
    # don't reach business handlers.
    app.add_middleware(AdminSessionGateMiddleware)
    # LIFO: RequestIdMiddleware added LAST so it runs FIRST, populating
    # request.state.request_id before exception handlers inspect it.
    app.add_middleware(RequestIdMiddleware)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/health")
    async def health_check():
        checks: dict = {"status": "ok"}

        # Check database
        try:
            from src.models.database import get_session_factory
            from sqlalchemy import text
            _sf = get_session_factory()
            async with _sf() as session:
                await session.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception:
            checks["database"] = "error"
            checks["status"] = "degraded"

        # GPU availability
        from src.services.gpu_monitor import get_gpu_stats
        gpus = get_gpu_stats()
        checks["gpus"] = len(gpus)

        # Loaded models + resident-preload failures (spec 4.3). A non-empty
        # load_failures dict means at least one resident model failed to
        # preload — the Dashboard renders a degraded banner + Retry from this.
        mgr = getattr(app.state, "model_manager", None)
        checks["models_loaded"] = len(mgr.loaded_model_ids) if mgr else 0
        load_failures = dict(mgr._load_failures) if mgr else {}
        checks["load_failures"] = load_failures
        if load_failures:
            checks["status"] = "degraded"

        # Per-runner state (spec 4.2). runner_supervisors is populated by Lane K
        # lifespan wiring; until then it's unset and runners is []. LLMRunner
        # (主进程对象, app.state.llm_runner) 也并入此列表 —— 它有自己的
        # health_snapshot()，让前端 TaskPanel 用同一个 runners 列表渲染所有泳道。
        # 一个 runner 不 running（crashed / mid-restart）→ degraded.
        supervisors = getattr(app.state, "runner_supervisors", [])
        runners = [s.health_snapshot() for s in supervisors]
        _llm = getattr(app.state, "llm_runner", None)
        if _llm is not None:
            runners.append(_llm.health_snapshot())
        checks["runners"] = runners
        if any(not r.get("running", False) for r in runners):
            checks["status"] = "degraded"

        return checks
    app.include_router(understand.router)
    app.include_router(generate.router)
    app.include_router(tts.router)
    app.include_router(engines.router)
    app.include_router(models_routes.router)
    app.include_router(loras_routes.router)
    app.include_router(image_files_routes.router)
    from src.api.routes import components as components_routes
    app.include_router(components_routes.router)
    app.include_router(audio.router)
    app.include_router(voices.router)
    app.include_router(openai_compat.router)
    app.include_router(ollama_compat.router)
    app.include_router(api_gateway_routes.router)
    app.include_router(context_cache_routes.router)
    app.include_router(files_routes.router)
    from src.api.routes import responses as responses_routes
    app.include_router(responses_routes.router)
    app.include_router(settings.router)
    app.include_router(instances.router)
    app.include_router(predictions_routes.router)
    app.include_router(workflows.router)
    app.include_router(agents.router)
    app.include_router(skills.router)
    app.include_router(monitor.router)
    app.include_router(node_packages.router)
    app.include_router(execution_tasks.router)
    app.include_router(apps.router)
    app.include_router(services_routes.router)
    app.include_router(workflow_publish_routes.router)
    app.include_router(usage_routes.router)
    app.include_router(dashboard_routes.router)
    app.include_router(api_keys_routes.router)
    app.include_router(api_keys_routes.service_grants_router)
    app.include_router(anthropic_compat.router)
    app.include_router(observability.router)
    app.include_router(logs.router)
    from src.api.routes import memory as memory_routes
    app.include_router(memory_routes.router)
    from src.api.routes import admin_auth as admin_auth_routes
    app.include_router(admin_auth_routes.router)
    from src.api.routes import admin_passkey as admin_passkey_routes
    app.include_router(admin_passkey_routes.router)
    from src.api.routes import admin_totp as admin_totp_routes
    app.include_router(admin_totp_routes.router)

    from src.api.admin_session import websocket_is_authed

    async def _reject_unauthed_ws(websocket: WebSocket) -> bool:
        """Return True when the WS was rejected. Closes with policy code 4401."""
        if websocket_is_authed(websocket):
            return False
        await websocket.close(code=4401)
        return True

    @app.websocket("/ws/tasks/{task_id}")
    async def websocket_task(websocket: WebSocket, task_id: str):
        if await _reject_unauthed_ws(websocket):
            return
        await ws_manager.connect(task_id, websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(task_id, websocket)

    @app.websocket("/ws/tasks")
    async def websocket_tasks_global(websocket: WebSocket):
        """Global task list WebSocket — pushes task create/update/delete events."""
        if await _reject_unauthed_ws(websocket):
            return
        await ws_manager.subscribe_global(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.unsubscribe_global(websocket)

    @app.websocket("/ws/models")
    async def websocket_models(websocket: WebSocket):
        """Model loading status WebSocket -- pushes loading/loaded/failed events."""
        if await _reject_unauthed_ws(websocket):
            return
        await ws_manager.subscribe_models(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.unsubscribe_models(websocket)

    @app.websocket("/ws/tts")
    async def websocket_tts(websocket: WebSocket):
        if await _reject_unauthed_ws(websocket):
            return
        await handle_tts_websocket(websocket)

    @app.websocket("/ws/workflow/{instance_id}")
    async def workflow_progress_ws(websocket: WebSocket, instance_id: str):
        if await _reject_unauthed_ws(websocket):
            return
        await websocket.accept()
        if instance_id not in _ws_connections:
            _ws_connections[instance_id] = []
        _ws_connections[instance_id].append(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            # round4 #12:workflow_runner._broadcast 推送失败时会先 remove 本连接、桶空再
            # pop 掉整个 key。竞态下若本连接已被它剔除、key 已删,这里裸 `[instance_id].remove`
            # 会抛 KeyError/ValueError(从 except 逃逸成未处理 task 异常)。且非干净断开
            # (网络 reset 不抛 WebSocketDisconnect)早先完全不清理 → 连接 + key 永久泄漏。
            # 改 finally + 守卫:存在才 remove,空了再 pop。
            socks = _ws_connections.get(instance_id)
            if socks and websocket in socks:
                socks.remove(websocket)
                if not socks:
                    _ws_connections.pop(instance_id, None)

    _mount_frontend(app)
    _register_error_handlers(app)
    return app


# --------------------------------------------------------------------------- #
# Frontend static serving — production builds are served by the API process so
# the browser hits one origin (cloudflared can point at :8000 only). Vite dev
# on :9999 still works for local HMR; this only kicks in when `dist/` exists.
# --------------------------------------------------------------------------- #

# Path prefixes that belong to the API and must NOT fall through to the SPA.
# Anything else returns index.html so client-side routing handles deep links.
_API_PREFIXES = ("/v1", "/sys", "/api", "/ws", "/health", "/healthz", "/docs", "/openapi.json", "/redoc")


def _frontend_dist_dir() -> Path | None:
    # backend/src/api/main.py → repo_root = parents[3]
    candidate = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    return candidate if (candidate / "index.html").exists() else None


def _mount_frontend(app: FastAPI) -> None:
    # Test suites build the app and add their own routes after create_app().
    # The SPA catch-all (/{full_path:path}) would otherwise win route matching
    # against later-registered test endpoints. Tests set this env to opt out.
    import os
    if os.environ.get("NOUS_DISABLE_FRONTEND_MOUNT") == "1":
        return
    dist = _frontend_dist_dir()
    if dist is None:
        logger.info("frontend dist not found, skipping static mount (run `npm run build`)")
        return

    index_html = dist / "index.html"
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="frontend-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str, request: Request):
        path = "/" + full_path
        if any(path == p or path.startswith(p + "/") for p in _API_PREFIXES):
            raise HTTPException(status_code=404)
        candidate = dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_html)


# --------------------------------------------------------------------------- #
# Global exception handlers — convert everything into OpenAI-style JSON
# --------------------------------------------------------------------------- #

from src.errors import (
    NousError,
    InvalidRequestError,
    AuthenticationError,
    PermissionError as NousPermissionError,
    NotFoundError,
    RateLimitError,
    APIError,
)

_HTTP_STATUS_TO_ERROR = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: NousPermissionError,
    404: NotFoundError,
    409: InvalidRequestError,
    422: InvalidRequestError,
    429: RateLimitError,
}

# Default `code` field for statuses where the shared error type needs disambiguation
_STATUS_DEFAULT_CODE = {
    409: "conflict",
    422: "validation_error",
}


def _detail_to_message_and_param(detail) -> tuple[str, str | None]:
    """Parse HTTPException.detail into (message, param).

    detail can be str, list (Pydantic-style), or anything else.
    """
    if isinstance(detail, str):
        return detail, None
    if isinstance(detail, list) and detail:
        first = detail[0] if isinstance(detail[0], dict) else {}
        msg = first.get("msg") or "; ".join(
            e.get("msg", str(e)) if isinstance(e, dict) else str(e) for e in detail
        )
        loc = first.get("loc") or []
        param = ".".join(str(x) for x in loc if x != "body") or None
        return msg, param
    return str(detail), None


def _response(err: NousError) -> JSONResponse:
    headers = {"X-Request-Id": err.request_id} if err.request_id else {}
    return JSONResponse(err.to_dict(), status_code=err.http_status, headers=headers)


def _with_request_id(err: NousError, request) -> NousError:
    err.request_id = getattr(request.state, "request_id", None)
    return err


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NousError)
    async def _nous(request, exc: NousError):
        return _response(_with_request_id(exc, request))

    @app.exception_handler(HTTPException)
    async def _http(request, exc: HTTPException):
        status = exc.status_code
        cls = _HTTP_STATUS_TO_ERROR.get(status)
        if cls is None:
            cls = InvalidRequestError if 400 <= status < 500 else APIError
        msg, param = _detail_to_message_and_param(exc.detail)
        err = cls(msg, param=param, code=_STATUS_DEFAULT_CODE.get(status))
        err.http_status = status  # preserve original 4xx nuance
        return _response(_with_request_id(err, request))

    @app.exception_handler(RequestValidationError)
    async def _validation(request, exc: RequestValidationError):
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        err = InvalidRequestError(
            message=first.get("msg", "Invalid request"),
            code="validation_error",
            param=loc or None,
        )
        return _response(_with_request_id(err, request))

    @app.exception_handler(Exception)
    async def _unhandled(request, exc: Exception):
        # Outer safety net: this handler itself must never raise.
        rid = None
        try:
            rid = getattr(request.state, "request_id", None)
            try:
                logger.exception(
                    "unhandled exception | req_id=%s | %s %s",
                    rid, request.method, request.url.path,
                )
            except Exception:
                pass  # logging failure must not crash the handler
            err = APIError(
                "Internal server error", code="internal_error", request_id=rid
            )
            return _response(err)
        except Exception:
            headers = {"X-Request-Id": rid} if rid else {}
            return JSONResponse(
                {"error": {
                    "message": "Internal server error",
                    "type": "api_error",
                    "code": "internal_error",
                }},
                status_code=500,
                headers=headers,
            )


app = create_app()
