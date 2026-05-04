import asyncio
import logging

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import tasks, understand, generate, tts, engines, audio, voices, openai_compat, ollama_compat, api_gateway as api_gateway_routes, settings, instances, instance_keys, instance_service, workflows, agents, skills, monitor, node_packages, execution_tasks, apps, logs, context_cache as context_cache_routes, files as files_routes, services as services_routes, workflow_publish as workflow_publish_routes, usage as usage_routes, dashboard as dashboard_routes, api_keys as api_keys_routes, anthropic_compat, observability, loras as loras_routes
from src.api.websocket import ws_manager
from src.api.ws_tts import handle_tts_websocket
from src.services.gpu_monitor import memory_guard_loop

logger = logging.getLogger(__name__)

# Active WebSocket connections for workflow progress updates, keyed by instance_id
_ws_connections: dict[str, list[WebSocket]] = {}


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

    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    logger.info("Database tables ensured")

    # Initialize log database
    from src.services.log_db import init_log_db
    init_log_db()
    logger.info("Log database initialized")

    # Install DB log handler for application logs
    from src.services.log_collector import DbLogHandler
    import logging as _logging
    db_handler = DbLogHandler()
    db_handler.setLevel(_logging.INFO)
    _logging.getLogger("src").addHandler(db_handler)
    _logging.getLogger("nous").addHandler(db_handler)
    logger.info("Application log collector installed")

    # Auto-sync model metadata for any new engines
    from src.models.database import create_session_factory
    from src.services.model_metadata_service import sync_metadata

    # Wave 1 MemoryProvider: init PGMemoryProvider + expose via app.state
    from src.services.memory.pg_provider import PGMemoryProvider
    try:
        app.state.memory_provider = PGMemoryProvider(
            session_factory=create_session_factory()
        )
        await app.state.memory_provider.initialize()
        logger.info("MemoryProvider initialized (pg)")
    except Exception as e:
        logger.warning("MemoryProvider init failed (non-fatal): %s", e)

    try:
        sf = create_session_factory()
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

    # Wire ModelManager into workflow executor
    from src.services.workflow_executor import set_model_manager
    set_model_manager(model_mgr)

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

    if not _bg_tasks_disabled:
        # Load workflow dependencies in background (non-blocking)
        async def _load_wf_deps():
            for dep in wf_model_deps:
                try:
                    await model_mgr.load_model(dep["key"])
                except Exception as e:
                    logger.warning("Failed to load model %s for workflow %s: %s", dep["key"], dep["wf_id"], e)

        if wf_model_deps:
            asyncio.create_task(_load_wf_deps())

        # Image models marked resident: preload in the background so the
        # ~120s diffusers compose doesn't block /health (cloudflared /
        # systemd probes would mark backend down). Each spec gets its own
        # task so one failure doesn't poison the others. On success:
        # invalidate the engines cache + push a ws/models event so the UI
        # flips the badge within 1s. On failure: write the reason into
        # mm._load_failures so subsequent get_loaded_adapter raises a
        # ModelLoadError instead of retrying indefinitely.
        async def _preload_image_model(spec_id: str):
            try:
                await model_mgr.load_model(spec_id)
                logger.info("Image preload succeeded: %s", spec_id)
                from src.api.response_cache import invalidate as _invalidate
                _invalidate("models", "engines")
                from src.api.websocket import ws_manager as _ws
                await _ws.broadcast_model_status(spec_id, "loaded")
            except Exception as e:
                detail = f"{type(e).__name__}: {e}"
                model_mgr._load_failures[spec_id] = detail
                logger.warning("Image preload failed for %s: %s", spec_id, detail)
                try:
                    from src.api.websocket import ws_manager as _ws
                    await _ws.broadcast_model_status(spec_id, "error", detail)
                except Exception:
                    pass

        image_specs = [
            s for s in registry.specs
            if s.model_type == "image" and s.resident and s.id not in reconnected
        ]
        # Persist task refs so 3.11+ doesn't garbage-collect a still-running
        # background coroutine and silently drop the load.
        app.state._image_preload_tasks = [
            asyncio.create_task(_preload_image_model(s.id)) for s in image_specs
        ]
        if image_specs:
            logger.info(
                "Started background preload for %d resident image model(s): %s",
                len(image_specs), [s.id for s in image_specs],
            )

        # Start idle model checker background task
        async def idle_checker():
            while True:
                await asyncio.sleep(60)
                try:
                    await model_mgr.check_idle_models()
                except Exception as e:
                    logger.warning("Idle model check failed: %s", e)

        asyncio.create_task(idle_checker())
        asyncio.create_task(memory_guard_loop(reserved_gb=4.0))

        async def log_cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # Every hour
                try:
                    from src.services.log_db import cleanup_logs
                    cleanup_logs()
                except Exception as e:
                    logger.warning("Log cleanup failed: %s", e)

        asyncio.create_task(log_cleanup_loop())

        async def context_cache_cleanup_loop(interval_seconds: int = 3600):
            from src.services.context_cache_service import cleanup_expired
            from src.models.database import create_session_factory as _csf
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
            from src.models.database import create_session_factory as _csf
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
        # Drain partial-write worker
        if partial_worker is not None:
            from src.api.routes import responses as responses_routes
            if responses_routes._partial_write_queue is not None:
                await responses_routes._partial_write_queue.put(None)
                try:
                    await asyncio.wait_for(partial_worker, timeout=5.0)
                except asyncio.TimeoutError:
                    partial_worker.cancel()


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
            from src.models.database import create_session_factory
            from sqlalchemy import text
            _sf = create_session_factory()
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

        # Loaded models
        mgr = getattr(app.state, "model_manager", None)
        checks["models_loaded"] = len(mgr.loaded_model_ids) if mgr else 0

        return checks
    app.include_router(tasks.router)
    app.include_router(understand.router)
    app.include_router(generate.router)
    app.include_router(tts.router)
    app.include_router(engines.router)
    app.include_router(loras_routes.router)
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
    app.include_router(instance_keys.router)
    app.include_router(instance_service.router)
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
            _ws_connections[instance_id].remove(websocket)

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
