import asyncio
import logging

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import tasks, understand, generate, tts, engines, audio, voices, openai_compat, settings, instances, instance_keys, instance_service, workflows, agents, skills, monitor, node_packages, execution_tasks, apps, logs
from src.api.websocket import ws_manager
from src.api.ws_tts import handle_tts_websocket
from src.services.gpu_monitor import memory_guard_loop

logger = logging.getLogger(__name__)

# Active WebSocket connections for workflow progress updates, keyed by instance_id
_ws_connections: dict[str, list[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup."""
    from src.models.database import Base, create_engine
    import src.models.voice_preset  # noqa: F401
    import src.models.tts_usage  # noqa: F401
    import src.models.service_instance  # noqa: F401
    import src.models.instance_api_key  # noqa: F401
    import src.models.model_metadata  # noqa: F401
    import src.models.workflow  # noqa: F401
    import src.models.execution_task  # noqa: F401
    import src.models.workflow_app  # noqa: F401

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

    # Auto-load resident models
    for spec in registry.specs:
        if spec.resident:
            try:
                logger.info(f"Auto-loading resident model: {spec.id}")
                await model_mgr.load_model(spec.id)
            except Exception as e:
                logger.warning(f"Failed to auto-load {spec.id}: {e}")

    # Auto-detect running vLLM instances and register matching adapters
    import httpx as _httpx
    vllm_urls_checked: dict[str, list[str]] = {}  # url -> list of served model paths
    for spec in registry.specs:
        if spec.model_type != "llm":
            continue
        port = spec.params.get("vllm_port") or spec.params.get("vllm_base_url", "").split(":")[-1]
        if not port:
            continue
        base_url = f"http://localhost:{port}"
        if base_url not in vllm_urls_checked:
            try:
                async with _httpx.AsyncClient(timeout=3) as client:
                    resp = await client.get(f"{base_url}/v1/models")
                    if resp.status_code == 200:
                        models_data = resp.json().get("data", [])
                        vllm_urls_checked[base_url] = [m.get("id", "") for m in models_data]
                    else:
                        vllm_urls_checked[base_url] = []
            except Exception:
                vllm_urls_checked[base_url] = []
        # Check if this spec's model path matches any served model
        served = vllm_urls_checked.get(base_url, [])
        if any(s.rstrip("/").endswith(spec.path.rstrip("/")) for s in served):
            logger.info("Auto-detected running vLLM model for %s", spec.id)
            try:
                await model_mgr.load_model(spec.id)
            except Exception as e:
                logger.warning("Failed to auto-register %s: %s", spec.id, e)

    # Re-register model references for published workflows
    from sqlalchemy import select
    from src.models.workflow import Workflow
    async with sf() as session:
        stmt = select(Workflow).where(Workflow.status == "published")
        result = await session.execute(stmt)
        for wf in result.scalars():
            deps = model_mgr.get_model_dependencies({"nodes": wf.nodes, "edges": wf.edges})
            for dep in deps:
                model_mgr.add_reference(dep["key"], str(wf.id))
                try:
                    await model_mgr.load_model(dep["key"])
                except Exception as e:
                    logger.warning(f"Failed to load model {dep['key']} for workflow {wf.id}: {e}")

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

    yield


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
    )

    from src.api.middleware import RequestLoggingMiddleware, AuditMiddleware
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

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
    app.include_router(audio.router)
    app.include_router(voices.router)
    app.include_router(openai_compat.router)
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
    app.include_router(logs.router)

    @app.websocket("/ws/tasks/{task_id}")
    async def websocket_task(websocket: WebSocket, task_id: str):
        await ws_manager.connect(task_id, websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(task_id, websocket)

    @app.websocket("/ws/tasks")
    async def websocket_tasks_global(websocket: WebSocket):
        """Global task list WebSocket — pushes task create/update/delete events."""
        await ws_manager.subscribe_global(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.unsubscribe_global(websocket)

    @app.websocket("/ws/tts")
    async def websocket_tts(websocket: WebSocket):
        await handle_tts_websocket(websocket)

    @app.websocket("/ws/workflow/{instance_id}")
    async def workflow_progress_ws(websocket: WebSocket, instance_id: str):
        await websocket.accept()
        if instance_id not in _ws_connections:
            _ws_connections[instance_id] = []
        _ws_connections[instance_id].append(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            _ws_connections[instance_id].remove(websocket)

    return app


app = create_app()
