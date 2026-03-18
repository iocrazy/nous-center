import asyncio
import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import tasks, understand, generate, tts, engines, audio, voices, openai_compat, settings, instances, instance_keys, instance_service, workflows, agents, skills
from src.api.websocket import ws_manager
from src.api.ws_tts import handle_tts_websocket
from src.services import model_scheduler

logger = logging.getLogger(__name__)


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

    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    logger.info("Database tables ensured")

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

    # Start idle model checker background task
    async def idle_checker():
        while True:
            await asyncio.sleep(60)
            try:
                await model_scheduler.check_idle_models()
            except Exception as e:
                logger.warning("Idle model check failed: %s", e)

    asyncio.create_task(idle_checker())

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

    app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
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

    @app.websocket("/ws/tasks/{task_id}")
    async def websocket_task(websocket: WebSocket, task_id: str):
        await ws_manager.connect(task_id, websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(task_id, websocket)

    @app.websocket("/ws/tts")
    async def websocket_tts(websocket: WebSocket):
        await handle_tts_websocket(websocket)

    return app


app = create_app()
