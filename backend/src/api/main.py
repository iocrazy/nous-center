from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import tasks, understand, generate, tts, engines, audio, voices, openai_compat, settings, preset_keys, preset_service
from src.api.websocket import ws_manager
from src.api.ws_tts import handle_tts_websocket


def create_app() -> FastAPI:
    app = FastAPI(title="Nous Center", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
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
    app.include_router(preset_keys.router)
    app.include_router(preset_service.router)

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
