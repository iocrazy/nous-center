from fastapi import FastAPI

from src.api.routes import tasks


def create_app() -> FastAPI:
    app = FastAPI(title="Mind Center", version="0.1.0")

    app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
    app.include_router(tasks.router)

    return app


app = create_app()
