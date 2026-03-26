import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# Stub out heavy GPU dependencies so tests run without torch/torchaudio/etc.
for mod_name in [
    "torch", "torch.nn", "torch.nn.functional", "torch.cuda",
    "torchaudio", "torchaudio.transforms",
    "modelscope", "cosyvoice",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from src.models.database import Base, get_async_session
from src.api.main import create_app
import src.models.voice_preset  # noqa: F401 — ensure models registered with Base
import src.models.tts_usage  # noqa: F401 — register model
import src.models.service_instance  # noqa: F401 — register model
import src.models.instance_api_key  # noqa: F401 — register model
import src.models.model_metadata  # noqa: F401 — register model
import src.models.workflow  # noqa: F401 — register model
import src.models.execution_task  # noqa: F401 — register model


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def db_client(tmp_path):
    """Client with a real (SQLite) test database for voice preset tests."""
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with session_factory() as session:
            yield session

    test_app = create_app()
    test_app.dependency_overrides[get_async_session] = override_session

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await engine.dispose()


@pytest.fixture
async def db_session(tmp_path):
    """Raw async session with all tables created (no app)."""
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()
