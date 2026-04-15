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
import src.models.workflow_app  # noqa: F401 — register model
import src.models.context_cache  # noqa: F401 — register model


def _mock_model_manager():
    """Create a mock ModelManager for tests."""
    mgr = MagicMock()
    mgr.load_model = MagicMock(side_effect=lambda *a, **kw: _async_noop())
    mgr.unload_model = MagicMock(side_effect=lambda *a, **kw: _async_noop())
    mgr.add_reference = MagicMock()
    mgr.remove_reference = MagicMock()
    mgr.get_model_dependencies = MagicMock(return_value=[])
    mgr.loaded_model_ids = []
    mgr.get_status = MagicMock(return_value={"loaded": [], "references": {}, "last_used": {}})
    mgr.check_idle_models = MagicMock(side_effect=lambda: _async_noop())
    return mgr


async def _async_noop():
    pass


@pytest.fixture
def app():
    _app = create_app()
    _app.state.model_manager = _mock_model_manager()
    return _app


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
    test_app.state.model_manager = _mock_model_manager()
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

# Auto-use fixture: isolate tests from real logs.db
from .conftest_logs import _silence_db_log_handler  # noqa: F401


@pytest.fixture
async def sample_instance(db_session):
    from src.models.service_instance import ServiceInstance
    inst = ServiceInstance(
        source_type="model",
        source_name="qwen3.5-35b-test",
        name="test instance",
        type="llm",
        status="active",
    )
    db_session.add(inst)
    await db_session.commit()
    await db_session.refresh(inst)
    return inst


@pytest.fixture
async def other_instance(db_session):
    from src.models.service_instance import ServiceInstance
    inst = ServiceInstance(
        source_type="model",
        source_name="other-model",
        name="other instance",
        type="llm",
        status="active",
    )
    db_session.add(inst)
    await db_session.commit()
    await db_session.refresh(inst)
    return inst


@pytest.fixture
async def sample_api_key(db_session, sample_instance):
    """Returns the plaintext key string. Inserts the bcrypt-hashed row."""
    import bcrypt
    import secrets as _secrets
    from src.models.instance_api_key import InstanceApiKey

    raw = f"sk-test-{_secrets.token_hex(8)}"
    k = InstanceApiKey(
        instance_id=sample_instance.id,
        label="test",
        key_hash=bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode(),
        key_prefix=raw[:10],
        is_active=True,
    )
    db_session.add(k)
    await db_session.commit()
    return raw

