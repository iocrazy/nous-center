"""Integration tests for /api/v1/memory/sync + /prefetch (Wave 1 Task 5.3)."""

import bcrypt
import pytest
import pytest_asyncio
import secrets as _secrets
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from unittest.mock import MagicMock

from src.api.main import create_app
from src.models.database import Base, get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.memory.pg_provider import PGMemoryProvider

# Import registers models on Base.metadata
import src.models.memory  # noqa: F401
import src.models.voice_preset  # noqa: F401
import src.models.tts_usage  # noqa: F401
import src.models.service_instance  # noqa: F401
import src.models.instance_api_key  # noqa: F401
import src.models.model_metadata  # noqa: F401
import src.models.workflow  # noqa: F401
import src.models.execution_task  # noqa: F401
import src.models.workflow_app  # noqa: F401
import src.models.context_cache  # noqa: F401
import src.models.response_session  # noqa: F401


def _mock_model_manager():
    mgr = MagicMock()

    async def _noop(*a, **kw):
        return None

    mgr.load_model = MagicMock(side_effect=_noop)
    mgr.unload_model = MagicMock(side_effect=_noop)
    mgr.add_reference = MagicMock()
    mgr.remove_reference = MagicMock()
    mgr.get_model_dependencies = MagicMock(return_value=[])
    mgr.loaded_model_ids = []
    mgr.get_status = MagicMock(
        return_value={"loaded": [], "references": {}, "last_used": {}}
    )
    mgr.check_idle_models = MagicMock(side_effect=_noop)
    return mgr


@pytest_asyncio.fixture
async def api_client_and_key(tmp_path):
    """Spin up app with SQLite DB, seed an active instance + API key,
    attach a PGMemoryProvider on app.state.memory_provider.

    Yields (client, plaintext_key).
    """
    db_path = tmp_path / "api_memory_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Seed instance + API key
    raw_key = f"sk-test-{_secrets.token_hex(8)}"
    async with session_factory() as session:
        inst = ServiceInstance(
            source_type="model",
            source_name="qwen-test",
            name="memory-test-instance",
            type="llm",
            status="active",
        )
        session.add(inst)
        await session.commit()
        await session.refresh(inst)

        key = InstanceApiKey(
            instance_id=inst.id,
            label="memory-test",
            key_hash=bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode(),
            key_prefix=raw_key[:10],
            is_active=True,
        )
        session.add(key)
        await session.commit()

    async def override_session():
        async with session_factory() as session:
            yield session

    test_app = create_app()
    test_app.state.model_manager = _mock_model_manager()
    test_app.dependency_overrides[get_async_session] = override_session

    # Inject memory provider (lifespan does not run under ASGITransport here).
    provider = PGMemoryProvider(session_factory=session_factory)
    await provider.initialize()
    test_app.state.memory_provider = provider

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, raw_key

    await engine.dispose()


@pytest.fixture
def api_client(api_client_and_key):
    return api_client_and_key[0]


@pytest.fixture
def bearer_headers(api_client_and_key):
    return {"Authorization": f"Bearer {api_client_and_key[1]}"}


@pytest.mark.asyncio
async def test_sync_endpoint_writes_entries(api_client, bearer_headers):
    resp = await api_client.post(
        "/api/v1/memory/sync",
        json={
            "entries": [
                {"category": "preference", "content": "用户喜欢简洁回复", "context_key": "proj-1"},
            ],
            "context_key": "proj-1",
        },
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    assert "entry_ids" in resp.json()
    assert len(resp.json()["entry_ids"]) == 1


@pytest.mark.asyncio
async def test_prefetch_endpoint_returns_entries(api_client, bearer_headers):
    await api_client.post(
        "/api/v1/memory/sync",
        json={"entries": [{"category": "fact", "content": "Tokyo", "context_key": None}]},
        headers=bearer_headers,
    )
    resp = await api_client.get(
        "/api/v1/memory/prefetch?q=Tokyo&limit=5",
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    assert len(resp.json()["entries"]) == 1


@pytest.mark.asyncio
async def test_sync_over_100_entries_returns_400(api_client, bearer_headers):
    entries = [{"category": "fact", "content": "x", "context_key": None}] * 101
    resp = await api_client.post(
        "/api/v1/memory/sync",
        json={"entries": entries},
        headers=bearer_headers,
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_sync_empty_list_ok(api_client, bearer_headers):
    resp = await api_client.post(
        "/api/v1/memory/sync",
        json={"entries": []},
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["entry_ids"] == []
