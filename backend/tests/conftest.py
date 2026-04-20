import os

# CRITICAL SAFETY: disable lifespan background tasks BEFORE any app import.
# The default set includes memory_guard_loop which polls nvidia-smi via
# subprocess every 5s. When test files using starlette/fastapi TestClient
# trigger lifespan (test_ws_tts.py, test_api_errors.py), concurrent
# nvidia-smi calls can crash the NVIDIA driver → X session logout. This
# env var (consumed in src/api/main.py lifespan) gates all asyncio.create_task
# calls so tests never spawn those loops.
os.environ.setdefault("NOUS_DISABLE_BG_TASKS", "1")

# CRITICAL SAFETY: hide CUDA so any torch/vllm import sees zero devices,
# preventing libcudart dlopen / CUDA context init during tests.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = ""

import sys
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
import src.models.response_session  # noqa: F401 — register model


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
def on_progress_capture(monkeypatch):
    """Capture all events emitted via workflow_executor._on_progress_ref.

    Used by Wave 1 event-type tests. Resets the module-level reference so the
    LLM node's streaming branch routes events through our capture callable.
    """
    class _Cap:
        def __init__(self):
            self.events: list[dict] = []

        async def __call__(self, ev):
            self.events.append(ev)

    cap = _Cap()
    from src.services import workflow_executor
    monkeypatch.setattr(workflow_executor, "_on_progress_ref", cap)
    return cap


@pytest.fixture
def mock_llm_stream(monkeypatch):
    """Replace workflow_executor._stream_llm with a deterministic 3-chunk stream.

    `_stream_llm` is a module-level coroutine with signature
    `(base_url, params, on_token=None) -> str`. It pushes each token via
    `on_token` and writes the final usage dict into the module-level
    `_last_stream_usage` (what `_exec_llm` reads after awaiting). The fake
    replicates that contract without any HTTP.
    """
    from src.services import workflow_executor

    async def _fake_stream(base_url, params, on_token=None):
        tokens = ["hel", "lo"]
        for tok in tokens:
            if on_token is not None:
                await on_token(tok)
        workflow_executor._last_stream_usage = {
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "total_tokens": 4,
        }
        return "".join(tokens)

    monkeypatch.setattr(workflow_executor, "_stream_llm", _fake_stream)
    return _fake_stream


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

