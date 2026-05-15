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

# Tests must not inherit the developer's local admin password from .env —
# pydantic-settings reads .env automatically, which would 401 every admin-gated
# route. Force-empty here disables the cookie gate (dev mode) for the whole
# suite. Bare assignment (not setdefault) overrides whatever .env provides.
os.environ["ADMIN_PASSWORD"] = ""
# image_generate node demands a signing secret since p2-polish-3 (no more
# base64 fallback). Tests don't need a real secret — anything non-empty
# unlocks the URL path. Specific tests that exercise the missing-secret
# error monkeypatch this back to "".
os.environ["ADMIN_SESSION_SECRET"] = "tests-only-do-not-use-in-prod"

# Skip mounting the SPA catch-all in tests — its /{full_path:path} would shadow
# routes that test fixtures register on the app after create_app() returns.
os.environ["NOUS_DISABLE_FRONTEND_MOUNT"] = "1"

# Image output storage writes to ~/.gstack/outputs/images by default.
# Redirect to a tmp dir so tests never pollute the developer's real home.
import tempfile as _tempfile
os.environ.setdefault("NOUS_IMAGE_OUTPUTS", _tempfile.mkdtemp(prefix="nous-test-img-"))

import sys
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# Lane J (spec §5.6): register shared integration/chaos fixtures so any
# test under tests/ can request `hardware_2gpu` / `hardware_3gpu` /
# `fake_runner` / `fake_vllm` without local imports.
pytest_plugins = [
    "tests.fixtures.hardware_topo",
    "tests.fixtures.fake_runner",
]

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
import src.models.context_cache  # noqa: F401 — register model
import src.models.response_session  # noqa: F401 — register model
import src.models.memory  # noqa: F401 — register model
import src.models.api_gateway  # noqa: F401 — register model


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


def _install_fake_adapter(monkeypatch, *, stream_tokens=("hel", "lo"),
                          usage=None, nonstream_text="non-stream response",
                          nonstream_usage=None):
    """Install a fake v2 adapter on workflow_executor._model_manager.

    LLMNode.stream / invoke now resolve the adapter via
    `mm.get_loaded_adapter(model_key)` and call `adapter.infer_stream(req)` /
    `adapter.infer(req)`. This helper replaces the module-level model manager
    with one that returns a fake adapter wired with predictable behavior.
    """
    from unittest.mock import AsyncMock, MagicMock

    from src.services import workflow_executor
    from src.services.inference.base import (
        InferenceResult,
        StreamEvent,
        UsageMeter,
    )

    stream_usage = usage or {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    nonstream_usage = nonstream_usage or {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }

    async def _infer_stream(req):
        for tok in stream_tokens:
            yield StreamEvent(type="delta", payload={"content": tok})
        yield StreamEvent(type="done", payload={"usage": stream_usage})

    raw_body = {
        "choices": [{"message": {"content": nonstream_text}}],
        "usage": nonstream_usage,
    }

    async def _infer(req):
        return InferenceResult(
            media_type="application/json",
            data=b"{}",
            metadata={"raw": raw_body},
            usage=UsageMeter(
                input_tokens=nonstream_usage.get("prompt_tokens"),
                output_tokens=nonstream_usage.get("completion_tokens"),
                latency_ms=1,
            ),
        )

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.infer = _infer
    adapter.infer_stream = _infer_stream
    adapter.max_model_len = 4096

    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    mgr.get_adapter = MagicMock(return_value=adapter)
    monkeypatch.setattr(workflow_executor, "_model_manager", mgr)
    return adapter


@pytest.fixture
def mock_llm_stream(monkeypatch):
    """Install a fake v2 adapter that streams two delta events ('hel','lo')
    and a final done event with usage. LLMNode.stream iterates infer_stream
    and reads the final usage from the done event.
    """
    return _install_fake_adapter(monkeypatch)


@pytest.fixture
def mock_llm_stream_v2(monkeypatch):
    """Alias of mock_llm_stream for tests still binding the v2 fixture name."""
    return _install_fake_adapter(monkeypatch)


@pytest.fixture
def mock_llm_nonstream(monkeypatch):
    """Install a fake v2 adapter whose infer() returns a canned non-stream
    OpenAI-format response wrapped in the InferenceResult envelope.
    """
    return _install_fake_adapter(monkeypatch)


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


# ---------- Fixtures for /v1/responses agent binding tests ---------- #


@pytest.fixture
def fixtures_home(monkeypatch):
    """Point NOUS_CENTER_HOME at tests/fixtures for agent/skill lookups."""
    from pathlib import Path
    from src.config import get_settings
    from src.services.prompt_composer import _persona as _persona_mod

    fixtures = Path(__file__).parent / "fixtures"
    monkeypatch.setenv("NOUS_CENTER_HOME", str(fixtures))
    # Settings is cached; must clear so the new env var is picked up.
    get_settings.cache_clear()
    # persona _load_cached is keyed on (agent_id, mtime); clear to be safe in
    # case a prior test loaded a different fixtures_home.
    _persona_mod._load_cached.cache_clear()
    return fixtures


@pytest.fixture
def mock_vllm(monkeypatch):
    """Intercept outgoing httpx requests to vLLM, record last body.

    Patches ``httpx.AsyncClient.post`` class-wide but only rewrites responses
    whose URL does NOT target the ASGI test transport (base ``http://test``).
    Internal ASGI calls ``api_client.post("/v1/responses", ...)`` still go to
    the real httpx POST path; only outbound vLLM calls get mocked.

    Also stubs out record_llm_usage (which creates a separate engine against
    the real DATABASE_URL) so tests don't require Postgres.
    """
    import httpx

    real_post = httpx.AsyncClient.post

    class _Recorder:
        def __init__(self):
            self.last_request_body: dict | None = None
            self.last_usage: dict | None = None

    recorder = _Recorder()

    async def _patched_post(self, url, *args, **kwargs):
        # Distinguish outbound (upstream vLLM) vs inbound (ASGI test transport)
        # by looking at the full request URL. Inbound api_client.post uses the
        # AsyncClient's base_url "http://test", so the resolved URL has host
        # "test" — never "test-vllm.invalid". httpx merges client.base_url with
        # relative paths, and the "url" arg to AsyncClient.post can be a bare
        # path — we must inspect the merged absolute form, not the raw arg.
        from httpx import URL as _URL
        base = getattr(self, "base_url", None)
        raw = _URL(url) if not isinstance(url, _URL) else url
        full = base.join(raw) if base else raw
        host = full.host or ""
        if host == "test-vllm.invalid" or (not host and "test-vllm.invalid" in str(url)):
            body = kwargs.get("json")
            recorder.last_request_body = body
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-mock",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                },
                request=httpx.Request("POST", url),
            )
        return await real_post(self, url, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", _patched_post)

    # Stub record_llm_usage — it opens its own engine against DATABASE_URL.
    async def _noop_record(**kwargs):
        recorder.last_usage = kwargs
    import src.services.usage_service as _usage
    monkeypatch.setattr(_usage, "record_llm_usage", _noop_record)

    return recorder


@pytest.fixture
async def api_client(tmp_path):
    """Async client with a SQLite-backed app, a loaded-LLM adapter mock,
    and a LLM-type service instance + API key preseeded.

    Exposes:
        api_client.app.state.async_session_factory  — for tests to query DB
        api_client.headers default Authorization Bearer <key>
    """
    import bcrypt
    import secrets as _secrets
    from unittest.mock import MagicMock

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from src.api.main import create_app
    from src.models.database import Base, get_async_session
    from src.models.instance_api_key import InstanceApiKey
    from src.models.service_instance import ServiceInstance

    db_path = tmp_path / "api_client.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Seed a model-type instance + an active API key.
    raw_key = f"sk-test-{_secrets.token_hex(8)}"
    async with session_factory() as s:
        inst = ServiceInstance(
            source_type="model",
            source_name="qwen3.5",
            name="qwen3.5 test instance",
            type="llm",
            status="active",
        )
        s.add(inst)
        await s.commit()
        await s.refresh(inst)
        key = InstanceApiKey(
            instance_id=inst.id,
            label="test",
            key_hash=bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode(),
            key_prefix=raw_key[:10],
            is_active=True,
        )
        s.add(key)
        await s.commit()

    async def override_session():
        async with session_factory() as session:
            yield session

    # Force services that create their own engine (usage_service,
    # responses.py's `_csf = create_session_factory`) to use the SQLite engine.
    import src.models.database as _db_mod
    orig_create_session_factory = _db_mod.create_session_factory

    def _patched_csf(engine_arg=None):
        return session_factory

    _db_mod.create_session_factory = _patched_csf

    test_app = create_app()

    # Mock model_manager to return a loaded adapter for engine_name 'qwen3.5'.
    mgr = MagicMock()
    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.base_url = "http://test-vllm.invalid"
    adapter.max_model_len = 4096
    mgr.get_adapter = MagicMock(return_value=adapter)
    test_app.state.model_manager = mgr
    # Expose session factory for tests.
    test_app.state.async_session_factory = session_factory

    test_app.dependency_overrides[get_async_session] = override_session

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # Stash app + raw_key so tests/fixtures can find them.
        c.app = test_app  # type: ignore[attr-defined]
        c.raw_key = raw_key  # type: ignore[attr-defined]
        yield c

    _db_mod.create_session_factory = orig_create_session_factory
    await engine.dispose()


@pytest.fixture
def bearer_headers(api_client):
    return {"Authorization": f"Bearer {api_client.raw_key}"}

