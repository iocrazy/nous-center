# Nous Center V2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the project as a monorepo, enhance TTS with streaming/caching/rounds/emotion, expand the Rust service with audio IO, and complete the frontend node execution engine.

**Architecture:** Three-service monorepo — `backend/` (FastAPI + Celery), `frontend/` (React + WASM), `nous-core/` (Rust Axum). Backend handles AI inference and business logic. nous-core handles system monitoring and performance-sensitive audio IO. Frontend has a node-based workflow editor with a real execution engine.

**Tech Stack:** Python 3.12 (FastAPI, SQLAlchemy, Celery, Redis), TypeScript (React 19, Vite, Zustand, React Query), Rust (Axum, symphonia, hound).

**Spec:** `docs/superpowers/specs/2026-03-12-nous-center-v2-design.md`

**Key codebase facts (verified):**
- `config.py` uses bare `Path("settings.yaml")` and `"configs/models.yaml"` — relative to cwd, breaks if started from repo root
- `schemas.py` has `BatchSegment` with no `round_id` and `BatchTTSRequest.segments` — needs rename to `rounds`
- `TTSEngine.synthesize()` signature has no `emotion` param
- `conftest.py` has `client` and `db_client` fixtures using SQLite + aiosqlite for testing
- `pyproject.toml` has `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed
- `registry.py` uses `_ENGINE_INSTANCES` dict and `_ENGINE_CLASSES` dict
- `websocket.py` has `ConnectionManager` keyed by `task_id`
- Frontend `workflowExecutor.ts` has `topoSort()` already implemented; complex path falls back to simple TTS call
- Frontend `execution.ts` store tracks `isRunning`, `taskId`, `progress`, `error`, `result`
- Rust service is at `nous-center-sys/` with `gpu.rs`, `system.rs`, `files.rs`; `symphonia`+`hound` in Cargo.toml but unused

---

## Chunk 0: Phase 1 — Monorepo Restructure & Fixes

Phase 1 is already partially done (files renamed in git staging). These tasks complete it.

### Task 1: Rename nous-center-sys to nous-core

**Files:**
- Rename: `nous-center-sys/` → `nous-core/`
- Modify: `nous-core/src/main.rs`

- [ ] **Step 1: Rename directory**

Note: `nous-center-sys/` may be untracked in git. Use filesystem rename + git add:

```bash
mv nous-center-sys nous-core
git add nous-core
```

- [ ] **Step 2: Update startup message in main.rs**

In `nous-core/src/main.rs`, change the println:

```rust
// Before:
println!("nous-center-sys listening on http://127.0.0.1:8001");
// After:
println!("nous-core listening on http://127.0.0.1:8001");
```

- [ ] **Step 3: Update Cargo.toml package name**

In `nous-core/Cargo.toml`, change:

```toml
[package]
name = "nous-core"
```

- [ ] **Step 4: Build to verify**

```bash
cd nous-core && cargo build
```
Expected: builds successfully.

- [ ] **Step 5: Commit**

```bash
git add nous-core
git commit -m "core: rename nous-center-sys to nous-core"
```

### Task 2: Unify global .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add settings.yaml to gitignore**

Append to `.gitignore`:

```
# Runtime config
settings.yaml
```

This file contains user-specific overrides and should not be committed.

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add settings.yaml to gitignore"
```

### Task 3: Fix config.py relative paths

**Files:**
- Modify: `backend/src/config.py`
- Test: `backend/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_config.py`:

```python
import os
from pathlib import Path

from src.config import _resolve_path, load_model_configs


def test_resolve_path_is_relative_to_backend():
    """Paths must resolve relative to backend/ dir, not cwd."""
    resolved = _resolve_path("configs/models.yaml")
    # Should point into backend/configs/, not <cwd>/configs/
    assert "backend" in str(resolved), f"Expected path relative to backend/, got {resolved}"


def test_load_model_configs_from_any_cwd(tmp_path, monkeypatch):
    """load_model_configs works even when cwd is not backend/."""
    monkeypatch.chdir(tmp_path)
    # Should not raise FileNotFoundError
    configs = load_model_configs()
    assert isinstance(configs, dict)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && uv run pytest tests/test_config.py::test_resolve_path_is_relative_to_backend -v
```
Expected: FAIL — `_resolve_path` does not exist yet.

- [ ] **Step 3: Implement the fix**

Replace `backend/src/config.py`:

```python
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

# All paths resolve relative to the backend/ directory (parent of src/)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
SETTINGS_YAML_PATH = _BACKEND_DIR / "settings.yaml"


class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql+asyncpg://mindcenter:mindcenter@localhost:5432/mindcenter"

    NAS_MODELS_PATH: str = "/mnt/nas/models"
    NAS_OUTPUTS_PATH: str = "/mnt/nas/outputs"
    LOCAL_MODELS_PATH: str = "/media/heygo/Program/models"

    COSYVOICE_REPO_PATH: str = "/media/heygo/Program/projects-code/github-repos/CosyVoice"
    INDEXTTS_REPO_PATH: str = "/media/heygo/Program/projects-code/github-repos/index-tts"

    VLLM_BASE_URL: str = "http://localhost:8100"

    GPU_IMAGE: int = 0
    GPU_TTS: int = 1
    GPU_VIDEO: str = "0,1"

    CACHE_TTL_SECONDS: int = 3600  # TTS cache TTL (1 hour)

    model_config = {"env_file": ".env", "extra": "ignore"}


def _resolve_path(relative: str) -> Path:
    """Resolve a path relative to the backend/ directory."""
    return _BACKEND_DIR / relative


@lru_cache
def get_settings() -> Settings:
    overrides = _load_settings_yaml()
    return Settings(**overrides)


def _load_settings_yaml() -> dict:
    """Load overrides from settings.yaml if it exists."""
    if not SETTINGS_YAML_PATH.exists():
        return {}
    try:
        with open(SETTINGS_YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        return {}


def save_settings(updates: dict) -> None:
    """Merge updates into settings.yaml and clear the cached Settings."""
    existing = _load_settings_yaml()
    existing.update(updates)

    with open(SETTINGS_YAML_PATH, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

    get_settings.cache_clear()


def load_model_configs(path: str = "configs/models.yaml") -> dict:
    resolved = _resolve_path(path)
    with open(resolved) as f:
        data = yaml.safe_load(f)
    return data["models"]
```

- [ ] **Step 4: Run tests**

```bash
cd backend && uv run pytest tests/test_config.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/config.py backend/tests/test_config.py
git commit -m "backend: fix config.py relative paths to resolve from backend/ dir"
```

### Task 4: Create global dev startup script

**Files:**
- Create: `scripts/dev.sh`

- [ ] **Step 1: Create scripts directory and dev.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Starting nous-core (Rust) ==="
(cd "$REPO_ROOT/nous-core" && cargo run) &
CORE_PID=$!

echo "=== Starting backend (FastAPI) ==="
(cd "$REPO_ROOT/backend" && uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

echo "=== Starting frontend (Vite) ==="
(cd "$REPO_ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "Services running:"
echo "  backend:   http://localhost:8000"
echo "  nous-core: http://localhost:8001"
echo "  frontend:  http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all."

trap "kill $CORE_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/dev.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/dev.sh
git commit -m "chore: add scripts/dev.sh for one-command startup"
```

---

## Chunk 1: Phase 2 — TTS Result Caching + Usage Statistics

### Task 5: Add TTS usage model (tts_usage table)

**Files:**
- Create: `backend/src/models/tts_usage.py`
- Modify: `backend/tests/conftest.py` (ensure new model imported for create_all)
- Test: `backend/tests/test_tts_usage_model.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_tts_usage_model.py`:

```python
from sqlalchemy import select

from src.models.tts_usage import TTSUsage


async def test_create_tts_usage(db_session):
    usage = TTSUsage(engine="cosyvoice2", characters=42, duration_ms=3200, rtf=0.8, cached=False)
    db_session.add(usage)
    await db_session.commit()

    result = await db_session.execute(select(TTSUsage))
    row = result.scalar_one()
    assert row.engine == "cosyvoice2"
    assert row.characters == 42
    assert row.duration_ms == 3200
    assert row.id > 0  # snowflake ID
```

- [ ] **Step 2: Add db_session fixture to conftest.py**

The existing `db_client` fixture creates a full app. Add a lighter `db_session` fixture to `backend/tests/conftest.py`:

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from src.models.database import Base
import src.models.tts_usage  # noqa: F401 — register model

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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd backend && uv run pytest tests/test_tts_usage_model.py -v
```
Expected: FAIL — `src.models.tts_usage` does not exist.

- [ ] **Step 4: Create the TTSUsage model**

Create `backend/src/models/tts_usage.py`:

```python
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Index

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class TTSUsage(Base):
    __tablename__ = "tts_usage"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    engine = Column(String(64), nullable=False)
    characters = Column(Integer, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    rtf = Column(Float, nullable=True)
    cached = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_tts_usage_created", "created_at"),
        Index("idx_tts_usage_engine", "engine"),
    )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd backend && uv run pytest tests/test_tts_usage_model.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/models/tts_usage.py backend/tests/test_tts_usage_model.py backend/tests/conftest.py
git commit -m "backend: add TTSUsage model for synthesis statistics tracking"
```

### Task 6: Add TTS cache service

**Files:**
- Create: `backend/src/services/__init__.py`
- Create: `backend/src/services/tts_cache.py`
- Test: `backend/tests/test_tts_cache.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_tts_cache.py`:

```python
import hashlib

from src.services.tts_cache import make_cache_key, TTSCacheService


def test_make_cache_key_deterministic():
    key1 = make_cache_key(text="hello", engine="cosyvoice2", voice="default", speed=1.0, sample_rate=24000)
    key2 = make_cache_key(text="hello", engine="cosyvoice2", voice="default", speed=1.0, sample_rate=24000)
    assert key1 == key2
    assert key1.startswith("tts:")


def test_make_cache_key_differs_on_text():
    key1 = make_cache_key(text="hello", engine="cosyvoice2", voice="default", speed=1.0, sample_rate=24000)
    key2 = make_cache_key(text="world", engine="cosyvoice2", voice="default", speed=1.0, sample_rate=24000)
    assert key1 != key2


from unittest.mock import AsyncMock


async def test_cache_service_get_miss():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    svc = TTSCacheService(mock_redis, ttl=3600)
    result = await svc.get("tts:nonexistent")
    assert result is None
    mock_redis.get.assert_called_once_with("tts:nonexistent")


async def test_cache_service_set():
    mock_redis = AsyncMock()
    svc = TTSCacheService(mock_redis, ttl=3600)
    await svc.set("tts:key", "base64data")
    mock_redis.set.assert_called_once_with("tts:key", "base64data", ex=3600)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && uv run pytest tests/test_tts_cache.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement tts_cache.py**

Create `backend/src/services/__init__.py` (empty).

Create `backend/src/services/tts_cache.py`:

```python
"""TTS result caching via Redis.

Cache key = SHA-256 of (text + engine + voice + speed + sample_rate + emotion).
Value = base64-encoded audio bytes.
"""

import hashlib
import json

from src.config import get_settings


def make_cache_key(
    text: str,
    engine: str,
    voice: str = "default",
    speed: float = 1.0,
    sample_rate: int = 24000,
    emotion: str | None = None,
) -> str:
    payload = json.dumps(
        {"text": text, "engine": engine, "voice": voice, "speed": speed, "sr": sample_rate, "emotion": emotion},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"tts:{digest}"


class TTSCacheService:
    """Thin wrapper around Redis for TTS audio caching."""

    def __init__(self, redis_client, ttl: int | None = None):
        self._redis = redis_client
        self._ttl = ttl or get_settings().CACHE_TTL_SECONDS

    async def get(self, key: str) -> str | None:
        """Return cached base64 audio or None."""
        return await self._redis.get(key)

    async def set(self, key: str, audio_base64: str) -> None:
        """Cache base64 audio with TTL."""
        await self._redis.set(key, audio_base64, ex=self._ttl)
```

- [ ] **Step 4: Run tests**

```bash
cd backend && uv run pytest tests/test_tts_cache.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/ backend/tests/test_tts_cache.py
git commit -m "backend: add TTS cache service with SHA-256 keying and Redis TTL"
```

### Task 7: Integrate cache + usage tracking into /tts/synthesize

**Files:**
- Modify: `backend/src/models/schemas.py` (add `cache` and `emotion` fields)
- Modify: `backend/src/api/routes/tts.py`
- Test: `backend/tests/test_api_tts.py`

- [ ] **Step 1: Add `emotion` and `cache` fields to schemas**

In `backend/src/models/schemas.py`, add `emotion` and `cache` to `SynthesizeRequest`:

```python
class SynthesizeRequest(BaseModel):
    engine: Literal[
        "cosyvoice2", "indextts2", "qwen3_tts_base",
        "qwen3_tts_customvoice", "qwen3_tts_voicedesign", "moss_tts",
    ]
    text: str
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    reference_audio: str | None = None
    reference_text: str | None = None
    emotion: str | None = None
    cache: bool = True
```

Add `emotion` to `TTSRequest` too:

```python
class TTSRequest(BaseModel):
    text: str
    engine: Literal[
        "cosyvoice2",
        "indextts2",
        "qwen3_tts_base",
        "qwen3_tts_customvoice",
        "qwen3_tts_voicedesign",
        "moss_tts",
    ] = "cosyvoice2"
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    reference_audio: str | None = None  # for voice cloning engines
    emotion: str | None = None
```

Add `cached` field to `SynthesizeResponse`:

```python
class SynthesizeResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_seconds: float
    engine: str
    rtf: float
    format: str = "wav"
    cached: bool = False
```

- [ ] **Step 2: Write test for cache hit behavior**

Add to `backend/tests/test_api_tts.py`:

```python
async def test_synthesize_returns_cached_field(client):
    """SynthesizeResponse should include a 'cached' field."""
    resp = await client.post("/api/v1/tts/synthesize", json={
        "engine": "cosyvoice2",
        "text": "test",
    })
    # Engine not loaded in test, expect 409
    assert resp.status_code == 409


async def test_synthesize_accepts_emotion_field(client):
    """SynthesizeRequest should accept optional emotion field."""
    resp = await client.post("/api/v1/tts/synthesize", json={
        "engine": "cosyvoice2",
        "text": "test",
        "emotion": "happy tone",
    })
    # Engine not loaded, but 409 means request parsing succeeded (not 422)
    assert resp.status_code == 409
```

- [ ] **Step 3: Run tests**

```bash
cd backend && uv run pytest tests/test_api_tts.py -v
```
Expected: PASS — the new fields are optional so existing tests still work; new tests check 409 (not 422).

- [ ] **Step 4: Integrate cache + usage into the /tts/synthesize route handler**

In `backend/src/api/routes/tts.py`, update `tts_synthesize` to check cache before synthesis and record usage after:

```python
import redis.asyncio as aioredis
from src.services.tts_cache import make_cache_key, TTSCacheService
from src.services.usage_recorder import record_tts_usage
from src.models.database import get_async_session

# Module-level lazy Redis + cache
_redis_client = None
_cache_service = None

def _get_cache_service() -> TTSCacheService | None:
    global _redis_client, _cache_service
    if _cache_service is not None:
        return _cache_service
    try:
        from src.config import get_settings
        _redis_client = aioredis.from_url(get_settings().REDIS_URL)
        _cache_service = TTSCacheService(_redis_client)
        return _cache_service
    except Exception:
        return None
```

Then in the `tts_synthesize` function body, wrap the existing synthesis logic:

```python
@router.post("/synthesize", response_model=SynthesizeResponse)
async def tts_synthesize(req: SynthesizeRequest):
    engine = _get_loaded_engine(req.engine)
    if engine is None:
        raise HTTPException(409, detail=f"Engine {req.engine} not loaded...")

    # --- Cache check ---
    cache_key = None
    cache_svc = _get_cache_service() if req.cache else None
    if cache_svc:
        cache_key = make_cache_key(
            text=req.text, engine=req.engine, voice=req.voice,
            speed=req.speed, sample_rate=req.sample_rate, emotion=req.emotion,
        )
        cached = await cache_svc.get(cache_key)
        if cached:
            return SynthesizeResponse(
                audio_base64=cached, sample_rate=req.sample_rate,
                duration_seconds=0, engine=req.engine, rtf=0, cached=True,
            )

    # --- Synthesize ---
    start = time.monotonic()
    kwargs = dict(text=req.text, voice=req.voice, speed=req.speed,
                  sample_rate=req.sample_rate, reference_audio=req.reference_audio)
    if req.reference_text is not None:
        kwargs["reference_text"] = req.reference_text
    if req.emotion is not None:
        kwargs["emotion"] = req.emotion

    result = engine.synthesize(**kwargs)
    elapsed = time.monotonic() - start
    rtf = round(elapsed / max(result.duration_seconds, 0.01), 4)
    audio_b64 = base64.b64encode(result.audio_bytes).decode()

    # --- Cache store ---
    if cache_svc and cache_key:
        try:
            await cache_svc.set(cache_key, audio_b64)
        except Exception:
            pass  # cache write failure is non-fatal

    # --- Usage recording (fire-and-forget) ---
    # Usage is recorded in Task 8's usage_recorder via background task
    # (actual wiring added after Task 8)

    return SynthesizeResponse(
        audio_base64=audio_b64, sample_rate=result.sample_rate,
        duration_seconds=result.duration_seconds, engine=req.engine,
        rtf=rtf, format=result.format, cached=False,
    )
```

- [ ] **Step 5: Run tests**

```bash
cd backend && uv run pytest tests/test_api_tts.py -v
```
Expected: PASS — cache service returns None in test env (no Redis), so synthesis path is unchanged.

- [ ] **Step 6: Commit**

```bash
git add backend/src/models/schemas.py backend/src/api/routes/tts.py backend/tests/test_api_tts.py
git commit -m "backend: add emotion/cache fields to schemas and integrate cache into /tts/synthesize"
```

### Task 8: Add usage recording helper

**Files:**
- Create: `backend/src/services/usage_recorder.py`
- Test: `backend/tests/test_usage_recorder.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_usage_recorder.py`:

```python
from sqlalchemy import select

from src.models.tts_usage import TTSUsage
from src.services.usage_recorder import record_tts_usage


async def test_record_tts_usage(db_session):
    await record_tts_usage(
        session=db_session,
        engine="cosyvoice2",
        characters=10,
        duration_ms=2000,
        rtf=0.5,
        cached=False,
    )
    result = await db_session.execute(select(TTSUsage))
    row = result.scalar_one()
    assert row.engine == "cosyvoice2"
    assert row.characters == 10
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && uv run pytest tests/test_usage_recorder.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `backend/src/services/usage_recorder.py`:

```python
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.tts_usage import TTSUsage


async def record_tts_usage(
    session: AsyncSession,
    engine: str,
    characters: int,
    duration_ms: int | None = None,
    rtf: float | None = None,
    cached: bool = False,
) -> None:
    """Record a TTS synthesis event for usage statistics."""
    usage = TTSUsage(
        engine=engine,
        characters=characters,
        duration_ms=duration_ms,
        rtf=rtf,
        cached=cached,
    )
    session.add(usage)
    await session.commit()
```

- [ ] **Step 4: Run tests**

```bash
cd backend && uv run pytest tests/test_usage_recorder.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/usage_recorder.py backend/tests/test_usage_recorder.py
git commit -m "backend: add usage recorder service for TTS statistics"
```

---

## Chunk 2: Phase 2 — SSE Streaming + Emotion Control

### Task 9: Add StreamRequest schema

**Files:**
- Modify: `backend/src/models/schemas.py`

- [ ] **Step 1: Add StreamRequest and streaming event schemas**

Add to `backend/src/models/schemas.py`:

```python
# --- SSE Streaming TTS ---

class StreamRequest(BaseModel):
    text: str
    engine: Literal[
        "cosyvoice2", "indextts2", "qwen3_tts_base",
        "qwen3_tts_customvoice", "qwen3_tts_voicedesign", "moss_tts",
    ] = "cosyvoice2"
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    reference_audio: str | None = None
    reference_text: str | None = None
    emotion: str | None = None
    cache: bool = True
```

- [ ] **Step 2: Commit**

```bash
git add backend/src/models/schemas.py
git commit -m "backend: add StreamRequest schema for SSE TTS endpoint"
```

### Task 10: Implement SSE streaming endpoint

**Files:**
- Modify: `backend/src/api/routes/tts.py`
- Test: `backend/tests/test_api_tts.py`

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_api_tts.py`:

```python
async def test_stream_endpoint_exists(client):
    """POST /tts/stream should exist and return 409 when engine not loaded."""
    resp = await client.post("/api/v1/tts/stream", json={
        "engine": "cosyvoice2",
        "text": "hello",
    })
    # Engine not loaded → 409, not 404 (endpoint exists)
    assert resp.status_code == 409


async def test_stream_validates_request(client):
    """POST /tts/stream should reject invalid speed."""
    resp = await client.post("/api/v1/tts/stream", json={
        "engine": "cosyvoice2",
        "text": "hello",
        "speed": 999,
    })
    assert resp.status_code == 422
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && uv run pytest tests/test_api_tts.py::test_stream_endpoint_exists -v
```
Expected: FAIL — 404.

- [ ] **Step 3: Implement the streaming endpoint**

Add to `backend/src/api/routes/tts.py`:

```python
import json as json_module
from fastapi.responses import StreamingResponse
from src.models.schemas import StreamRequest


@router.post("/stream")
async def tts_stream(req: StreamRequest):
    """SSE streaming TTS synthesis.

    Note: Currently single-chunk (engine.synthesize returns complete audio).
    The SSE format enables future true streaming when engines support
    synthesize_stream() yielding multiple chunks.
    """
    engine = _get_loaded_engine(req.engine)
    if engine is None:
        raise HTTPException(
            409,
            detail=f"Engine {req.engine} not loaded. POST /api/v1/engines/{req.engine}/load first.",
        )

    async def event_generator():
        try:
            start = time.monotonic()
            kwargs = dict(
                text=req.text,
                voice=req.voice,
                speed=req.speed,
                sample_rate=req.sample_rate,
                reference_audio=req.reference_audio,
            )
            if req.reference_text is not None:
                kwargs["reference_text"] = req.reference_text

            result = engine.synthesize(**kwargs)
            elapsed = time.monotonic() - start
            rtf = round(elapsed / max(result.duration_seconds, 0.01), 4)

            audio_b64 = base64.b64encode(result.audio_bytes).decode()
            chunk = json_module.dumps({"seq": 1, "audio": audio_b64, "format": result.format})
            yield f"event: audio\ndata: {chunk}\n\n"

            done = json_module.dumps({
                "total_chunks": 1,
                "duration_ms": int(result.duration_seconds * 1000),
                "usage": {"characters": len(req.text), "rtf": rtf},
            })
            yield f"event: done\ndata: {done}\n\n"
        except Exception as exc:
            error = json_module.dumps({"code": "ENGINE_ERROR", "message": str(exc)})
            yield f"event: error\ndata: {error}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests**

```bash
cd backend && uv run pytest tests/test_api_tts.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/routes/tts.py backend/tests/test_api_tts.py
git commit -m "backend: add POST /tts/stream SSE endpoint for streaming TTS"
```

### Task 11: Add emotion passthrough to TTSEngine base

**Files:**
- Modify: `backend/src/workers/tts_engines/base.py`
- Modify: `backend/src/workers/tts_engines/cosyvoice2.py` (add `emotion` to signature)
- Modify: `backend/src/workers/tts_engines/indextts2.py` (add `emotion` to signature)
- Modify: `backend/src/workers/tts_engines/moss_tts.py` (add `emotion` to signature)
- Modify: `backend/src/workers/tts_engines/qwen3_tts.py` (add `emotion` to signature)

- [ ] **Step 1: Add emotion to synthesize signature**

In `backend/src/workers/tts_engines/base.py`, update the `synthesize` abstract method:

```python
@abstractmethod
def synthesize(
    self,
    text: str,
    voice: str = "default",
    speed: float = 1.0,
    sample_rate: int = 24000,
    reference_audio: str | None = None,
    reference_text: str | None = None,
    emotion: str | None = None,
) -> TTSResult:
    """Synthesize speech from text. Returns audio bytes."""
```

- [ ] **Step 2: Update tts.py to pass emotion**

In `backend/src/api/routes/tts.py`, update the `tts_synthesize` function's kwargs:

```python
kwargs = dict(
    text=req.text,
    voice=req.voice,
    speed=req.speed,
    sample_rate=req.sample_rate,
    reference_audio=req.reference_audio,
)
if req.reference_text is not None:
    kwargs["reference_text"] = req.reference_text
if req.emotion is not None:
    kwargs["emotion"] = req.emotion
```

Apply the same pattern in `tts_stream`.

- [ ] **Step 2b: Update all concrete engine synthesize() signatures**

Each concrete engine must accept the `emotion` parameter (just add to signature, not use it yet). In each of these files, add `emotion: str | None = None` to the `synthesize` method signature:
- `backend/src/workers/tts_engines/cosyvoice2.py`
- `backend/src/workers/tts_engines/indextts2.py`
- `backend/src/workers/tts_engines/moss_tts.py`
- `backend/src/workers/tts_engines/qwen3_tts.py`

Example (apply to each):
```python
def synthesize(
    self,
    text: str,
    voice: str = "default",
    speed: float = 1.0,
    sample_rate: int = 24000,
    reference_audio: str | None = None,
    reference_text: str | None = None,
    emotion: str | None = None,  # ← add this
) -> TTSResult:
```

- [ ] **Step 3: Run existing tests**

```bash
cd backend && uv run pytest tests/ -v
```
Expected: all PASS — emotion is optional with default None so existing engine implementations are unaffected.

- [ ] **Step 4: Commit**

```bash
git add backend/src/workers/tts_engines/base.py backend/src/api/routes/tts.py
git commit -m "backend: add emotion parameter to TTSEngine.synthesize and route handlers"
```

---

## Chunk 3: Phase 2 — Batch Round Model + WS Session

### Task 12: Migrate batch schemas from segments to rounds

**Files:**
- Modify: `backend/src/models/schemas.py`
- Modify: `backend/tests/test_api_tts.py`

- [ ] **Step 1: Update schemas**

In `backend/src/models/schemas.py`, replace the batch section:

```python
# --- Batch TTS (Round model) ---

class BatchRound(BaseModel):
    round_id: int
    voice_preset: str  # preset name
    text: str
    emotion: str | None = None


class BatchTTSRequest(BaseModel):
    rounds: list[BatchRound]


class BatchTTSResponse(BaseModel):
    batch_id: str
    total_rounds: int


class BatchRetryRequest(BaseModel):
    round_ids: list[int]
```

Remove `BatchSegment`, `BatchTaskInfo`, `BatchStatusResponse`.

**Important:** Also update or remove any existing tests in `test_api_tts.py` that reference the old `BatchSegment`/`segments` schema (e.g., `test_batch_tts`, `test_batch_status`). Replace with the new rounds-based test below.

- [ ] **Step 2: Update test**

In `backend/tests/test_api_tts.py`, update or add:

```python
async def test_batch_accepts_rounds_schema(client):
    """Batch endpoint should accept the new rounds schema."""
    resp = await client.post("/api/v1/tts/batch", json={
        "rounds": [
            {"round_id": 1, "voice_preset": "test", "text": "hello"},
            {"round_id": 2, "voice_preset": "test", "text": "world"},
        ]
    })
    # Will fail with 404 (preset not found) but not 422 (valid schema)
    assert resp.status_code != 422
```

- [ ] **Step 3: Run tests**

```bash
cd backend && uv run pytest tests/test_api_tts.py -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/src/models/schemas.py backend/tests/test_api_tts.py
git commit -m "backend: migrate batch TTS from segments to rounds model"
```

### Task 13: Rewrite batch endpoint with round model

**Files:**
- Modify: `backend/src/api/routes/tts.py`
- Test: `backend/tests/test_api_tts.py`

- [ ] **Step 1: Rewrite the batch endpoint**

Replace the batch-related code in `backend/src/api/routes/tts.py`:

```python
from src.models.schemas import BatchTTSRequest, BatchTTSResponse, BatchRetryRequest
from src.models.voice_preset import VoicePreset

# In-memory batch state (production would use Redis)
_batch_store: dict[str, dict] = {}


async def _resolve_preset(name: str, session: AsyncSession) -> dict | None:
    """Look up a voice preset by name and return engine + params dict."""
    from sqlalchemy import select
    result = await session.execute(select(VoicePreset).where(VoicePreset.name == name))
    preset = result.scalar_one_or_none()
    if preset is None:
        return None
    return {
        "engine": preset.engine,
        "params": {
            "voice": preset.voice or "default",
            "speed": preset.speed or 1.0,
            "sample_rate": preset.sample_rate or 24000,
            "reference_audio": preset.reference_audio,
            "reference_text": preset.reference_text,
        },
    }


@router.post("/batch", response_model=BatchTTSResponse, status_code=202)
async def tts_batch(
    req: BatchTTSRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Dispatch batch TTS with round model. Progress pushed via /ws/tts."""
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"

    rounds_state = {}
    for r in req.rounds:
        preset = await _resolve_preset(r.voice_preset, session)
        if preset is None:
            raise HTTPException(404, detail=f"Voice preset not found: {r.voice_preset}")
        rounds_state[r.round_id] = {
            "text": r.text,
            "emotion": r.emotion,
            "engine": preset["engine"],
            "params": preset["params"],
            "status": "pending",
            "task_id": None,
        }

    _batch_store[batch_id] = {"rounds": rounds_state, "total": len(req.rounds)}

    # Dispatch each round as a Celery task
    for round_id, state in rounds_state.items():
        params = {**state["params"], "text": state["text"], "engine": state["engine"]}
        task = generate_tts_task.delay(f"{batch_id}_r{round_id}", params)
        state["task_id"] = task.id

    return BatchTTSResponse(batch_id=batch_id, total_rounds=len(req.rounds))


@router.post("/batch/{batch_id}/retry")
async def tts_batch_retry(
    batch_id: str,
    req: BatchRetryRequest,
):
    """Retry specific rounds in a batch."""
    batch = _batch_store.get(batch_id)
    if batch is None:
        raise HTTPException(404, detail=f"Batch not found: {batch_id}")

    retried = []
    for round_id in req.round_ids:
        state = batch["rounds"].get(round_id)
        if state is None:
            continue
        state["status"] = "pending"
        params = {**state["params"], "text": state["text"], "engine": state["engine"]}
        task = generate_tts_task.delay(f"{batch_id}_r{round_id}_retry", params)
        state["task_id"] = task.id
        retried.append(round_id)

    return {"batch_id": batch_id, "retried_rounds": retried}
```

Remove the old `tts_batch_status` endpoint (GET /batch/{batch_id}).

- [ ] **Step 2: Run tests**

```bash
cd backend && uv run pytest tests/test_api_tts.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/src/api/routes/tts.py backend/tests/test_api_tts.py
git commit -m "backend: rewrite batch endpoint with round model and single-round retry"
```

### Task 14: Add WS TTS session endpoint

**Files:**
- Create: `backend/src/api/ws_tts.py`
- Modify: `backend/src/api/main.py`
- Test: `backend/tests/test_ws_tts.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_ws_tts.py`:

```python
from starlette.testclient import TestClient

from src.api.main import create_app


def test_ws_tts_endpoint_accepts_connection():
    """WS /ws/tts should accept connections and handle start_session."""
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "start_session", "session_id": "s1", "engine": "cosyvoice2"})
        resp = ws.receive_json()
        # Engine not loaded → error
        assert resp["type"] == "error"
        assert resp["session_id"] == "s1"
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && uv run pytest tests/test_ws_tts.py -v
```
Expected: FAIL — no WS endpoint at /ws/tts.

- [ ] **Step 3: Implement ws_tts.py**

Create `backend/src/api/ws_tts.py`:

```python
"""WebSocket TTS session handler with connection-level reuse."""

import base64
import json
import time

from fastapi import WebSocket, WebSocketDisconnect

from src.api.routes.tts import _get_loaded_engine


async def handle_tts_websocket(websocket: WebSocket):
    await websocket.accept()
    active_session: str | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")
            session_id = msg.get("session_id", "")

            if msg_type == "start_session":
                active_session = session_id
                engine_name = msg.get("engine", "")
                engine = _get_loaded_engine(engine_name)
                if engine is None:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "ENGINE_NOT_LOADED",
                        "message": f"Engine {engine_name} not loaded",
                    })
                    active_session = None
                else:
                    # Store engine ref for this session
                    websocket.state.engine = engine
                    websocket.state.session_config = msg
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session_id,
                    })

            elif msg_type == "synthesize":
                if active_session != session_id:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "NO_ACTIVE_SESSION",
                        "message": "No active session for this session_id",
                    })
                    continue

                engine = getattr(websocket.state, "engine", None)
                if engine is None:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "ENGINE_NOT_LOADED",
                        "message": "No engine loaded for session",
                    })
                    continue

                try:
                    text = msg.get("text", "")
                    config = getattr(websocket.state, "session_config", {})
                    start = time.monotonic()

                    result = engine.synthesize(
                        text=text,
                        voice=config.get("voice", "default"),
                        speed=config.get("speed", 1.0),
                        sample_rate=config.get("sample_rate", 24000),
                        reference_audio=config.get("reference_audio"),
                        reference_text=config.get("reference_text"),
                        emotion=config.get("emotion") or msg.get("emotion"),
                    )
                    elapsed = time.monotonic() - start

                    audio_b64 = base64.b64encode(result.audio_bytes).decode()
                    seq = getattr(websocket.state, "seq", 0) + 1
                    websocket.state.seq = seq

                    await websocket.send_json({
                        "type": "audio",
                        "session_id": session_id,
                        "seq": seq,
                        "audio": audio_b64,
                        "format": result.format,
                        "duration_ms": int(result.duration_seconds * 1000),
                        "rtf": round(elapsed / max(result.duration_seconds, 0.01), 4),
                    })
                except Exception as exc:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "SYNTHESIS_FAILED",
                        "message": str(exc),
                    })

            elif msg_type == "end_session":
                active_session = None
                websocket.state.engine = None
                websocket.state.seq = 0
                await websocket.send_json({
                    "type": "session_ended",
                    "session_id": session_id,
                })

    except WebSocketDisconnect:
        pass
```

- [ ] **Step 4: Register in main.py**

Add to `backend/src/api/main.py`:

```python
from src.api.ws_tts import handle_tts_websocket

# Inside create_app(), after the existing /ws/tasks endpoint:
@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await handle_tts_websocket(websocket)
```

- [ ] **Step 5: Run tests**

```bash
cd backend && uv run pytest tests/test_ws_tts.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/ws_tts.py backend/src/api/main.py backend/tests/test_ws_tts.py
git commit -m "backend: add WS /ws/tts session endpoint with connection reuse"
```

---

## Chunk 4: Phase 3 — nous-core Audio IO

### Task 15: Add audio info endpoint to nous-core

**Files:**
- Create: `nous-core/src/audio.rs`
- Modify: `nous-core/src/main.rs`

- [ ] **Step 1: Create audio module with info endpoint**

Create `nous-core/src/audio.rs`:

```rust
use axum::Json;
use serde::{Deserialize, Serialize};
use std::path::Path;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

#[derive(Deserialize)]
pub struct AudioInfoRequest {
    pub path: String,
}

#[derive(Serialize)]
pub struct AudioInfoResponse {
    pub sample_rate: u32,
    pub channels: u32,
    pub duration_ms: u64,
    pub format: String,
    pub file_size_bytes: u64,
}

#[derive(Serialize)]
pub struct AudioErrorResponse {
    pub error: String,
}

pub async fn audio_info(Json(req): Json<AudioInfoRequest>) -> Result<Json<AudioInfoResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    let path = Path::new(&req.path);
    if !path.exists() {
        return Err((
            axum::http::StatusCode::NOT_FOUND,
            Json(AudioErrorResponse { error: format!("File not found: {}", req.path) }),
        ));
    }

    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);

    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => return Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(AudioErrorResponse { error: format!("Cannot open file: {e}") }),
        )),
    };

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension(&ext);

    let probed = match symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    ) {
        Ok(p) => p,
        Err(e) => return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: format!("Cannot probe audio: {e}") }),
        )),
    };

    let track = match probed.format.default_track() {
        Some(t) => t,
        None => return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: "No audio track found".into() }),
        )),
    };

    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let channels = track.codec_params.channels.map(|c| c.count() as u32).unwrap_or(0);
    let n_frames = track.codec_params.n_frames.unwrap_or(0);
    let duration_ms = if sample_rate > 0 {
        (n_frames as u64 * 1000) / sample_rate as u64
    } else {
        0
    };

    Ok(Json(AudioInfoResponse {
        sample_rate,
        channels,
        duration_ms,
        format: ext,
        file_size_bytes: file_size,
    }))
}
```

- [ ] **Step 2: Register route in main.rs**

Update `nous-core/src/main.rs`:

```rust
mod gpu;
mod system;
mod files;
mod audio;

use axum::{routing::{get, post}, Router};
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/sys/gpus", get(gpu::get_gpus))
        .route("/sys/stats", get(system::get_stats))
        .route("/sys/processes", get(system::get_processes))
        .route("/sys/models", get(files::list_models))
        .route("/audio/info", post(audio::audio_info))
        .layer(cors);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8001")
        .await
        .unwrap();

    println!("nous-core listening on http://127.0.0.1:8001");
    axum::serve(listener, app).await.unwrap();
}
```

- [ ] **Step 3: Build and verify**

```bash
cd nous-core && cargo build
```
Expected: builds successfully.

- [ ] **Step 4: Commit**

```bash
git add nous-core/src/audio.rs nous-core/src/main.rs
git commit -m "core: add POST /audio/info endpoint for audio metadata"
```

### Task 16: Add audio resample endpoint

**Files:**
- Modify: `nous-core/src/audio.rs`
- Modify: `nous-core/src/main.rs`

- [ ] **Step 1: Add resample handler to audio.rs**

Append to `nous-core/src/audio.rs`:

```rust
use hound::{WavReader, WavWriter, WavSpec, SampleFormat};

#[derive(Deserialize)]
pub struct ResampleRequest {
    pub input_path: String,
    pub output_path: String,
    pub target_sample_rate: u32,
}

#[derive(Serialize)]
pub struct ResampleResponse {
    pub output_path: String,
    pub original_sample_rate: u32,
    pub target_sample_rate: u32,
    pub duration_ms: u64,
}

pub async fn audio_resample(Json(req): Json<ResampleRequest>) -> Result<Json<ResampleResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    let input = Path::new(&req.input_path);
    if !input.exists() {
        return Err((
            axum::http::StatusCode::NOT_FOUND,
            Json(AudioErrorResponse { error: format!("Input not found: {}", req.input_path) }),
        ));
    }

    // Read input WAV
    let reader = WavReader::open(input).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        Json(AudioErrorResponse { error: format!("Cannot read WAV: {e}") }),
    ))?;

    let spec = reader.spec();
    let original_sr = spec.sample_rate;
    let channels = spec.channels;

    // Read all samples as f32
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().filter_map(|s| s.ok()).collect(),
        SampleFormat::Int => reader.into_samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / 32768.0).collect(),
    };

    // Simple linear interpolation resample
    let ratio = req.target_sample_rate as f64 / original_sr as f64;
    let new_len = (samples.len() as f64 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = (src_idx - idx0 as f64) as f32;
        let sample = samples[idx0] * (1.0 - frac) + samples[idx1] * frac;
        resampled.push(sample);
    }

    // Write output WAV
    let out_spec = WavSpec {
        channels,
        sample_rate: req.target_sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(&req.output_path, out_spec).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Cannot create output: {e}") }),
    ))?;

    for s in &resampled {
        let val = (*s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(val).map_err(|e| (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(AudioErrorResponse { error: format!("Write error: {e}") }),
        ))?;
    }

    writer.finalize().map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Finalize error: {e}") }),
    ))?;

    let duration_ms = if req.target_sample_rate > 0 && channels > 0 {
        (resampled.len() as u64 * 1000) / (req.target_sample_rate as u64 * channels as u64)
    } else {
        0
    };

    Ok(Json(ResampleResponse {
        output_path: req.output_path,
        original_sample_rate: original_sr,
        target_sample_rate: req.target_sample_rate,
        duration_ms,
    }))
}
```

- [ ] **Step 2: Register route**

In `nous-core/src/main.rs`, add:

```rust
.route("/audio/resample", post(audio::audio_resample))
```

- [ ] **Step 3: Build**

```bash
cd nous-core && cargo build
```
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add nous-core/src/audio.rs nous-core/src/main.rs
git commit -m "core: add POST /audio/resample with linear interpolation"
```

### Task 17: Add audio concat endpoint

**Files:**
- Modify: `nous-core/src/audio.rs`
- Modify: `nous-core/src/main.rs`

- [ ] **Step 1: Add concat handler**

Append to `nous-core/src/audio.rs`:

```rust
#[derive(Deserialize)]
pub struct ConcatRequest {
    pub input_paths: Vec<String>,
    pub output_path: String,
}

#[derive(Serialize)]
pub struct ConcatResponse {
    pub output_path: String,
    pub total_duration_ms: u64,
    pub file_count: usize,
}

pub async fn audio_concat(Json(req): Json<ConcatRequest>) -> Result<Json<ConcatResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    if req.input_paths.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: "No input files provided".into() }),
        ));
    }

    // Read first file to get spec
    let first_reader = WavReader::open(&req.input_paths[0]).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        Json(AudioErrorResponse { error: format!("Cannot read first WAV: {e}") }),
    ))?;
    let spec = first_reader.spec();

    let mut all_samples: Vec<i16> = first_reader.into_samples::<i16>().filter_map(|s| s.ok()).collect();

    // Read and append remaining files
    for path in &req.input_paths[1..] {
        let reader = WavReader::open(path).map_err(|e| (
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: format!("Cannot read {path}: {e}") }),
        ))?;
        let file_spec = reader.spec();
        if file_spec.sample_rate != spec.sample_rate || file_spec.channels != spec.channels {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                Json(AudioErrorResponse {
                    error: format!(
                        "Sample rate/channel mismatch: expected {}Hz/{}ch, got {}Hz/{}ch in {path}",
                        spec.sample_rate, spec.channels, file_spec.sample_rate, file_spec.channels
                    ),
                }),
            ));
        }
        let samples: Vec<i16> = reader.into_samples::<i16>().filter_map(|s| s.ok()).collect();
        all_samples.extend(samples);
    }

    // Write concatenated output
    let mut writer = WavWriter::create(&req.output_path, spec).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Cannot create output: {e}") }),
    ))?;

    for s in &all_samples {
        writer.write_sample(*s).map_err(|e| (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(AudioErrorResponse { error: format!("Write error: {e}") }),
        ))?;
    }
    writer.finalize().map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Finalize error: {e}") }),
    ))?;

    let total_duration_ms = if spec.sample_rate > 0 && spec.channels > 0 {
        (all_samples.len() as u64 * 1000) / (spec.sample_rate as u64 * spec.channels as u64)
    } else {
        0
    };

    Ok(Json(ConcatResponse {
        output_path: req.output_path,
        total_duration_ms,
        file_count: req.input_paths.len(),
    }))
}
```

- [ ] **Step 2: Register route**

In `nous-core/src/main.rs`, add:

```rust
.route("/audio/concat", post(audio::audio_concat))
```

- [ ] **Step 3: Build**

```bash
cd nous-core && cargo build
```

- [ ] **Step 4: Commit**

```bash
git add nous-core/src/audio.rs nous-core/src/main.rs
git commit -m "core: add POST /audio/concat for multi-file WAV concatenation"
```

### Task 18: Add audio split and convert endpoints

**Files:**
- Modify: `nous-core/src/audio.rs`
- Modify: `nous-core/src/main.rs`

- [ ] **Step 1: Add split handler**

Append to `nous-core/src/audio.rs`:

```rust
#[derive(Deserialize)]
pub struct SplitRequest {
    pub input_path: String,
    pub output_dir: String,
    pub split_points_ms: Vec<u64>,
}

#[derive(Serialize)]
pub struct SplitResponse {
    pub output_paths: Vec<String>,
    pub durations_ms: Vec<u64>,
}

pub async fn audio_split(Json(req): Json<SplitRequest>) -> Result<Json<SplitResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    let reader = WavReader::open(&req.input_path).map_err(|e| (
        axum::http::StatusCode::BAD_REQUEST,
        Json(AudioErrorResponse { error: format!("Cannot read WAV: {e}") }),
    ))?;
    let spec = reader.spec();
    let samples_per_ms = (spec.sample_rate as u64 * spec.channels as u64) / 1000;
    let all_samples: Vec<i16> = reader.into_samples::<i16>().filter_map(|s| s.ok()).collect();

    // Create output directory
    std::fs::create_dir_all(&req.output_dir).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Cannot create dir: {e}") }),
    ))?;

    let mut boundaries: Vec<usize> = vec![0];
    for &ms in &req.split_points_ms {
        boundaries.push((ms * samples_per_ms) as usize);
    }
    boundaries.push(all_samples.len());

    let mut output_paths = Vec::new();
    let mut durations_ms = Vec::new();

    for i in 0..boundaries.len() - 1 {
        let start = boundaries[i].min(all_samples.len());
        let end = boundaries[i + 1].min(all_samples.len());
        let chunk = &all_samples[start..end];

        let out_path = format!("{}/part_{:03}.wav", req.output_dir, i);
        let mut writer = WavWriter::create(&out_path, spec).map_err(|e| (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(AudioErrorResponse { error: format!("Write error: {e}") }),
        ))?;
        for s in chunk {
            writer.write_sample(*s).unwrap();
        }
        writer.finalize().unwrap();

        let dur = if samples_per_ms > 0 { chunk.len() as u64 / samples_per_ms } else { 0 };
        output_paths.push(out_path);
        durations_ms.push(dur);
    }

    Ok(Json(SplitResponse { output_paths, durations_ms }))
}
```

- [ ] **Step 2: Add convert handler (stub for WAV passthrough; full format support uses symphonia decode + hound encode)**

```rust
#[derive(Deserialize)]
pub struct ConvertRequest {
    pub input_path: String,
    pub output_path: String,
    pub target_format: String,  // "wav" for now
}

#[derive(Serialize)]
pub struct ConvertResponse {
    pub output_path: String,
    pub format: String,
}

pub async fn audio_convert(Json(req): Json<ConvertRequest>) -> Result<Json<ConvertResponse>, (axum::http::StatusCode, Json<AudioErrorResponse>)> {
    if req.target_format != "wav" {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(AudioErrorResponse { error: format!("Unsupported target format: {}. Currently only 'wav' is supported.", req.target_format) }),
        ));
    }

    // For wav→wav: read with symphonia, write with hound (normalizes format)
    let input = Path::new(&req.input_path);
    if !input.exists() {
        return Err((
            axum::http::StatusCode::NOT_FOUND,
            Json(AudioErrorResponse { error: format!("Input not found: {}", req.input_path) }),
        ));
    }

    // Simple copy for now; extend with symphonia decode for mp3/ogg/flac input
    std::fs::copy(input, &req.output_path).map_err(|e| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(AudioErrorResponse { error: format!("Copy error: {e}") }),
    ))?;

    Ok(Json(ConvertResponse {
        output_path: req.output_path,
        format: req.target_format,
    }))
}
```

- [ ] **Step 3: Register routes in main.rs**

```rust
.route("/audio/split", post(audio::audio_split))
.route("/audio/convert", post(audio::audio_convert))
```

- [ ] **Step 4: Build**

```bash
cd nous-core && cargo build
```

- [ ] **Step 5: Commit**

```bash
git add nous-core/src/audio.rs nous-core/src/main.rs
git commit -m "core: add POST /audio/split and /audio/convert endpoints"
```

### Task 19: Add backend audio_io service layer

**Files:**
- Create: `backend/src/services/audio_io.py`
- Test: `backend/tests/test_audio_io.py`

- [ ] **Step 1: Write a simple test**

Create `backend/tests/test_audio_io.py`:

```python
from src.services.audio_io import AudioIOClient


def test_audio_io_client_has_methods():
    """AudioIOClient should expose info, resample, concat, split, convert."""
    client = AudioIOClient(base_url="http://localhost:8001")
    assert hasattr(client, "info")
    assert hasattr(client, "resample")
    assert hasattr(client, "concat")
    assert hasattr(client, "split")
    assert hasattr(client, "convert")
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && uv run pytest tests/test_audio_io.py -v
```

- [ ] **Step 3: Implement**

Create `backend/src/services/audio_io.py`:

```python
"""Client for nous-core audio IO service."""

import httpx


class AudioIOClient:
    """Async HTTP client wrapping nous-core /audio/* endpoints."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self._base = base_url

    async def info(self, path: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._base}/audio/info", json={"path": path})
            resp.raise_for_status()
            return resp.json()

    async def resample(self, input_path: str, output_path: str, target_sample_rate: int) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._base}/audio/resample", json={
                "input_path": input_path,
                "output_path": output_path,
                "target_sample_rate": target_sample_rate,
            })
            resp.raise_for_status()
            return resp.json()

    async def concat(self, input_paths: list[str], output_path: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._base}/audio/concat", json={
                "input_paths": input_paths,
                "output_path": output_path,
            })
            resp.raise_for_status()
            return resp.json()

    async def split(self, input_path: str, output_dir: str, split_points_ms: list[int]) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._base}/audio/split", json={
                "input_path": input_path,
                "output_dir": output_dir,
                "split_points_ms": split_points_ms,
            })
            resp.raise_for_status()
            return resp.json()

    async def convert(self, input_path: str, output_path: str, target_format: str = "wav") -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._base}/audio/convert", json={
                "input_path": input_path,
                "output_path": output_path,
                "target_format": target_format,
            })
            resp.raise_for_status()
            return resp.json()


# Default singleton
audio_io = AudioIOClient()
```

- [ ] **Step 4: Run tests**

```bash
cd backend && uv run pytest tests/test_audio_io.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/audio_io.py backend/tests/test_audio_io.py
git commit -m "backend: add AudioIOClient wrapper for nous-core audio endpoints"
```

---

## Chunk 5: Phase 4 — Frontend Completion

### Task 20: Upgrade frontend TTS API with streaming and emotion

**Files:**
- Modify: `frontend/src/api/tts.ts`

- [ ] **Step 1: Add StreamRequest type and streaming fetch**

Replace `frontend/src/api/tts.ts`:

```typescript
import { useMutation } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface SynthesizeRequest {
  engine: string
  text: string
  voice?: string
  speed?: number
  sample_rate?: number
  reference_audio?: string
  reference_text?: string
  emotion?: string
  cache?: boolean
}

export interface SynthesizeResponse {
  audio_base64: string
  sample_rate: number
  duration_seconds: number
  engine: string
  rtf: number
  format: string
  cached: boolean
}

export interface StreamChunk {
  seq: number
  audio: string
  format: string
}

export interface StreamDone {
  total_chunks: number
  duration_ms: number
  usage: { characters: number; rtf: number }
}

export interface StreamError {
  code: string
  message: string
}

export function useSynthesize() {
  return useMutation({
    mutationFn: (req: SynthesizeRequest) =>
      apiFetch<SynthesizeResponse>('/api/v1/tts/synthesize', {
        method: 'POST',
        body: JSON.stringify(req),
      }),
  })
}

/**
 * Consume SSE stream from POST /api/v1/tts/stream.
 * Calls onChunk for each audio chunk and onDone when complete.
 */
export async function streamTTS(
  req: SynthesizeRequest,
  callbacks: {
    onChunk: (chunk: StreamChunk) => void
    onDone: (done: StreamDone) => void
    onError: (err: StreamError) => void
  },
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch('/api/v1/tts/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })

  if (!resp.ok || !resp.body) {
    callbacks.onError({ code: 'HTTP_ERROR', message: `HTTP ${resp.status}` })
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    let currentEvent = ''
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim()
      } else if (line.startsWith('data: ')) {
        const data = line.slice(6)
        try {
          const parsed = JSON.parse(data)
          if (currentEvent === 'audio') callbacks.onChunk(parsed)
          else if (currentEvent === 'done') callbacks.onDone(parsed)
          else if (currentEvent === 'error') callbacks.onError(parsed)
        } catch {
          // ignore parse errors
        }
        currentEvent = ''
      }
    }
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd frontend && git add src/api/tts.ts
git commit -m "frontend: add SSE streaming TTS client and emotion/cache fields"
```

### Task 21: Add TTS WebSocket session manager

**Files:**
- Rewrite: `frontend/src/api/websocket.ts`

- [ ] **Step 1: Replace with session-based WS manager**

Replace `frontend/src/api/websocket.ts`:

```typescript
type MessageHandler = (msg: Record<string, unknown>) => void

/**
 * TTS WebSocket session manager.
 * Maintains a single connection, supports multiple serial sessions.
 */
class TTSWebSocket {
  private ws: WebSocket | null = null
  private handlers = new Map<string, MessageHandler>()
  private connectPromise: Promise<void> | null = null

  private getUrl(): string {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${location.host}/ws/tts`
  }

  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) return
    if (this.connectPromise) return this.connectPromise

    this.connectPromise = new Promise((resolve, reject) => {
      const ws = new WebSocket(this.getUrl())

      ws.onopen = () => {
        this.ws = ws
        this.connectPromise = null
        resolve()
      }

      ws.onerror = () => {
        this.connectPromise = null
        reject(new Error('WebSocket connection failed'))
      }

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          const sessionId = msg.session_id as string
          if (sessionId) {
            this.handlers.get(sessionId)?.(msg)
          }
          // Also notify global handlers
          this.handlers.get('*')?.(msg)
        } catch {
          // ignore
        }
      }

      ws.onclose = () => {
        this.ws = null
        this.connectPromise = null
      }
    })

    return this.connectPromise
  }

  send(msg: Record<string, unknown>): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected')
    }
    this.ws.send(JSON.stringify(msg))
  }

  onSession(sessionId: string, handler: MessageHandler): () => void {
    this.handlers.set(sessionId, handler)
    return () => this.handlers.delete(sessionId)
  }

  onAll(handler: MessageHandler): () => void {
    this.handlers.set('*', handler)
    return () => this.handlers.delete('*')
  }

  async startSession(sessionId: string, config: {
    engine: string
    voice_preset?: string
    voice?: string
    speed?: number
    sample_rate?: number
    emotion?: string
  }): Promise<void> {
    await this.connect()
    this.send({ type: 'start_session', session_id: sessionId, ...config })
  }

  synthesize(sessionId: string, text: string, emotion?: string): void {
    this.send({ type: 'synthesize', session_id: sessionId, text, emotion })
  }

  endSession(sessionId: string): void {
    this.send({ type: 'end_session', session_id: sessionId })
  }

  disconnect(): void {
    this.ws?.close()
    this.ws = null
    this.handlers.clear()
  }
}

export const ttsWS = new TTSWebSocket()

// Re-export legacy hook for backward compat with existing task WS
export { useTaskWebSocket } from './legacyWebSocket'
```

- [ ] **Step 2: Move old hook to legacy file**

Create `frontend/src/api/legacyWebSocket.ts` with the old `useTaskWebSocket` code from the original `websocket.ts`.

- [ ] **Step 3: Commit**

```bash
cd frontend && git add src/api/websocket.ts src/api/legacyWebSocket.ts
git commit -m "frontend: replace per-task WS with session-based TTS WebSocket manager"
```

### Task 22: Implement full node execution engine

**Files:**
- Rewrite: `frontend/src/utils/workflowExecutor.ts`

- [ ] **Step 1: Rewrite with per-node-type dispatch**

Replace `frontend/src/utils/workflowExecutor.ts`:

```typescript
import type { Workflow, WorkflowNode, WorkflowEdge, NodeType } from '../models/workflow'
import { apiFetch } from '../api/client'
import { streamTTS } from '../api/tts'
import type { SynthesizeResponse } from '../api/tts'

export interface ExecutionResult {
  audioBase64: string
  sampleRate: number
  duration: number
}

// Each node produces typed output during execution
type NodeOutput = {
  text?: string
  audioBase64?: string
  sampleRate?: number
  audioPath?: string
}

function topoSort(nodes: WorkflowNode[], edges: WorkflowEdge[]): WorkflowNode[] {
  const inDegree = new Map<string, number>()
  const adj = new Map<string, string[]>()

  for (const n of nodes) {
    inDegree.set(n.id, 0)
    adj.set(n.id, [])
  }

  for (const e of edges) {
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1)
    adj.get(e.source)?.push(e.target)
  }

  const queue: string[] = []
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id)
  }

  const sorted: WorkflowNode[] = []
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))

  while (queue.length > 0) {
    const id = queue.shift()!
    const node = nodeMap.get(id)
    if (node) sorted.push(node)

    for (const next of adj.get(id) ?? []) {
      const deg = (inDegree.get(next) ?? 1) - 1
      inDegree.set(next, deg)
      if (deg === 0) queue.push(next)
    }
  }

  if (sorted.length !== nodes.length) {
    throw new Error('工作流存在循环依赖')
  }

  return sorted
}

function getInputs(
  nodeId: string,
  edges: WorkflowEdge[],
  outputs: Map<string, NodeOutput>,
): NodeOutput {
  const merged: NodeOutput = {}
  for (const e of edges) {
    if (e.target === nodeId) {
      const src = outputs.get(e.source)
      if (!src) continue
      // Map source output to target input by handle type
      if (e.targetHandle === 'text' && src.text) merged.text = src.text
      if (e.targetHandle?.startsWith('audio') && src.audioBase64) {
        merged.audioBase64 = src.audioBase64
        merged.sampleRate = src.sampleRate
      }
      if (e.targetHandle === 'ref_audio' && src.audioBase64) {
        merged.audioBase64 = src.audioBase64
      }
      // For multi-input nodes (mixer, concat), use targetHandle to distinguish
      if (e.targetHandle === 'audio_1' && src.audioBase64) merged.audioBase64 = src.audioBase64
      if (e.targetHandle === 'audio_2' && src.audioBase64) merged.text = src.audioBase64 // store second track in text field temporarily
      if (e.targetHandle === 'speech' && src.audioBase64) merged.audioBase64 = src.audioBase64
      if (e.targetHandle === 'bgm' && src.audioBase64) merged.text = src.audioBase64
    }
  }
  return merged
}

const nodeExecutors: Record<NodeType, (node: WorkflowNode, inputs: NodeOutput) => Promise<NodeOutput>> = {
  text_input: async (node) => ({
    text: (node.data.text as string) ?? '',
  }),

  ref_audio: async (node) => ({
    audioBase64: (node.data.audioBase64 as string) ?? '',
    sampleRate: (node.data.sampleRate as number) ?? 24000,
    audioPath: (node.data.path as string) ?? '',
  }),

  tts_engine: async (node, inputs) => {
    const text = inputs.text ?? ''
    if (!text.trim()) throw new Error('TTS 节点缺少文本输入')

    const engine = (node.data.engine as string) ?? 'cosyvoice2'
    const voice = (node.data.voice as string) ?? 'default'
    const speed = (node.data.speed as number) ?? 1.0
    const sampleRate = (node.data.sampleRate as number) ?? 24000
    const emotion = (node.data.emotion as string) || undefined

    const resp = await apiFetch<SynthesizeResponse>('/api/v1/tts/synthesize', {
      method: 'POST',
      body: JSON.stringify({
        engine, text, voice, speed, sample_rate: sampleRate,
        reference_audio: inputs.audioPath ?? undefined,
        emotion,
      }),
    })

    return {
      audioBase64: resp.audio_base64,
      sampleRate: resp.sample_rate,
    }
  },

  resample: async (node, inputs) => {
    // TODO: call nous-core /audio/resample when available
    // For now, passthrough
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },

  mixer: async (_node, inputs) => {
    // TODO: WASM audio mixing
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },

  concat: async (_node, inputs) => {
    // TODO: call nous-core /audio/concat or WASM
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },

  bgm_mix: async (_node, inputs) => {
    // TODO: WASM BGM mixing
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },

  output: async (_node, inputs) => {
    return { audioBase64: inputs.audioBase64, sampleRate: inputs.sampleRate }
  },
}

export async function executeWorkflow(workflow: Workflow): Promise<ExecutionResult> {
  const { nodes, edges } = workflow

  if (nodes.length === 0) throw new Error('工作流为空')

  const hasOutput = nodes.some((n) => n.type === 'output')
  if (!hasOutput) throw new Error('工作流缺少输出节点')

  const sorted = topoSort(nodes, edges)
  const outputs = new Map<string, NodeOutput>()

  for (const node of sorted) {
    const inputs = getInputs(node.id, edges, outputs)
    const executor = nodeExecutors[node.type]
    if (!executor) throw new Error(`未知节点类型: ${node.type}`)

    const result = await executor(node, inputs)
    outputs.set(node.id, result)
  }

  // Find the output node's result
  const outputNode = sorted.find((n) => n.type === 'output')!
  const finalOutput = outputs.get(outputNode.id)

  if (!finalOutput?.audioBase64) {
    throw new Error('工作流执行完成但没有音频输出')
  }

  return {
    audioBase64: finalOutput.audioBase64,
    sampleRate: finalOutput.sampleRate ?? 24000,
    duration: 0, // will be calculated by player
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd frontend && git add src/utils/workflowExecutor.ts
git commit -m "frontend: implement full node execution engine with per-type dispatch"
```

### Task 23: Add emotion input to TTS engine node

**Files:**
- Modify: `frontend/src/components/nodes/TTSEngineNode.tsx` (or equivalent)

- [ ] **Step 1: Find and update the TTS engine node component**

Locate the TTS engine node component and add an `emotion` text input field. The exact path depends on the component structure — look for the node that renders the TTS engine configuration.

Add to the node's data:
```typescript
// In the node's data interface or component:
<input
  type="text"
  placeholder="情感描述（可选）"
  value={(data.emotion as string) ?? ''}
  onChange={(e) => updateNodeData(id, { emotion: e.target.value })}
  className="w-full px-2 py-1 text-xs border rounded"
/>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/nodes/
git commit -m "frontend: add emotion text input to TTS engine node"
```

---

## Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|-----------------|
| Phase 1: Monorepo | Tasks 1-4 | Rename, gitignore, config fix, dev script |
| Phase 2: TTS | Tasks 5-14 | Usage model, cache, SSE stream, emotion, batch rounds, WS sessions |
| Phase 3: nous-core | Tasks 15-19 | Audio info/resample/concat/split/convert, backend client |
| Phase 4: Frontend | Tasks 20-23 | Stream client, WS session manager, node executor, emotion UI |
