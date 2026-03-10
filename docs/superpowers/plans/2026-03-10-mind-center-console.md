# Mind Center Console Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the mind-center backend APIs (engine management, sync TTS, voice presets, audio upload, batch TTS) and a React developer console for TTS debugging and voice preset management.

**Architecture:** Two repos — `mind-center` (existing FastAPI backend, add new endpoints) and `mind-center-console` (new Vite+React frontend). Backend-first approach: build and test all APIs before starting the frontend. Routes restructured from `/api/v1/generate/*` to domain-grouped `/api/v1/tts/*`, `/api/v1/engines/*`, `/api/v1/voices/*`.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 (async), PostgreSQL, Celery+Redis, React 19, TypeScript, Vite, TailwindCSS, React Query, Zustand.

**Spec:** `docs/superpowers/specs/2026-03-10-mind-center-console-design.md`

**Key codebase facts (verified):**
- Registry uses `_ENGINE_INSTANCES` (not `_engines`) — see `src/workers/tts_engines/registry.py`
- `TTSEngine.synthesize()` signature has no `reference_text` param — needs adding to `base.py`
- `database.py` has no `get_async_session` — needs adding before voice preset routes
- Existing generate routes return `JSONResponse(content={...})`, not `TaskResponse` model
- `pyproject.toml` has `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed on tests
- All 6 engine implementations already exist and register via decorator

---

## Chunk 0: Shared Test Infrastructure

### Task 0: Create shared conftest.py

All API tests need a shared `client` fixture. Create this first to avoid per-file duplication.

**Files:**
- Create: `tests/conftest.py`
- Modify: `src/models/database.py` — add `get_async_session`

- [ ] **Step 1: Add `get_async_session` to database.py**

```python
# Add to src/models/database.py at the end:

_session_factory = None

async def get_async_session():
    """FastAPI dependency for async DB sessions."""
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory()
    async with _session_factory() as session:
        yield session
```

- [ ] **Step 2: Create tests/conftest.py**

```python
# tests/conftest.py
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

- [ ] **Step 3: Run existing tests to ensure nothing breaks**

Run: `python -m pytest tests/ -v`
Expected: All existing tests PASS (fixtures that define their own `client` will shadow the conftest one).

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py src/models/database.py
git commit -m "feat: add shared test fixtures and get_async_session dependency"
```

---

## Chunk 1: Backend Foundation — CORS + Route Restructure + Engine API

### Task 1: Add CORS Middleware

**Files:**
- Modify: `src/api/main.py`
- Test: `tests/test_api_cors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_cors.py
# Uses shared `client` fixture from conftest.py


async def test_cors_preflight(client):
    resp = await client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


async def test_cors_allows_console_origin(client):
    resp = await client.get(
        "/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /media/heygo/Program/projects-code/_playground/mind-center && python -m pytest tests/test_api_cors.py -v`
Expected: FAIL — no CORS headers in response.

- [ ] **Step 3: Implement CORS middleware**

In `src/api/main.py`, add inside `create_app()` after `app = FastAPI(...)`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # alternate dev port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_cors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/main.py tests/test_api_cors.py
git commit -m "feat: add CORS middleware for console cross-origin access"
```

---

### Task 2: Restructure TTS Route Path

Change `/api/v1/generate/tts` → `/api/v1/tts/generate` to group TTS endpoints by domain.

**Files:**
- Modify: `src/api/routes/generate.py` — split into `src/api/routes/tts.py` (TTS routes) and keep `generate.py` (image/video only)
- Create: `src/api/routes/tts.py`
- Modify: `src/api/main.py` — include new tts router
- Modify: `tests/test_api_generate.py` — update TTS test paths

- [ ] **Step 1: Update existing TTS generate test to use new path**

In `tests/test_api_generate.py`, find the TTS test and update the path from `/api/v1/generate/tts` to `/api/v1/tts/generate`. If no TTS-specific test exists, add one:

```python
# In tests/test_api_generate.py or create tests/test_api_tts.py
async def test_tts_generate_dispatches_task(client, mocker):
    mocker.patch(
        "src.api.routes.tts.generate_tts_task.delay",
        return_value=mocker.Mock(id="fake-task-id"),
    )
    resp = await client.post(
        "/api/v1/tts/generate",
        json={"text": "hello", "engine": "cosyvoice2"},
    )
    assert resp.status_code == 202
    assert "task_id" in resp.json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_generate.py -v -k tts`
Expected: FAIL — 404, route not found at new path.

- [ ] **Step 3: Create `src/api/routes/tts.py`**

```python
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.models.schemas import TTSRequest
from src.workers.tts_worker import generate_tts_task

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])


@router.post("/generate", status_code=202)
async def tts_generate(req: TTSRequest):
    """Dispatch async TTS generation task via Celery."""
    task_id = str(uuid.uuid4())
    generate_tts_task.delay(task_id, req.model_dump())
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "pending", "type": "tts"},
    )
```

Note: Copy the existing TTS dispatch logic from `generate.py` into this new file. Remove TTS route from `generate.py`.

- [ ] **Step 4: Remove TTS from `generate.py`, register new router in `main.py`**

In `src/api/routes/generate.py`: remove the TTS endpoint (keep image and video).

In `src/api/main.py`: add `from src.api.routes import tts` and `app.include_router(tts.router)`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS, TTS now at `/api/v1/tts/generate`.

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/tts.py src/api/routes/generate.py src/api/main.py tests/
git commit -m "refactor: move TTS route to /api/v1/tts/generate (domain-grouped)"
```

---

### Task 3: Engine Management API

**Files:**
- Create: `src/api/routes/engines.py`
- Modify: `src/api/main.py` — register router
- Modify: `src/models/schemas.py` — add response schemas
- Modify: `src/gpu/model_manager.py` — may need to expose engine load status
- Test: `tests/test_api_engines.py`

- [ ] **Step 1: Add engine response schemas**

In `src/models/schemas.py`:

```python
class EngineInfo(BaseModel):
    name: str
    display_name: str
    type: str
    status: Literal["loaded", "unloaded"]
    gpu: int
    vram_gb: float
    resident: bool


class EngineLoadResponse(BaseModel):
    name: str
    status: Literal["loaded", "unloaded"]
    load_time_seconds: float | None = None
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_api_engines.py
# Uses shared `client` fixture from conftest.py


async def test_list_engines(client):
    resp = await client.get("/api/v1/engines")
    assert resp.status_code == 200
    engines = resp.json()
    assert isinstance(engines, list)
    assert len(engines) > 0
    engine = engines[0]
    assert "name" in engine
    assert "status" in engine
    assert engine["status"] in ("loaded", "unloaded")


async def test_list_engines_contains_all_tts(client):
    resp = await client.get("/api/v1/engines")
    names = {e["name"] for e in resp.json()}
    assert "cosyvoice2" in names
    assert "indextts2" in names
    assert "moss_tts" in names
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_api_engines.py -v`
Expected: FAIL — 404.

- [ ] **Step 4: Implement engine list endpoint**

```python
# src/api/routes/engines.py
from fastapi import APIRouter, HTTPException

from src.config import load_model_configs
from src.models.schemas import EngineInfo
from src.workers.tts_engines.registry import list_engines, get_engine

router = APIRouter(prefix="/api/v1/engines", tags=["engines"])


@router.get("", response_model=list[EngineInfo])
async def list_all_engines():
    """List all engines with their load status."""
    configs = load_model_configs()
    registered = set(list_engines())
    result = []
    for key, cfg in configs.items():
        if cfg.get("type") != "tts":
            continue
        result.append(EngineInfo(
            name=key,
            display_name=cfg["name"],
            type=cfg["type"],
            status="loaded" if _is_engine_loaded(key) else "unloaded",
            gpu=cfg.get("gpu", 1),
            vram_gb=cfg.get("vram_gb", 0),
            resident=cfg.get("resident", False),
        ))
    return result


def _is_engine_loaded(name: str) -> bool:
    """Check if engine singleton exists and is loaded."""
    from src.workers.tts_engines import registry
    engine = registry._ENGINE_INSTANCES.get(name)
    return engine is not None and engine.is_loaded
```

Register in `main.py`: `from src.api.routes import engines` → `app.include_router(engines.router)`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_api_engines.py -v`
Expected: PASS

- [ ] **Step 6: Write load/unload tests (including success paths)**

Add to `tests/test_api_engines.py`:

```python
from unittest.mock import patch, MagicMock


async def test_load_unknown_engine(client):
    resp = await client.post("/api/v1/engines/nonexistent/load")
    assert resp.status_code == 404


async def test_unload_unknown_engine(client):
    resp = await client.post("/api/v1/engines/nonexistent/unload")
    assert resp.status_code == 404


async def test_load_engine_success(client):
    mock_engine = MagicMock()
    mock_engine.is_loaded = False
    with patch("src.api.routes.engines.get_engine", return_value=mock_engine):
        resp = await client.post("/api/v1/engines/cosyvoice2/load")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "cosyvoice2"
    assert data["status"] == "loaded"
    mock_engine.load.assert_called_once()


async def test_unload_resident_engine_rejected(client):
    resp = await client.post("/api/v1/engines/cosyvoice2/unload")
    assert resp.status_code == 409  # cosyvoice2 is resident


async def test_unload_resident_engine_with_force(client):
    mock_engine = MagicMock()
    mock_engine.is_loaded = True
    with patch.dict("src.workers.tts_engines.registry._ENGINE_INSTANCES", {"cosyvoice2": mock_engine}):
        resp = await client.post("/api/v1/engines/cosyvoice2/unload?force=true")
    assert resp.status_code == 200
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_engines.py -v -k "load or unload"`
Expected: FAIL — endpoints not implemented yet.

Add to `src/api/routes/engines.py`:

```python
import time
from pathlib import Path

from src.config import get_settings


@router.post("/{name}/load", response_model=EngineLoadResponse)
async def load_engine(name: str):
    """Load an engine onto GPU."""
    configs = load_model_configs()
    if name not in configs or configs[name].get("type") != "tts":
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    cfg = configs[name]
    settings = get_settings()
    model_path = Path(settings.LOCAL_MODELS_PATH) / cfg["local_path"]
    device = f"cuda:{cfg.get('gpu', 1)}"

    start = time.monotonic()
    engine = get_engine(name, model_path=model_path, device=device)
    if not engine.is_loaded:
        engine.load()
    elapsed = round(time.monotonic() - start, 2)

    return EngineLoadResponse(name=name, status="loaded", load_time_seconds=elapsed)


@router.post("/{name}/unload", response_model=EngineLoadResponse)
async def unload_engine(name: str, force: bool = False):
    """Unload an engine from GPU."""
    configs = load_model_configs()
    if name not in configs or configs[name].get("type") != "tts":
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    cfg = configs[name]
    if cfg.get("resident", False) and not force:
        raise HTTPException(409, detail=f"Engine {name} is resident. Use force=true to unload.")

    from src.workers.tts_engines import registry
    engine = registry._ENGINE_INSTANCES.get(name)
    if engine and engine.is_loaded:
        engine.unload()

    return EngineLoadResponse(name=name, status="unloaded")
```

- [ ] **Step 7: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/api/routes/engines.py src/models/schemas.py src/api/main.py tests/test_api_engines.py
git commit -m "feat: add engine management API (list, load, unload)"
```

---

## Chunk 2: Sync TTS Synthesize + Audio Upload

### Task 4: Sync TTS Synthesize Endpoint

The core debug endpoint — calls engine.synthesize() directly (no Celery), returns audio as base64.

**Files:**
- Modify: `src/api/routes/tts.py`
- Modify: `src/models/schemas.py` — add SynthesizeRequest, SynthesizeResponse
- Test: `tests/test_api_tts.py`

- [ ] **Step 1: Add request/response schemas**

In `src/models/schemas.py`:

```python
import base64


class SynthesizeRequest(BaseModel):
    engine: Literal[
        "cosyvoice2", "indextts2", "qwen3_tts_base",
        "qwen3_tts_customvoice", "qwen3_tts_voicedesign", "moss_tts",
    ]
    text: str
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    reference_audio: str | None = None  # UUID from upload API or server path
    reference_text: str | None = None   # needed by qwen3_tts_base


class SynthesizeResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_seconds: float
    engine: str
    rtf: float
    format: str = "wav"
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_api_tts.py
# Uses shared `client` fixture from conftest.py
from unittest.mock import MagicMock, patch

from src.workers.tts_engines.base import TTSResult


FAKE_WAV = b"RIFF" + b"\x00" * 100  # minimal fake WAV


async def test_synthesize_returns_audio(client):
    fake_result = TTSResult(
        audio_bytes=FAKE_WAV,
        sample_rate=24000,
        duration_seconds=1.5,
        format="wav",
    )
    mock_engine = MagicMock()
    mock_engine.is_loaded = True
    mock_engine.synthesize.return_value = fake_result

    with patch("src.api.routes.tts._get_loaded_engine", return_value=mock_engine):
        resp = await client.post(
            "/api/v1/tts/synthesize",
            json={"engine": "cosyvoice2", "text": "hello"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "audio_base64" in data
    assert data["engine"] == "cosyvoice2"
    assert data["duration_seconds"] == 1.5
    assert data["rtf"] > 0


async def test_synthesize_engine_not_loaded(client):
    with patch("src.api.routes.tts._get_loaded_engine", return_value=None):
        resp = await client.post(
            "/api/v1/tts/synthesize",
            json={"engine": "cosyvoice2", "text": "hello"},
        )

    assert resp.status_code == 409
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_api_tts.py -v`
Expected: FAIL

- [ ] **Step 4: Implement sync synthesize endpoint**

Add to `src/api/routes/tts.py`:

```python
import base64
import time

from fastapi import HTTPException

from src.models.schemas import SynthesizeRequest, SynthesizeResponse
from src.workers.tts_engines.registry import _ENGINE_INSTANCES


def _get_loaded_engine(name: str):
    """Get a loaded engine or None."""
    engine = _ENGINE_INSTANCES.get(name)
    if engine and engine.is_loaded:
        return engine
    return None


@router.post("/synthesize", response_model=SynthesizeResponse)
async def tts_synthesize(req: SynthesizeRequest):
    """Synchronous TTS synthesis for debugging. Returns audio as base64."""
    engine = _get_loaded_engine(req.engine)
    if engine is None:
        raise HTTPException(
            409,
            detail=f"Engine {req.engine} not loaded. POST /api/v1/engines/{req.engine}/load first.",
        )

    start = time.monotonic()
    result = engine.synthesize(
        text=req.text,
        voice=req.voice,
        speed=req.speed,
        sample_rate=req.sample_rate,
        reference_audio=req.reference_audio,
    )
    elapsed = time.monotonic() - start
    rtf = round(elapsed / max(result.duration_seconds, 0.01), 2)

    return SynthesizeResponse(
        audio_base64=base64.b64encode(result.audio_bytes).decode(),
        sample_rate=result.sample_rate,
        duration_seconds=result.duration_seconds,
        engine=req.engine,
        rtf=rtf,
        format=result.format,
    )
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_api_tts.py tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/tts.py src/models/schemas.py tests/test_api_tts.py
git commit -m "feat: add sync /tts/synthesize endpoint for debug console"
```

---

### Task 5: Audio Upload Endpoint

Upload reference audio files for use in TTS synthesis.

**Files:**
- Create: `src/api/routes/audio.py`
- Modify: `src/api/main.py`
- Modify: `src/models/schemas.py`
- Test: `tests/test_api_audio.py`

- [ ] **Step 1: Add response schema**

In `src/models/schemas.py`:

```python
class AudioUploadResponse(BaseModel):
    id: str
    path: str
    duration_seconds: float | None = None
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_api_audio.py
# Uses shared `client` fixture from conftest.py
from unittest.mock import patch
from pathlib import Path


async def test_upload_audio(client, tmp_path):
    # Patch _get_upload_dir to return tmp_path
    with patch("src.api.routes.audio._get_upload_dir", return_value=tmp_path):
        # Create a minimal WAV file (44-byte header + 100 bytes of silence)
        import struct
        data_size = 100
        wav = bytearray()
        wav.extend(b"RIFF")
        wav.extend(struct.pack("<I", 36 + data_size))
        wav.extend(b"WAVEfmt ")
        wav.extend(struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16))
        wav.extend(b"data")
        wav.extend(struct.pack("<I", data_size))
        wav.extend(b"\x00" * data_size)

        resp = await client.post(
            "/api/v1/audio/upload",
            files={"file": ("test.wav", bytes(wav), "audio/wav")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["path"].endswith(".wav")


async def test_upload_rejects_non_audio(client):
    resp = await client.post(
        "/api/v1/audio/upload",
        files={"file": ("test.txt", b"not audio", "text/plain")},
    )
    assert resp.status_code == 422
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_api_audio.py -v`
Expected: FAIL — 404.

- [ ] **Step 4: Implement audio upload endpoint**

```python
# src/api/routes/audio.py
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from src.models.schemas import AudioUploadResponse

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _get_upload_dir() -> Path:
    """Resolve upload directory at call time (testable)."""
    return Path(os.getenv("AUDIO_UPLOAD_DIR", "assets/voices/uploads"))


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """Upload a reference audio file for TTS voice cloning."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(422, detail=f"Unsupported format: {ext}. Use: {ALLOWED_EXTENSIONS}")

    file_id = str(uuid.uuid4())
    upload_dir = _get_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{file_id}{ext}"

    content = await file.read()
    dest.write_bytes(content)

    return AudioUploadResponse(id=file_id, path=str(dest))


@router.get("/{audio_id}")
async def get_audio_info(audio_id: str):
    """Get info about an uploaded audio file."""
    upload_dir = _get_upload_dir()
    matches = list(upload_dir.glob(f"{audio_id}.*"))
    if not matches:
        raise HTTPException(404, detail=f"Audio not found: {audio_id}")
    path = matches[0]
    return {"id": audio_id, "path": str(path), "filename": path.name}
```

Register in `main.py`: `from src.api.routes import audio` → `app.include_router(audio.router)`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_api_audio.py tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/audio.py src/models/schemas.py src/api/main.py tests/test_api_audio.py
git commit -m "feat: add audio upload endpoint for reference audio management"
```

---

## Chunk 3: Voice Presets

### Task 6: Voice Preset Database Model

**Files:**
- Create: `src/models/voice_preset.py`
- Modify: `src/models/database.py` — ensure Base is shared
- Test: `tests/test_voice_preset_model.py`

- [ ] **Step 1: Write test for model instantiation**

```python
# tests/test_voice_preset_model.py
from src.models.voice_preset import VoicePreset, VoicePresetGroup


def test_voice_preset_creation():
    preset = VoicePreset(
        name="test-voice",
        engine="cosyvoice2",
        params={"voice": "default", "speed": 1.0, "sample_rate": 24000},
    )
    assert preset.name == "test-voice"
    assert preset.engine == "cosyvoice2"
    assert preset.params["speed"] == 1.0


def test_voice_preset_group_creation():
    group = VoicePresetGroup(
        name="test-group",
        presets=[{"role": "host", "voice_preset_id": "some-id"}],
    )
    assert group.name == "test-group"
    assert len(group.presets) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_voice_preset_model.py -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement VoicePreset model**

```python
# src/models/voice_preset.py
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, JSON, DateTime
from sqlalchemy.dialects.postgresql import UUID

from src.models.database import Base


class VoicePreset(Base):
    __tablename__ = "voice_presets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    engine = Column(String(50), nullable=False)
    params = Column(JSON, default=dict)
    reference_audio_path = Column(String(500), nullable=True)
    reference_text = Column(String(1000), nullable=True)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class VoicePresetGroup(Base):
    __tablename__ = "voice_preset_groups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    presets = Column(JSON, default=list)  # [{"role": "host", "voice_preset_id": "uuid"}]
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_voice_preset_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/voice_preset.py tests/test_voice_preset_model.py
git commit -m "feat: add VoicePreset and VoicePresetGroup SQLAlchemy models"
```

---

### Task 7: Voice Preset CRUD API

**Files:**
- Create: `src/api/routes/voices.py`
- Modify: `src/api/main.py`
- Modify: `src/models/schemas.py` — add Pydantic schemas
- Test: `tests/test_api_voices.py`

- [ ] **Step 1: Add Pydantic schemas**

In `src/models/schemas.py`:

```python
class VoicePresetCreate(BaseModel):
    name: str
    engine: str
    params: dict = {}
    reference_audio_path: str | None = None
    reference_text: str | None = None
    tags: list[str] = []


class VoicePresetUpdate(BaseModel):
    name: str | None = None
    engine: str | None = None
    params: dict | None = None
    reference_audio_path: str | None = None
    reference_text: str | None = None
    tags: list[str] | None = None


class VoicePresetOut(BaseModel):
    id: str
    name: str
    engine: str
    params: dict
    reference_audio_path: str | None
    reference_text: str | None
    tags: list[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_api_voices.py
# Uses shared `db_client` fixture from conftest.py (has test SQLite DB)


async def test_create_and_list_presets(db_client):
    client = db_client  # alias for readability
    # Create
    resp = await client.post("/api/v1/voices", json={
        "name": "test-voice",
        "engine": "cosyvoice2",
        "params": {"voice": "default", "speed": 1.0},
        "tags": ["test"],
    })
    assert resp.status_code == 201
    preset = resp.json()
    assert preset["name"] == "test-voice"
    preset_id = preset["id"]

    # List
    resp = await client.get("/api/v1/voices")
    assert resp.status_code == 200
    presets = resp.json()
    assert any(p["id"] == preset_id for p in presets)

    # Get by ID
    resp = await client.get(f"/api/v1/voices/{preset_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "test-voice"

    # Update
    resp = await client.put(f"/api/v1/voices/{preset_id}", json={"name": "renamed"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "renamed"

    # Delete
    resp = await client.delete(f"/api/v1/voices/{preset_id}")
    assert resp.status_code == 204


async def test_get_nonexistent_preset(db_client):
    resp = await db_client.get("/api/v1/voices/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
```

Note: These tests need a test database. Use an in-memory SQLite or test PostgreSQL. The exact fixture setup depends on the project's test infrastructure. If using SQLite for tests, add a `conftest.py` fixture that overrides `get_async_session` with a SQLite async engine and creates all tables.

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_api_voices.py -v`
Expected: FAIL — 404 or import error.

- [ ] **Step 4: Implement voice preset CRUD routes**

```python
# src/api/routes/voices.py
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.schemas import VoicePresetCreate, VoicePresetUpdate, VoicePresetOut
from src.models.voice_preset import VoicePreset

router = APIRouter(prefix="/api/v1/voices", tags=["voices"])


@router.get("", response_model=list[VoicePresetOut])
async def list_presets(session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(VoicePreset).order_by(VoicePreset.created_at))
    return result.scalars().all()


@router.post("", response_model=VoicePresetOut, status_code=201)
async def create_preset(
    data: VoicePresetCreate,
    session: AsyncSession = Depends(get_async_session),
):
    preset = VoicePreset(**data.model_dump())
    session.add(preset)
    await session.commit()
    await session.refresh(preset)
    return preset


@router.get("/{preset_id}", response_model=VoicePresetOut)
async def get_preset(
    preset_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Voice preset not found")
    return preset


@router.put("/{preset_id}", response_model=VoicePresetOut)
async def update_preset(
    preset_id: UUID,
    data: VoicePresetUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Voice preset not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(preset, key, value)
    await session.commit()
    await session.refresh(preset)
    return preset


@router.delete("/{preset_id}", status_code=204)
async def delete_preset(
    preset_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Voice preset not found")
    await session.delete(preset)
    await session.commit()
```

Note: Need to add `get_async_session` to `src/models/database.py` if it doesn't exist:

```python
async def get_async_session():
    async with async_session_factory() as session:
        yield session
```

Register in `main.py`: `from src.api.routes import voices` → `app.include_router(voices.router)`.

- [ ] **Step 5: Update shared conftest.py with DB support**

Update `tests/conftest.py` to add a `db_client` fixture that uses SQLite for DB tests:

```python
# Add to tests/conftest.py:
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from src.models.database import Base, get_async_session
from src.api.main import create_app


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
```

Update `tests/test_api_voices.py` to use `db_client` instead of `client`:

```python
async def test_create_and_list_presets(db_client):
    # ... (same test body, replace `client` with `db_client`)
```

Add `aiosqlite` to dev dependencies in `pyproject.toml`.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_api_voices.py tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/api/routes/voices.py src/models/schemas.py src/models/database.py src/api/main.py tests/test_api_voices.py tests/conftest.py pyproject.toml
git commit -m "feat: add voice preset CRUD API with DB model"
```

---

## Chunk 4: Batch TTS

### Task 8: Batch TTS Endpoint

**Files:**
- Modify: `src/api/routes/tts.py`
- Modify: `src/models/schemas.py`
- Test: `tests/test_api_tts.py` (add batch tests)

- [ ] **Step 1: Add batch schemas**

In `src/models/schemas.py`:

```python
class BatchSegment(BaseModel):
    voice_preset: str  # preset name or ID
    text: str


class BatchTTSRequest(BaseModel):
    segments: list[BatchSegment]


class BatchTaskInfo(BaseModel):
    index: int
    task_id: str


class BatchTTSResponse(BaseModel):
    batch_id: str
    tasks: list[BatchTaskInfo]


class BatchStatusResponse(BaseModel):
    batch_id: str
    status: Literal["pending", "partial", "completed", "failed"]
    tasks: list[dict]
```

- [ ] **Step 2: Write failing test**

```python
# Add to tests/test_api_tts.py
async def test_batch_tts(client, mocker):
    mock_delay = mocker.patch(
        "src.api.routes.tts.generate_tts_task.delay",
        return_value=mocker.Mock(id="fake-id"),
    )
    # Need voice presets to exist — mock the lookup
    mocker.patch(
        "src.api.routes.tts._resolve_preset",
        return_value={"engine": "cosyvoice2", "params": {"voice": "default"}},
    )

    resp = await client.post("/api/v1/tts/batch", json={
        "segments": [
            {"voice_preset": "host", "text": "Hello"},
            {"voice_preset": "guest", "text": "Hi"},
        ]
    })
    assert resp.status_code == 202
    data = resp.json()
    assert "batch_id" in data
    assert len(data["tasks"]) == 2
    assert data["tasks"][0]["index"] == 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_api_tts.py::test_batch_tts -v`
Expected: FAIL

- [ ] **Step 4: Implement batch endpoint**

Add to `src/api/routes/tts.py`:

```python
import uuid as uuid_mod

from src.models.schemas import BatchTTSRequest, BatchTTSResponse, BatchTaskInfo, BatchStatusResponse


def _resolve_preset(name: str) -> dict | None:
    """Look up voice preset by name. Returns dict with engine + params or None."""
    # TODO: DB lookup in production. For now, return None to trigger 404.
    return None


@router.post("/batch", response_model=BatchTTSResponse, status_code=202)
async def tts_batch(req: BatchTTSRequest):
    """Dispatch multiple TTS tasks for multi-character scenarios."""
    batch_id = f"batch_{uuid_mod.uuid4().hex[:12]}"
    tasks = []

    for i, segment in enumerate(req.segments):
        preset = _resolve_preset(segment.voice_preset)
        if preset is None:
            raise HTTPException(404, detail=f"Voice preset not found: {segment.voice_preset}")

        params = {**preset["params"], "text": segment.text, "engine": preset["engine"]}
        task = generate_tts_task.delay(task_id=f"{batch_id}_{i}", params=params)
        tasks.append(BatchTaskInfo(index=i, task_id=task.id))

    # Store batch_id → task_ids mapping in Redis for status lookup
    from src.workers.celery_app import celery_app
    celery_app.backend.set(
        f"batch:{batch_id}",
        [t.task_id for t in tasks],
    )

    return BatchTTSResponse(batch_id=batch_id, tasks=tasks)


@router.get("/batch/{batch_id}", response_model=BatchStatusResponse)
async def tts_batch_status(batch_id: str):
    """Query status of a batch TTS request."""
    from src.workers.celery_app import celery_app
    task_ids = celery_app.backend.get(f"batch:{batch_id}")
    if task_ids is None:
        raise HTTPException(404, detail=f"Batch not found: {batch_id}")

    tasks = []
    statuses = set()
    for i, tid in enumerate(task_ids):
        result = celery_app.AsyncResult(tid)
        status = result.status.lower()
        statuses.add(status)
        tasks.append({"index": i, "task_id": tid, "status": status})

    if all(s == "success" for s in statuses):
        overall = "completed"
    elif "failure" in statuses and not any(s in ("pending", "started") for s in statuses):
        overall = "failed"
    elif any(s in ("success", "started") for s in statuses):
        overall = "partial"
    else:
        overall = "pending"

    return BatchStatusResponse(batch_id=batch_id, status=overall, tasks=tasks)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_api_tts.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/tts.py src/models/schemas.py tests/test_api_tts.py
git commit -m "feat: add batch TTS endpoint for multi-character synthesis"
```

---

## Chunk 5: Console Frontend Setup

### Task 9: Initialize mind-center-console Project

This task creates the React frontend as a sibling project.

**Files:**
- Create: `/media/heygo/Program/projects-code/_playground/mind-center-console/` (new project)

- [ ] **Step 1: Scaffold Vite + React + TypeScript project**

```bash
cd /media/heygo/Program/projects-code/_playground
npm create vite@latest mind-center-console -- --template react-ts
cd mind-center-console
npm install
```

- [ ] **Step 2: Install dependencies**

```bash
npm install @tanstack/react-query zustand
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 3: Configure TailwindCSS**

Update `vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
```

Replace `src/index.css` content with:

```css
@import "tailwindcss";
```

- [ ] **Step 4: Set up React Query provider**

Replace `src/main.tsx`:

```typescript
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import './index.css'

const queryClient = new QueryClient()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
```

- [ ] **Step 5: Create basic layout with nav**

Replace `src/App.tsx`:

```typescript
import { useState } from 'react'

type Page = 'tts' | 'models'

export default function App() {
  const [page, setPage] = useState<Page>('tts')

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-6">
        <h1 className="text-lg font-semibold">mind-center console</h1>
        <nav className="flex gap-2">
          <button
            onClick={() => setPage('tts')}
            className={`px-3 py-1 rounded text-sm ${page === 'tts' ? 'bg-gray-700' : 'hover:bg-gray-800'}`}
          >
            TTS
          </button>
          <button
            onClick={() => setPage('models')}
            className={`px-3 py-1 rounded text-sm ${page === 'models' ? 'bg-gray-700' : 'hover:bg-gray-800'}`}
          >
            Models
          </button>
        </nav>
      </header>
      <main className="p-6">
        {page === 'tts' && <div>TTS Playground (coming next)</div>}
        {page === 'models' && <div>Model Dashboard (later)</div>}
      </main>
    </div>
  )
}
```

- [ ] **Step 6: Verify it runs**

```bash
npm run dev
# Open http://localhost:5173 — should see dark header with TTS/Models nav
```

- [ ] **Step 7: Init git and commit**

```bash
cd /media/heygo/Program/projects-code/_playground/mind-center-console
git init
git add .
git commit -m "feat: scaffold mind-center-console (Vite + React + TS + Tailwind)"
```

---

### Task 10: API Client Layer

**Files:**
- Create: `src/api/client.ts` — base fetch wrapper
- Create: `src/api/engines.ts` — engine API hooks
- Create: `src/api/tts.ts` — TTS API hooks
- Create: `src/api/voices.ts` — voice preset API hooks

- [ ] **Step 1: Create API client**

```typescript
// src/api/client.ts
const BASE = ''  // proxied via vite config

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    throw new Error(body.detail || `API error: ${resp.status}`)
  }
  return resp.json()
}
```

- [ ] **Step 2: Create engine hooks**

```typescript
// src/api/engines.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface EngineInfo {
  name: string
  display_name: string
  type: string
  status: 'loaded' | 'unloaded'
  gpu: number
  vram_gb: number
  resident: boolean
}

export function useEngines() {
  return useQuery({
    queryKey: ['engines'],
    queryFn: () => apiFetch<EngineInfo[]>('/api/v1/engines'),
    refetchInterval: 5000,
  })
}

export function useLoadEngine() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiFetch(`/api/v1/engines/${name}/load`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}

export function useUnloadEngine() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiFetch(`/api/v1/engines/${name}/unload`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}
```

- [ ] **Step 3: Create TTS hooks**

```typescript
// src/api/tts.ts
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
}

export interface SynthesizeResponse {
  audio_base64: string
  sample_rate: number
  duration_seconds: number
  engine: string
  rtf: number
  format: string
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
```

- [ ] **Step 4: Create voices hooks**

```typescript
// src/api/voices.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface VoicePreset {
  id: string
  name: string
  engine: string
  params: Record<string, unknown>
  reference_audio_path: string | null
  reference_text: string | null
  tags: string[]
}

export function useVoicePresets() {
  return useQuery({
    queryKey: ['voices'],
    queryFn: () => apiFetch<VoicePreset[]>('/api/v1/voices'),
  })
}

export function useCreatePreset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: Omit<VoicePreset, 'id'>) =>
      apiFetch('/api/v1/voices', { method: 'POST', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['voices'] }),
  })
}

export function useDeletePreset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiFetch(`/api/v1/voices/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['voices'] }),
  })
}
```

- [ ] **Step 5: Commit**

```bash
git add src/api/
git commit -m "feat: add API client layer with React Query hooks"
```

---

## Chunk 6: TTS Playground Page

### Task 11: Engine Panel Component

**Files:**
- Create: `src/components/EnginePanel.tsx`

- [ ] **Step 1: Implement EnginePanel**

```typescript
// src/components/EnginePanel.tsx
import { useEngines, useLoadEngine, useUnloadEngine } from '../api/engines'

export default function EnginePanel({ onSelect }: { onSelect: (name: string) => void }) {
  const { data: engines, isLoading } = useEngines()
  const loadEngine = useLoadEngine()
  const unloadEngine = useUnloadEngine()

  if (isLoading) return <div className="text-gray-500 text-sm">Loading engines...</div>

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Engines</h3>
      {engines?.map((e) => (
        <div
          key={e.name}
          className="flex items-center justify-between p-2 rounded bg-gray-900 hover:bg-gray-800 cursor-pointer"
          onClick={() => onSelect(e.name)}
        >
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${e.status === 'loaded' ? 'bg-green-500' : 'bg-gray-600'}`} />
            <span className="text-sm">{e.name}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">{e.vram_gb}GB</span>
            {e.status === 'loaded' ? (
              <button
                className="text-xs text-red-400 hover:text-red-300"
                onClick={(ev) => { ev.stopPropagation(); unloadEngine.mutate(e.name) }}
              >
                Unload
              </button>
            ) : (
              <button
                className="text-xs text-blue-400 hover:text-blue-300"
                onClick={(ev) => { ev.stopPropagation(); loadEngine.mutate(e.name) }}
              >
                Load
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add src/components/EnginePanel.tsx
git commit -m "feat: add EnginePanel component"
```

---

### Task 12: Synthesize Form + Audio Player

**Files:**
- Create: `src/components/SynthesizeForm.tsx`
- Create: `src/components/AudioPlayer.tsx`
- Create: `src/stores/history.ts` — Zustand store for synthesis history

- [ ] **Step 1: Create history store**

```typescript
// src/stores/history.ts
import { create } from 'zustand'
import type { SynthesizeResponse } from '../api/tts'

export interface HistoryEntry {
  id: string
  text: string
  engine: string
  response: SynthesizeResponse
  timestamp: number
}

interface HistoryState {
  entries: HistoryEntry[]
  add: (entry: HistoryEntry) => void
  clear: () => void
}

export const useHistoryStore = create<HistoryState>((set) => ({
  entries: [],
  add: (entry) => set((s) => ({ entries: [entry, ...s.entries].slice(0, 50) })),
  clear: () => set({ entries: [] }),
}))
```

- [ ] **Step 2: Create AudioPlayer**

```typescript
// src/components/AudioPlayer.tsx
import { useRef, useState, useEffect } from 'react'
import type { SynthesizeResponse } from '../api/tts'

export default function AudioPlayer({ result }: { result: SynthesizeResponse | null }) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)

  useEffect(() => { setPlaying(false) }, [result])

  if (!result) return null

  const src = `data:audio/wav;base64,${result.audio_base64}`

  return (
    <div className="p-4 bg-gray-900 rounded space-y-2">
      <audio ref={audioRef} src={src} onEnded={() => setPlaying(false)} />
      <div className="flex items-center gap-3">
        <button
          className="px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-sm"
          onClick={() => {
            if (playing) { audioRef.current?.pause(); setPlaying(false) }
            else { audioRef.current?.play(); setPlaying(true) }
          }}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <span className="text-sm text-gray-400">
          {result.duration_seconds}s | {result.sample_rate}Hz | RTF: {result.rtf}
        </span>
      </div>
      <div className="text-xs text-gray-500">Engine: {result.engine}</div>
    </div>
  )
}
```

- [ ] **Step 3: Create SynthesizeForm**

```typescript
// src/components/SynthesizeForm.tsx
import { useState } from 'react'
import { useSynthesize, type SynthesizeRequest, type SynthesizeResponse } from '../api/tts'
import { useHistoryStore } from '../stores/history'
import AudioPlayer from './AudioPlayer'

export default function SynthesizeForm({ selectedEngine }: { selectedEngine: string }) {
  const [text, setText] = useState('')
  const [voice, setVoice] = useState('default')
  const [speed, setSpeed] = useState(1.0)
  const [sampleRate, setSampleRate] = useState(24000)
  const [refAudio, setRefAudio] = useState('')
  const [refText, setRefText] = useState('')

  const synth = useSynthesize()
  const addHistory = useHistoryStore((s) => s.add)
  const [lastResult, setLastResult] = useState<SynthesizeResponse | null>(null)

  const handleSubmit = () => {
    if (!text.trim() || !selectedEngine) return
    const req: SynthesizeRequest = {
      engine: selectedEngine,
      text,
      voice: voice || 'default',
      speed,
      sample_rate: sampleRate,
      ...(refAudio ? { reference_audio: refAudio } : {}),
      ...(refText ? { reference_text: refText } : {}),
    }
    synth.mutate(req, {
      onSuccess: (resp) => {
        setLastResult(resp)
        addHistory({
          id: crypto.randomUUID(),
          text: text.slice(0, 50),
          engine: selectedEngine,
          response: resp,
          timestamp: Date.now(),
        })
      },
    })
  }

  return (
    <div className="space-y-4">
      <textarea
        className="w-full h-32 bg-gray-900 border border-gray-700 rounded p-3 text-sm resize-none focus:outline-none focus:border-blue-500"
        placeholder="Enter text to synthesize..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Engine</label>
          <input
            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm"
            value={selectedEngine}
            readOnly
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Voice</label>
          <input
            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm"
            value={voice}
            onChange={(e) => setVoice(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Speed: {speed}</label>
          <input
            type="range" min="0.5" max="2.0" step="0.1"
            className="w-full"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Sample Rate</label>
          <select
            className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm"
            value={sampleRate}
            onChange={(e) => setSampleRate(parseInt(e.target.value))}
          >
            <option value={16000}>16000</option>
            <option value={22050}>22050</option>
            <option value={24000}>24000</option>
            <option value={44100}>44100</option>
            <option value={48000}>48000</option>
          </select>
        </div>
      </div>

      {/* Reference audio fields — shown for voice-clone engines */}
      {['cosyvoice2', 'indextts2', 'qwen3_tts_base', 'moss_tts'].includes(selectedEngine) && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Reference Audio (path/UUID)</label>
            <input
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm"
              placeholder="Leave empty for default"
              value={refAudio}
              onChange={(e) => setRefAudio(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Reference Text (Qwen3 Base)</label>
            <input
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm"
              placeholder="Optional transcript of reference audio"
              value={refText}
              onChange={(e) => setRefText(e.target.value)}
            />
          </div>
        </div>
      )}

      <div className="flex gap-3">
        <button
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm disabled:opacity-50"
          onClick={handleSubmit}
          disabled={synth.isPending || !text.trim()}
        >
          {synth.isPending ? 'Synthesizing...' : 'Synthesize'}
        </button>
      </div>

      {synth.isError && (
        <div className="text-red-400 text-sm">{(synth.error as Error).message}</div>
      )}

      <AudioPlayer result={lastResult} />
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add src/components/ src/stores/
git commit -m "feat: add SynthesizeForm, AudioPlayer, and history store"
```

---

### Task 13: Wire Up TTS Playground Page

**Files:**
- Create: `src/pages/TTSPlayground.tsx`
- Create: `src/components/HistoryList.tsx`
- Create: `src/components/VoicePresetPanel.tsx`
- Modify: `src/App.tsx`

- [ ] **Step 1: Create HistoryList**

```typescript
// src/components/HistoryList.tsx
import { useHistoryStore } from '../stores/history'
import AudioPlayer from './AudioPlayer'
import { useState } from 'react'

export default function HistoryList() {
  const entries = useHistoryStore((s) => s.entries)
  const [expanded, setExpanded] = useState<string | null>(null)

  if (entries.length === 0) return null

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">History</h3>
      {entries.map((e) => (
        <div key={e.id} className="p-2 bg-gray-900 rounded">
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpanded(expanded === e.id ? null : e.id)}
          >
            <span className="text-sm truncate max-w-[200px]">{e.text}</span>
            <span className="text-xs text-gray-500">{e.engine} | {e.response.duration_seconds}s</span>
          </div>
          {expanded === e.id && <AudioPlayer result={e.response} />}
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Create VoicePresetPanel (minimal)**

```typescript
// src/components/VoicePresetPanel.tsx
import { useVoicePresets } from '../api/voices'

interface Props {
  onSelect: (preset: { engine: string; params: Record<string, unknown> }) => void
}

export default function VoicePresetPanel({ onSelect }: Props) {
  const { data: presets, isLoading } = useVoicePresets()

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Voice Presets</h3>
      {isLoading && <div className="text-gray-500 text-xs">Loading...</div>}
      {presets?.map((p) => (
        <div
          key={p.id}
          className="p-2 bg-gray-900 rounded hover:bg-gray-800 cursor-pointer"
          onClick={() => onSelect({ engine: p.engine, params: p.params })}
        >
          <div className="text-sm">{p.name}</div>
          <div className="text-xs text-gray-500">{p.engine}</div>
        </div>
      ))}
      {presets?.length === 0 && <div className="text-gray-500 text-xs">No presets yet</div>}
    </div>
  )
}
```

- [ ] **Step 3: Create TTSPlayground page**

```typescript
// src/pages/TTSPlayground.tsx
import { useState } from 'react'
import EnginePanel from '../components/EnginePanel'
import VoicePresetPanel from '../components/VoicePresetPanel'
import SynthesizeForm from '../components/SynthesizeForm'
import HistoryList from '../components/HistoryList'

export default function TTSPlayground() {
  const [selectedEngine, setSelectedEngine] = useState('cosyvoice2')

  return (
    <div className="flex gap-6 h-[calc(100vh-64px)]">
      {/* Left sidebar */}
      <div className="w-64 shrink-0 space-y-6 overflow-y-auto">
        <EnginePanel onSelect={setSelectedEngine} />
        <VoicePresetPanel onSelect={({ engine }) => setSelectedEngine(engine)} />
      </div>

      {/* Main content */}
      <div className="flex-1 space-y-6 overflow-y-auto">
        <h2 className="text-xl font-semibold">TTS Playground</h2>
        <SynthesizeForm selectedEngine={selectedEngine} />
        <HistoryList />
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Wire into App.tsx**

Update `src/App.tsx` to import and render `TTSPlayground`:

```typescript
import TTSPlayground from './pages/TTSPlayground'

// In the main area:
{page === 'tts' && <TTSPlayground />}
```

- [ ] **Step 5: Verify it runs**

```bash
npm run dev
# Open http://localhost:5173
# Should see sidebar with engines + presets, main area with synthesize form
```

- [ ] **Step 6: Commit**

```bash
git add src/
git commit -m "feat: complete TTS Playground page with all components"
```

---

## Summary

| Chunk | Tasks | What it delivers |
|-------|-------|-----------------|
| 0: Test Infra | 0 | Shared conftest, get_async_session |
| 1: Backend Foundation | 1-3 | CORS, route restructure, engine management API |
| 2: Sync TTS + Audio | 4-5 | Debug synthesize endpoint, audio upload |
| 3: Voice Presets | 6-7 | DB model + CRUD API |
| 4: Batch TTS | 8 | Multi-character synthesis |
| 5: Console Setup | 9-10 | React project + API client layer |
| 6: TTS Playground | 11-13 | Complete TTS debugging UI |

**Total tasks:** 14 (Task 0-13)
**Infra:** Task 0 (shared fixtures, DB session dependency)
**Backend tasks:** 1-8 (CORS, routes, engines, synthesize, audio, presets, batch)
**Frontend tasks:** 9-13 (scaffold, API hooks, components, page)

After completing all tasks, the developer console will be functional for TTS debugging: select engine → load → type text → synthesize → listen → save as preset.
