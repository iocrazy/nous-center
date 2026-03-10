# Mind Center Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an AI media workstation with FastAPI gateway, Celery task queue, GPU model management, and WebSocket notifications on Ubuntu dual-3090.

**Architecture:** FastAPI receives requests, dispatches GPU tasks via Celery+Redis (Mac), models managed by ModelManager with VRAM tracking. vLLM runs as a separate process for LLM tasks. Results stored on NAS, metadata in PostgreSQL (Mac).

**Tech Stack:** Python 3.12, uv, FastAPI, Celery, Redis, PostgreSQL (SQLAlchemy + asyncpg), vLLM, PyTorch, Pydantic v2

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/api/__init__.py`
- Create: `src/api/routes/__init__.py`
- Create: `src/workers/__init__.py`
- Create: `src/gpu/__init__.py`
- Create: `src/storage/__init__.py`
- Create: `src/models/__init__.py`
- Create: `configs/models.yaml`
- Create: `.gitignore`

**Step 1: Initialize uv project**

```bash
cd /media/heygo/Program/projects-code/_playground/mind-center
uv init --python 3.12
```

**Step 2: Set Python version**

`.python-version`:
```
3.12
```

**Step 3: Configure pyproject.toml**

```toml
[project]
name = "mind-center"
version = "0.1.0"
description = "AI media workstation - image/video/tts generation & multimodal understanding"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "celery[redis]>=5.4",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.30",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "httpx>=0.28",
    "websockets>=14.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.28",
    "ruff>=0.9",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
```

**Step 4: Create .env.example**

```env
# Redis (Mac)
REDIS_URL=redis://mac-ip:6379/0

# PostgreSQL (Mac)
DATABASE_URL=postgresql+asyncpg://mind:mind@mac-ip:5432/mind_center

# NAS
NAS_MODELS_PATH=/mnt/nas/models
NAS_OUTPUTS_PATH=/mnt/nas/outputs

# vLLM
VLLM_BASE_URL=http://localhost:8100

# GPU
GPU_IMAGE=0
GPU_TTS=1
GPU_VIDEO=0,1
```

**Step 5: Create directory structure and __init__.py files**

```bash
mkdir -p src/api/routes src/workers src/gpu src/storage src/models tests configs
touch src/__init__.py src/api/__init__.py src/api/routes/__init__.py
touch src/workers/__init__.py src/gpu/__init__.py src/storage/__init__.py src/models/__init__.py
touch tests/__init__.py
```

**Step 6: Create configs/models.yaml**

```yaml
models:
  sdxl:
    name: "stabilityai/stable-diffusion-xl-base-1.0"
    type: image
    gpu: 0
    vram_gb: 10
    resident: true

  cosyvoice2:
    name: "CosyVoice2-0.5B"
    type: tts
    gpu: 1
    vram_gb: 3
    resident: true

  qwen_tts:
    name: "Qwen-TTS"
    type: tts
    gpu: 1
    vram_gb: 3
    resident: true

  qwen25_vl:
    name: "Qwen/Qwen2.5-VL-7B-Instruct"
    type: understand
    engine: vllm
    gpu: 1
    vram_gb: 16
    resident: false

  wan21:
    name: "Wan2.1"
    type: video
    gpu: [0, 1]
    vram_gb: 40
    resident: false
    exclusive: true
```

**Step 7: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.env
.venv/
*.egg-info/
dist/
.ruff_cache/
.pytest_cache/
```

**Step 8: Install dependencies and verify**

```bash
uv sync
```

**Step 9: Initialize git and commit**

```bash
git init
git add -A
git commit -m "chore: scaffold mind-center project with uv"
```

---

### Task 2: Settings & Configuration

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

`tests/test_config.py`:
```python
from src.config import Settings, load_model_configs


def test_settings_defaults():
    settings = Settings(
        REDIS_URL="redis://localhost:6379/0",
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/db",
    )
    assert settings.REDIS_URL == "redis://localhost:6379/0"
    assert settings.NAS_OUTPUTS_PATH == "/mnt/nas/outputs"
    assert settings.VLLM_BASE_URL == "http://localhost:8100"


def test_load_model_configs():
    configs = load_model_configs("configs/models.yaml")
    assert "sdxl" in configs
    assert configs["sdxl"]["type"] == "image"
    assert configs["wan21"]["exclusive"] is True
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

**Step 3: Write minimal implementation**

`src/config.py`:
```python
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql+asyncpg://mind:mind@localhost:5432/mind_center"

    NAS_MODELS_PATH: str = "/mnt/nas/models"
    NAS_OUTPUTS_PATH: str = "/mnt/nas/outputs"

    VLLM_BASE_URL: str = "http://localhost:8100"

    GPU_IMAGE: int = 0
    GPU_TTS: int = 1
    GPU_VIDEO: str = "0,1"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_model_configs(path: str = "configs/models.yaml") -> dict:
    with open(Path(path)) as f:
        data = yaml.safe_load(f)
    return data["models"]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_config.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add settings and model config loading"
```

---

### Task 3: Database Models (Task ORM)

**Files:**
- Create: `src/models/database.py`
- Create: `src/models/task.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test**

`tests/test_models.py`:
```python
import uuid
from datetime import datetime, timezone

from src.models.task import TaskStatus, TaskRecord


def test_task_record_creation():
    task = TaskRecord(
        id=uuid.uuid4(),
        task_type="image",
        status=TaskStatus.PENDING,
        params={"prompt": "a cat"},
    )
    assert task.status == TaskStatus.PENDING
    assert task.task_type == "image"
    assert task.params["prompt"] == "a cat"
    assert task.result is None


def test_task_status_values():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.RUNNING == "running"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_models.py -v
```

Expected: FAIL

**Step 3: Write database setup**

`src/models/database.py`:
```python
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from src.config import get_settings


class Base(DeclarativeBase):
    pass


def create_engine():
    settings = get_settings()
    return create_async_engine(settings.DATABASE_URL)


def create_session_factory(engine=None):
    if engine is None:
        engine = create_engine()
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

**Step 4: Write task model**

`src/models/task.py`:
```python
import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRecord(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[TaskStatus] = mapped_column(String(20), default=TaskStatus.PENDING)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_models.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/models/database.py src/models/task.py tests/test_models.py
git commit -m "feat: add task database model with status enum"
```

---

### Task 4: Pydantic Schemas (Request/Response)

**Files:**
- Create: `src/models/schemas.py`
- Create: `tests/test_schemas.py`

**Step 1: Write the failing test**

`tests/test_schemas.py`:
```python
import uuid
from src.models.schemas import (
    ImageGenerateRequest,
    TTSRequest,
    VideoGenerateRequest,
    ImageUnderstandRequest,
    TaskResponse,
    TaskStatus,
)


def test_image_request():
    req = ImageGenerateRequest(prompt="a cat in space")
    assert req.prompt == "a cat in space"
    assert req.width == 1024
    assert req.height == 1024
    assert req.num_steps == 30


def test_tts_request():
    req = TTSRequest(text="Hello world", engine="cosyvoice2")
    assert req.engine == "cosyvoice2"


def test_tts_request_invalid_engine():
    import pytest
    with pytest.raises(ValueError):
        TTSRequest(text="Hello", engine="invalid")


def test_video_request():
    req = VideoGenerateRequest(prompt="a sunset timelapse")
    assert req.num_frames == 81


def test_image_understand_request():
    req = ImageUnderstandRequest(image_url="/path/to/img.png", question="What is this?")
    assert req.question == "What is this?"


def test_task_response():
    tid = uuid.uuid4()
    resp = TaskResponse(
        id=tid,
        task_type="image",
        status=TaskStatus.PENDING,
    )
    assert resp.id == tid
    assert resp.status == "pending"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: FAIL

**Step 3: Write schemas**

`src/models/schemas.py`:
```python
import uuid
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Requests ---

class ImageGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    num_steps: int = Field(default=30, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int | None = None


class VideoGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=832, ge=256, le=1280)
    height: int = Field(default=480, ge=256, le=720)
    num_frames: int = Field(default=81, ge=1, le=161)
    seed: int | None = None


class TTSRequest(BaseModel):
    text: str
    engine: Literal["cosyvoice2", "qwen_tts"] = "cosyvoice2"
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class ImageUnderstandRequest(BaseModel):
    image_url: str
    question: str = "Describe this image in detail."


# --- Responses ---

class TaskResponse(BaseModel):
    id: uuid.UUID
    task_type: str
    status: TaskStatus
    progress: int = 0
    result: dict | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/schemas.py tests/test_schemas.py
git commit -m "feat: add pydantic request/response schemas"
```

---

### Task 5: Celery App Configuration

**Files:**
- Create: `src/workers/celery_app.py`
- Create: `tests/test_celery_app.py`

**Step 1: Write the failing test**

`tests/test_celery_app.py`:
```python
from src.workers.celery_app import celery_app


def test_celery_app_configured():
    assert celery_app.main == "mind-center"


def test_celery_queues():
    routes = celery_app.conf.task_routes
    assert routes["src.workers.image_worker.*"]["queue"] == "image"
    assert routes["src.workers.tts_worker.*"]["queue"] == "tts"
    assert routes["src.workers.video_worker.*"]["queue"] == "video"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_celery_app.py -v
```

Expected: FAIL

**Step 3: Write Celery app**

`src/workers/celery_app.py`:
```python
from celery import Celery

from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "mind-center",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_routes={
        "src.workers.image_worker.*": {"queue": "image"},
        "src.workers.tts_worker.*": {"queue": "tts"},
        "src.workers.video_worker.*": {"queue": "video"},
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_celery_app.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/workers/celery_app.py tests/test_celery_app.py
git commit -m "feat: configure celery app with task routing"
```

---

### Task 6: VRAM Tracker

**Files:**
- Create: `src/gpu/vram_tracker.py`
- Create: `tests/test_vram_tracker.py`

**Step 1: Write the failing test**

`tests/test_vram_tracker.py`:
```python
from src.gpu.vram_tracker import VRAMTracker


def test_tracker_init():
    tracker = VRAMTracker(gpu_count=2, vram_per_gpu_gb=24)
    assert tracker.get_free(0) == 24.0
    assert tracker.get_free(1) == 24.0


def test_allocate_and_release():
    tracker = VRAMTracker(gpu_count=2, vram_per_gpu_gb=24)

    assert tracker.allocate(gpu=0, model_name="sdxl", vram_gb=10.0) is True
    assert tracker.get_free(0) == 14.0

    assert tracker.allocate(gpu=0, model_name="big_model", vram_gb=20.0) is False
    assert tracker.get_free(0) == 14.0

    tracker.release(gpu=0, model_name="sdxl")
    assert tracker.get_free(0) == 24.0


def test_get_loaded_models():
    tracker = VRAMTracker(gpu_count=2, vram_per_gpu_gb=24)
    tracker.allocate(gpu=0, model_name="sdxl", vram_gb=10.0)
    tracker.allocate(gpu=1, model_name="cosyvoice2", vram_gb=3.0)

    loaded = tracker.get_loaded_models()
    assert loaded[0] == [("sdxl", 10.0)]
    assert loaded[1] == [("cosyvoice2", 3.0)]


def test_release_all():
    tracker = VRAMTracker(gpu_count=2, vram_per_gpu_gb=24)
    tracker.allocate(gpu=0, model_name="sdxl", vram_gb=10.0)
    tracker.allocate(gpu=1, model_name="cosyvoice2", vram_gb=3.0)

    tracker.release_all()
    assert tracker.get_free(0) == 24.0
    assert tracker.get_free(1) == 24.0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_vram_tracker.py -v
```

Expected: FAIL

**Step 3: Write implementation**

`src/gpu/vram_tracker.py`:
```python
import threading


class VRAMTracker:
    def __init__(self, gpu_count: int = 2, vram_per_gpu_gb: float = 24.0):
        self._lock = threading.Lock()
        self._gpu_count = gpu_count
        self._vram_total = vram_per_gpu_gb
        # {gpu_id: {model_name: vram_gb}}
        self._allocated: dict[int, dict[str, float]] = {
            i: {} for i in range(gpu_count)
        }

    def get_free(self, gpu: int) -> float:
        with self._lock:
            used = sum(self._allocated[gpu].values())
            return self._vram_total - used

    def allocate(self, gpu: int, model_name: str, vram_gb: float) -> bool:
        with self._lock:
            used = sum(self._allocated[gpu].values())
            if used + vram_gb > self._vram_total:
                return False
            self._allocated[gpu][model_name] = vram_gb
            return True

    def release(self, gpu: int, model_name: str) -> None:
        with self._lock:
            self._allocated[gpu].pop(model_name, None)

    def release_all(self) -> None:
        with self._lock:
            for gpu in self._allocated:
                self._allocated[gpu].clear()

    def get_loaded_models(self) -> dict[int, list[tuple[str, float]]]:
        with self._lock:
            return {
                gpu: [(name, vram) for name, vram in models.items()]
                for gpu, models in self._allocated.items()
            }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_vram_tracker.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/gpu/vram_tracker.py tests/test_vram_tracker.py
git commit -m "feat: add VRAM tracker for GPU memory management"
```

---

### Task 7: Model Manager

**Files:**
- Create: `src/gpu/model_manager.py`
- Create: `tests/test_model_manager.py`

**Step 1: Write the failing test**

`tests/test_model_manager.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from src.gpu.model_manager import ModelManager


@pytest.fixture
def model_configs():
    return {
        "sdxl": {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "type": "image",
            "gpu": 0,
            "vram_gb": 10,
            "resident": True,
        },
        "cosyvoice2": {
            "name": "CosyVoice2-0.5B",
            "type": "tts",
            "gpu": 1,
            "vram_gb": 3,
            "resident": True,
        },
        "wan21": {
            "name": "Wan2.1",
            "type": "video",
            "gpu": [0, 1],
            "vram_gb": 40,
            "resident": False,
            "exclusive": True,
        },
    }


def test_manager_init(model_configs):
    manager = ModelManager(model_configs, gpu_count=2, vram_per_gpu_gb=24)
    assert manager.is_loaded("sdxl") is False
    status = manager.gpu_status()
    assert status[0]["free_gb"] == 24.0


def test_can_load(model_configs):
    manager = ModelManager(model_configs, gpu_count=2, vram_per_gpu_gb=24)
    assert manager.can_load("sdxl") is True
    assert manager.can_load("wan21") is False  # needs exclusive, nothing to unload yet


def test_get_model_config(model_configs):
    manager = ModelManager(model_configs, gpu_count=2, vram_per_gpu_gb=24)
    config = manager.get_model_config("sdxl")
    assert config["type"] == "image"
    assert manager.get_model_config("nonexistent") is None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model_manager.py -v
```

Expected: FAIL

**Step 3: Write implementation**

`src/gpu/model_manager.py`:
```python
import threading
from typing import Any

from src.gpu.vram_tracker import VRAMTracker


class ModelManager:
    def __init__(
        self,
        model_configs: dict[str, dict],
        gpu_count: int = 2,
        vram_per_gpu_gb: float = 24.0,
    ):
        self._lock = threading.Lock()
        self._configs = model_configs
        self._tracker = VRAMTracker(gpu_count, vram_per_gpu_gb)
        self._loaded: dict[str, Any] = {}  # model_name -> model instance

    def get_model_config(self, model_name: str) -> dict | None:
        return self._configs.get(model_name)

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self._loaded

    def can_load(self, model_name: str) -> bool:
        config = self._configs.get(model_name)
        if config is None:
            return False

        gpus = config["gpu"] if isinstance(config["gpu"], list) else [config["gpu"]]
        vram = config["vram_gb"]

        if config.get("exclusive"):
            # Exclusive models need all GPUs free
            for gpu in gpus:
                if self._tracker.get_free(gpu) < self._tracker._vram_total:
                    return False
            return True

        vram_per_gpu = vram / len(gpus)
        return all(self._tracker.get_free(gpu) >= vram_per_gpu for gpu in gpus)

    def register_loaded(self, model_name: str, instance: Any) -> bool:
        config = self._configs.get(model_name)
        if config is None:
            return False

        gpus = config["gpu"] if isinstance(config["gpu"], list) else [config["gpu"]]
        vram_per_gpu = config["vram_gb"] / len(gpus)

        with self._lock:
            for gpu in gpus:
                if not self._tracker.allocate(gpu, model_name, vram_per_gpu):
                    # Rollback
                    for g in gpus:
                        self._tracker.release(g, model_name)
                    return False
            self._loaded[model_name] = instance
            return True

    def unload(self, model_name: str) -> None:
        config = self._configs.get(model_name)
        if config is None or model_name not in self._loaded:
            return

        gpus = config["gpu"] if isinstance(config["gpu"], list) else [config["gpu"]]
        with self._lock:
            for gpu in gpus:
                self._tracker.release(gpu, model_name)
            instance = self._loaded.pop(model_name, None)
            del instance

    def unload_all(self) -> list[str]:
        unloaded = list(self._loaded.keys())
        for name in unloaded:
            self.unload(name)
        return unloaded

    def get_instance(self, model_name: str) -> Any | None:
        return self._loaded.get(model_name)

    def gpu_status(self) -> dict[int, dict]:
        loaded = self._tracker.get_loaded_models()
        return {
            gpu: {
                "free_gb": self._tracker.get_free(gpu),
                "models": models,
            }
            for gpu, models in loaded.items()
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model_manager.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/gpu/model_manager.py tests/test_model_manager.py
git commit -m "feat: add model manager with load/unload and VRAM tracking"
```

---

### Task 8: Storage Service (NAS)

**Files:**
- Create: `src/storage/nas.py`
- Create: `tests/test_storage.py`

**Step 1: Write the failing test**

`tests/test_storage.py`:
```python
import os
import tempfile
from pathlib import Path

from src.storage.nas import StorageService


def test_save_and_get_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageService(outputs_path=tmpdir)
        content = b"fake image data"
        file_path = storage.save(content, task_id="abc123", filename="output.png")
        assert Path(file_path).exists()
        assert storage.get_url("abc123", "output.png") == f"{tmpdir}/abc123/output.png"


def test_list_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageService(outputs_path=tmpdir)
        storage.save(b"data1", task_id="t1", filename="a.png")
        storage.save(b"data2", task_id="t1", filename="b.png")
        files = storage.list_files("t1")
        assert len(files) == 2


def test_delete_task_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageService(outputs_path=tmpdir)
        storage.save(b"data", task_id="t2", filename="x.png")
        storage.delete_task("t2")
        assert not Path(tmpdir, "t2").exists()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_storage.py -v
```

Expected: FAIL

**Step 3: Write implementation**

`src/storage/nas.py`:
```python
import shutil
from pathlib import Path

from src.config import get_settings


class StorageService:
    def __init__(self, outputs_path: str | None = None):
        self._base = Path(outputs_path or get_settings().NAS_OUTPUTS_PATH)

    def save(self, content: bytes, task_id: str, filename: str) -> str:
        task_dir = self._base / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        file_path = task_dir / filename
        file_path.write_bytes(content)
        return str(file_path)

    def get_url(self, task_id: str, filename: str) -> str:
        return str(self._base / task_id / filename)

    def list_files(self, task_id: str) -> list[str]:
        task_dir = self._base / task_id
        if not task_dir.exists():
            return []
        return [str(f) for f in task_dir.iterdir() if f.is_file()]

    def delete_task(self, task_id: str) -> None:
        task_dir = self._base / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_storage.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/storage/nas.py tests/test_storage.py
git commit -m "feat: add NAS storage service"
```

---

### Task 9: FastAPI App & Task Routes

**Files:**
- Create: `src/api/main.py`
- Create: `src/api/deps.py`
- Create: `src/api/routes/tasks.py`
- Create: `tests/test_api_tasks.py`

**Step 1: Write the failing test**

`tests/test_api_tasks.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_models_endpoint(client):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpu_status" in data
    assert "models" in data
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_api_tasks.py -v
```

Expected: FAIL

**Step 3: Write FastAPI app**

`src/api/deps.py`:
```python
from functools import lru_cache

from src.config import load_model_configs
from src.gpu.model_manager import ModelManager
from src.storage.nas import StorageService


@lru_cache
def get_model_manager() -> ModelManager:
    configs = load_model_configs()
    return ModelManager(configs, gpu_count=2, vram_per_gpu_gb=24.0)


@lru_cache
def get_storage() -> StorageService:
    return StorageService()
```

`src/api/routes/tasks.py`:
```python
from fastapi import APIRouter

from src.api.deps import get_model_manager

router = APIRouter(prefix="/api/v1")


@router.get("/models")
async def list_models():
    manager = get_model_manager()
    return {
        "gpu_status": manager.gpu_status(),
        "models": {
            name: {
                "type": cfg["type"],
                "loaded": manager.is_loaded(name),
                "gpu": cfg["gpu"],
                "vram_gb": cfg["vram_gb"],
            }
            for name, cfg in manager._configs.items()
        },
    }
```

`src/api/main.py`:
```python
from fastapi import FastAPI

from src.api.routes import tasks


def create_app() -> FastAPI:
    app = FastAPI(title="Mind Center", version="0.1.0")

    app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
    app.include_router(tasks.router)

    return app


app = create_app()
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_api_tasks.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/api/main.py src/api/deps.py src/api/routes/tasks.py tests/test_api_tasks.py
git commit -m "feat: add FastAPI app with health and models endpoints"
```

---

### Task 10: Generate Routes (Image/Video/TTS)

**Files:**
- Create: `src/api/routes/generate.py`
- Create: `tests/test_api_generate.py`

**Step 1: Write the failing test**

`tests/test_api_generate.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_generate_image_returns_task_id(client):
    with patch("src.api.routes.generate.dispatch_task") as mock:
        mock.return_value = "fake-task-id"
        resp = await client.post(
            "/api/v1/generate/image",
            json={"prompt": "a cat"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["task_id"] == "fake-task-id"
        assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_generate_tts_returns_task_id(client):
    with patch("src.api.routes.generate.dispatch_task") as mock:
        mock.return_value = "fake-task-id"
        resp = await client.post(
            "/api/v1/generate/tts",
            json={"text": "hello world"},
        )
        assert resp.status_code == 202


@pytest.mark.asyncio
async def test_generate_video_returns_task_id(client):
    with patch("src.api.routes.generate.dispatch_task") as mock:
        mock.return_value = "fake-task-id"
        resp = await client.post(
            "/api/v1/generate/video",
            json={"prompt": "sunset timelapse"},
        )
        assert resp.status_code == 202
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_api_generate.py -v
```

Expected: FAIL

**Step 3: Write generate routes**

`src/api/routes/generate.py`:
```python
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.models.schemas import (
    ImageGenerateRequest,
    VideoGenerateRequest,
    TTSRequest,
)

router = APIRouter(prefix="/api/v1/generate")


def dispatch_task(task_type: str, params: dict) -> str:
    """Dispatch a task to Celery. Returns task_id."""
    task_id = str(uuid.uuid4())
    # Celery integration will be wired in Task 12
    return task_id


@router.post("/image", status_code=202)
async def generate_image(req: ImageGenerateRequest):
    task_id = dispatch_task("image", req.model_dump())
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "pending", "type": "image"},
    )


@router.post("/video", status_code=202)
async def generate_video(req: VideoGenerateRequest):
    task_id = dispatch_task("video", req.model_dump())
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "pending", "type": "video"},
    )


@router.post("/tts", status_code=202)
async def generate_tts(req: TTSRequest):
    task_id = dispatch_task("tts", req.model_dump())
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "pending", "type": "tts"},
    )
```

**Step 4: Register router in main.py**

Add to `src/api/main.py`:
```python
from src.api.routes import tasks, generate

def create_app() -> FastAPI:
    app = FastAPI(title="Mind Center", version="0.1.0")
    app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
    app.include_router(tasks.router)
    app.include_router(generate.router)
    return app
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_api_generate.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/api/routes/generate.py src/api/main.py tests/test_api_generate.py
git commit -m "feat: add generate routes for image/video/tts"
```

---

### Task 11: Understand Route (vLLM Proxy)

**Files:**
- Create: `src/api/routes/understand.py`
- Create: `tests/test_api_understand.py`

**Step 1: Write the failing test**

`tests/test_api_understand.py`:
```python
import pytest
from unittest.mock import patch, AsyncMock
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_understand_image(client):
    mock_response = {"text": "A cat sitting on a table", "model": "qwen25-vl"}
    with patch("src.api.routes.understand.call_vllm", new_callable=AsyncMock) as mock:
        mock.return_value = mock_response
        resp = await client.post(
            "/api/v1/understand/image",
            json={"image_url": "/path/to/img.png", "question": "What is this?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "A cat sitting on a table"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_api_understand.py -v
```

Expected: FAIL

**Step 3: Write understand route**

`src/api/routes/understand.py`:
```python
import httpx
from fastapi import APIRouter

from src.config import get_settings
from src.models.schemas import ImageUnderstandRequest

router = APIRouter(prefix="/api/v1/understand")


async def call_vllm(image_url: str, question: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.VLLM_BASE_URL}/v1/chat/completions",
            json={
                "model": "Qwen2.5-VL-7B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            },
            timeout=120.0,
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "model": data.get("model", "qwen25-vl")}


@router.post("/image")
async def understand_image(req: ImageUnderstandRequest):
    result = await call_vllm(req.image_url, req.question)
    return result
```

**Step 4: Register router in main.py**

Add `understand` import and `app.include_router(understand.router)` in `create_app()`.

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_api_understand.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/api/routes/understand.py src/api/main.py tests/test_api_understand.py
git commit -m "feat: add understand route proxying to vLLM"
```

---

### Task 12: Wire Celery Dispatch into Generate Routes

**Files:**
- Modify: `src/api/routes/generate.py`
- Create: `src/workers/image_worker.py`
- Create: `src/workers/tts_worker.py`
- Create: `src/workers/video_worker.py`
- Create: `tests/test_workers.py`

**Step 1: Write the failing test**

`tests/test_workers.py`:
```python
from src.workers.image_worker import generate_image_task
from src.workers.tts_worker import generate_tts_task
from src.workers.video_worker import generate_video_task


def test_image_task_is_celery_task():
    assert hasattr(generate_image_task, "delay")
    assert generate_image_task.name == "src.workers.image_worker.generate_image_task"


def test_tts_task_is_celery_task():
    assert hasattr(generate_tts_task, "delay")


def test_video_task_is_celery_task():
    assert hasattr(generate_video_task, "delay")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_workers.py -v
```

Expected: FAIL

**Step 3: Write worker stubs**

`src/workers/image_worker.py`:
```python
from src.workers.celery_app import celery_app


@celery_app.task(bind=True, name="src.workers.image_worker.generate_image_task")
def generate_image_task(self, task_id: str, params: dict):
    """Generate image using diffusers. GPU model loading handled here."""
    self.update_state(state="RUNNING", meta={"progress": 0})

    # TODO: Load model via ModelManager, run inference, save to NAS
    # Placeholder for now
    return {"task_id": task_id, "status": "completed", "file": f"{task_id}/output.png"}
```

`src/workers/tts_worker.py`:
```python
from src.workers.celery_app import celery_app


@celery_app.task(bind=True, name="src.workers.tts_worker.generate_tts_task")
def generate_tts_task(self, task_id: str, params: dict):
    """Generate speech using CosyVoice2 or Qwen TTS."""
    self.update_state(state="RUNNING", meta={"progress": 0})

    # TODO: Load model, run inference, save audio to NAS
    return {"task_id": task_id, "status": "completed", "file": f"{task_id}/output.wav"}
```

`src/workers/video_worker.py`:
```python
from src.workers.celery_app import celery_app


@celery_app.task(bind=True, name="src.workers.video_worker.generate_video_task")
def generate_video_task(self, task_id: str, params: dict):
    """Generate video using Wan2.1. Requires exclusive dual-GPU access."""
    self.update_state(state="RUNNING", meta={"progress": 0})

    # TODO: Unload all models, load Wan2.1 on dual GPU, run inference, restore models
    return {"task_id": task_id, "status": "completed", "file": f"{task_id}/output.mp4"}
```

**Step 4: Update generate.py dispatch_task to use Celery**

```python
import uuid
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.models.schemas import ImageGenerateRequest, VideoGenerateRequest, TTSRequest
from src.workers.image_worker import generate_image_task
from src.workers.tts_worker import generate_tts_task
from src.workers.video_worker import generate_video_task

router = APIRouter(prefix="/api/v1/generate")

TASK_MAP = {
    "image": generate_image_task,
    "video": generate_video_task,
    "tts": generate_tts_task,
}


def dispatch_task(task_type: str, params: dict) -> str:
    task_id = str(uuid.uuid4())
    celery_task = TASK_MAP[task_type]
    celery_task.delay(task_id, params)
    return task_id
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_workers.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/workers/ src/api/routes/generate.py tests/test_workers.py
git commit -m "feat: add celery worker stubs and wire dispatch"
```

---

### Task 13: WebSocket Notifications

**Files:**
- Create: `src/api/websocket.py`
- Create: `tests/test_websocket.py`

**Step 1: Write the failing test**

`tests/test_websocket.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.api.websocket import ConnectionManager


def test_connection_manager_init():
    manager = ConnectionManager()
    assert len(manager.active_connections) == 0


@pytest.mark.asyncio
async def test_websocket_endpoint():
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Just verify the websocket route is registered
        routes = [r.path for r in app.routes]
        assert "/ws/tasks/{task_id}" in routes
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_websocket.py -v
```

Expected: FAIL

**Step 3: Write WebSocket manager**

`src/api/websocket.py`:
```python
import json
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    def __init__(self):
        # {task_id: [websocket, ...]}
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)

    def disconnect(self, task_id: str, websocket: WebSocket):
        if task_id in self.active_connections:
            self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]

    async def send_update(self, task_id: str, data: dict):
        if task_id in self.active_connections:
            message = json.dumps(data)
            for ws in self.active_connections[task_id]:
                await ws.send_text(message)


ws_manager = ConnectionManager()
```

**Step 4: Add WebSocket route to main.py**

Add to `create_app()`:
```python
from fastapi import WebSocket, WebSocketDisconnect
from src.api.websocket import ws_manager

@app.websocket("/ws/tasks/{task_id}")
async def websocket_task(websocket: WebSocket, task_id: str):
    await ws_manager.connect(task_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(task_id, websocket)
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_websocket.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/api/websocket.py src/api/main.py tests/test_websocket.py
git commit -m "feat: add WebSocket connection manager and task notifications"
```

---

### Task 14: Startup Scripts

**Files:**
- Create: `scripts/start_vllm.sh`
- Create: `scripts/start_workers.sh`
- Create: `scripts/start_api.sh`

**Step 1: Write startup scripts**

`scripts/start_vllm.sh`:
```bash
#!/bin/bash
set -e
echo "Starting vLLM on GPU1..."
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen2.5-VL-7B-Instruct \
    --port 8100 \
    --max-model-len 4096 \
    --trust-remote-code
```

`scripts/start_workers.sh`:
```bash
#!/bin/bash
set -e

echo "Starting Celery workers..."

# Image + TTS worker on GPU0/GPU1
CUDA_VISIBLE_DEVICES=0 celery -A src.workers.celery_app worker \
    --queue=image \
    --concurrency=1 \
    --hostname=image@%h \
    -l info &

CUDA_VISIBLE_DEVICES=1 celery -A src.workers.celery_app worker \
    --queue=tts \
    --concurrency=1 \
    --hostname=tts@%h \
    -l info &

# Video worker (will use both GPUs when active)
celery -A src.workers.celery_app worker \
    --queue=video \
    --concurrency=1 \
    --hostname=video@%h \
    -l info &

wait
```

`scripts/start_api.sh`:
```bash
#!/bin/bash
set -e
echo "Starting FastAPI..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Step 2: Make scripts executable**

```bash
chmod +x scripts/*.sh
```

**Step 3: Commit**

```bash
git add scripts/
git commit -m "feat: add startup scripts for vLLM, workers, and API"
```

---

### Task 15: Run All Tests & Final Verification

**Step 1: Run full test suite**

```bash
uv run pytest -v
```

Expected: All tests PASS

**Step 2: Verify FastAPI starts (no GPU required)**

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
sleep 2
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/models
kill %1
```

Expected: Health returns `{"status": "ok"}`, models returns GPU status.

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: final adjustments from integration testing"
```

---

## Summary

| Task | What | Depends On |
|------|------|------------|
| 1 | Project scaffolding | — |
| 2 | Settings & config | 1 |
| 3 | Database models | 1 |
| 4 | Pydantic schemas | 1 |
| 5 | Celery app config | 2 |
| 6 | VRAM tracker | 1 |
| 7 | Model manager | 6 |
| 8 | Storage service | 2 |
| 9 | FastAPI app + task routes | 2, 7 |
| 10 | Generate routes | 4, 9 |
| 11 | Understand route | 2, 9 |
| 12 | Wire Celery dispatch | 5, 10 |
| 13 | WebSocket | 9 |
| 14 | Startup scripts | 12 |
| 15 | Integration test | All |

**Parallelizable groups:**
- Tasks 3, 4, 5, 6, 8 can run in parallel after Task 2
- Tasks 10, 11, 13 can run in parallel after Task 9
