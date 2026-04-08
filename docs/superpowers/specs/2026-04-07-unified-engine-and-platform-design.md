# Unified Engine & Platform Enhancement Design

> Date: 2026-04-07
> Status: Approved
> Scope: 4 modules — unified inference, workflow app publish, task queue, LLM node enhancement

---

## 1. Unified Inference Engine Framework

### Goal

Replace separate TTS/LLM/Image engine management with a single `InferenceAdapter` protocol. All model types share the same lifecycle (load/unload/infer) and are managed by one `ModelManager`.

### Core Interface

```python
# backend/src/services/inference/base.py

@dataclass(frozen=True)
class InferenceResult:
    data: bytes
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

class InferenceAdapter(ABC):
    model_type: str            # "tts" | "llm" | "image"
    estimated_vram_mb: int

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self._model = None

    @abstractmethod
    async def load(self, device: str) -> None: ...
    def unload(self) -> None:
        self._model = None
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    @abstractmethod
    async def infer(self, params: dict) -> InferenceResult: ...
```

Note: Use ABC instead of Protocol because adapters share common state (`model_path`, `device`, `_model`) and default `unload()`/`is_loaded` implementations. The existing `TTSEngine` already uses this pattern.

### TTS Engine Migration

Existing `TTSEngine` base class inherits `InferenceAdapter`. The `synthesize()` method is preserved as a TTS-specific convenience; `infer()` delegates to `synthesize()` and wraps the result as `InferenceResult`. The existing `TTSResult` dataclass is kept for backward compatibility within TTS code paths.

Current `TTSEngine.load()` is synchronous — the `InferenceAdapter.load()` is async, so `ModelManager` wraps sync loads with `asyncio.to_thread()` (as `model_scheduler.py` already does).

```
InferenceAdapter
├── TTSEngine (migrated)
│   ├── CosyVoice2Engine
│   ├── IndexTTS2Engine
│   ├── MossTTSEngine
│   └── Qwen3TTSEngine
├── VLLMAdapter          # httpx client to external vLLM
└── ImageAdapter         # reserved, PyTorch direct load
```

### Model Registry

`configs/models.yaml` declares all available models:

```yaml
models:
  - id: cosyvoice2-0.5b
    type: tts
    adapter: src.workers.tts_engines.cosyvoice2.CosyVoice2Engine
    path: /media/heygo/Program/models/nous/tts/cosyvoice2-0.5b
    vram_mb: 2000

  - id: qwen3.5-35b-a3b
    type: llm
    adapter: src.services.inference.llm_vllm.VLLMAdapter
    path: /media/heygo/Program/models/nous/llm/Qwen3.5-35B-A3B
    vram_mb: 0  # managed by external vLLM
    params:
      vllm_base_url: http://localhost:8100
```

### ModelManager

Refactored from existing `model_scheduler.py`. The current scheduler uses module-level dicts (`_loaded_models`, `_references`, `_last_used`) with a shared `asyncio.Lock`. The refactor moves this into a `ModelManager` class for testability and cleaner dependency injection.

Key behaviors preserved from `model_scheduler.py`:
- Reference counting: workflows register/unregister model usage
- Idle timeout: unload models after configurable TTL (default 300s)
- Resident models: pinned models skip eviction and idle timeout
- `asyncio.to_thread()` for blocking PyTorch load/unload calls

New behaviors:
- Per-model async lock: prevents concurrent load/unload on same model (current code has one global lock)
- GPUAllocator: extracted from `gpu_monitor.py`, picks optimal GPU based on free VRAM (current `get_device_for_engine` is static config-based)
- Unified adapter registry: replaces separate `tts_engines/registry.py` and `llm_engines/registry.py`

Migration: `gpu_monitor.py`'s `check_and_evict()` currently reaches into `model_scheduler._lock` and `_references` directly. After refactor, `ModelManager` exposes a clean `evict_lru(gpu_index)` method instead.

### File Changes

| Action | File |
|--------|------|
| New | `src/services/inference/base.py` |
| New | `src/services/inference/registry.py` |
| New | `src/services/inference/llm_vllm.py` |
| New | `src/services/inference/image_base.py` |
| Migrate | `src/workers/tts_engines/base.py` (inherit InferenceAdapter) |
| Refactor | `src/services/model_scheduler.py` → `src/services/model_manager.py` |
| Refactor | `src/services/gpu_monitor.py` (extract GPUAllocator logic) |

---

## 2. Workflow App Publish

### Goal

Allow users to publish a workflow as a standalone API service. External systems call it via API Key.

### Data Model

```python
# backend/src/models/workflow_app.py

class WorkflowApp(Base):
    id: int                     # Snowflake ID
    name: str                   # URL-safe slug, e.g. "my-tts-pipeline"
    display_name: str
    description: str
    workflow_id: int            # linked Workflow
    workflow_snapshot: dict     # frozen DAG at publish time
    active: bool
    exposed_inputs: list[ExposedParam]   # JSON column
    exposed_outputs: list[ExposedParam]  # JSON column
    created_at: datetime
    updated_at: datetime

class ExposedParam(BaseModel):
    node_id: str
    param_key: str
    api_name: str               # external-facing name
    param_type: str             # "string" | "number" | "file"
    description: str
    required: bool
    default: Any
```

### API Endpoints

**Management (admin token required):**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/workflows/{id}/publish` | Publish workflow as App |
| DELETE | `/api/v1/apps/{name}` | Unpublish App |
| GET | `/api/v1/apps` | List all Apps |

**External (API Key required):**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/apps` | Discover available Apps |
| POST | `/v1/apps/{name}` | Execute App |

### Execution Flow

```
External request → API Key auth → merge params into workflow_snapshot
→ TaskQueue.submit() → WorkflowExecutor runs DAG → return result + record usage
```

The DAG is frozen at publish time so subsequent edits don't affect the live service.

### Publish Wizard UI (3-step)

**Step 1 — Basic Info:**
- Name: auto-generated slug from display name, editable
- Real-time uniqueness validation
- Display name + multi-line description

**Step 2 — Parameter Configuration:**
- Semi-transparent node graph as background, highlight exposable nodes
- Click node to see parameter list, checkbox to expose
- Each exposed param: API name, description, required flag, default value
- Right side: live JSON Schema preview

**Step 3 — Preview & Confirm:**
- Full API documentation preview (endpoint, request body, response body)
- Code snippets: curl / Python / JavaScript, one-click copy
- "Publish" button → redirect to App management

### Post-Publish Topbar State

| State | Display |
|-------|---------|
| Unpublished | Blue "Publish" button |
| Published | Green badge "Published" + "Update" / "Unpublish" buttons |
| Has uncommitted changes | Yellow badge "Has Changes" with update prompt |

### App Management

In the API Management overlay, add an "Apps" tab:
- Card list: name, status, call count, last called time
- Click card to expand: API docs, Key management, call logs

### File Changes

| Action | File |
|--------|------|
| New | `src/models/workflow_app.py` |
| New | `src/api/routes/apps.py` |
| Modify | `src/models/schemas.py` (App schemas) |
| New | `frontend/src/components/overlays/PublishWizard.tsx` |
| New | `frontend/src/components/overlays/AppDetailPanel.tsx` |
| Modify | `frontend/src/components/layout/Topbar.tsx` (publish badge) |
| Modify | `frontend/src/components/overlays/ApiManagementOverlay.tsx` (Apps tab) |
| New | `frontend/src/api/apps.ts` |

---

## 3. Async Task Queue

### Goal

Replace simple `ExecutionTask` recording with a full task queue supporting concurrency limits, timeouts, priorities, and retries.

### Core Service

```python
# backend/src/services/task_queue.py

class TaskQueue:
    def __init__(self, max_concurrent: int = 4, default_timeout: int = 300):
        ...

    async def submit(
        self,
        func: Callable,
        params: dict,
        priority: int = 0,        # 0=normal, 1=high, 2=urgent
        timeout: int | None = None,
        max_retries: int = 0,
    ) -> str:  # returns task_id

    async def cancel(self, task_id: str) -> bool
    async def get_status(self, task_id: str) -> TaskStatus
```

### Mechanisms

**Concurrency control:** `asyncio.Semaphore` limits simultaneous tasks. TTS and LLM inference share GPU resources. Default max 4 concurrent (configurable per model type).

**Priority queue:** `asyncio.PriorityQueue`. External API calls = normal priority, frontend interactions = high priority. Ensures user operations aren't blocked by batch API requests.

**Timeout:** Each task wrapped in `asyncio.wait_for`. Auto-cancel on timeout. Defaults: TTS short text 30s, LLM conversation 600s, general 300s.

**Retry:** Exponential backoff (1s, 2s, 4s...). Only retries transient errors (GPU OOM, temporary failures). Parameter errors fail immediately.

### Task Status Flow

```
pending → queued → running → completed
                          → failed → retrying → running
                          → timeout
              → cancelled
```

### Integration

| Existing | After |
|----------|-------|
| `ExecutionTask` model | Preserved as persistent record |
| `WorkflowExecutor` | Submits tasks to `TaskQueue`, no longer directly awaits |
| `instance_service.py /run` | Submit via `TaskQueue.submit()`, return task_id for polling |

Note: `WorkflowExecutor` currently uses a synchronous `on_progress` callback for WebSocket updates. The `TaskQueue` wrapper must preserve this — the queue runs the executor in a task, and the progress callback still pushes to WebSocket connections. The queue adds lifecycle management (timeout, retry) around the executor, not inside it.

### File Changes

| Action | File |
|--------|------|
| New | `src/services/task_queue.py` |
| Modify | `src/api/routes/execution_tasks.py` (integrate queue status) |
| Modify | `src/services/workflow_executor.py` (submit through queue) |
| Modify | `src/api/routes/instance_service.py` (async submit + poll) |

---

## 4. LLM Node Enhancement

### Goal

Enhance the existing LLM Node in the declarative node framework with streaming output, dynamic model selection, and real-time canvas feedback.

### Node Parameters

```
LLM Node
├── model: dropdown (fetched from ModelManager's loaded LLM list)
├── system_prompt: multiline text
├── user_prompt: multiline text (supports {{input}} template vars)
├── temperature: slider 0-2
├── max_tokens: number input
└── stream: toggle (default on)
```

### VLLMAdapter Implementation

```python
# backend/src/services/inference/llm_vllm.py

class VLLMAdapter(InferenceAdapter):
    model_type = "llm"
    estimated_vram_mb = 0  # managed by external vLLM

    async def load(self, device: str) -> None:
        resp = await self._client.get(f"{self._base_url}/v1/models")
        self._loaded = resp.status_code == 200

    async def infer(self, params: dict) -> InferenceResult:
        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions", json=params
        )
        return InferenceResult(data=resp.content, content_type="application/json")

    async def infer_stream(self, params: dict) -> AsyncIterator[bytes]:
        async with self._client.stream(
            "POST", f"{self._base_url}/v1/chat/completions",
            json={**params, "stream": True}
        ) as resp:
            async for line in resp.aiter_lines():
                yield line.encode() + b"\n"
```

### Streaming Execution Feedback

When the workflow executor hits an LLM Node with `stream=True`, it pushes tokens through the existing WebSocket progress channel:

```json
{"type": "node_stream", "node_id": "llm_1", "token": "Hello"}
{"type": "node_stream", "node_id": "llm_1", "token": " world"}
{"type": "node_complete", "node_id": "llm_1", "output": "Hello world"}
```

The frontend renders streaming text inside the LLM Node on the canvas.

### File Changes

| Action | File |
|--------|------|
| New | `src/services/inference/llm_vllm.py` |
| Modify | `src/services/workflow_executor.py` (LLM node streaming execution) |
| Modify | `frontend/src/components/nodes/DeclarativeNode.tsx` (render streaming text) |
| Modify | `frontend/src/config/engineParams.ts` (LLM node param config) |

---

## Available Models

| Type | Model | Path |
|------|-------|------|
| TTS | cosyvoice2-0.5b | `/media/heygo/Program/models/nous/tts/cosyvoice2-0.5b` |
| TTS | indextts-2 | `/media/heygo/Program/models/nous/tts/indextts-2` |
| TTS | moss-tts | `/media/heygo/Program/models/nous/tts/moss-tts` |
| TTS | qwen3-tts-1.7b-base | `/media/heygo/Program/models/nous/tts/qwen3-tts-1.7b-base` |
| TTS | qwen3-tts-1.7b-customvoice | `/media/heygo/Program/models/nous/tts/qwen3-tts-1.7b-customvoice` |
| TTS | qwen3-tts-1.7b-voicedesign | `/media/heygo/Program/models/nous/tts/qwen3-tts-1.7b-voicedesign` |
| LLM | gemma-4-26B-A4B-it | `/media/heygo/Program/models/nous/llm/gemma-4-26B-A4B-it` |
| LLM | Qwen3.5-35B-A3B | `/media/heygo/Program/models/nous/llm/Qwen3.5-35B-A3B` |

## Hardware

- 2x NVIDIA RTX 3090 (24GB VRAM each, 48GB total)
- vLLM: external process with `--tensor-parallel-size 2`
