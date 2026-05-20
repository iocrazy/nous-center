# Image Component Multi-GPU — PR-5a (Backend: component state + preload) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backend plumbing for the component loader UI (PR-5b): batch-preload a unet+clip+vae combo into the image runner, and expose each component's live load state (loaded/loading/cold/failed) via a query endpoint + a `/ws/models` `component_state_changed` push.

**Architecture:** Event-driven (matches the existing `/ws/models` `model_status` pattern). The runner subprocess holds the real component L1 cache; it emits a new `ComponentEvent` (loading→loaded/failed) whenever a component loads — on BOTH the workflow-run path and the new preload path. The RunnerClient forwards those events to a backend in-memory `ComponentStateRegistry`, which the `GET /state` endpoint reads and which fans out as a WS broadcast. `POST /preload` sends a new `PreloadComponents` message to the image runner and returns `202` immediately; progress arrives via WS. No blocking backend→runner RPC.

**Tech Stack:** Python 3.12 / FastAPI / multiprocessing + msgpack pipe (runner IPC) / pytest + pytest-asyncio.

**Branch:** `feat/image-component-multigpu-pr5a-backend` (from master).

**Scope split (user decision 2026-05-20):** PR-5a = backend only (this plan). PR-5b = frontend (4 node React components + `useComponentState` hook + `ComponentSelectWidget` + palette subcategory + 四态 display), separate plan/PR, verified in the real vite app.

**Spec:** `docs/superpowers/specs/2026-05-19-image-component-multi-gpu-design.md` §6.2 (preload batch endpoint), §6.3 (state query + WS), §6.1 (four states).

**Prereqs (merged):** PR-1..PR-4 (#112–#116). `ModelManager.get_or_load_image_adapter` + `get_or_load_component` + `is_component_loaded` + `to_component_key` exist. CI is green (#117).

---

## Key design decisions (read first)

1. **State source = runner events, not RPC.** The runner ModelManager holds the truth; it emits `ComponentEvent` on every component state transition. Backend keeps an in-memory mirror (`ComponentStateRegistry`). `GET /state` returns the mirror; unknown keys default to `"cold"`. Backend restart → mirror empty (matches L1 being in-memory). Avoids a per-poll blocking RPC and reuses the WS push model the frontend already consumes via `useLiveChannel('/ws/models')`.
2. **Component wire key** = `component_state_key(spec)` — a stable string derived from `to_component_key` (file|device|dtype|lora_sig). Same function the runner uses to tag events and the endpoint uses to key the registry. PR-5b computes the identical string from the loader-node descriptor to match. Defined once in `component_spec.py`.
3. **Events fire on both paths.** `get_or_load_image_adapter` gains an optional `on_event` async callback; the runner passes one that sends `ComponentEvent` over the pipe. Both `_node_executor` (run) and `_handle_preload_components` (preload) call `get_or_load_image_adapter`, so state stays in sync regardless of trigger.
4. **Preload is fire-and-forget + 202.** `POST /preload` validates, generates a task_id, sends `PreloadComponents` to the image runner, returns `202 {task_id}`. The runner loads (emitting `ComponentEvent`s) → WS push. No awaiting the load in the request.
5. **WS event name** = `component_state_changed` (new `ws_manager.broadcast_component_state`), distinct from `model_status`, so the frontend can route it cleanly.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `backend/src/runner/protocol.py` | Modify | `PreloadComponents` (main→runner) + `ComponentEvent` (runner→main) dataclasses + codec registration |
| `backend/src/services/inference/component_spec.py` | Modify | `component_state_key(spec) -> str` |
| `backend/src/services/model_manager.py` | Modify | `get_or_load_image_adapter(..., on_event=None)` emits per-component loading/loaded/failed |
| `backend/src/runner/runner_process.py` | Modify | `_handle_preload_components`; wire `on_event` (run + preload) → `ComponentEvent` over pipe |
| `backend/src/runner/client.py` | Modify | `_demux_loop` ComponentEvent → `on_component_event` callback; `preload_components(...)` send |
| `backend/src/services/component_state.py` | Create | `ComponentStateRegistry` (in-memory state mirror) |
| `backend/src/api/websocket.py` | Modify | `broadcast_component_state(component_key, state, error)` → event `component_state_changed` |
| `backend/src/api/main.py` | Modify | lifespan: create registry → `app.state.component_state_registry`; register image client `on_component_event` → registry + WS |
| `backend/src/api/routes/models.py` | Modify | `GET /api/v1/models/components/state` + `POST /api/v1/models/components/preload` |
| `backend/tests/test_*` | Create/Modify | per task |

---

## Task 1: Protocol — `PreloadComponents` + `ComponentEvent`

**Files:**
- Modify: `backend/src/runner/protocol.py`
- Test: `backend/tests/test_runner_protocol.py`

- [ ] **Step 1: failing test** — add two messages to `ALL_MESSAGES` in `tests/test_runner_protocol.py` (the parametrized round-trip covers msgpack+json):

```python
    P.PreloadComponents(
        task_id=7,
        components={
            "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
            "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
            "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
        },
        pipeline_class="Flux2KleinPipeline",
    ),
    P.ComponentEvent(component_key="/m/u.safe|cuda:1|bfloat16|", state="loaded", error=None),
```

- [ ] **Step 2: run → FAIL**

Run: `cd backend && uv run pytest tests/test_runner_protocol.py -q`
Expected: FAIL — `AttributeError: module 'src.runner.protocol' has no attribute 'PreloadComponents'`.

- [ ] **Step 3: implement** — in `protocol.py`, add the dataclasses (PreloadComponents in the "主进程 -> runner" section, ComponentEvent in the "runner -> 主进程" section):

```python
@dataclass(frozen=True)
class PreloadComponents:
    """主进程 → image runner：批量预热一组 unet+clip+vae(spec §6.2)。
    components = {"unet": <spec dict>, "clip": <spec dict>, "vae": <spec dict>}。
    runner 走 get_or_load_image_adapter,过程中发 ComponentEvent。"""
    task_id: int
    components: dict[str, Any]
    pipeline_class: str = "Flux2KleinPipeline"
    kind: Literal["preload_components"] = "preload_components"
```

```python
@dataclass(frozen=True)
class ComponentEvent:
    """image runner → 主进程:单个组件加载状态迁移(spec §6.1 四态)。
    component_key = component_state_key(spec)(file|device|dtype|lora_sig)。"""
    component_key: str
    state: Literal["loading", "loaded", "failed", "cold"]
    error: str | None = None
    kind: Literal["component_event"] = "component_event"
```

Register both in `_KIND_TO_CLASS` and extend the `Message` union:

```python
    "preload_components": PreloadComponents,
    "component_event": ComponentEvent,
```
```python
Message = (
    LoadModel | UnloadModel | RunNode | Abort | Ping | PreloadComponents
    | Ready | NodeResult | NodeProgress | ModelEvent | Pong | ComponentEvent
)
```

- [ ] **Step 4: run → PASS**

Run: `cd backend && uv run pytest tests/test_runner_protocol.py -q`
Expected: PASS.

- [ ] **Step 5: commit**

```bash
git add backend/src/runner/protocol.py backend/tests/test_runner_protocol.py
git commit -m "feat(runner): PR-5a — PreloadComponents + ComponentEvent protocol messages"
```

---

## Task 2: `component_state_key(spec)`

**Files:**
- Modify: `backend/src/services/inference/component_spec.py`
- Test: `backend/tests/test_component_spec.py`

- [ ] **Step 1: failing test** — append to `tests/test_component_spec.py`:

```python
def test_component_state_key_stable_and_lora_aware():
    from src.services.inference.base import LoRASpec
    from src.services.inference.component_spec import ComponentSpec, component_state_key

    base = ComponentSpec(kind="unet", file="/m/u.safe", device="cuda:1", dtype="bfloat16", adapter_arch="flux2")
    assert component_state_key(base) == "/m/u.safe|cuda:1|bfloat16|"
    # lora-order-independent, strength-sensitive
    a = base.model_copy(update={"loras": [LoRASpec(name="x", strength=0.8), LoRASpec(name="y", strength=0.4)]})
    b = base.model_copy(update={"loras": [LoRASpec(name="y", strength=0.4), LoRASpec(name="x", strength=0.8)]})
    assert component_state_key(a) == component_state_key(b)
    assert component_state_key(a) != component_state_key(base)
```

- [ ] **Step 2: run → FAIL** — `cd backend && uv run pytest tests/test_component_spec.py -q` → ImportError on `component_state_key`.

- [ ] **Step 3: implement** — add to `component_spec.py` (after `to_component_key`):

```python
def component_state_key(spec: ComponentSpec) -> str:
    """Stable wire/UI string key for one component's load state. Derived from
    to_component_key so it matches the L1 cache identity: file|device|dtype|loras.
    LoRAs are sorted (order-independent) as 'name@strength' joined by '+'. The
    frontend (PR-5b) computes the identical string from the loader-node descriptor."""
    file, device, dtype, lora_set = to_component_key(spec)
    lora_sig = "+".join(sorted(f"{name}@{strength}" for name, strength in lora_set))
    return f"{file}|{device}|{dtype}|{lora_sig}"
```

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_component_spec.py -q`.

- [ ] **Step 5: commit**

```bash
git add backend/src/services/inference/component_spec.py backend/tests/test_component_spec.py
git commit -m "feat(image): PR-5a — component_state_key (stable wire/UI key)"
```

---

## Task 3: `get_or_load_image_adapter(on_event=...)` emits per-component state

**Files:**
- Modify: `backend/src/services/model_manager.py`
- Test: `backend/tests/test_get_or_load_image_adapter.py`

> The method already resolves devices, loads base modules via `get_or_load_component`, assembles + caches the adapter. Add an optional async `on_event(component_key, state, error)` callback fired around each base-module load: `loading` before, `loaded` after success, `failed` on exception. Fire for unet/clip/vae. On a combo cache HIT, fire `loaded` for all three (so a re-run still reports current state).

- [ ] **Step 1: failing test** — append:

```python
@pytest.mark.asyncio
async def test_image_adapter_emits_component_events(stubbed):
    from src.services.inference.component_spec import component_state_key
    mm, module_loads, assemble_calls = stubbed
    events = []

    async def _on_event(key, state, error):
        events.append((key, state, error))

    comps = _comps()
    await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline", on_event=_on_event)

    # each component: loading then loaded (resolved devices → keys)
    keys = {component_state_key(mm._resolve_component_device(comps[k])) for k in ("unet", "clip", "vae")}
    loaded = {k for (k, s, _e) in events if s == "loaded"}
    loading = {k for (k, s, _e) in events if s == "loading"}
    assert keys <= loaded
    assert keys <= loading


@pytest.mark.asyncio
async def test_image_adapter_emits_failed_on_load_error(mm, monkeypatch):
    from src.services.inference.component_spec import component_state_key
    def _boom(spec):
        raise RuntimeError("synthetic load fail")
    monkeypatch.setattr(mm, "_load_component_module", _boom)
    events = []
    async def _on_event(key, state, error):
        events.append((key, state, error))
    with pytest.raises(RuntimeError):
        await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline", on_event=_on_event)
    assert any(s == "failed" for (_k, s, _e) in events)
```

- [ ] **Step 2: run → FAIL** — `get_or_load_image_adapter` has no `on_event` kwarg → TypeError.

- [ ] **Step 3: implement** — change `get_or_load_image_adapter` signature + body. Add `on_event=None`. Wrap each base-load with emit. Use a helper:

```python
    async def get_or_load_image_adapter(self, components: dict, pipeline_class: str = "Flux2KleinPipeline", on_event=None):
        from src.services.inference.component_spec import to_component_key, component_state_key
        from src.services.inference.image_diffusers import DiffusersImageBackend

        async def _emit(spec, state, error=None):
            if on_event is not None:
                await on_event(component_state_key(spec), state, error)

        resolved = {k: self._resolve_component_device(s) for k, s in components.items()}
        combo_key = (pipeline_class,) + tuple(
            to_component_key(resolved[k]) for k in ("unet", "clip", "vae"))

        async with self._image_adapter_lock_for(combo_key):
            cached = self._image_adapters.get(combo_key)
            if cached is not None:
                for k in ("unet", "clip", "vae"):
                    await _emit(resolved[k], "loaded")
                return cached

            for attempt in range(2):
                try:
                    base = {k: self._base_spec(resolved[k]) for k in ("unet", "clip", "vae")}
                    loaded_modules = {}
                    for k in ("unet", "clip", "vae"):
                        await _emit(resolved[k], "loading")
                        try:
                            loaded_modules[k] = await self.get_or_load_component(base[k])
                        except Exception as e:  # noqa: BLE001
                            await _emit(resolved[k], "failed", f"{type(e).__name__}: {e}")
                            raise
                        await _emit(resolved[k], "loaded")
                    modules = {
                        "transformer": loaded_modules["unet"]["module"],
                        "text_encoder": loaded_modules["clip"]["module"],
                        "tokenizer": loaded_modules["clip"]["tokenizer"],
                        "vae": loaded_modules["vae"]["module"],
                    }
                    adapter = DiffusersImageBackend.from_loaded_components(modules, resolved, pipeline_class)
                    self._image_adapters[combo_key] = adapter
                    return adapter
                except Exception as e:  # noqa: BLE001
                    if self._is_oom(e) and attempt == 0:
                        evicted = await self.evict_lru()
                        logger.warning(
                            "get_or_load_image_adapter OOM; evicted=%r; %s",
                            evicted, "retrying" if evicted else "nothing to evict, retry likely fails")
                        continue
                    raise
```

(Note: the `failed` emit + re-raise inside the per-component loop means the outer OOM-retry still works; on the retry pass it re-emits loading/loaded.)

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_get_or_load_image_adapter.py -q`.

- [ ] **Step 5: regression** — `cd backend && uv run pytest tests/test_runner_components_dispatch.py tests/test_image_components_e2e.py -q` (callers pass no `on_event` → default None, unchanged).

- [ ] **Step 6: commit**

```bash
git add backend/src/services/model_manager.py backend/tests/test_get_or_load_image_adapter.py
git commit -m "feat(image): PR-5a — get_or_load_image_adapter emits per-component state events"
```

---

## Task 4: Runner — preload handler + emit ComponentEvent (run + preload)

**Files:**
- Modify: `backend/src/runner/runner_process.py`
- Test: `backend/tests/test_runner_components_dispatch.py`

> Add `_handle_preload_components` (mirrors `_handle_load_model`). Add an `_on_component_event` factory that sends `ComponentEvent` over the pipe. Wire it into BOTH the preload handler and the existing components branch of `_node_executor` (pass `on_event=` to `get_or_load_image_adapter`). Pipe-reader dispatches `PreloadComponents`.

- [ ] **Step 1: failing test** — append (drives `_handle_preload_components` directly with a fake MM that emits events):

```python
@pytest.mark.asyncio
async def test_preload_components_emits_events_via_pipe():
    from src.runner.runner_process import _handle_preload_components

    class _MM:
        async def get_or_load_image_adapter(self, components, pipeline_class, on_event=None):
            await on_event("/m/u.safe|cuda:1|bfloat16|", "loading", None)
            await on_event("/m/u.safe|cuda:1|bfloat16|", "loaded", None)
            return object()

    state = _RunnerState("r", "image", [0, 1, 2], _MM())
    ch = _Collect()
    msg = P.PreloadComponents(
        task_id=3,
        components={
            "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
            "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
            "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
        },
        pipeline_class="Flux2KleinPipeline")
    await _handle_preload_components(state, ch, msg)
    evs = [m for m in ch.sent if isinstance(m, P.ComponentEvent)]
    assert ("loading" in [e.state for e in evs]) and ("loaded" in [e.state for e in evs])


@pytest.mark.asyncio
async def test_preload_components_emits_failed_on_error():
    from src.runner.runner_process import _handle_preload_components

    class _MM:
        async def get_or_load_image_adapter(self, components, pipeline_class, on_event=None):
            await on_event("/m/u.safe|cuda:1|bfloat16|", "failed", "boom")
            raise RuntimeError("boom")

    state = _RunnerState("r", "image", [0, 1, 2], _MM())
    ch = _Collect()
    msg = P.PreloadComponents(task_id=4, components={
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
    })
    await _handle_preload_components(state, ch, msg)  # must NOT raise — runner stays alive
    assert any(e.state == "failed" for e in ch.sent if isinstance(e, P.ComponentEvent))
```

- [ ] **Step 2: run → FAIL** — no `_handle_preload_components`.

- [ ] **Step 3: implement** — add the component-event sender + handler; wire pipe-reader + the run path.

Add a helper near `_build_request`:

```python
def _make_component_event_sender(ch: PipeChannel):
    """Returns an async on_event(component_key, state, error) that sends a
    ComponentEvent over the pipe — passed to ModelManager.get_or_load_image_adapter
    so component state transitions reach the backend (spec §6.1)."""
    async def _on_event(component_key: str, state: str, error: str | None = None) -> None:
        await ch.send_message(P.ComponentEvent(component_key=component_key, state=state, error=error))
    return _on_event
```

Add the handler:

```python
async def _handle_preload_components(state: _RunnerState, ch: PipeChannel, msg: P.PreloadComponents) -> None:
    """PreloadComponents → get_or_load_image_adapter（发 ComponentEvent）。不抛 —— 失败
    已通过 ComponentEvent(state=failed) 报告,runner 不能崩。"""
    from src.services.inference.component_spec import ComponentSpec
    try:
        components = {k: ComponentSpec(**v) for k, v in msg.components.items()}
    except Exception as e:  # noqa: BLE001 — bad descriptor
        await ch.send_message(P.ComponentEvent(component_key="?", state="failed", error=f"bad spec: {e}"))
        return
    on_event = _make_component_event_sender(ch)
    try:
        await state.mm.get_or_load_image_adapter(components, msg.pipeline_class, on_event=on_event)
    except Exception:  # noqa: BLE001 — already reported per-component via on_event
        pass
```

In `_pipe_reader`, add a branch (next to `LoadModel`):

```python
        elif isinstance(msg, P.PreloadComponents):
            await _handle_preload_components(state, ch, msg)
```

In `_node_executor` components branch, pass `on_event` so RUN-path loads also emit (find the `get_or_load_image_adapter` call from PR-4 and add the kwarg):

```python
            if components:
                adapter = await state.mm.get_or_load_image_adapter(
                    components, getattr(req, "pipeline_class", "Flux2KleinPipeline"),
                    on_event=_make_component_event_sender(ch))
```

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_runner_components_dispatch.py -q`.

- [ ] **Step 5: regression** — `cd backend && uv run pytest tests/ -q -k "runner" 2>&1 | tail -5`.

- [ ] **Step 6: commit**

```bash
git add backend/src/runner/runner_process.py backend/tests/test_runner_components_dispatch.py
git commit -m "feat(runner): PR-5a — preload handler + emit ComponentEvent (run + preload paths)"
```

---

## Task 5: RunnerClient — ComponentEvent dispatch + `preload_components`

**Files:**
- Modify: `backend/src/runner/client.py`
- Test: `backend/tests/test_runner_client_component.py` (new)

- [ ] **Step 1: failing test** (`backend/tests/test_runner_client_component.py`):

```python
"""PR-5a: RunnerClient routes ComponentEvent → on_component_event; preload_components sends."""
from __future__ import annotations

import asyncio
import pytest

from src.runner import protocol as P
from src.runner.client import RunnerClient


class _FakeChannel:
    def __init__(self):
        self.sent = []
        self._incoming = asyncio.Queue()

    async def send_message(self, m):
        self.sent.append(m)

    async def recv_message(self):
        return await self._incoming.get()

    def feed(self, m):
        self._incoming.put_nowait(m)


@pytest.mark.asyncio
async def test_component_event_routed_to_callback():
    ch = _FakeChannel()
    client = RunnerClient(ch)
    got = []
    client.on_component_event = lambda evt: got.append(evt)
    task = asyncio.create_task(client._demux_loop())
    ch.feed(P.ComponentEvent(component_key="/m/u|cuda:1|bfloat16|", state="loaded", error=None))
    await asyncio.sleep(0.05)
    task.cancel()
    assert got and got[0].state == "loaded" and got[0].component_key == "/m/u|cuda:1|bfloat16|"


@pytest.mark.asyncio
async def test_preload_components_sends_message():
    ch = _FakeChannel()
    client = RunnerClient(ch)
    await client.preload_components(task_id=9, components={"unet": {}, "clip": {}, "vae": {}}, pipeline_class="Flux2KleinPipeline")
    sent = [m for m in ch.sent if isinstance(m, P.PreloadComponents)]
    assert sent and sent[0].task_id == 9
```

> Check the real `RunnerClient.__init__` signature first (it takes a channel; match how existing client tests construct it — see `tests/` for `RunnerClient(`). Adjust the fixture construction to match. If `_demux_loop` needs `_connected`/`is_ready` state, set the minimal attributes the loop reads.

- [ ] **Step 2: run → FAIL** — `on_component_event` / `preload_components` don't exist.

- [ ] **Step 3: implement** — in `client.py`:

`__init__`: add `self.on_component_event: Callable[[P.ComponentEvent], None] | None = None`.

`_demux_loop`: add a branch (next to the `ModelEvent` branch):

```python
            elif isinstance(msg, P.ComponentEvent):
                cb = self.on_component_event
                if cb is not None:
                    cb(msg)
```

Add the send method (near `load_model`):

```python
    async def preload_components(self, task_id: int, components: dict, pipeline_class: str = "Flux2KleinPipeline") -> None:
        """发 PreloadComponents —— fire-and-forget;状态走 ComponentEvent → on_component_event。"""
        if not self._connected:
            raise ConnectionError("runner disconnected")
        await self._ch.send_message(P.PreloadComponents(
            task_id=task_id, components=components, pipeline_class=pipeline_class))
```

(If the field is named differently than `self._connected`, match the existing guard used by `load_model`.)

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_runner_client_component.py -q`.

- [ ] **Step 5: regression** — `cd backend && uv run pytest tests/ -q -k "client or runner" 2>&1 | tail -5`.

- [ ] **Step 6: commit**

```bash
git add backend/src/runner/client.py backend/tests/test_runner_client_component.py
git commit -m "feat(runner): PR-5a — RunnerClient ComponentEvent dispatch + preload_components"
```

---

## Task 6: `ComponentStateRegistry` + WS broadcast

**Files:**
- Create: `backend/src/services/component_state.py`
- Modify: `backend/src/api/websocket.py`
- Test: `backend/tests/test_component_state_registry.py` (new)

- [ ] **Step 1: failing test** (`backend/tests/test_component_state_registry.py`):

```python
"""PR-5a: ComponentStateRegistry mirror."""
from __future__ import annotations

from src.services.component_state import ComponentStateRegistry


def test_registry_defaults_cold_and_updates():
    reg = ComponentStateRegistry()
    assert reg.get("/m/u|cuda:1|bfloat16|") == {"key": "/m/u|cuda:1|bfloat16|", "state": "cold", "error": None}
    reg.update("/m/u|cuda:1|bfloat16|", "loading", None)
    reg.update("/m/u|cuda:1|bfloat16|", "loaded", None)
    assert reg.get("/m/u|cuda:1|bfloat16|")["state"] == "loaded"


def test_registry_query_many_and_all():
    reg = ComponentStateRegistry()
    reg.update("a", "loaded", None)
    reg.update("b", "failed", "boom")
    rows = reg.query(["a", "b", "c"])
    by = {r["key"]: r for r in rows}
    assert by["a"]["state"] == "loaded"
    assert by["b"]["state"] == "failed" and by["b"]["error"] == "boom"
    assert by["c"]["state"] == "cold"   # unknown → cold
    assert len(reg.all()) == 2
```

- [ ] **Step 2: run → FAIL** — module missing.

- [ ] **Step 3: implement** (`backend/src/services/component_state.py`):

```python
"""PR-5a: in-memory mirror of runner component load state (spec §6.1).

The runner subprocess owns the real L1 cache; it emits ComponentEvent on every
state transition. The backend keeps this best-effort mirror so GET
/api/v1/models/components/state can answer without a blocking RPC. Unknown keys
default to 'cold'. Lost on backend restart (matches L1 being in-memory)."""
from __future__ import annotations

import time
from typing import Any


class ComponentStateRegistry:
    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def update(self, key: str, state: str, error: str | None = None) -> None:
        self._states[key] = {"key": key, "state": state, "error": error, "updated_at": time.time()}

    def get(self, key: str) -> dict[str, Any]:
        entry = self._states.get(key)
        if entry is None:
            return {"key": key, "state": "cold", "error": None}
        return {"key": key, "state": entry["state"], "error": entry["error"]}

    def query(self, keys: list[str]) -> list[dict[str, Any]]:
        return [self.get(k) for k in keys]

    def all(self) -> list[dict[str, Any]]:
        return [{"key": k, "state": v["state"], "error": v["error"]} for k, v in self._states.items()]
```

In `websocket.py`, add to the manager (next to `broadcast_model_status`):

```python
    async def broadcast_component_state(self, component_key: str, state: str, error: str | None = None) -> None:
        """Push a component load-state change to /ws/models subscribers (spec §6.3)."""
        message = json.dumps({
            "event": "component_state_changed",
            "component_key": component_key,
            "state": state,
            "error": error,
        })
        await self._broadcast_to_model_subscribers(message)
```

> Match the existing `broadcast_model_status` body: it iterates `self._model_subscribers` and sends, pruning dead sockets. If there's no `_broadcast_to_model_subscribers` helper, inline the same loop `broadcast_model_status` uses (copy its send/prune logic).

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_component_state_registry.py -q`.

- [ ] **Step 5: commit**

```bash
git add backend/src/services/component_state.py backend/src/api/websocket.py backend/tests/test_component_state_registry.py
git commit -m "feat(image): PR-5a — ComponentStateRegistry + WS broadcast_component_state"
```

---

## Task 7: Lifespan wiring — registry + image-client callback

**Files:**
- Modify: `backend/src/api/main.py`
- Test: `backend/tests/test_component_state_wiring.py` (new)

> In lifespan, after `runner_clients` is built (~main.py:286), create the registry, expose on `app.state`, and register the image client's `on_component_event` to (a) update the registry and (b) WS-broadcast. The callback is sync (called from RunnerClient `_demux_loop`); schedule the async WS broadcast via `asyncio.create_task`.

- [ ] **Step 1: failing test** (`backend/tests/test_component_state_wiring.py`) — unit-test the wiring helper rather than full lifespan:

```python
"""PR-5a: image client component-event callback updates registry + schedules WS."""
from __future__ import annotations

import asyncio
import pytest

from src.runner import protocol as P
from src.services.component_state import ComponentStateRegistry
from src.api.main import _make_component_event_handler


@pytest.mark.asyncio
async def test_handler_updates_registry_and_broadcasts():
    reg = ComponentStateRegistry()
    broadcasts = []

    class _WS:
        async def broadcast_component_state(self, key, state, error=None):
            broadcasts.append((key, state, error))

    handler = _make_component_event_handler(reg, _WS())
    handler(P.ComponentEvent(component_key="k1", state="loaded", error=None))
    await asyncio.sleep(0.05)  # let the scheduled broadcast run
    assert reg.get("k1")["state"] == "loaded"
    assert broadcasts == [("k1", "loaded", None)]
```

- [ ] **Step 2: run → FAIL** — `_make_component_event_handler` missing.

- [ ] **Step 3: implement** — add the factory at module level in `main.py` (near other helpers):

```python
def _make_component_event_handler(registry, ws):
    """Build the sync callback RunnerClient.on_component_event uses: update the
    backend mirror + fan out a WS push. WS broadcast is async → scheduled."""
    def _handler(evt) -> None:
        registry.update(evt.component_key, evt.state, evt.error)
        try:
            asyncio.get_running_loop().create_task(
                ws.broadcast_component_state(evt.component_key, evt.state, evt.error))
        except RuntimeError:
            pass  # no running loop (shouldn't happen in lifespan) — registry still updated
    return _handler
```

In lifespan, after `app.state.runner_clients = runner_clients` (~main.py:286):

```python
    # PR-5a: component-state mirror fed by the image runner's ComponentEvents.
    from src.services.component_state import ComponentStateRegistry
    app.state.component_state_registry = ComponentStateRegistry()
    _img_client = runner_clients.get("image")
    if _img_client is not None:
        _img_client.on_component_event = _make_component_event_handler(
            app.state.component_state_registry, ws_manager)
```

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_component_state_wiring.py -q`.

- [ ] **Step 5: regression** — `cd backend && uv run pytest tests/test_websocket.py -q` + app still imports/boots: `cd backend && uv run pytest tests/test_components_routes.py -q`.

- [ ] **Step 6: commit**

```bash
git add backend/src/api/main.py backend/tests/test_component_state_wiring.py
git commit -m "feat(image): PR-5a — lifespan wires component-state registry + image client callback"
```

---

## Task 8: Endpoints — `GET /components/state` + `POST /components/preload`

**Files:**
- Modify: `backend/src/api/routes/models.py`
- Test: `backend/tests/test_components_state_routes.py` (new)

- [ ] **Step 1: failing test** (`backend/tests/test_components_state_routes.py`):

```python
"""PR-5a: GET components/state + POST components/preload."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.services.component_state import ComponentStateRegistry


@pytest.fixture
def app_with_state(monkeypatch):
    app = create_app()
    reg = ComponentStateRegistry()
    reg.update("/m/u|cuda:1|bfloat16|", "loaded", None)
    app.state.component_state_registry = reg
    return app


def test_get_state_known_and_unknown(app_with_state):
    client = TestClient(app_with_state)
    resp = client.get("/api/v1/models/components/state",
                      params={"keys": "/m/u|cuda:1|bfloat16|,/m/x|cuda:0|bfloat16|"})
    assert resp.status_code == 200
    by = {r["key"]: r for r in resp.json()["components"]}
    assert by["/m/u|cuda:1|bfloat16|"]["state"] == "loaded"
    assert by["/m/x|cuda:0|bfloat16|"]["state"] == "cold"


def test_get_state_all_when_no_keys(app_with_state):
    client = TestClient(app_with_state)
    resp = client.get("/api/v1/models/components/state")
    assert resp.status_code == 200
    assert any(r["state"] == "loaded" for r in resp.json()["components"])


def test_preload_dispatches_to_image_runner(app_with_state):
    sent = {}

    class _Client:
        _connected = True
        async def preload_components(self, task_id, components, pipeline_class="Flux2KleinPipeline"):
            sent["task_id"] = task_id
            sent["components"] = components

    app_with_state.state.runner_clients = {"image": _Client()}
    client = TestClient(app_with_state)
    body = {"components": {
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
    }}
    resp = client.post("/api/v1/models/components/preload", json=body)
    assert resp.status_code == 202
    assert "task_id" in resp.json()
    assert sent["components"]["unet"]["file"] == "/m/u.safe"


def test_preload_no_runner_returns_503(app_with_state):
    app_with_state.state.runner_clients = {}
    client = TestClient(app_with_state)
    resp = client.post("/api/v1/models/components/preload", json={"components": {
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "auto", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "auto", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "auto", "dtype": "bfloat16"},
    }})
    assert resp.status_code == 503
```

- [ ] **Step 2: run → FAIL** — `cd backend && uv run pytest tests/test_components_state_routes.py -q`.

- [ ] **Step 3: implement** — append to `models.py` (router prefix `/api/v1/models`). Use `require_admin` on preload like `components.py:/scan`. Validate the descriptor via `ComponentSpec` before dispatch (rejects bad input early). Do NOT `@cached` the state route (it changes too often).

```python
import time as _time
from fastapi import Body, HTTPException, Query, Request, Response
from src.api.deps_admin import require_admin
from src.services.inference.component_spec import ComponentSpec


@router.get("/components/state")
async def get_components_state(request: Request, keys: str | None = Query(default=None)):
    """Batch component load-state (spec §6.3). `keys` = comma-separated
    component_state_key list; omitted → all known. Unknown keys → 'cold'."""
    reg = getattr(request.app.state, "component_state_registry", None)
    if reg is None:
        return {"components": []}
    if keys:
        wanted = [k for k in keys.split(",") if k]
        return {"components": reg.query(wanted)}
    return {"components": reg.all()}


@router.post("/components/preload", status_code=202, dependencies=[Depends(require_admin)])
async def preload_components(request: Request, body: dict = Body(...)):
    """Batch-warm a unet+clip+vae combo on the image runner (spec §6.2). Returns
    202 + task_id immediately; state arrives via /ws/models component_state_changed."""
    raw = body.get("components") or {}
    missing = {"unet", "clip", "vae"} - set(raw)
    if missing:
        raise HTTPException(422, f"missing component kinds: {sorted(missing)}")
    try:
        components = {k: ComponentSpec(**raw[k]).model_dump() for k in ("unet", "clip", "vae")}
    except Exception as e:  # noqa: BLE001 — invalid descriptor
        raise HTTPException(422, f"invalid component spec: {e}") from e

    client = (getattr(request.app.state, "runner_clients", {}) or {}).get("image")
    if client is None or not getattr(client, "_connected", True):
        raise HTTPException(503, "image runner not available")

    task_id = int(_time.time() * 1000) % (2**31)
    pipeline_class = str(body.get("pipeline_class") or "Flux2KleinPipeline")
    await client.preload_components(task_id=task_id, components=components, pipeline_class=pipeline_class)
    return {"task_id": task_id, "status": "accepted"}
```

> Check `models.py` imports — it already imports `APIRouter`, `Depends`. Add the new imports without duplicating. `require_admin` no-ops in tests (ADMIN_TOKEN empty per conftest).

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_components_state_routes.py -q`.

- [ ] **Step 5: regression** — `cd backend && uv run pytest tests/test_components_routes.py tests/test_websocket.py -q`.

- [ ] **Step 6: commit**

```bash
git add backend/src/api/routes/models.py backend/tests/test_components_state_routes.py
git commit -m "feat(image): PR-5a — GET components/state + POST components/preload endpoints"
```

---

## Task 9: Integration — preload → events → registry → endpoint + WS

**Files:**
- Test: `backend/tests/test_components_preload_e2e.py` (new)

> End-to-end through the real `_node_executor`/`_handle_preload_components` + real `RunnerClient` demux + real registry + real handler, with a fake ModelManager (no GPU). Verifies: PreloadComponents → runner emits ComponentEvents → client routes to handler → registry updated → GET /state reflects it. WS broadcast asserted via a fake ws.

- [ ] **Step 1: write the test**

```python
"""PR-5a §9: preload → ComponentEvent → registry → /state, end-to-end (no GPU)."""
from __future__ import annotations

import asyncio
import pytest

from src.runner import protocol as P
from src.runner.client import RunnerClient
from src.runner.runner_process import _RunnerState, _handle_preload_components
from src.services.component_state import ComponentStateRegistry
from src.api.main import _make_component_event_handler


class _PairChannel:
    """Two-ended in-memory channel: runner sends → client recv."""
    def __init__(self):
        self.to_client = asyncio.Queue()
    async def send_message(self, m):  # runner side
        self.to_client.put_nowait(m)
    async def recv_message(self):     # client side
        return await self.to_client.get()


class _FakeMM:
    async def get_or_load_image_adapter(self, components, pipeline_class, on_event=None):
        from src.services.inference.component_spec import component_state_key
        for k in ("unet", "clip", "vae"):
            key = component_state_key(components[k])
            await on_event(key, "loading", None)
            await on_event(key, "loaded", None)
        return object()


class _WS:
    def __init__(self): self.calls = []
    async def broadcast_component_state(self, key, state, error=None):
        self.calls.append((key, state, error))


@pytest.mark.asyncio
async def test_preload_to_registry_e2e():
    ch = _PairChannel()
    registry = ComponentStateRegistry()
    ws = _WS()
    client = RunnerClient(ch)
    client.on_component_event = _make_component_event_handler(registry, ws)
    demux = asyncio.create_task(client._demux_loop())

    state = _RunnerState("r", "image", [0, 1, 2], _FakeMM())
    comps = {
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
    }
    await _handle_preload_components(state, ch, P.PreloadComponents(task_id=1, components=comps))
    await asyncio.sleep(0.1)
    demux.cancel()

    from src.services.inference.component_spec import ComponentSpec, component_state_key
    unet_key = component_state_key(ComponentSpec(**comps["unet"]))
    assert registry.get(unet_key)["state"] == "loaded"
    assert any(s == "loaded" for (_k, s, _e) in ws.calls)
```

> If `RunnerClient(ch)` needs extra init args/state for `_demux_loop` (e.g. `_connected`), set them after construction to match the real loop's expectations (mirror Task 5's fixture).

- [ ] **Step 2: run → PASS** — `cd backend && uv run pytest tests/test_components_preload_e2e.py -q`.

- [ ] **Step 3: commit**

```bash
git add backend/tests/test_components_preload_e2e.py
git commit -m "test(image): PR-5a — preload→event→registry→state e2e (no GPU)"
```

---

## Full verification (after all tasks)

- [ ] **Lint + full suite:**

```bash
cd backend && uv run ruff check src tests
cd backend && DATABASE_URL="sqlite+aiosqlite:///:memory:" uv run pytest -q   # CI-equivalent DB; expect green
```

- [ ] **finishing-a-development-branch** → push `feat/image-component-multigpu-pr5a-backend` → PR → CI (now green baseline) → merge on green ([[feedback-auto-merge]]).

> No real-model smoke needed for PR-5a (no new GPU math — preload reuses PR-4's verified `get_or_load_image_adapter`; tests use fake MM). The component path itself was real-model-verified in PR-4.

---

## Self-Review (vs spec §6)

- **§6.2 preload batch endpoint** (POST, 202+task_id, batch unet+clip+vae, WS push): Task 8 endpoint + Task 4 runner handler + Task 5 client send + Task 7 WS fan-out. ✓
- **§6.3 state query** (batch GET by keys, WS component_state_changed): Task 8 GET + Task 6 registry + Task 2 key + Task 6 WS event. ✓
- **§6.1 four states** (loaded/loading/cold/failed): Task 3 emits loading/loaded/failed; registry defaults cold. ✓
- **Runner→backend visibility** (the hard part): event-driven via ComponentEvent (Task 4) → RunnerClient (Task 5) → handler (Task 7) → registry (Task 6). ✓
- **Both load paths report**: run path (Task 4 `_node_executor` on_event) + preload path (Task 4 handler). ✓
- **Type consistency**: `component_state_key` (Task 2) used by Task 3/4/9; `on_event(key,state,error)` signature consistent Task 3↔4; `ComponentEvent(component_key,state,error)` Task 1↔4↔5; `preload_components(task_id,components,pipeline_class)` Task 5↔8. ✓
- **No placeholders.** ✓
- **Out of scope (PR-5b)**: frontend nodes/hook/widget/palette/四态 display; ModelsOverlay GPU form. **Deferred**: per-component preload result aggregation/task tracking beyond fire-and-forget (events suffice for the UI).
