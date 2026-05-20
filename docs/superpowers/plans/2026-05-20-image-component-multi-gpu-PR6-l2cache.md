# Image Component Multi-GPU — PR-6 (L2 image_generate output cache) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Deterministic `image_generate` re-runs return instantly from an in-runner L2 output cache (no model load, no sampling) — re-signing a fresh URL each hit so the cached PNG is always servable. A `cached` flag surfaces to the UI.

**Architecture:** A per-runner LRU(50) cache in `_RunnerState`, keyed by a deterministic hash of everything that affects the image (components/model_key + LoRAs + prompt + negative + steps/width/height/cfg + seed + pipeline_class). Only `node.is_deterministic` (seed non-empty, set by `workflow_executor` in PR-4) participates. The cache stores a disk **anchor** (`image_uuid`, `date`, `ext`, `meta`) — NOT a signed URL (URLs expire). On hit, `_node_executor` checks the cache BEFORE fetching the adapter: if the PNG still exists on disk, it re-signs a fresh URL (HMAC, microseconds) and returns immediately; if the PNG was reaped, it's a miss → normal inference + cache write. `cached: true` rides in `NodeResult.outputs` → the executor's `node_complete` event → a "(cached)" hint on the node.

**Tech Stack:** Python 3.12 (runner subprocess) + HMAC URL signing (existing `image_output_storage`) / React (one tiny node-display tweak) / pytest + vitest.

**Branch:** `feat/image-component-multigpu-pr6-l2cache` (from master).

**Prereqs (merged):** PR-1..PR-5b (#112–#119). `RunNode.is_deterministic` exists (PR-4). `image_output_storage._sign` / `write_image` / `resolve_path` exist. CI green.

**Spec:** design doc §3.3 (L2 entry schema + serve_l2_hit re-sign) + §8 (PR-6 row) + §2 (success: 同 seed 二跑 ≤ 0.1s).

---

## Key design decisions

1. **Check L2 before adapter fetch** → a hit skips BOTH model load and sampling (the spec's "≤ 0.1s" / "<2s combined" wins). Place the check right after `req = _build_request(node)` and before the adapter-fetch block in `_node_executor`.
2. **Store an anchor, re-sign on hit** (spec §3.3): the entry holds `image_uuid/date/ext/meta/width/height`, never a URL. `serve_image_l2` validates `resolve_path(...).exists()` (TTL reaper may have deleted it) → miss if gone → re-run; else re-sign with the node's `url_ttl_seconds`.
3. **Key includes device** (via `to_component_key`) — cross-arch output differs (PR-4: SSIM 0.98), so a different device combo is correctly a different entry.
4. **Random seed never caches** — gated by `node.is_deterministic`. Backend restart loses L2 (in-memory; spec OK).
5. **`cached` flag, not a new WS message** — rides in `NodeResult.outputs["cached"]` → executor adds it to the existing `node_complete` event → the node's image-stage line shows "(cached)". (Spec mentions a `node_cache_hit` WS event + TaskPanel badge; the outputs-flag path is simpler, reuses existing plumbing, and covers the UX. A dedicated TaskPanel badge is deferred polish.)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `backend/src/services/image_output_storage.py` | Modify | `write_image` returns `date`; new `sign_existing_image(date, uuid, ext, ttl) -> (url, expires)` |
| `backend/src/services/inference/image_l2_cache.py` | Create | `image_l2_key(node, req)`, `ImageOutputCache` (LRU 50), `serve_image_l2(entry, ttl)` |
| `backend/src/runner/runner_process.py` | Modify | `_RunnerState.image_l2`; L2 check before adapter fetch; store after write_image; `cached` in outputs |
| `backend/src/services/workflow_executor.py` | Modify | `node_complete` event carries `cached` |
| `frontend/src/components/nodes/DeclarativeNode.tsx` | Modify | image-stage "done" line shows "(cached)" |
| tests | Create/Modify | per task |

---

## Task 1: `image_output_storage` — `date` + `sign_existing_image`

**Files:** Modify `backend/src/services/image_output_storage.py`. Test: `backend/tests/test_image_output_storage.py`.

- [ ] **Step 1: failing test** — append:

```python
def test_write_image_returns_date(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    from src.services import image_output_storage as ios
    rec = ios.write_image(b"\x89PNG\r\n", ext="png", ttl_seconds=3600)
    assert "date" in rec and rec["date"] == __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d")


def test_sign_existing_image_roundtrips_token(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "sek")
    from src.services import image_output_storage as ios
    url, expires = ios.sign_existing_image("2026-05-20", "abc123", "png", ttl_seconds=3600)
    assert url is not None and "/files/images/2026-05-20/abc123.png?token=" in url
    # the signed token verifies for that uuid+expires
    tok = url.split("token=", 1)[1].split("&", 1)[0]
    assert ios.verify_token("abc123", expires, tok)


def test_sign_existing_image_no_secret_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "")
    from src.services import image_output_storage as ios
    url, expires = ios.sign_existing_image("2026-05-20", "abc", "png")
    assert url is None and expires is None
```

- [ ] **Step 2: run → FAIL** — `cd backend && uv run pytest tests/test_image_output_storage.py -q`.

- [ ] **Step 3: implement** — in `image_output_storage.py`:

(a) Add `"date": today,` to the dict `write_image` returns (today is already computed at the top of the function).

(b) Add a module function (after `write_image`):

```python
def sign_existing_image(date: str, uuid_str: str, ext: str, ttl_seconds: int = 3600) -> tuple[str | None, int | None]:
    """Re-sign a fresh URL for an already-on-disk image (L2 cache hit, spec §3.3).
    HMAC is microseconds, so a long-lived cache entry always serves a URL valid
    for the next ttl window. Returns (None, None) when no signing secret."""
    if _signing_key() is None:
        return None, None
    expires = int(time.time()) + max(60, int(ttl_seconds))
    token = _sign(uuid_str, expires)
    return f"/files/images/{date}/{uuid_str}.{ext}?token={token}&expires={expires}", expires
```

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_image_output_storage.py -q`.

- [ ] **Step 5: commit**

```bash
git add backend/src/services/image_output_storage.py backend/tests/test_image_output_storage.py
git commit -m "feat(image): PR-6 — write_image returns date + sign_existing_image (re-sign for L2)"
```

---

## Task 2: `image_l2_cache` — key + LRU + serve

**Files:** Create `backend/src/services/inference/image_l2_cache.py`. Test: `backend/tests/test_image_l2_cache.py`.

- [ ] **Step 1: failing test** (`backend/tests/test_image_l2_cache.py`):

```python
"""PR-6: L2 image output cache — deterministic key + LRU + serve/re-sign."""
from __future__ import annotations

from src.runner import protocol as P
from src.services.inference.base import ImageRequest, LoRASpec
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_l2_cache import ImageOutputCache, image_l2_key, serve_image_l2


def _node(model_key=None, deterministic=True):
    return P.RunNode(task_id=1, node_id="g", node_type="image", model_key=model_key,
                     inputs={}, is_deterministic=deterministic)


def _req_components(seed=42, prompt="a cat"):
    comps = {
        "unet": ComponentSpec(kind="unet", file="/m/u.safe", device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16", clip_arch="flux2"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }
    return ImageRequest(request_id="r", prompt=prompt, seed=seed, steps=9, width=512, height=512, components=comps)


def test_key_stable_and_sensitive():
    k1 = image_l2_key(_node(), _req_components(seed=42))
    k2 = image_l2_key(_node(), _req_components(seed=42))
    assert k1 == k2                                   # deterministic
    assert k1 != image_l2_key(_node(), _req_components(seed=43))      # seed
    assert k1 != image_l2_key(_node(), _req_components(prompt="a dog"))  # prompt


def test_key_legacy_model_key_path():
    req = ImageRequest(request_id="r", prompt="x", seed=1, loras=[LoRASpec(name="s", strength=0.8)])
    k = image_l2_key(_node(model_key="flux2-klein-9b"), req)
    k2 = image_l2_key(_node(model_key="other"), req)
    assert k != k2                                    # model_key in key


def test_lru_evicts_oldest():
    c = ImageOutputCache(maxsize=2)
    c.put("a", {"image_uuid": "a"}); c.put("b", {"image_uuid": "b"})
    c.get("a")                                        # touch a → b now oldest
    c.put("c", {"image_uuid": "c"})                   # evicts b
    assert c.get("a") is not None and c.get("c") is not None and c.get("b") is None


def test_serve_miss_when_png_gone(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    entry = {"image_uuid": "ghost", "date": "2026-05-20", "ext": "png", "meta": {}, "width": 512, "height": 512}
    assert serve_image_l2(entry, ttl=3600) is None    # file not on disk → miss


def test_serve_hit_resigns(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "sek")
    d = tmp_path / "2026-05-20"; d.mkdir()
    (d / "real.png").write_bytes(b"\x89PNG")
    entry = {"image_uuid": "real", "date": "2026-05-20", "ext": "png", "meta": {"seed": 42}, "width": 512, "height": 512}
    out = serve_image_l2(entry, ttl=3600)
    assert out is not None and out["cached"] is True
    assert "/files/images/2026-05-20/real.png?token=" in out["image_url"]
    assert out["image_uuid"] == "real" and out["width"] == 512
```

- [ ] **Step 2: run → FAIL** — module missing.

- [ ] **Step 3: implement** (`backend/src/services/inference/image_l2_cache.py`):

```python
"""PR-6: L2 image_generate output cache (spec §3.3). Per-runner, in-memory,
LRU(50). Keyed by everything that determines the image; only deterministic
(seeded) runs participate. Stores a disk anchor (uuid/date/ext/meta) and
re-signs a fresh URL on each hit — never caches an expiring URL."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any


def image_l2_key(node, req) -> str:
    """Deterministic hash of all output-affecting inputs. `node` supplies the
    legacy model_key; `req` supplies prompt/sampling/components."""
    from src.services.inference.component_spec import to_component_key

    comps = getattr(req, "components", None)
    if comps:
        comp_part: list = []
        for kind in ("unet", "clip", "vae"):
            f, dev, dt, lset = to_component_key(comps[kind])
            comp_part.append([kind, f, dev, dt, sorted(list(lset))])
        model_part: dict[str, Any] = {
            "pipeline_class": getattr(req, "pipeline_class", None),
            "components": comp_part,
        }
    else:
        model_part = {
            "model_key": getattr(node, "model_key", None),
            "loras": sorted([(lo.name, float(lo.strength)) for lo in getattr(req, "loras", [])]),
        }
    payload = {
        "model": model_part,
        "prompt": req.prompt,
        "negative": req.negative_prompt,
        "steps": req.steps,
        "w": req.width,
        "h": req.height,
        "cfg": req.cfg_scale,
        "seed": req.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class ImageOutputCache:
    """LRU anchor cache. Entry: {image_uuid, date, ext, meta, width, height}."""

    def __init__(self, maxsize: int = 50) -> None:
        self._d: "OrderedDict[str, dict]" = OrderedDict()
        self._max = maxsize

    def get(self, key: str) -> dict | None:
        entry = self._d.get(key)
        if entry is not None:
            self._d.move_to_end(key)
        return entry

    def put(self, key: str, entry: dict) -> None:
        self._d[key] = entry
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)


def serve_image_l2(entry: dict, ttl: int) -> dict | None:
    """Build a NodeResult outputs payload from a cache entry, re-signing a fresh
    URL. Returns None when the underlying PNG was reaped (caller treats as miss)."""
    from src.services.image_output_storage import resolve_path, sign_existing_image

    path = resolve_path(entry["date"], entry["image_uuid"], entry["ext"])
    if not path.exists():
        return None
    url, expires = sign_existing_image(entry["date"], entry["image_uuid"], entry["ext"], ttl_seconds=ttl)
    return {
        "image_url": url,
        "image_uuid": entry["image_uuid"],
        "image_expires": expires,
        "width": entry.get("width"),
        "height": entry.get("height"),
        "meta": entry.get("meta"),
        "media_type": f"image/{entry['ext']}",
        "cached": True,
    }
```

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_image_l2_cache.py -q`.

- [ ] **Step 5: commit**

```bash
git add backend/src/services/inference/image_l2_cache.py backend/tests/test_image_l2_cache.py
git commit -m "feat(image): PR-6 — image_l2_cache (key + LRU + serve/re-sign)"
```

---

## Task 3: Runner — L2 check (skip load+infer) + store + `cached`

**Files:** Modify `backend/src/runner/runner_process.py`. Test: `backend/tests/test_runner_l2_cache.py`.

> Add `self.image_l2 = ImageOutputCache()` to `_RunnerState.__init__`. In `_node_executor`, after `req = _build_request(node)` and BEFORE the adapter-fetch block, insert an L2 check for deterministic image nodes: hit → send NodeResult(completed, outputs incl `cached=True`) + continue (no adapter, no infer). After the existing `write_image` block, store the entry when deterministic.

- [ ] **Step 1: failing test** (`backend/tests/test_runner_l2_cache.py`):

```python
"""PR-6: runner L2 cache — deterministic image re-run hits cache (no infer)."""
from __future__ import annotations

import asyncio
import threading

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel
from src.runner.runner_process import _RunnerState, _node_executor


class _CountingAdapter:
    def __init__(self): self.calls = 0
    is_loaded = True
    async def infer(self, req, **kw):
        from src.services.inference.base import InferenceResult, UsageMeter
        self.calls += 1
        return InferenceResult(media_type="image/png", data=b"\x89PNG\r\n",
                               metadata={"width": req.width, "height": req.height, "seed": req.seed},
                               usage=UsageMeter(image_count=1, latency_ms=1))


class _MM:
    def __init__(self, adapter): self.adapter = adapter
    async def get_or_load(self, key): return self.adapter
    async def get_or_load_image_adapter(self, components, pipeline_class, on_event=None): return self.adapter


class _Collect(PipeChannel):
    def __init__(self): self.sent = []
    async def send_message(self, m): self.sent.append(m)


def _img_node(task_id, seed):
    return P.RunNode(task_id=task_id, node_id="g", node_type="image", model_key="m",
                     inputs={"prompt": "a cat", "seed": seed, "width": 64, "height": 64, "steps": 2},
                     is_deterministic=True)


async def _run(state, ch, node):
    state.cancel_flags[node.task_id] = threading.Event()
    state.run_queue.put_nowait(node)
    t = asyncio.create_task(_node_executor(state, ch))
    await asyncio.sleep(0.15)
    state.shutdown.set()
    await asyncio.wait_for(t, timeout=2)


@pytest.mark.asyncio
async def test_deterministic_rerun_hits_l2(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "sek")
    adapter = _CountingAdapter()
    state = _RunnerState("r", "image", [0], _MM(adapter))

    # first run → infer + write + store
    ch1 = _Collect(); await _run(state, ch1, _img_node(1, seed=42))
    r1 = [m for m in ch1.sent if isinstance(m, P.NodeResult)][-1]
    assert r1.status == "completed" and adapter.calls == 1
    assert not r1.outputs.get("cached")

    # second identical deterministic run → L2 hit, NO infer
    ch2 = _Collect(); await _run(state, ch2, _img_node(2, seed=42))
    r2 = [m for m in ch2.sent if isinstance(m, P.NodeResult)][-1]
    assert r2.status == "completed" and adapter.calls == 1   # still 1 → cache hit
    assert r2.outputs.get("cached") is True
    assert r2.outputs.get("image_url")   # re-signed


@pytest.mark.asyncio
async def test_random_seed_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "sek")
    adapter = _CountingAdapter()
    state = _RunnerState("r", "image", [0], _MM(adapter))
    n1 = _img_node(1, seed=7); n1 = P.RunNode(**{**n1.__dict__, "is_deterministic": False})
    n2 = _img_node(2, seed=7); n2 = P.RunNode(**{**n2.__dict__, "is_deterministic": False})
    ch1 = _Collect(); await _run(state, ch1, n1)
    ch2 = _Collect(); await _run(state, ch2, n2)
    assert adapter.calls == 2   # non-deterministic → both infer
```

- [ ] **Step 2: run → FAIL** — no L2 wiring (adapter.calls == 2 on rerun).

- [ ] **Step 3: implement** — in `runner_process.py`:

(a) `_RunnerState.__init__`: add
```python
        from src.services.inference.image_l2_cache import ImageOutputCache
        self.image_l2 = ImageOutputCache()
```

(b) In `_node_executor`, immediately AFTER the `req = _build_request(node)` try/except (before the `# adapter 获取` block), insert:

```python
        # PR-6: L2 output cache —— 确定性 image 节点二跑命中则跳过 load+infer。
        l2_key = None
        if node.node_type == "image" and getattr(node, "is_deterministic", False):
            from src.services.inference.image_l2_cache import image_l2_key, serve_image_l2
            l2_key = image_l2_key(node, req)
            entry = state.image_l2.get(l2_key)
            if entry is not None:
                ttl = int(node.inputs.get("url_ttl_seconds") or 3600)
                hit = serve_image_l2(entry, ttl)
                if hit is not None:
                    hit_outputs = {"meta": hit["meta"], "media_type": hit["media_type"],
                                   "image_url": hit["image_url"], "image_uuid": hit["image_uuid"],
                                   "image_expires": hit["image_expires"],
                                   "width": hit["width"], "height": hit["height"], "cached": True}
                    await ch.send_message(P.NodeResult(
                        task_id=node.task_id, node_id=node.node_id, status="completed",
                        outputs=hit_outputs, error=None,
                        duration_ms=int((time.monotonic() - started) * 1000)))
                    state.cancel_flags.pop(node.task_id, None)
                    continue
                # PNG reaped → drop the stale entry, fall through to recompute
                state.image_l2._d.pop(l2_key, None)
```

(c) In the existing `write_image` block (after `record = write_image(...)` + the `outputs.update({...})`), store the entry when deterministic:

```python
            if l2_key is not None:
                state.image_l2.put(l2_key, {
                    "image_uuid": record["uuid"], "date": record["date"], "ext": ext,
                    "meta": result.metadata, "width": meta.get("width"), "height": meta.get("height"),
                })
```

(`l2_key` is in scope — computed earlier in the loop iteration, None for non-deterministic.)

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_runner_l2_cache.py -q`.

- [ ] **Step 5: regression** — `cd backend && uv run pytest tests/ -q -k "runner" 2>&1 | tail -6`.

- [ ] **Step 6: commit**

```bash
git add backend/src/runner/runner_process.py backend/tests/test_runner_l2_cache.py
git commit -m "feat(image): PR-6 — runner L2 cache (skip load+infer on deterministic hit) + store"
```

---

## Task 4: Executor — `cached` in `node_complete`

**Files:** Modify `backend/src/services/workflow_executor.py`. Test: `backend/tests/test_workflow_executor_cached_event.py`.

- [ ] **Step 1: failing test**

```python
"""PR-6: node_complete event carries cached flag from runner outputs."""
from __future__ import annotations

import pytest

from src.runner import protocol as P
from src.services.workflow_executor import WorkflowExecutor


class _Client:
    async def run_node(self, spec, *, workflow_name=""):
        return P.NodeResult(task_id=spec.task_id, node_id=spec.node_id, status="completed",
                            outputs={"image_url": "u", "cached": True}, error=None, duration_ms=3)


@pytest.mark.asyncio
async def test_node_complete_includes_cached():
    events = []
    async def on_prog(e): events.append(e)
    wf = {"nodes": [{"id": "g", "type": "image_generate", "data": {"model_key": "m", "seed": 1}}], "edges": []}
    ex = WorkflowExecutor(wf, on_progress=on_prog, runner_clients={"image": _Client()}, task_id=5)
    await ex.execute()
    complete = [e for e in events if e.get("type") == "node_complete" and e["node_id"] == "g"]
    assert complete and complete[-1].get("cached") is True
```

- [ ] **Step 2: run → FAIL** — `cd backend && uv run pytest tests/test_workflow_executor_cached_event.py -q` (no `cached` in event).

- [ ] **Step 3: implement** — in `workflow_executor.py` `execute()`, where `complete_event` is built (it already conditionally adds `usage` / `duration_ms` from `output`), add:

```python
                if output.get("cached"):
                    complete_event["cached"] = True
```
(Place it inside the existing `if isinstance(output, dict):` block, next to the usage/duration handling.)

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_workflow_executor_cached_event.py -q`.

- [ ] **Step 5: regression** — `cd backend && uv run pytest tests/test_workflow_executor.py -q`.

- [ ] **Step 6: commit**

```bash
git add backend/src/services/workflow_executor.py backend/tests/test_workflow_executor_cached_event.py
git commit -m "feat(image): PR-6 — node_complete event carries cached flag"
```

---

## Task 5: Frontend — "(cached)" on the image node

**Files:** Modify `frontend/src/components/nodes/DeclarativeNode.tsx`. Test: extend an existing DeclarativeNode test or add `frontend/src/components/nodes/CachedHint.test.tsx`.

> The node's `node-progress` window-event handler already sets `imageStage` to `{phase:'done', elapsedSec}` on `node_complete`. Carry a `cached` boolean into `imageStage` and append "(cached)" to the done line.

- [ ] **Step 1: implement** — in `DeclarativeNode.tsx`:

(a) Extend the `imageStage` state type to include `cached?: boolean`.

(b) In the `node_complete` branch of the `node-progress` handler (where it sets `setImageStage({ phase: 'done', elapsedSec: realElapsed })`), include `cached: !!data.cached`:
```typescript
          setImageStage({ phase: 'done', elapsedSec: realElapsed, cached: !!data.cached })
```

(c) In the image-stage render, change the done line to show "(cached)" when cached:
```typescript
            {imageStage.phase === 'done' && `完成 · ${Math.round(imageStage.elapsedSec * 10) / 10}s${imageStage.cached ? ' (cached)' : ''}`}
```

- [ ] **Step 2: test** — add a focused vitest that dispatches a `node-progress` `node_complete` CustomEvent with `cached: true` to a rendered `image_generate` DeclarativeNode and asserts "(cached)" appears. Mirror the existing DeclarativeNode test setup (QueryClientProvider + ReactFlow wrapper if needed — inspect an existing nodes test; if rendering a DeclarativeNode standalone needs ReactFlow context, use the same wrapper the repo's node tests use, or assert via the simpler path used elsewhere). If a full render is too heavy, at minimum assert the done-line string builder includes "(cached)" by extracting it to a tiny pure helper and testing that. Keep a real assertion (not trivial).

```tsx
// Example shape (adapt wrapper to repo conventions):
import { describe, it, expect, vi } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { ReactFlowProvider } from '@xyflow/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
vi.mock('../../api/client', () => ({ apiFetch: vi.fn(() => Promise.resolve({ components: [] })) }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
import DeclarativeNode from './DeclarativeNode'

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}><ReactFlowProvider>{ui}</ReactFlowProvider></QueryClientProvider>)
}

it('shows (cached) on a cached image_generate completion', () => {
  wrap(<DeclarativeNode id="g" type="image_generate" data={{}} selected={false} {...({} as any)} />)
  act(() => {
    window.dispatchEvent(new CustomEvent('node-progress', { detail: { type: 'node_complete', node_id: 'g', duration_ms: 50, cached: true } }))
  })
  expect(screen.getByText(/\(cached\)/)).toBeInTheDocument()
})
```
(If `DeclarativeNode` needs more props/context to render, inspect a sibling node test and match it. If unrenderable in isolation, extract the done-line text into a pure function `imageDoneLabel(elapsedSec, cached)` and unit-test that instead — still a real assertion.)

- [ ] **Step 3: run → PASS** — `cd frontend && npx vitest run src/components/nodes/CachedHint.test.tsx` (or the file you used).

- [ ] **Step 4: gates + commit** — `cd frontend && npx tsc --noEmit && npx vitest run && npm run build`.
```bash
git add frontend/src/components/nodes/DeclarativeNode.tsx frontend/src/components/nodes/CachedHint.test.tsx
git commit -m "feat(image): PR-6 — image node shows (cached) on deterministic cache hit"
```

---

## Task 6: Real-model smoke — same-seed cache hit

**Files:** extend `backend/scripts/smoke_pr4_components.py` or a new `backend/scripts/smoke_pr6_l2.py`. Non-TDD verification (GPU + real model + ADMIN_SESSION_SECRET).

> The runner L2 lives in `_RunnerState`; the cleanest live check drives `_node_executor` with a real ModelManager + the bf16 components twice (same seed) and asserts the 2nd run skips infer. But `_node_executor` needs a real runner harness. Simpler: a standalone script that exercises `image_l2_key` + `serve_image_l2` against a real `write_image` output, plus a manual two-run timing via the running backend if convenient.

- [ ] **Step 1** — standalone: write a real PNG via `write_image`, build an `ImageOutputCache`, `put` the anchor under the deterministic key, then `serve_image_l2` → assert a fresh valid signed URL + `cached=True`; delete the PNG → assert miss (None). Confirms the anchor/re-sign/reaped-miss logic on real disk + real signing.

```python
# backend/scripts/smoke_pr6_l2.py — run: cd backend && set -a && source .env && set +a && PYTHONPATH=. .venv/bin/python scripts/smoke_pr6_l2.py
from src.services.image_output_storage import write_image, verify_token
from src.services.inference.image_l2_cache import ImageOutputCache, serve_image_l2

rec = write_image(b"\x89PNG\r\n", ext="png", ttl_seconds=3600)
print("[1] wrote", rec["uuid"], "date", rec["date"], "url?", bool(rec["url"]))
cache = ImageOutputCache()
entry = {"image_uuid": rec["uuid"], "date": rec["date"], "ext": "png", "meta": {"seed": 42}, "width": 512, "height": 512}
cache.put("k", entry)
hit = serve_image_l2(cache.get("k"), ttl=3600)
tok = hit["image_url"].split("token=")[1].split("&")[0]; exp = int(hit["image_url"].split("expires=")[1])
print("[2] L2 hit cached?", hit["cached"], "url valid?", verify_token(rec["uuid"], exp, tok))
rec["path"].unlink()
print("[3] after reap, serve →", serve_image_l2(cache.get("k"), ttl=3600), "(expect None = miss)")
```

- [ ] **Step 2** — (optional, if a backend + image runner is up) two-run timing: dispatch the same deterministic `image_generate` (real bf16 components, fixed seed) twice; assert 2nd ≤ 0.1s (no GPU work). Reuse the PR-4 smoke harness if available; otherwise the Step-1 unit-level smoke + the Task-3 fake-adapter integration test together cover the cache-hit path.

- [ ] **Step 3** — record results; commit the script.

```bash
git add backend/scripts/smoke_pr6_l2.py
git commit -m "test(image): PR-6 — L2 anchor/re-sign/reaped-miss smoke"
```

---

## Full verification + finish

- [ ] `cd backend && uv run ruff check src tests && DATABASE_URL="sqlite+aiosqlite:///:memory:" uv run pytest -q` (CI-equivalent; expect green).
- [ ] `cd frontend && npx tsc --noEmit && npx vitest run && npm run build` green.
- [ ] superpowers:finishing-a-development-branch → push → PR → CI green → merge ([[feedback-auto-merge]]).

---

## Self-Review (vs spec §3.3 / §8)

- **L2 LRU 50** (Task 2 `ImageOutputCache(maxsize=50)` — runner passes default). ✓
- **Key = components/model_key + prompt + sampling + seed** (Task 2 `image_l2_key`). ✓
- **`is_deterministic` gate** (Task 3 — only seeded runs cache). ✓
- **Anchor entry + re-sign on hit** (§3.3 `L2CacheEntry` + `serve_l2_hit`): Task 1 `sign_existing_image` + Task 2 `serve_image_l2`, validates `path.exists()` → miss if reaped. ✓
- **Hit skips load+infer** (Task 3 — check before adapter fetch). ✓
- **`cached` to UI** (§8 "node_cache_hit + TaskPanel 角标"): Task 3 outputs flag → Task 4 node_complete event → Task 5 node "(cached)". (Dedicated `node_cache_hit` WS message + TaskPanel badge → **deferred**; the outputs-flag path covers the UX with less surface.) ✓
- **Type consistency**: `image_l2_key(node, req)` Task 2↔3; entry shape `{image_uuid,date,ext,meta,width,height}` Task 2↔3; `serve_image_l2(entry, ttl)` Task 2↔3; `record["date"]` (Task 1) consumed by Task 3 store. ✓
- **No real GPU math** — reuses verified write_image/signing; cache hit is pure disk+HMAC.
