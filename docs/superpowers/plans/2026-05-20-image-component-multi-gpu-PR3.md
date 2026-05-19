# Image Component Multi-GPU Loader — PR-3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `component_scanner` service that enumerates available unet / clip / vae / lora files by role (globbing the directories declared in a new `model_paths.yaml`), expose it via `GET /api/v1/components?role=...` + `POST /api/v1/components/scan`, and broadcast a `component_index_changed` WS event on rescan. This is the data source for PR-4's loader-node file dropdowns.

**Architecture:** Pure read-side infra, modeled on the existing `lora_scanner.py` (scan + module-level cache + invalidate). No inference path touched. `model_paths.yaml` declares role→glob-dirs (analogous to ComfyUI `extra_model_paths.yaml`). The scanner globs those dirs (non-recursive, O(hundreds of files)), detects `quant_type` per file (bf16/fp16/fp8mixed/mxfp8mixed/nvfp4mixed/gguf) by filename + safetensors header sniff, and caches the result. A FastAPI route serves the cached index; admin-only rescan refreshes it + broadcasts WS.

**Tech Stack:** Python 3.12 / pydantic v2 / FastAPI / PyYAML / pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-19-image-component-multi-gpu-design.md` §4.6.

**Branch:** `feat/image-component-multi-gpu-pr3` (already created off master post-PR-2 + hotfix merge).

**Out of scope** (per spec §8): workflow loader nodes (PR-4), frontend hook + palette (PR-5), L2 cache (PR-6).

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `backend/configs/model_paths.yaml` | role → glob-pattern-dirs mapping (unet/clip/vae/loras). base_path from `LOCAL_MODELS_PATH`. |
| `backend/src/services/component_scanner.py` | `scan_components(role, *, force_refresh)` + `get_component_index()` + `invalidate_component_cache()` + `quant_type` detection. Module-level cache like lora_scanner. |
| `backend/src/api/routes/components.py` | `GET /api/v1/components?role=...` + `POST /api/v1/components/scan` (admin). |
| `backend/tests/test_component_scanner.py` | Unit tests using tmp_path fixtures with synthetic files. |
| `backend/tests/test_components_routes.py` | Route tests (GET filters by role, POST scan requires admin, malformed role 400). |

### Modified files

| Path | Change |
|---|---|
| `backend/src/api/main.py` | (a) `app.include_router(components_routes.router)`; (b) in lifespan, build initial index into `app.state.component_index` after model_manager is set up. |

### NOT touched

- Any inference file (image_diffusers / image_sampler / model_manager / quant_loaders)
- Frontend
- workflow_executor / runner_process / nodes

---

## Task 1: `model_paths.yaml` + config loader

**Files:**
- Create: `backend/configs/model_paths.yaml`
- Create: `backend/tests/test_component_scanner.py` (config-loader portion)

- [ ] **Step 1: Write the failing test for config loading**

```python
# backend/tests/test_component_scanner.py
"""component_scanner: model_paths config + role glob + quant detection."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.services.component_scanner import load_model_paths_config, ROLE_DIRS


def test_load_model_paths_config_returns_role_dirs():
    cfg = load_model_paths_config()
    assert "unet" in cfg
    assert "clip" in cfg
    assert "vae" in cfg
    assert "loras" in cfg
    # Each role maps to a non-empty list of glob patterns (str)
    for role, patterns in cfg.items():
        assert isinstance(patterns, list)
        assert all(isinstance(p, str) for p in patterns)


def test_role_dirs_constant_matches_config_keys():
    """ROLE_DIRS is the canonical role set."""
    cfg = load_model_paths_config()
    assert set(ROLE_DIRS) == set(cfg.keys())
```

- [ ] **Step 2: Run — fail**

```bash
cd backend
.venv/bin/pytest tests/test_component_scanner.py::test_load_model_paths_config_returns_role_dirs -v
```

Expected: ImportError.

- [ ] **Step 3: Create `model_paths.yaml`**

```yaml
# backend/configs/model_paths.yaml
# role → glob-pattern dirs (relative to LOCAL_MODELS_PATH). Analogous to ComfyUI
# extra_model_paths.yaml. component_scanner globs these to populate loader-node
# file dropdowns (PR-4). Non-recursive globs — keep dirs flat.
#
# base_path is NOT in this file — it's read from settings.LOCAL_MODELS_PATH
# (backend/.env → /media/heygo/Program/models/nous).

roles:
  unet:
    - image/diffusion_models/*.safetensors
    - image/diffusion_models/*.gguf
    - image/diffusers/*/transformer/*.safetensors
  clip:
    - image/text_encoders/*.safetensors
    - image/diffusers/*/text_encoder/*.safetensors
  vae:
    - image/vae/*.safetensors
    - image/diffusers/*/vae/*.safetensors
  loras:
    - image/loras/*.safetensors
    - image/loras/**/*.safetensors
```

- [ ] **Step 4: Implement config loader (start `component_scanner.py`)**

```python
# backend/src/services/component_scanner.py
"""component_scanner — enumerate unet/clip/vae/lora files by role for PR-4 loader nodes.

Modeled on lora_scanner.py: glob role-dirs declared in model_paths.yaml, detect
quant_type per file, cache module-level, expose invalidate. base_path from
settings.LOCAL_MODELS_PATH.

Spec §4.6.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROLE_DIRS = ("unet", "clip", "vae", "loras")

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "model_paths.yaml"


def load_model_paths_config() -> dict[str, list[str]]:
    """Load role → glob-patterns from model_paths.yaml. Fail-soft to empty
    pattern lists if the file is missing (keeps the scanner from crashing the
    app on a fresh checkout without the config)."""
    if not _CONFIG_PATH.exists():
        logger.warning("model_paths.yaml not found at %s; component index will be empty", _CONFIG_PATH)
        return {role: [] for role in ROLE_DIRS}
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f) or {}
    roles = data.get("roles", {})
    return {role: list(roles.get(role, [])) for role in ROLE_DIRS}
```

- [ ] **Step 5: Run — pass**

```bash
.venv/bin/pytest tests/test_component_scanner.py -v
```

Expected: 2/2 PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/configs/model_paths.yaml backend/src/services/component_scanner.py backend/tests/test_component_scanner.py
git commit -m "feat(scanner): model_paths.yaml + config loader (PR-3 Task 1)"
```

---

## Task 2: `scan_components` + quant detection + cache

**Files:**
- Modify: `backend/src/services/component_scanner.py`
- Modify: `backend/tests/test_component_scanner.py`

- [ ] **Step 1: Add failing tests**

Append to `backend/tests/test_component_scanner.py`:

```python
def _make_file(root: Path, rel: str, content: bytes = b"\x00" * 64) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def test_scan_components_globs_role_dirs(tmp_path, monkeypatch):
    # Build a fake LOCAL_MODELS_PATH tree
    _make_file(tmp_path, "image/diffusion_models/Flux2-bf16.safetensors")
    _make_file(tmp_path, "image/diffusion_models/Flux2-fp8mixed.safetensors")
    _make_file(tmp_path, "image/text_encoders/qwen3.safetensors")
    _make_file(tmp_path, "image/vae/flux2-vae.safetensors")
    _make_file(tmp_path, "image/loras/style.safetensors")

    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)

    from src.services.component_scanner import scan_components
    unet = scan_components("unet", force_refresh=True)
    names = {e["filename"] for e in unet}
    assert "Flux2-bf16.safetensors" in names
    assert "Flux2-fp8mixed.safetensors" in names

    clip = scan_components("clip", force_refresh=True)
    assert {e["filename"] for e in clip} == {"qwen3.safetensors"}

    vae = scan_components("vae", force_refresh=True)
    assert {e["filename"] for e in vae} == {"flux2-vae.safetensors"}

    loras = scan_components("loras", force_refresh=True)
    assert {e["filename"] for e in loras} == {"style.safetensors"}


def test_scan_components_entry_shape(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/x-bf16.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import scan_components
    entry = scan_components("unet", force_refresh=True)[0]
    assert set(entry.keys()) >= {"filename", "abs_path", "size_mb", "quant_type"}
    assert entry["abs_path"].endswith("x-bf16.safetensors")
    assert isinstance(entry["size_mb"], (int, float))


def test_quant_type_detection_by_filename(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/M-bf16.safetensors")
    _make_file(tmp_path, "image/diffusion_models/M-fp8mixed.safetensors")
    _make_file(tmp_path, "image/diffusion_models/M-mxfp8mixed.safetensors")
    _make_file(tmp_path, "image/diffusion_models/M-nvfp4mixed.safetensors")
    _make_file(tmp_path, "image/diffusion_models/M-Q4_K.gguf")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import scan_components
    by_name = {e["filename"]: e["quant_type"] for e in scan_components("unet", force_refresh=True)}
    assert by_name["M-bf16.safetensors"] == "bf16"
    assert by_name["M-fp8mixed.safetensors"] == "fp8mixed"
    assert by_name["M-mxfp8mixed.safetensors"] == "mxfp8mixed"
    assert by_name["M-nvfp4mixed.safetensors"] == "nvfp4mixed"
    assert by_name["M-Q4_K.gguf"] == "gguf"


def test_scan_components_caches_until_invalidate(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/vae/v1.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import scan_components, invalidate_component_cache

    first = scan_components("vae")  # populate cache (no force_refresh)
    # Add a new file — should NOT appear until invalidate
    _make_file(tmp_path, "image/vae/v2.safetensors")
    second = scan_components("vae")
    assert {e["filename"] for e in first} == {e["filename"] for e in second}  # cached

    invalidate_component_cache()
    third = scan_components("vae")
    assert {e["filename"] for e in third} == {"v1.safetensors", "v2.safetensors"}


def test_get_component_index_returns_all_roles(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/u.safetensors")
    _make_file(tmp_path, "image/text_encoders/c.safetensors")
    _make_file(tmp_path, "image/vae/v.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import get_component_index, invalidate_component_cache
    invalidate_component_cache()
    idx = get_component_index()
    assert set(idx.keys()) == {"unet", "clip", "vae", "loras"}
    assert len(idx["unet"]) == 1
    assert len(idx["clip"]) == 1
    assert len(idx["vae"]) == 1
```

- [ ] **Step 2: Run — fail**

```bash
.venv/bin/pytest tests/test_component_scanner.py -v
```

Expected: failures on scan_components / get_component_index / invalidate_component_cache not defined.

- [ ] **Step 3: Implement scanner**

Append to `backend/src/services/component_scanner.py`:

```python
import glob as _glob


def _base_path() -> Path:
    """LOCAL_MODELS_PATH from settings. Wrapped in a function so tests can monkeypatch."""
    from src.config import get_settings
    return Path(get_settings().LOCAL_MODELS_PATH)


def _detect_quant_type(path: Path) -> str:
    """Filename-substring + extension based quant type. Cheap (no file read for
    safetensors unless we need to disambiguate). Mirrors quant_loaders matchers.
    """
    name = path.name.lower()
    if name.endswith(".gguf"):
        return "gguf"
    if "nvfp4mixed" in name:
        return "nvfp4mixed"
    if "mxfp8mixed" in name:
        return "mxfp8mixed"
    if "fp8mixed" in name:
        return "fp8mixed"
    if "fp16" in name or "float16" in name:
        return "fp16"
    return "bf16"  # default for plain safetensors


# Module-level cache: role -> list[entry dict]. None = not yet scanned.
_cache: dict[str, list[dict[str, Any]]] | None = None


def scan_components(role: str, *, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Return list of available files for a role. Each entry:
      {filename, abs_path, size_mb, quant_type, mtime}
    Cached module-level; force_refresh re-globs.
    """
    if role not in ROLE_DIRS:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLE_DIRS}")

    global _cache
    if _cache is None or force_refresh:
        _cache = _scan_all()
    return _cache.get(role, [])


def _scan_all() -> dict[str, list[dict[str, Any]]]:
    """Glob every role's patterns under base_path; build the full index."""
    base = _base_path()
    cfg = load_model_paths_config()
    index: dict[str, list[dict[str, Any]]] = {}
    seen_per_role: dict[str, set[str]] = {}
    for role, patterns in cfg.items():
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in _glob.glob(str(base / pattern), recursive=True):
                p = Path(match)
                if not p.is_file():
                    continue
                abs_path = str(p.resolve())
                if abs_path in seen:
                    continue  # de-dup across overlapping patterns
                seen.add(abs_path)
                try:
                    stat = p.stat()
                    size_mb = round(stat.st_size / (1024 * 1024), 1)
                    mtime = stat.st_mtime
                except OSError:
                    size_mb, mtime = 0.0, 0.0
                entries.append({
                    "filename": p.name,
                    "abs_path": abs_path,
                    "size_mb": size_mb,
                    "quant_type": _detect_quant_type(p),
                    "mtime": mtime,
                })
        # Stable ordering by filename for deterministic dropdowns
        entries.sort(key=lambda e: e["filename"])
        index[role] = entries
        seen_per_role[role] = seen
    total = sum(len(v) for v in index.values())
    logger.info("component_scanner: indexed %d files across %d roles", total, len(index))
    return index


def get_component_index() -> dict[str, list[dict[str, Any]]]:
    """Full role → entries index (all roles). Populates cache if cold."""
    global _cache
    if _cache is None:
        _cache = _scan_all()
    return dict(_cache)


def invalidate_component_cache() -> None:
    """Drop the cache so the next scan re-globs. Called by POST /scan."""
    global _cache
    _cache = None
```

- [ ] **Step 4: Run — pass**

```bash
.venv/bin/pytest tests/test_component_scanner.py -v
```

Expected: 7/7 PASS (2 from Task 1 + 5 new).

- [ ] **Step 5: Ruff**

```bash
.venv/bin/ruff check src/services/component_scanner.py tests/test_component_scanner.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/component_scanner.py backend/tests/test_component_scanner.py
git commit -m "feat(scanner): scan_components + quant detection + cache (PR-3 Task 2)"
```

---

## Task 3: `GET /api/v1/components` + `POST /scan` routes

**Files:**
- Create: `backend/src/api/routes/components.py`
- Create: `backend/tests/test_components_routes.py`

- [ ] **Step 1: Write failing route tests**

```python
# backend/tests/test_components_routes.py
"""Routes for /api/v1/components — GET role index + POST rescan."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


def _make_file(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * 64)


@pytest.fixture
def client(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/u-bf16.safetensors")
    _make_file(tmp_path, "image/vae/v.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import invalidate_component_cache
    invalidate_component_cache()
    app = create_app()
    return TestClient(app)


def test_get_components_by_role(client):
    resp = client.get("/api/v1/components?role=unet")
    assert resp.status_code == 200
    data = resp.json()
    assert "components" in data
    names = {c["filename"] for c in data["components"]}
    assert "u-bf16.safetensors" in names


def test_get_components_unknown_role_400(client):
    resp = client.get("/api/v1/components?role=bogus")
    assert resp.status_code == 400


def test_get_components_no_role_returns_all(client):
    resp = client.get("/api/v1/components")
    assert resp.status_code == 200
    data = resp.json()
    # all-roles shape: {index: {unet: [...], clip: [...], ...}}
    assert "index" in data
    assert set(data["index"].keys()) == {"unet", "clip", "vae", "loras"}


def test_post_scan_refreshes_index(client, tmp_path):
    # POST scan should re-glob; admin gate is off in tests (conftest ADMIN_PASSWORD="")
    _make_file(tmp_path, "image/vae/v2-new.safetensors")
    resp = client.post("/api/v1/components/scan")
    assert resp.status_code == 200
    # After rescan, new file appears
    vae = client.get("/api/v1/components?role=vae").json()["components"]
    assert any(c["filename"] == "v2-new.safetensors" for c in vae)
```

- [ ] **Step 2: Run — fail**

```bash
.venv/bin/pytest tests/test_components_routes.py -v
```

Expected: 404s (route not registered).

- [ ] **Step 3: Implement routes**

```python
# backend/src/api/routes/components.py
"""GET /api/v1/components + POST /scan — file index for PR-4 loader nodes.

GET ?role=unet|clip|vae|loras → that role's files.
GET (no role)                 → full index across all roles.
POST /scan                    → admin-only re-glob + cache invalidate + WS broadcast.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.middleware import require_admin
from src.services.component_scanner import (
    ROLE_DIRS,
    get_component_index,
    invalidate_component_cache,
    scan_components,
)

router = APIRouter(prefix="/api/v1/components", tags=["components"])


@router.get("")
async def list_components(role: str | None = Query(default=None)):
    """List component files. With ?role= → that role; without → full index."""
    if role is None:
        return {"index": get_component_index()}
    if role not in ROLE_DIRS:
        raise HTTPException(400, detail=f"unknown role {role!r}; expected one of {list(ROLE_DIRS)}")
    return {"components": scan_components(role)}


@router.post("/scan", dependencies=[Depends(require_admin)])
async def rescan_components():
    """Re-glob the model dirs + invalidate cache + broadcast WS event."""
    invalidate_component_cache()
    index = get_component_index()  # repopulates
    total = sum(len(v) for v in index.values())
    from src.api.websocket import ws_manager
    await ws_manager.broadcast_model_status("__components__", "index_changed", f"{total} files")
    return {"status": "rescanned", "total": total}
```

**Note on the WS event:** spec §4.6 calls it `component_index_changed`. We reuse the existing `broadcast_model_status` channel with a sentinel model_id `"__components__"` + status `"index_changed"` so the frontend's existing `/ws/models` subscription picks it up without a new WS channel. If `require_admin` import path differs, grep `backend/src/api/middleware.py` for the actual dependency name.

- [ ] **Step 4: Register router in main.py**

Open `backend/src/api/main.py`. Add to the imports line (alongside other route imports):

```python
from src.api.routes import components as components_routes
```

And add the include_router call near the other `app.include_router(...)` calls:

```python
    app.include_router(components_routes.router)
```

- [ ] **Step 5: Run — pass**

```bash
.venv/bin/pytest tests/test_components_routes.py -v
```

Expected: 4/4 PASS.

- [ ] **Step 6: Ruff**

```bash
.venv/bin/ruff check src/api/routes/components.py tests/test_components_routes.py src/api/main.py
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add backend/src/api/routes/components.py backend/tests/test_components_routes.py backend/src/api/main.py
git commit -m "feat(scanner): GET/POST /api/v1/components routes + main wiring (PR-3 Task 3)"
```

---

## Task 4: Lifespan index warm-up + app.state

**Files:**
- Modify: `backend/src/api/main.py`
- Modify: `backend/tests/test_components_routes.py` (verify app.state populated)

- [ ] **Step 1: Add failing test**

Append to `backend/tests/test_components_routes.py`:

```python
def test_lifespan_warms_component_index(tmp_path, monkeypatch):
    """On app startup, app.state.component_index should be populated."""
    _make_file(tmp_path, "image/diffusion_models/warm-bf16.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import invalidate_component_cache
    invalidate_component_cache()

    app = create_app()
    with TestClient(app):  # triggers lifespan startup
        assert hasattr(app.state, "component_index")
        assert "unet" in app.state.component_index
```

- [ ] **Step 2: Run — fail**

```bash
.venv/bin/pytest tests/test_components_routes.py::test_lifespan_warms_component_index -v
```

Expected: AttributeError on app.state.component_index.

- [ ] **Step 3: Wire into lifespan**

Open `backend/src/api/main.py`. In the `lifespan` function, AFTER `app.state.model_manager` is set (search for `app.state.model_manager = model_mgr`), add:

```python
    # PR-3: warm the component file index for loader-node dropdowns (PR-4).
    # Fail-soft — a scan error must not block app startup.
    try:
        from src.services.component_scanner import get_component_index
        app.state.component_index = get_component_index()
        _ci_total = sum(len(v) for v in app.state.component_index.values())
        logger.info("PR-3: component index warmed — %d files", _ci_total)
    except Exception:  # noqa: BLE001 — index is non-critical at boot
        logger.exception("PR-3: component index warm-up failed; serving empty index")
        app.state.component_index = {role: [] for role in ("unet", "clip", "vae", "loras")}
```

- [ ] **Step 4: Run — pass**

```bash
.venv/bin/pytest tests/test_components_routes.py -v
```

Expected: 5/5 PASS (4 from Task 3 + 1 new).

- [ ] **Step 5: Ruff**

```bash
.venv/bin/ruff check src/api/main.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/main.py backend/tests/test_components_routes.py
git commit -m "feat(scanner): warm component index in lifespan → app.state (PR-3 Task 4)"
```

---

## Task 5: PR Wrap-up

- [ ] **Step 1: Full suite — no regressions**

```bash
cd backend
.venv/bin/pytest -q 2>&1 | tail -10
```

Expected: all green (component_scanner + components_routes added, nothing broken).

- [ ] **Step 2: Ruff full**

```bash
.venv/bin/ruff check src/ tests/ 2>&1 | tail -3
```

- [ ] **Step 3: Real-dir smoke (verify against actual model dir)**

```bash
.venv/bin/python -c "
from src.services.component_scanner import get_component_index, invalidate_component_cache
invalidate_component_cache()
idx = get_component_index()
for role, entries in idx.items():
    print(f'{role}: {len(entries)} files')
    for e in entries[:3]:
        print(f'  {e[\"filename\"]} ({e[\"quant_type\"]}, {e[\"size_mb\"]}MB)')
"
```

Expected: lists real Flux2 files — unet should show the 6 quant variants (bf16/fp8mixed/mxfp8mixed/nvfp4mixed + GGUFs) under image/diffusion_models/ + the diffusers transformer; clip shows text_encoders; vae shows vae files. This confirms the globs match the real layout.

- [ ] **Step 4: Push + PR + merge**

```bash
git push -u origin feat/image-component-multi-gpu-pr3
gh pr create --base master \
  --title "feat(scanner): component-multi-gpu PR-3 — component_scanner + /api/v1/components" \
  --body "PR-3 of image-component-multi-gpu. component_scanner globs role dirs (model_paths.yaml) → file index for PR-4 loader-node dropdowns. GET /api/v1/components?role= + POST /scan + lifespan warm-up + WS broadcast. Full suite green, ruff clean, real-dir smoke confirms globs match Flux2 layout."
gh pr merge --squash --delete-branch
```

---

## Self-Review Checklist

### Spec coverage (§4.6)

| Requirement | Task |
|---|---|
| model_paths.yaml role→dirs | Task 1 |
| scan once at startup → app.state.component_index | Task 4 |
| quant_type detection (fp8mixed/mxfp8/nvfp4/gguf) | Task 2 |
| GET /api/v1/components?role= | Task 3 |
| POST /api/v1/components/scan (admin) | Task 3 |
| WS component_index_changed | Task 3 (via broadcast_model_status sentinel) |
| O(hundreds) glob not full-disk-scan | Task 2 (`_glob.glob` on declared patterns) |

### Placeholder scan
- No TBD/TODO/FIXME.
- The WS event reuses `broadcast_model_status("__components__", "index_changed", ...)` — documented choice (avoid new WS channel). Frontend (PR-5) treats `model_id == "__components__"` as the index-changed signal.

### Type consistency
- `ROLE_DIRS` tuple consistent across scanner + routes.
- entry dict shape `{filename, abs_path, size_mb, quant_type, mtime}` consistent across scan + tests + route response.
- `scan_components(role, *, force_refresh)` / `get_component_index()` / `invalidate_component_cache()` signatures consistent across scanner + routes + tests.

### Decomposition
- Task 1: config layer alone.
- Task 2: scanner core alone (testable without FastAPI).
- Task 3: routes (depends on scanner).
- Task 4: lifespan wiring (depends on scanner).
- Task 5: wrap-up.

PR-3 ships ~250 LOC across 4 new files + 1 modified (main.py). Each task independently revertable.
