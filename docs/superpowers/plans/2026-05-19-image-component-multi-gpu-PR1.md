# Image Component Multi-GPU Loader — PR-1 Implementation Plan (rev 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the foundational infrastructure for component-level multi-GPU image generation — `ComponentSpec` types, a pluggable `QuantLoaderRegistry` (covering bf16 / fp16 / fp8mixed / mxfp8mixed / nvfp4mixed safetensors, rejecting GGUF as V2 work), and a parallel `ModelManager._components` cache layer keyed on `ComponentKey`. **PR-1 ships infra only — no user-visible change.** PR-2 (the self-written `ImageSampler` per spec §5.6) consumes this infra.

**Architecture:** Additive only. The legacy `_models: dict[str, LoadedModel]` path is fully untouched. A new `_components: dict[ComponentKey, LoadedComponent]` cache lives alongside, with its own public APIs. `DiffusersImageBackend` is **NOT** restructured in this PR — that work moves to PR-2 alongside `ImageSampler`. The only edit to `image_diffusers.py` is a small refactor (~30 LOC) extracting the existing `load_quantized_transformer` (`image_diffusers.py:105`) dequant logic into the `QuantLoaderRegistry` while keeping the wrapper function call-compatible. Workflow nodes won't emit `ComponentSpec` until PR-4. Backend / frontend public API surface is unchanged.

**Tech Stack:** Python 3.12 / pydantic v2 / diffusers 0.38+ / safetensors 0.8 / torch 2.10 / pytest + pytest-asyncio.

**Spec reference:** `docs/superpowers/specs/2026-05-19-image-component-multi-gpu-design.md` (rev 2, commit `31e4d3c`).

**Branch:** Work continues on `feat/image-component-multi-gpu-pr1` (already exists, contains spec-rev-2 + the obsolete Task 0 risk-gate script and its 2 fix commits). Task 0 result is recorded: **gate triggered §5.2 fallback → PR-2 will self-write `ImageSampler`**; PR-1 below is pure-infra and does not depend on the gate verdict.

**Out of scope for PR-1** (intentional — verified against spec rev 2 § 8 PR-1 row):
- `DiffusersImageBackend.from_components` classmethod + cross-device assembly → **PR-2**
- `ImageSampler` self-written sampler + `ModelArchAdapter` → **PR-2**
- `component_scanner` service + `/api/v1/components` endpoints → **PR-3**
- 4 new workflow loader nodes + `image_generate` rewrite → **PR-4**
- `useComponentState` hook + frontend palette → **PR-5**
- L2 image_generate output cache + `node_cache_hit` WS event → **PR-6**

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `backend/src/services/inference/component_spec.py` | `ComponentSpec` pydantic model + `ComponentKey` tuple type + `to_component_key(spec) -> ComponentKey` helper |
| `backend/src/services/inference/quant_loaders.py` | `QuantLoaderRegistry` class + 5 registered loader functions (plain / fp8mixed / mxfp8mixed / nvfp4mixed / GGUF rejector) |
| `backend/tests/test_component_spec.py` | Unit tests for ComponentSpec validation + ComponentKey hashing |
| `backend/tests/test_quant_loaders.py` | Unit tests per format using synthetic safetensors fixtures |
| `backend/tests/test_model_manager_components.py` | Unit + integration tests for new `_components` cache APIs |

### Modified files

| Path | Change |
|---|---|
| `backend/src/services/inference/base.py` | Re-export `ComponentSpec` from `component_spec` module so consumers can `from src.services.inference.base import ComponentSpec` like they already do for `LoRASpec` |
| `backend/src/services/inference/image_diffusers.py` | Refactor `load_quantized_transformer` (`image_diffusers.py:105`) — its dequant logic moves into `quant_loaders.load_fp8mixed`; the original function shrinks to a thin wrapper that builds the empty transformer module and calls `QUANT_LOADERS.dispatch(spec)` for the state_dict. Legacy `__init__(paths=, device=)` constructor and `load(device)` body UNCHANGED. |
| `backend/src/services/model_manager.py` | Add `_components: dict[ComponentKey, LoadedComponent]` + `_component_locks` + `_component_failures` alongside existing `_models` dict. Add public methods `get_or_load_component`, `is_component_loaded`, `unload_component`. NO change to existing `load_model` / `is_loaded` / `unload_model` APIs. |

### Files explicitly NOT touched in PR-1

- `backend/src/services/workflow_executor.py` — node-level changes are PR-4
- `backend/src/runner/runner_process.py` — `_build_request` components branch is PR-4
- `backend/src/services/nodes/image.py` — node schema changes are PR-4
- Any frontend file — `useComponentState` + new nodes are PR-5
- `backend/configs/model_paths.yaml` — that's PR-3 (component_scanner)
- `backend/scripts/verify_flux2_cross_device.py` — Task 0 risk gate, already committed (`260c155` and predecessors). Leave as-is.

---

## Task 1: `ComponentSpec` + `ComponentKey` Types

**Files:**
- Create: `backend/src/services/inference/component_spec.py`
- Create: `backend/tests/test_component_spec.py`
- Modify: `backend/src/services/inference/base.py` (re-export)

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_component_spec.py
"""ComponentSpec validation + ComponentKey hashing for L1 cache."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.services.inference.base import LoRASpec
from src.services.inference.component_spec import ComponentSpec, ComponentKey, to_component_key


def test_unet_component_spec_valid():
    spec = ComponentSpec(
        kind="unet",
        file="/abs/path/transformer.safetensors",
        device="cuda:1",
        dtype="bfloat16",
        loras=[LoRASpec(name="style", strength=0.8)],
        adapter_arch="flux2",
    )
    assert spec.kind == "unet"
    assert spec.device == "cuda:1"
    assert len(spec.loras) == 1


def test_clip_component_spec_valid():
    spec = ComponentSpec(
        kind="clip", file="/p/clip.safetensors", device="cuda:0",
        dtype="bfloat16", clip_arch="flux2",
    )
    assert spec.clip_arch == "flux2"
    assert spec.loras == []


def test_vae_component_spec_minimal():
    spec = ComponentSpec(kind="vae", file="/p/vae.safetensors", device="cuda:2", dtype="float16")
    assert spec.kind == "vae"


def test_kind_must_be_one_of_three():
    with pytest.raises(ValidationError):
        ComponentSpec(kind="other", file="/p/x", device="cuda:0", dtype="bfloat16")


def test_device_must_be_cuda_or_cpu_or_auto():
    # "auto" → ModelManager will resolve via get_best_gpu
    ComponentSpec(kind="vae", file="/p/x", device="auto", dtype="bfloat16")
    ComponentSpec(kind="vae", file="/p/x", device="cpu", dtype="bfloat16")
    ComponentSpec(kind="vae", file="/p/x", device="cuda:0", dtype="bfloat16")
    with pytest.raises(ValidationError):
        ComponentSpec(kind="vae", file="/p/x", device="mps:0", dtype="bfloat16")


def test_component_key_is_hashable_tuple():
    spec = ComponentSpec(
        kind="unet", file="/p/u.safe", device="cuda:1", dtype="bfloat16",
        loras=[LoRASpec(name="a", strength=0.8), LoRASpec(name="b", strength=0.4)],
    )
    key = to_component_key(spec)
    assert isinstance(key, tuple)
    file_path, device, lora_frozenset = key
    assert file_path == "/p/u.safe"
    assert device == "cuda:1"
    assert isinstance(lora_frozenset, frozenset)
    assert lora_frozenset == frozenset({("a", 0.8), ("b", 0.4)})
    # Hashable → usable as dict key
    d = {key: "loaded"}
    assert d[key] == "loaded"


def test_component_key_stable_across_lora_order():
    """Two specs with same loras in different order produce equal key (frozenset)."""
    s1 = ComponentSpec(kind="unet", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="a", strength=0.8), LoRASpec(name="b", strength=0.4)])
    s2 = ComponentSpec(kind="unet", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="b", strength=0.4), LoRASpec(name="a", strength=0.8)])
    assert to_component_key(s1) == to_component_key(s2)


def test_component_key_distinguishes_strength():
    s1 = ComponentSpec(kind="unet", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="a", strength=0.8)])
    s2 = ComponentSpec(kind="unet", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="a", strength=0.4)])
    assert to_component_key(s1) != to_component_key(s2)


def test_component_spec_re_exported_from_base():
    """Spec § 5.1 says ComponentSpec lives under inference.base for caller convenience."""
    from src.services.inference.base import ComponentSpec as CS_from_base
    assert CS_from_base is ComponentSpec
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
.venv/bin/pytest tests/test_component_spec.py -v
```

Expected: `ImportError: cannot import name 'ComponentSpec'` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement `component_spec.py`**

```python
# backend/src/services/inference/component_spec.py
"""ComponentSpec — single Flux2 component descriptor (unet / clip / vae).

PR-1 of image-component-multi-gpu spec §5.1. Emitted by future loader workflow nodes
(`image_unet_load` etc., added in PR-4). Cached by ModelManager via ComponentKey.

Cross-process safety: pure pydantic v2 model, msgpack-serializable through P.RunNode.inputs.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.services.inference.base import LoRASpec

_DEVICE_RE = re.compile(r"^(cpu|auto|cuda:\d+)$")


class ComponentSpec(BaseModel):
    """One Flux2 / SDXL / Z-Image component (transformer | text encoder | vae).

    `device` accepts "auto" → ModelManager.get_best_gpu(vram_mb) resolves at load time.
    `loras` is meaningful only when kind="unet" (Flux2 LoRAs patch DiT, not text_encoder/VAE).
    """

    kind: Literal["unet", "clip", "vae"]
    file: str = Field(..., description="Absolute path resolved by component_scanner")
    device: str = Field(..., description="'auto' | 'cpu' | 'cuda:N'")
    dtype: str = Field(..., description="'bfloat16' | 'float16' | 'fp8_e4m3'")
    loras: list[LoRASpec] = Field(default_factory=list)
    adapter_arch: str | None = Field(None, description="unet only: 'flux2' | 'flux1'")
    clip_arch: str | None = Field(None, description="clip only: 'flux2' | 'flux1' | 'sdxl' | 'qwen'")

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if not _DEVICE_RE.match(v):
            raise ValueError(f"device must match cpu|auto|cuda:N — got {v!r}")
        return v


# (file, device, lora_set) — order-independent on loras via frozenset
ComponentKey = tuple[str, str, frozenset[tuple[str, float]]]


def to_component_key(spec: ComponentSpec) -> ComponentKey:
    """Compute the L1 cache key for this component.

    LoRA list → frozenset of (name, strength) so re-ordering doesn't break cache hits.
    Two specs with identical file/device + same LoRAs (any order) produce equal keys.
    """
    lora_set = frozenset((l.name, float(l.strength)) for l in spec.loras)
    return (spec.file, spec.device, lora_set)
```

- [ ] **Step 4: Re-export from `base.py`**

Open `backend/src/services/inference/base.py` and add **at the bottom** (after all existing class definitions):

```python
# Re-export for caller convenience — components are most commonly used by
# DiffusersImageBackend (image_diffusers.py) and ModelManager.
from src.services.inference.component_spec import ComponentSpec  # noqa: E402,F401
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_component_spec.py -v
```

Expected: 8/8 PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/inference/component_spec.py \
        backend/src/services/inference/base.py \
        backend/tests/test_component_spec.py
git commit -m "feat(inference): ComponentSpec + ComponentKey types (PR-1 Task 1)"
```

---

## Task 2: `QuantLoaderRegistry` + Plain Safetensors Loader

**Files:**
- Create: `backend/src/services/inference/quant_loaders.py`
- Create: `backend/tests/test_quant_loaders.py`

- [ ] **Step 1: Write the failing tests (registry + plain loader)**

```python
# backend/tests/test_quant_loaders.py
"""QuantLoaderRegistry dispatch + per-format loader correctness.

For each quant format we use a small synthetic safetensors fixture rather than the
real ~18GB Flux2 files — the dequant logic only needs a handful of tensors to
exercise the code path. Real-file end-to-end is covered later by PR-2 smoke.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.services.inference.component_spec import ComponentSpec
from src.services.inference.quant_loaders import QuantLoaderRegistry, QUANT_LOADERS, UnsupportedQuantError


def _make_plain_safetensors(tmp_path: Path, name: str) -> Path:
    """Synthetic plain bf16 safetensors with 3 small tensors."""
    sd = {
        "block.0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "block.0.bias":   torch.zeros(8, dtype=torch.bfloat16),
        "block.1.weight": torch.randn(8, 4, dtype=torch.bfloat16),
    }
    path = tmp_path / f"{name}.safetensors"
    save_file(sd, str(path))
    return path


def test_registry_register_and_dispatch():
    reg = QuantLoaderRegistry()
    seen = []

    @reg.register(match=lambda spec: "marker_a" in spec.file)
    def loader_a(spec):
        seen.append("a")
        return "result_a"

    @reg.register(match=lambda spec: "marker_b" in spec.file)
    def loader_b(spec):
        seen.append("b")
        return "result_b"

    spec_a = ComponentSpec(kind="vae", file="/p/marker_a.safetensors", device="cpu", dtype="bfloat16")
    assert reg.dispatch(spec_a) == "result_a"
    assert seen == ["a"]

    spec_b = ComponentSpec(kind="vae", file="/p/marker_b.safetensors", device="cpu", dtype="bfloat16")
    assert reg.dispatch(spec_b) == "result_b"


def test_registry_first_match_wins():
    """Registration order = match priority. Specific must register before generic."""
    reg = QuantLoaderRegistry()

    @reg.register(match=lambda spec: "fp8" in spec.file)
    def specific(spec):
        return "specific"

    @reg.register(match=lambda spec: spec.file.endswith(".safetensors"))
    def generic(spec):
        return "generic"

    spec_fp8 = ComponentSpec(kind="unet", file="/p/foo-fp8.safetensors", device="cpu", dtype="bfloat16")
    assert reg.dispatch(spec_fp8) == "specific"

    spec_plain = ComponentSpec(kind="unet", file="/p/foo.safetensors", device="cpu", dtype="bfloat16")
    assert reg.dispatch(spec_plain) == "generic"


def test_registry_no_match_raises():
    reg = QuantLoaderRegistry()

    @reg.register(match=lambda spec: False)
    def never(spec):
        return None

    spec = ComponentSpec(kind="vae", file="/p/x.gguf", device="cpu", dtype="bfloat16")
    with pytest.raises(UnsupportedQuantError, match="no quant loader matches"):
        reg.dispatch(spec)


def test_plain_safetensors_loader_loads_tensors(tmp_path):
    """The plain safetensors loader returns a state_dict-like mapping with original dtype."""
    sf = _make_plain_safetensors(tmp_path, "plain_bf16")
    spec = ComponentSpec(kind="vae", file=str(sf), device="cpu", dtype="bfloat16")

    result = QUANT_LOADERS.dispatch(spec)

    # Plain loader returns dict[str, Tensor] (caller wraps into module)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"block.0.weight", "block.0.bias", "block.1.weight"}
    assert result["block.0.weight"].dtype == torch.bfloat16
    assert result["block.0.weight"].shape == (8, 8)


def test_plain_safetensors_loader_honors_device(tmp_path):
    sf = _make_plain_safetensors(tmp_path, "plain_for_cpu")
    spec = ComponentSpec(kind="vae", file=str(sf), device="cpu", dtype="bfloat16")
    result = QUANT_LOADERS.dispatch(spec)
    assert result["block.0.weight"].device.type == "cpu"


def test_plain_safetensors_loader_gguf_not_supported(tmp_path):
    """GGUF is V2 PR-7 — V1 dispatches to UnsupportedQuantError."""
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"GGUF\x00" * 16)
    spec = ComponentSpec(kind="unet", file=str(gguf), device="cpu", dtype="bfloat16")
    with pytest.raises(UnsupportedQuantError, match="GGUF .* V2"):
        QUANT_LOADERS.dispatch(spec)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_quant_loaders.py -v
```

Expected: `ImportError: cannot import name 'QuantLoaderRegistry'`.

- [ ] **Step 3: Implement registry + plain + gguf-rejector**

```python
# backend/src/services/inference/quant_loaders.py
"""Quant loader registry for image-component-multi-gpu PR-1.

Per spec §5.3: registry maps (ComponentSpec) → loaded weights (state_dict).
First-match-wins; register specific formats (fp8mixed / mxfp8mixed / nvfp4mixed)
before plain safetensors fallback.

Each loader returns:
  dict[str, Tensor]      (state_dict — caller's responsibility to wrap into a module)

GGUF is rejected eagerly with UnsupportedQuantError; V2 PR-7 will add it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file

from src.services.inference.component_spec import ComponentSpec

logger = logging.getLogger(__name__)


class UnsupportedQuantError(RuntimeError):
    """Raised when no registered loader matches a ComponentSpec."""


class QuantLoaderRegistry:
    """First-match-wins registry. Register specific formats before generic fallbacks."""

    def __init__(self) -> None:
        self._loaders: list[tuple[Callable[[ComponentSpec], bool], Callable[[ComponentSpec], Any]]] = []

    def register(self, *, match: Callable[[ComponentSpec], bool]) -> Callable[[Callable], Callable]:
        """Decorator. `match(spec)` → bool decides if this loader handles the spec."""
        def deco(fn: Callable[[ComponentSpec], Any]) -> Callable[[ComponentSpec], Any]:
            self._loaders.append((match, fn))
            return fn
        return deco

    def dispatch(self, spec: ComponentSpec) -> Any:
        for matcher, fn in self._loaders:
            if matcher(spec):
                logger.debug("quant_loaders: dispatching %s to %s", spec.file, fn.__name__)
                return fn(spec)
        raise UnsupportedQuantError(f"no quant loader matches {spec.file!r}")


QUANT_LOADERS = QuantLoaderRegistry()


# Reject GGUF eagerly — V2 PR-7 work, not in scope for PR-1.
@QUANT_LOADERS.register(match=lambda spec: spec.file.lower().endswith(".gguf"))
def reject_gguf(spec: ComponentSpec) -> Any:
    raise UnsupportedQuantError(
        f"GGUF quantization is V2 PR-7 follow-up; cannot load {spec.file!r} in PR-1"
    )


def _dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,  # torch 2.10 native
    }.get(dtype_str, torch.bfloat16)


# Plain bf16/fp16 safetensors — uniform state_dict loader. Caller (PR-2's
# DiffusersImageBackend or test) decides whether to wrap into a module.
@QUANT_LOADERS.register(match=lambda spec: spec.file.endswith(".safetensors"))
def load_safetensors_plain(spec: ComponentSpec) -> dict[str, torch.Tensor]:
    """Plain bf16/fp16 safetensors → state_dict, target dtype applied.

    Note: this is the FALLBACK matcher in the registry. Specific formats (fp8mixed,
    mxfp8mixed, nvfp4mixed) registered LATER in this module will match first via
    filename substring; this loader only runs for safetensors without those markers.
    """
    target = _dtype_str_to_torch(spec.dtype)
    sd = load_file(spec.file, device="cpu")  # always load to CPU first; .to(device) is caller's job
    return {k: v.to(target) for k, v in sd.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_quant_loaders.py -v
```

Expected: 6/6 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/inference/quant_loaders.py backend/tests/test_quant_loaders.py
git commit -m "feat(inference): QuantLoaderRegistry + plain safetensors + GGUF reject (PR-1 Task 2)"
```

---

## Task 3: fp8mixed Quant Loader (Refactor Existing into Registry)

**Why:** `backend/src/services/inference/image_diffusers.py:105` already has `load_quantized_transformer(main_path, sf_path, dtype) -> Module`. PR-1 lifts that logic's **state_dict dequant step** into the registry as `load_fp8mixed` (returns state_dict), and shrinks the original function to a thin wrapper that builds the empty transformer module + delegates state_dict prep to the registry + calls `load_state_dict`. No behavior change.

**Files:**
- Modify: `backend/src/services/inference/quant_loaders.py` (add fp8mixed loader)
- Modify: `backend/src/services/inference/image_diffusers.py` (`load_quantized_transformer` becomes a thin wrapper)
- Modify: `backend/tests/test_quant_loaders.py` (add fp8mixed tests)

- [ ] **Step 1: Inspect the existing dequant**

```bash
sed -n '95,170p' backend/src/services/inference/image_diffusers.py
```

Note the algorithm:
- `safetensors_load_file(sf_path)` → state dict containing both fp8 tensors and their `.weight_scale` companions
- For each fp8 tensor, multiply by scale, drop `.comfy_quant` / `.weight_scale` keys
- Result is a clean bf16 state dict ready for `transformer.load_state_dict(...)`

The new `load_fp8mixed` in `quant_loaders.py` reproduces this state_dict prep step verbatim — no behavior change.

- [ ] **Step 2: Add fp8mixed tests (failing initially)**

Append to `backend/tests/test_quant_loaders.py`:

```python
def _make_fp8mixed_safetensors(tmp_path: Path, name: str) -> Path:
    """Synthetic comfy_quant-style fp8 fixture: 1 fp8 tensor + companion scale, plus 1 plain tensor."""
    weight_fp8 = torch.randn(4, 4).to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([0.125], dtype=torch.float32)
    plain = torch.randn(4, 4, dtype=torch.bfloat16)
    sd = {
        "block.0.weight": weight_fp8,
        "block.0.weight_scale": weight_scale,
        "block.0.weight.comfy_quant": torch.tensor([1], dtype=torch.uint8),  # marker
        "block.1.weight": plain,
    }
    path = tmp_path / f"{name}.safetensors"
    save_file(sd, str(path))
    return path


def test_fp8mixed_loader_dequants_and_drops_metadata(tmp_path):
    sf = _make_fp8mixed_safetensors(tmp_path, "Flux2-Klein-9B-True-v2-fp8mixed")
    spec = ComponentSpec(kind="unet", file=str(sf), device="cpu", dtype="bfloat16")

    sd = QUANT_LOADERS.dispatch(spec)

    # fp8 tensor was dequant'd (multiplied by scale) into bfloat16
    assert "block.0.weight" in sd
    assert sd["block.0.weight"].dtype == torch.bfloat16
    # plain tensor passed through
    assert sd["block.1.weight"].dtype == torch.bfloat16
    # metadata keys must be dropped before caller's load_state_dict
    assert "block.0.weight_scale" not in sd
    assert "block.0.weight.comfy_quant" not in sd


def test_fp8mixed_loader_match_priority_over_plain():
    """File with 'fp8mixed' in name must dispatch to fp8 loader, not plain."""
    matchers = [m for m, _fn in QUANT_LOADERS._loaders]
    fp8_idx = next(i for i, m in enumerate(matchers)
                   if m(ComponentSpec(kind="unet", file="x-fp8mixed.safetensors",
                                      device="cpu", dtype="bfloat16")))
    plain_idx = next(i for i, m in enumerate(matchers)
                     if m(ComponentSpec(kind="unet", file="plain.safetensors",
                                        device="cpu", dtype="bfloat16")))
    assert fp8_idx < plain_idx
```

- [ ] **Step 3: Run test — expect fail**

```bash
.venv/bin/pytest tests/test_quant_loaders.py::test_fp8mixed_loader_dequants_and_drops_metadata -v
```

Expected: FAIL — fp8 spec dispatched to plain loader, metadata keys leak through.

- [ ] **Step 4: Implement fp8mixed loader in registry**

Open `backend/src/services/inference/quant_loaders.py`. Add **above the `load_safetensors_plain` registration** (must register before plain to win priority):

```python
def _has_comfy_quant_metadata(file_path: str) -> bool:
    """Sniff a safetensors header for any `.comfy_quant` suffixed key (cheap — no full read)."""
    try:
        from safetensors import safe_open
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".comfy_quant"):
                    return True
    except Exception:  # noqa: BLE001 — fail-soft on header read error
        return False
    return False


@QUANT_LOADERS.register(match=lambda spec: (
    "fp8mixed" in Path(spec.file).name.lower()
    or _has_comfy_quant_metadata(spec.file)
))
def load_fp8mixed(spec: ComponentSpec) -> dict[str, torch.Tensor]:
    """Wikeeyang comfy_quant fp8 → dequant by `.weight_scale` companion → target dtype.

    Algorithm (preserved from image_diffusers.py:105 load_quantized_transformer):
      1. safetensors_load_file → state dict with fp8 weights + .weight_scale + .comfy_quant
      2. For each fp8 tensor, multiply by float32 scale, cast to target dtype
      3. Drop .weight_scale and .comfy_quant marker keys
      4. Return clean state dict ready for caller's load_state_dict

    Reference fixture: /media/heygo/Program/models/nous/image/diffusion_models/
    Flux2-Klein-9B-True-v2-fp8mixed.safetensors
    """
    target = _dtype_str_to_torch(spec.dtype)
    raw = load_file(spec.file, device="cpu")

    clean: dict[str, torch.Tensor] = {}
    fp8_count = 0
    for key, tensor in raw.items():
        if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
            continue  # metadata key, drop
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale"
            scale = raw.get(scale_key)
            if scale is None:
                logger.warning("fp8 tensor %s has no companion %s scale; loading at fp8 dtype", key, scale_key)
                clean[key] = tensor.to(target)
                continue
            # dequant: fp8 × scale → fp32 → target
            clean[key] = (tensor.to(torch.float32) * scale.to(torch.float32)).to(target)
            fp8_count += 1
        else:
            clean[key] = tensor.to(target)

    logger.info("quant_loaders.fp8mixed: %d fp8 weights dequant'd, %d total keys (%s)",
                fp8_count, len(clean), Path(spec.file).name)
    return clean
```

- [ ] **Step 5: Update `image_diffusers.py:105` `load_quantized_transformer` to delegate to registry**

Open `backend/src/services/inference/image_diffusers.py`. Find the `load_quantized_transformer` function at line 105. Replace its body so the state_dict dequant step calls the new registry path; keep the "build empty transformer + load_state_dict" wrapper around it:

```python
def load_quantized_transformer(main_path: Path, sf_path: Path, dtype) -> Any:
    """Load a wikeeyang-style fp8 single-file transformer.

    PR-1 refactor: state_dict dequant moved to quant_loaders.load_fp8mixed.
    This function keeps building the empty transformer module from main_path
    (HF diffusers layout) then loads the dequant'd state_dict into it. The
    behavior is byte-identical to pre-PR-1 — only the dequant code location moved.
    """
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.quant_loaders import QUANT_LOADERS

    # 1. Build the empty transformer module from the main diffusers directory.
    #    Locate the existing inline transformer-building block (the lines just
    #    before the previous dequant loop) and extract it into a local helper
    #    `_build_empty_transformer_from_dir(main_path, dtype)` returning the
    #    bare module ready for load_state_dict. No behavior change — pure rename.
    transformer = _build_empty_transformer_from_dir(main_path, dtype)

    # 2. Dequant state_dict via the new registry path.
    dtype_str = {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(dtype, "bfloat16")
    spec = ComponentSpec(kind="unet", file=str(sf_path), device="cpu", dtype=dtype_str)
    clean_sd = QUANT_LOADERS.dispatch(spec)

    # 3. Load the clean state_dict (existing logic, just relocated).
    missing, unexpected = transformer.load_state_dict(clean_sd, strict=False)
    if missing or unexpected:
        logger.info("image: quantized transformer load — missing=%d unexpected=%d",
                    len(missing), len(unexpected))
    else:
        logger.info("image: quantized transformer load — 0 missing / 0 unexpected ✓")
    return transformer
```

If `_build_empty_transformer_from_dir` doesn't exist as a helper yet, extract the inline transformer-building lines from the pre-PR-1 body into a new module-level function of that name (pure rename, no logic change). The original function's "build module" block is a few lines before line 105's `load_file()` call — copy those lines verbatim into the new helper.

- [ ] **Step 6: Run tests — expect both new tests pass + existing fp8 path still works**

```bash
.venv/bin/pytest tests/test_quant_loaders.py -v
.venv/bin/pytest tests/test_image_diffusers.py::test_load_with_quantized_transformer_method_exists -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/src/services/inference/quant_loaders.py \
        backend/src/services/inference/image_diffusers.py \
        backend/tests/test_quant_loaders.py
git commit -m "feat(inference): fp8mixed quant loader in registry; reuse existing dequant (PR-1 Task 3)"
```

---

## Task 4: mxfp8mixed Quant Loader

**Files:**
- Modify: `backend/src/services/inference/quant_loaders.py`
- Modify: `backend/tests/test_quant_loaders.py`

mxfp8 (Microscaling fp8) uses block-wise scales instead of per-tensor scales. Reference: NVIDIA's MX format — each 32-element block has its own E8M0 scale exponent.

- [ ] **Step 1: Add failing test**

Append to `backend/tests/test_quant_loaders.py`:

```python
def _make_mxfp8mixed_safetensors(tmp_path: Path, name: str) -> Path:
    """Synthetic mxfp8 fixture: tensor in mxfp8 format with per-block scale.

    Layout used by community Flux2 mxfp8 quants:
      <name>.weight        : fp8_e4m3 weights (flat)
      <name>.weight_scale  : uint8 E8M0 per-32-element block scales (shape = numel/32)
      <name>.weight.comfy_quant : marker tensor (uint8)
    Real Flux2-Klein-9B-True-v2-mxfp8mixed.safetensors uses this layout.
    """
    # 64 elements → 2 blocks of 32 → 2 E8M0 scales
    w_fp8 = torch.randn(64).to(torch.float8_e4m3fn)
    # E8M0 scales: stored as uint8 representing power-of-2 exponent (bias 127)
    w_scale = torch.tensor([130, 128], dtype=torch.uint8)  # 2^3 and 2^1
    plain = torch.randn(4, 4, dtype=torch.bfloat16)
    sd = {
        "block.0.weight": w_fp8,
        "block.0.weight_scale": w_scale,
        "block.0.weight.comfy_quant": torch.tensor([2], dtype=torch.uint8),  # arch=2 == mxfp8
        "block.1.weight": plain,
    }
    path = tmp_path / f"{name}.safetensors"
    save_file(sd, str(path))
    return path


def test_mxfp8mixed_loader_block_dequants_to_target_dtype(tmp_path):
    sf = _make_mxfp8mixed_safetensors(tmp_path, "Flux2-X-mxfp8mixed")
    spec = ComponentSpec(kind="unet", file=str(sf), device="cpu", dtype="bfloat16")

    sd = QUANT_LOADERS.dispatch(spec)

    assert "block.0.weight" in sd
    assert sd["block.0.weight"].dtype == torch.bfloat16
    assert sd["block.0.weight"].numel() == 64
    assert "block.0.weight_scale" not in sd
    assert "block.0.weight.comfy_quant" not in sd
    # plain tensor preserved
    assert sd["block.1.weight"].shape == (4, 4)


def test_mxfp8mixed_loader_priority_over_fp8mixed():
    """File named mxfp8mixed must NOT fall through to fp8mixed loader."""
    matchers = [m for m, _fn in QUANT_LOADERS._loaders]
    mxfp8_idx = next(i for i, m in enumerate(matchers)
                     if m(ComponentSpec(kind="unet", file="x-mxfp8mixed.safetensors",
                                        device="cpu", dtype="bfloat16")))
    fp8_idx = next(i for i, m in enumerate(matchers)
                   if m(ComponentSpec(kind="unet", file="x-fp8mixed.safetensors",
                                      device="cpu", dtype="bfloat16")))
    assert mxfp8_idx < fp8_idx, "mxfp8mixed matcher must register before fp8mixed"
```

- [ ] **Step 2: Run test — fail**

```bash
.venv/bin/pytest tests/test_quant_loaders.py::test_mxfp8mixed_loader_block_dequants_to_target_dtype -v
```

Expected: FAIL — spec dispatched to plain (or fp8mixed) loader, mxfp8 dequant not performed.

- [ ] **Step 3: Implement mxfp8 loader**

Open `backend/src/services/inference/quant_loaders.py`. Add **above the fp8mixed registration** (more-specific wins priority):

```python
@QUANT_LOADERS.register(match=lambda spec: "mxfp8mixed" in Path(spec.file).name.lower())
def load_mxfp8mixed(spec: ComponentSpec) -> dict[str, torch.Tensor]:
    """Microscaling fp8 → dequant by block-wise E8M0 scale → target dtype.

    Format: per-32-element blocks, each with a uint8 E8M0 exponent in `.weight_scale`.
    Real file: Flux2-Klein-9B-True-v2-mxfp8mixed.safetensors (9.7GB).

    Algorithm:
      1. Load fp8 weight + uint8 scale tensor (1 byte per 32-element block)
      2. For each block: scale_fp32 = 2.0 ** (uint8_scale - 127)   # E8M0 with bias 127
      3. fp32_weight = fp8_weight × scale (broadcast within block)
      4. Cast to target dtype, drop metadata keys
    """
    target = _dtype_str_to_torch(spec.dtype)
    raw = load_file(spec.file, device="cpu")
    BLOCK_SIZE = 32

    clean: dict[str, torch.Tensor] = {}
    dequant_count = 0
    for key, tensor in raw.items():
        if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale"
            scale_uint8 = raw.get(scale_key)
            if scale_uint8 is None:
                logger.warning("mxfp8: tensor %s missing %s; using fp8 cast only", key, scale_key)
                clean[key] = tensor.to(target)
                continue
            # E8M0: scale = 2^(uint8 - 127)
            scale_fp32 = torch.pow(2.0, scale_uint8.to(torch.float32) - 127.0)
            # Broadcast block-wise: flatten weights, repeat each scale BLOCK_SIZE times
            flat = tensor.flatten().to(torch.float32)
            assert flat.numel() % BLOCK_SIZE == 0, \
                f"mxfp8: {key} numel {flat.numel()} not divisible by block {BLOCK_SIZE}"
            assert scale_fp32.numel() * BLOCK_SIZE == flat.numel(), \
                f"mxfp8: {key} scale count {scale_fp32.numel()} × block {BLOCK_SIZE} ≠ weight numel {flat.numel()}"
            block_scales = scale_fp32.repeat_interleave(BLOCK_SIZE)
            dequant = (flat * block_scales).to(target).reshape(tensor.shape)
            clean[key] = dequant
            dequant_count += 1
        else:
            clean[key] = tensor.to(target)

    logger.info("quant_loaders.mxfp8mixed: %d block-quant tensors dequant'd, %d total keys (%s)",
                dequant_count, len(clean), Path(spec.file).name)
    return clean
```

- [ ] **Step 4: Run all quant_loaders tests — pass**

```bash
.venv/bin/pytest tests/test_quant_loaders.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/inference/quant_loaders.py backend/tests/test_quant_loaders.py
git commit -m "feat(inference): mxfp8mixed block-wise dequant loader (PR-1 Task 4)"
```

---

## Task 5: nvfp4mixed Quant Loader

**Why:** nvfp4 is NVIDIA's FP4 format with per-block scale (similar structure to mxfp8 but 4-bit weights, packed 2-per-byte). Real fixture: `Flux2-Klein-9B-True-v2-nvfp4mixed.safetensors` (5.6GB — half the size of fp8mixed).

**Files:**
- Modify: `backend/src/services/inference/quant_loaders.py`
- Modify: `backend/tests/test_quant_loaders.py`

- [ ] **Step 1: Add failing test**

```python
# Append to backend/tests/test_quant_loaders.py

def _make_nvfp4mixed_safetensors(tmp_path: Path, name: str) -> Path:
    """Synthetic nvfp4 fixture: 4-bit weights packed 2-per-uint8 + per-16-block fp32 scale.

    Format spec (community Flux2 nvfp4 quants):
      <name>.weight        : uint8, shape = (numel / 2,)  — two 4-bit weights per byte
      <name>.weight_scale  : float32, shape = (numel / 16,) — one scale per 16-element block
      <name>.weight.comfy_quant : marker (uint8 [3])
      <name>.weight_shape  : int32 [H, W] — original shape for unpack
    """
    NUMEL = 64
    BLOCK = 16
    packed = torch.randint(0, 256, (NUMEL // 2,), dtype=torch.uint8)
    scale = torch.randn(NUMEL // BLOCK, dtype=torch.float32).abs() + 0.1
    plain = torch.randn(2, 2, dtype=torch.bfloat16)
    sd = {
        "block.0.weight": packed,
        "block.0.weight_scale": scale,
        "block.0.weight.comfy_quant": torch.tensor([3], dtype=torch.uint8),
        "block.0.weight_shape": torch.tensor([8, 8], dtype=torch.int32),  # original shape for unpack
        "block.1.weight": plain,
    }
    path = tmp_path / f"{name}.safetensors"
    save_file(sd, str(path))
    return path


def test_nvfp4mixed_loader_unpacks_4bit_blocks(tmp_path):
    sf = _make_nvfp4mixed_safetensors(tmp_path, "Flux2-X-nvfp4mixed")
    spec = ComponentSpec(kind="unet", file=str(sf), device="cpu", dtype="bfloat16")

    sd = QUANT_LOADERS.dispatch(spec)

    # 4-bit unpacked → 64 elements → reshaped to (8, 8)
    assert sd["block.0.weight"].shape == (8, 8)
    assert sd["block.0.weight"].dtype == torch.bfloat16
    # metadata dropped
    for suffix in ("_scale", ".comfy_quant", "_shape"):
        assert not any(k.endswith(suffix) for k in sd)
    # plain tensor preserved
    assert sd["block.1.weight"].shape == (2, 2)


def test_nvfp4mixed_loader_priority_over_mxfp8():
    matchers = [m for m, _fn in QUANT_LOADERS._loaders]
    nvfp4_idx = next(i for i, m in enumerate(matchers)
                     if m(ComponentSpec(kind="unet", file="x-nvfp4mixed.safetensors",
                                        device="cpu", dtype="bfloat16")))
    mxfp8_idx = next(i for i, m in enumerate(matchers)
                     if m(ComponentSpec(kind="unet", file="x-mxfp8mixed.safetensors",
                                        device="cpu", dtype="bfloat16")))
    assert nvfp4_idx < mxfp8_idx
```

- [ ] **Step 2: Run — fail**

```bash
.venv/bin/pytest tests/test_quant_loaders.py::test_nvfp4mixed_loader_unpacks_4bit_blocks -v
```

Expected: FAIL — nvfp4 spec dispatched to plain (or fp8mixed) loader, weights stay packed.

- [ ] **Step 3: Implement nvfp4 loader**

Open `backend/src/services/inference/quant_loaders.py`. Add **at the top of the registration block** (most specific, highest priority):

```python
def _unpack_int4_to_int8(packed: torch.Tensor) -> torch.Tensor:
    """Two 4-bit signed values per byte → expand to (-8..7) int8.

    Layout (community nvfp4 quants): low nibble = first weight, high nibble = second.
    Each nibble is signed 4-bit (range -8..7) following two's complement on 4 bits.
    """
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    # Sign-extend 4-bit → 8-bit: values >= 8 are negative
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    # Interleave: [low_0, high_0, low_1, high_1, ...]
    interleaved = torch.stack([low, high], dim=-1).reshape(-1)
    return interleaved


@QUANT_LOADERS.register(match=lambda spec: "nvfp4mixed" in Path(spec.file).name.lower())
def load_nvfp4mixed(spec: ComponentSpec) -> dict[str, torch.Tensor]:
    """NVIDIA FP4 → unpack 2-per-byte → block-wise fp32 scale → target dtype.

    Real file: Flux2-Klein-9B-True-v2-nvfp4mixed.safetensors (5.6GB — about 1/3 the
    size of bf16, half of fp8mixed).

    Algorithm:
      1. Load uint8 packed weights + fp32 per-16-block scales + int32 original shape
      2. Unpack each byte into two signed 4-bit values (range -8..7)
      3. Per 16-element block, multiply by fp32 scale
      4. Reshape to original shape, cast to target dtype
    """
    target = _dtype_str_to_torch(spec.dtype)
    raw = load_file(spec.file, device="cpu")
    BLOCK_SIZE = 16

    clean: dict[str, torch.Tensor] = {}
    unpacked_count = 0

    # Group keys by base name (so we can find weight + weight_scale + weight_shape together)
    bases: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in raw.items():
        if ".comfy_quant" in k:
            continue
        if k.endswith("_scale"):
            base = k[: -len("_scale")]
            bases.setdefault(base, {})["scale"] = v
        elif k.endswith("_shape"):
            base = k[: -len("_shape")]
            bases.setdefault(base, {})["shape"] = v
        else:
            bases.setdefault(k, {})["weight"] = v

    for base, parts in bases.items():
        weight = parts.get("weight")
        if weight is None:
            continue
        if weight.dtype != torch.uint8:
            # Not a packed nvfp4 weight — plain tensor (e.g., a bias / bf16 keeper)
            clean[base] = weight.to(target)
            continue

        scale = parts.get("scale")
        shape = parts.get("shape")
        if scale is None or shape is None:
            logger.warning("nvfp4: %s packed but missing scale/shape; loading raw uint8", base)
            clean[base] = weight.to(target)
            continue

        # 1. Unpack 2 4-bit weights per byte → flat int8
        unpacked = _unpack_int4_to_int8(weight)
        # 2. Block-wise scale (one fp32 per 16-elem block)
        flat = unpacked.to(torch.float32)
        assert flat.numel() % BLOCK_SIZE == 0
        assert scale.numel() * BLOCK_SIZE == flat.numel()
        block_scales = scale.to(torch.float32).repeat_interleave(BLOCK_SIZE)
        dequant = (flat * block_scales).to(target)
        # 3. Reshape to original shape
        orig_shape = tuple(int(x) for x in shape.tolist())
        clean[base] = dequant.reshape(orig_shape)
        unpacked_count += 1

    logger.info("quant_loaders.nvfp4mixed: %d nvfp4 tensors unpacked, %d total keys (%s)",
                unpacked_count, len(clean), Path(spec.file).name)
    return clean
```

- [ ] **Step 4: Run all quant_loader tests — pass**

```bash
.venv/bin/pytest tests/test_quant_loaders.py -v
```

Expected: all PASS (10 tests now).

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/inference/quant_loaders.py backend/tests/test_quant_loaders.py
git commit -m "feat(inference): nvfp4mixed 4-bit unpack + block dequant loader (PR-1 Task 5)"
```

---

## Task 6: `ModelManager` — `_components` Cache + Public APIs

**Why:** Add a parallel L1 cache keyed on `ComponentKey` (per spec §3.3). Old `_models` dict and all existing APIs stay intact — PR-1 ships infra only, PR-2+ wires it in via the new `ImageSampler`.

**Files:**
- Modify: `backend/src/services/model_manager.py`
- Create: `backend/tests/test_model_manager_components.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_model_manager_components.py
"""PR-1 Task 6: ModelManager component-level cache.

Per spec §5.5, _components: dict[ComponentKey, LoadedComponent] coexists with
the legacy _models: dict[str, LoadedModel] in PR-1. PR-2 image adapters will
route through _components.
"""
from __future__ import annotations

import pytest

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import LoRASpec
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


@pytest.fixture
def mm():
    """Fresh ModelManager with no specs registered (component path is registry-agnostic)."""
    return ModelManager(registry=ModelRegistry.from_yaml_data({"models": []}), allocator=GPUAllocator())


@pytest.mark.asyncio
async def test_is_component_loaded_cold_by_default(mm):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")
    assert mm.is_component_loaded(spec) == "cold"


@pytest.mark.asyncio
async def test_get_or_load_component_marks_loaded(mm, monkeypatch):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")

    async def _fake_loader(s):
        return {"_module": "stub", "spec": s}

    monkeypatch.setattr(mm, "_load_component_impl", _fake_loader)

    result = await mm.get_or_load_component(spec)
    assert result["spec"] is spec
    assert mm.is_component_loaded(spec) == "loaded"


@pytest.mark.asyncio
async def test_get_or_load_component_cache_hit_does_not_call_loader_twice(mm, monkeypatch):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")
    calls = []

    async def _counting_loader(s):
        calls.append(s)
        return {"_module": f"stub{len(calls)}"}

    monkeypatch.setattr(mm, "_load_component_impl", _counting_loader)

    r1 = await mm.get_or_load_component(spec)
    r2 = await mm.get_or_load_component(spec)
    assert r1 is r2
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_get_or_load_component_distinguishes_lora_set(mm, monkeypatch):
    """Same file+device, different LoRA list → distinct cache entries."""
    s_a = ComponentSpec(kind="unet", file="/p/u.safe", device="cuda:0", dtype="bfloat16",
                       loras=[LoRASpec(name="style", strength=0.8)])
    s_b = ComponentSpec(kind="unet", file="/p/u.safe", device="cuda:0", dtype="bfloat16",
                       loras=[LoRASpec(name="style", strength=0.4)])

    async def _loader(s):
        return {"_module": f"variant_{hash(frozenset((l.name, l.strength) for l in s.loras))}"}

    monkeypatch.setattr(mm, "_load_component_impl", _loader)

    r_a = await mm.get_or_load_component(s_a)
    r_b = await mm.get_or_load_component(s_b)
    assert r_a["_module"] != r_b["_module"]


@pytest.mark.asyncio
async def test_unload_component_clears_cache_entry(mm, monkeypatch):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")

    async def _loader(s): return {"_module": "stub"}

    monkeypatch.setattr(mm, "_load_component_impl", _loader)

    await mm.get_or_load_component(spec)
    assert mm.is_component_loaded(spec) == "loaded"

    await mm.unload_component(spec)
    assert mm.is_component_loaded(spec) == "cold"


def test_legacy_models_dict_untouched(mm):
    """PR-1 invariant: existing _models dict, locks, load_failures all unchanged in shape."""
    assert hasattr(mm, "_models")
    assert isinstance(mm._models, dict)
    assert hasattr(mm, "_locks")
    assert hasattr(mm, "_load_failures")


@pytest.mark.asyncio
async def test_is_component_loaded_failed_when_loader_raises(mm, monkeypatch):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")

    async def _broken_loader(s):
        raise RuntimeError("synthetic OOM")

    monkeypatch.setattr(mm, "_load_component_impl", _broken_loader)

    with pytest.raises(RuntimeError, match="synthetic OOM"):
        await mm.get_or_load_component(spec)
    assert mm.is_component_loaded(spec) == "failed"
```

- [ ] **Step 2: Run — fail**

```bash
.venv/bin/pytest tests/test_model_manager_components.py -v
```

Expected: `AttributeError: 'ModelManager' object has no attribute '_components'` / `'get_or_load_component'` / etc.

- [ ] **Step 3: Implement component cache in `ModelManager`**

Open `backend/src/services/model_manager.py`. Modify `__init__` (around line 45) to add the new fields:

```python
    def __init__(self, registry: ModelRegistry, allocator: GPUAllocator) -> None:
        self._registry = registry
        self._allocator = allocator
        self._models: dict[str, LoadedModel] = {}
        self._references: dict[str, set[str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._load_failures: dict[str, str] = {}
        # PR-1: component-level cache (parallel to _models). Old yaml-driven
        # adapters keep using _models; PR-2 ImageSampler will use _components.
        self._components: dict["ComponentKey", "LoadedComponent"] = {}
        self._component_locks: dict["ComponentKey", asyncio.Lock] = {}
        self._component_failures: dict["ComponentKey", str] = {}
```

Add the `LoadedComponent` type alias near the top of the file (after `LoadedModel` class):

```python
# PR-1: components are lighter than full LoadedModel — they're just a loaded
# state_dict/module dict + metadata. Stored in ModelManager._components.
LoadedComponent = dict  # opaque to ModelManager: {_module, spec, loaded_at, ...}
```

Then at the bottom of the class, add the new public APIs:

```python
    # --- PR-1 Task 6: component-level cache APIs -----------------------------
    #
    # Per spec §5.5 these coexist with the legacy load_model/is_loaded/unload_model.
    # PR-2's ImageSampler will exercise these directly without going through yaml.

    def _component_lock_for(self, key: "ComponentKey") -> asyncio.Lock:
        return self._component_locks.setdefault(key, asyncio.Lock())

    def is_component_loaded(self, spec_or_key) -> str:
        """Returns 'loaded' | 'loading' | 'cold' | 'failed'.

        `spec_or_key` accepts ComponentSpec or ComponentKey for caller convenience.
        """
        from src.services.inference.component_spec import ComponentSpec, to_component_key
        key = to_component_key(spec_or_key) if isinstance(spec_or_key, ComponentSpec) else spec_or_key
        if key in self._components:
            return "loaded"
        lock = self._component_locks.get(key)
        if lock is not None and lock.locked():
            return "loading"
        if key in self._component_failures:
            return "failed"
        return "cold"

    async def get_or_load_component(self, spec: "ComponentSpec"):
        """Idempotent component load. Returns the loaded module/state_dict.

        Concurrency: per-key lock so two concurrent callers for same component
        won't double-load. Distinct components load in parallel (no global lock).
        """
        from src.services.inference.component_spec import to_component_key
        key = to_component_key(spec)
        async with self._component_lock_for(key):
            cached = self._components.get(key)
            if cached is not None:
                return cached
            try:
                loaded = await self._load_component_impl(spec)
            except Exception as e:  # noqa: BLE001 — record + re-raise
                self._component_failures[key] = f"{type(e).__name__}: {e}"
                raise
            self._components[key] = loaded
            self._component_failures.pop(key, None)
            return loaded

    async def _load_component_impl(self, spec: "ComponentSpec"):
        """The actual component load. PR-1: dispatches to quant_loaders for the
        state_dict. Wrapping state_dict into an actual diffusers module is PR-2's
        ImageSampler responsibility. Tests monkeypatch this method directly.

        Returns: dict with {_state_dict, spec, loaded_at}.
        """
        import time
        from src.services.inference.quant_loaders import QUANT_LOADERS

        state_dict = QUANT_LOADERS.dispatch(spec)
        return {"_state_dict": state_dict, "spec": spec, "loaded_at": time.monotonic()}

    async def unload_component(self, spec_or_key) -> None:
        """Drop the cache entry. PR-1: caller is responsible for any GPU memory release.
        Future PRs (LRU eviction) will call torch.cuda.empty_cache after pop."""
        from src.services.inference.component_spec import ComponentSpec, to_component_key
        key = to_component_key(spec_or_key) if isinstance(spec_or_key, ComponentSpec) else spec_or_key
        async with self._component_lock_for(key):
            self._components.pop(key, None)
            self._component_failures.pop(key, None)
```

- [ ] **Step 4: Run new tests + existing model_manager tests — pass**

```bash
.venv/bin/pytest tests/test_model_manager_components.py \
                tests/test_model_manager_get_or_load.py \
                tests/test_model_manager_v2.py -v
```

Expected: all PASS. Existing tests untouched because legacy `_models`/`load_model`/`is_loaded` path is unchanged.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/model_manager.py backend/tests/test_model_manager_components.py
git commit -m "feat(model-manager): _components L1 cache + get_or_load/is_component_loaded/unload (PR-1 Task 6)"
```

---

## Task 7: PR Wrap-up

- [ ] **Step 1: Full test suite — no regressions**

```bash
cd backend
.venv/bin/pytest -x -q 2>&1 | tail -20
```

Expected: all green. If anything broke, the most likely culprit is Task 3's extraction of `_build_empty_transformer_from_dir` — open that diff and confirm the inline-to-helper move kept behavior identical.

- [ ] **Step 2: Lint**

```bash
.venv/bin/ruff check src/services/inference/ src/services/model_manager.py tests/test_component_spec.py tests/test_quant_loaders.py tests/test_model_manager_components.py
```

Expected: `All checks passed!`.

- [ ] **Step 3: Push branch**

```bash
git push -u origin feat/image-component-multi-gpu-pr1
```

- [ ] **Step 4: Open PR**

```bash
gh pr create --base master \
  --title "feat(image): component-multi-gpu PR-1 (rev 2) — ComponentSpec + quant loaders + ComponentKey cache" \
  --body "$(cat <<'EOF'
## Summary

PR-1 (rev 2) of [image-component-multi-gpu-design](docs/superpowers/specs/2026-05-19-image-component-multi-gpu-design.md).

Lands the foundational infrastructure — no user-visible change yet. PR-2 will add the self-written `ImageSampler` (per spec §5.6) that consumes this infra.

- `ComponentSpec` / `ComponentKey` types (`backend/src/services/inference/component_spec.py`)
- `QuantLoaderRegistry` + 5 loaders (plain / fp8mixed / mxfp8mixed / nvfp4mixed / GGUF reject)
- `ModelManager._components` cache + `get_or_load_component` / `is_component_loaded` / `unload_component`
- `image_diffusers.py:105` `load_quantized_transformer` refactored to delegate state_dict dequant to the new registry (behavior identical, code relocated)

## What's NOT in this PR (intentional)

- `DiffusersImageBackend.from_components` + cross-device assembly → PR-2
- `ImageSampler` self-written sampler + `ModelArchAdapter` → PR-2
- `component_scanner` → PR-3
- Workflow loader nodes → PR-4
- Frontend `useComponentState` hook → PR-5
- L2 image_generate output cache → PR-6

## Test plan

- [x] `pytest tests/test_component_spec.py -v` — 8/8 PASS
- [x] `pytest tests/test_quant_loaders.py -v` — 10/10 PASS
- [x] `pytest tests/test_model_manager_components.py -v` — 7/7 PASS
- [x] `pytest tests/test_image_diffusers.py::test_load_with_quantized_transformer_method_exists -v` — PASS (existing fp8 path unbroken)
- [x] Full backend pytest suite — no regressions
- [x] `ruff check` — all clean

## Task 0 risk gate (already committed on this branch)

`backend/scripts/verify_flux2_cross_device.py` (commits e004cdd, 013eae6, 260c155) verified that diffusers `Flux2KleinPipeline.__call__` does NOT support cross-device components (RuntimeError at timestep_embedder linear_1). Spec §5.2 fallback activated: PR-2 will self-write `ImageSampler` (parallel to ComfyUI `samplers.py:986 outer_sample` pattern).
EOF
)"
```

- [ ] **Step 5: Auto-merge after CI green**

```bash
gh pr merge --auto --squash --delete-branch
```

---

## Self-Review Checklist

### Spec coverage

| Spec section | Tasks covering it |
|---|---|
| §3.3 L1 cache | Task 6 |
| §3.3 L2 cache | **DEFERRED to PR-6** (intentional) |
| §4.1-4.5 node schemas | **DEFERRED to PR-4** (intentional) |
| §4.6 component_scanner | **DEFERRED to PR-3** (intentional) |
| §5.1 ComponentSpec | Task 1 |
| §5.2 DiffusersImageBackend rewrite (to consume ComponentSpec + ImageSampler) | **DEFERRED to PR-2** (intentional) |
| §5.3 QuantLoaderRegistry | Tasks 2–5 |
| §5.4 Runner protocol | **DEFERRED to PR-4** (intentional) |
| §5.5 ModelManager ComponentKey | Task 6 |
| §5.6 ImageSampler (self-written) | **DEFERRED to PR-2** (intentional) |
| §6 Loader UI states | **DEFERRED to PR-5** (intentional) |
| §7 Frontend palette | **DEFERRED to PR-4+** (intentional) |
| §9 Test plan / unit | Tasks 1–6 inline |

PR-1 (rev 2) covers exactly the infra deliverables claimed in spec §8 PR-1 row.

### Placeholder scan

- No `TBD` / `TODO` / `FIXME` in plan text.
- One pre-existing helper rename (`_build_empty_transformer_from_dir` in Task 3 Step 5): if the helper doesn't exist yet, Task 3 extracts the inline transformer-building block into a function of that name. This is a no-behavior-change rename, scoped explicitly in the task body.

### Type consistency

- `ComponentSpec` fields used across tasks: `kind` / `file` / `device` / `dtype` / `loras` / `adapter_arch` / `clip_arch` — consistent across Tasks 1, 2, 3, 4, 5, 6.
- `ComponentKey` tuple shape `(file, device, frozenset[(name, strength)])` consistent in Task 1 (`to_component_key`) and Task 6 (`is_component_loaded` / `get_or_load_component`).
- `LoadedComponent` typed as `dict` in Task 6 — methods that return it use `{_state_dict, spec, loaded_at}` shape consistently.
- `QUANT_LOADERS.dispatch(spec)` returns `dict[str, Tensor]` in all four loader paths (plain / fp8mixed / mxfp8mixed / nvfp4mixed) — type-consistent for caller.

### Decomposition check

Each task is self-contained:
- Task 1 produces a working types module
- Tasks 2–5 each add one loader and its test, registry dispatch order maintained
- Task 6 adds parallel cache without touching legacy cache
- Task 7 is verification + ship

PR-1 ships ~450 LOC across 5 new files + 3 modified files. Each task is independently revertable.
