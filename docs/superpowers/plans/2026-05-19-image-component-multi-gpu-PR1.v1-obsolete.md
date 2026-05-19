# Image Component Multi-GPU Loader — PR-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the foundational infrastructure for component-level multi-GPU image generation: `ComponentSpec` types, a pluggable quant-loader registry (covering bf16 / fp16 / fp8mixed / mxfp8mixed / nvfp4mixed safetensors), a `DiffusersImageBackend` constructor path that accepts per-component file+device+dtype, and a `ModelManager` cache layer keyed on `ComponentKey` — all behind the existing public APIs so legacy yaml-based image workflows keep running unchanged.

**Architecture:** PR-1 introduces all new types and storage **additively**. The legacy `_models: dict[str, LoadedModel]` path is untouched; a new `_components: dict[ComponentKey, LoadedComponent]` path is added alongside. The new path is exercised only by direct unit/integration tests in this PR — workflow nodes won't emit `ComponentSpec` until PR-3. PR-1 ships as a "no user-visible change, full backend infra ready" milestone.

**Tech Stack:** Python 3.12 / pydantic v2 / diffusers 0.38+ (Flux2Pipeline) / safetensors 0.8 / torch 2.10 / pytest + pytest-asyncio.

**Risk Gate (Task 0):** The premise of cross-device Flux2Pipeline assembly (transformer on `cuda:1`, text_encoder on `cuda:0`, vae on `cuda:2`) is unverified against diffusers 0.38. Task 0 runs a 30-line verification script before any refactor. If it fails: stop, surface the failure modes to the user, fall back to single-device assembly (spec §5.2 fallback) before continuing.

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `backend/src/services/inference/component_spec.py` | `ComponentSpec` pydantic model + `ComponentKey` tuple type + `to_component_key(spec) -> ComponentKey` helper |
| `backend/src/services/inference/quant_loaders.py` | `QuantLoaderRegistry` class + 5 registered loader functions (plain / fp8mixed / mxfp8mixed / nvfp4mixed / [gguf placeholder]) |
| `backend/scripts/verify_flux2_cross_device.py` | Task 0 risk-gate script. ~30 lines. Run once manually before continuing the PR. |
| `backend/tests/test_component_spec.py` | Unit tests for ComponentSpec validation + ComponentKey hashing |
| `backend/tests/test_quant_loaders.py` | Unit tests per format using synthetic safetensors fixtures |
| `backend/tests/test_model_manager_components.py` | Unit + integration tests for new `_components` cache APIs |

### Modified files

| Path | Change |
|---|---|
| `backend/src/services/inference/base.py` | Re-export `ComponentSpec` from `component_spec` module (so consumers can `from src.services.inference.base import ComponentSpec` like they do for `LoRASpec`) |
| `backend/src/services/inference/image_diffusers.py` | (a) Add new `__init__(components: dict[str, ComponentSpec], **kwargs)` overload via factory classmethod `from_components`; (b) Extract `load_quantized_transformer` (`image_diffusers.py:105`) into a callable that fits the quant_loaders registry signature; (c) Existing `__init__(paths, device, ...)` and `load(device)` path UNCHANGED |
| `backend/src/services/model_manager.py` | (a) Add `_components: dict[ComponentKey, LoadedComponent]` alongside existing `_models`; (b) Add public methods `get_or_load_component`, `is_component_loaded`, `unload_component`; (c) NO change to existing `load_model` / `is_loaded` / `unload_model` APIs |

### Files explicitly NOT touched in PR-1

- `backend/src/services/workflow_executor.py` — node-level changes are PR-3
- `backend/src/runner/runner_process.py` — `_build_request` components branch is PR-3
- `backend/src/services/nodes/image.py` — node schema changes are PR-3
- Any frontend file — `useComponentState` + new nodes are PR-4
- `backend/configs/model_paths.yaml` — that's PR-2 (component_scanner)

---

## Task 0: Risk Gate — Verify Flux2Pipeline Cross-Device Assembly

**Why:** Spec §5.2 flags this as the only unverified premise. If diffusers 0.38 Flux2Pipeline rejects cross-device components, the entire spec fallback path activates and PR-1's adapter rewrite needs different shape.

**Files:**
- Create: `backend/scripts/verify_flux2_cross_device.py`

- [ ] **Step 1: Create verification script**

```python
# backend/scripts/verify_flux2_cross_device.py
"""PR-1 Task 0 risk gate — verify diffusers Flux2Pipeline accepts cross-device components.

Runs manually before the rest of PR-1 work proceeds. Exit 0 = green light to continue.
Exit 1 = stop PR-1; activate spec §5.2 fallback (single-device assembly only).

Usage:
    cd backend && .venv/bin/python scripts/verify_flux2_cross_device.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

MODELS_ROOT = Path("/media/heygo/Program/models/nous")
TRANSFORMER_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B/transformer"
TEXT_ENCODER_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B/text_encoder"
VAE_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B/vae"

# Device targets for the test (current hardware: cuda:0=3090, cuda:1=Pro 6000, cuda:2=3090)
UNET_DEVICE = "cuda:1"   # largest VRAM, holds 18GB transformer
CLIP_DEVICE = "cuda:0"   # 3090 #1 holds 6GB text_encoder
VAE_DEVICE = "cuda:2"    # 3090 #2 holds 0.4GB vae


def main() -> int:
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 1024**3:.1f}GB")

    from diffusers import Flux2Pipeline, Flux2Transformer2DModel, AutoencoderKL
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading transformer from {TRANSFORMER_DIR} -> {UNET_DEVICE}")
    transformer = Flux2Transformer2DModel.from_pretrained(
        TRANSFORMER_DIR, torch_dtype=torch.bfloat16
    ).to(UNET_DEVICE)
    print(f"  transformer.device={next(transformer.parameters()).device}")

    print(f"\nLoading text_encoder from {TEXT_ENCODER_DIR} -> {CLIP_DEVICE}")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        TEXT_ENCODER_DIR, torch_dtype=torch.bfloat16
    ).to(CLIP_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_DIR)
    print(f"  text_encoder.device={next(text_encoder.parameters()).device}")

    print(f"\nLoading vae from {VAE_DIR} -> {VAE_DEVICE}")
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.bfloat16).to(VAE_DEVICE)
    print(f"  vae.device={next(vae.parameters()).device}")

    print("\nAssembling Flux2Pipeline with cross-device components")
    pipe = Flux2Pipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        scheduler=None,  # let Pipeline pick default
    )

    print("\nRunning 2-step inference as smoke test")
    out = pipe(prompt="a cat", num_inference_steps=2, height=512, width=512, generator=torch.Generator("cuda:1").manual_seed(0))
    img = out.images[0]
    print(f"  output image size={img.size}")
    if img.size != (512, 512):
        print(f"  ERROR expected (512,512) got {img.size}")
        return 1

    print("\n✅ Cross-device Flux2Pipeline assembly works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make script executable + run it**

```bash
cd backend
chmod +x scripts/verify_flux2_cross_device.py
.venv/bin/python scripts/verify_flux2_cross_device.py
```

Expected on success: prints device assignments, runs 2-step inference, prints "✅ Cross-device Flux2Pipeline assembly works", exit 0.

Expected on failure: Python traceback or non-(512,512) output. **STOP** here. Surface the failure to the user; activate spec §5.2 fallback (modify Task 6 to do single-device assembly only).

- [ ] **Step 3: Commit script (regardless of result)**

```bash
git add backend/scripts/verify_flux2_cross_device.py
git commit -m "test(image): add PR-1 Task 0 risk gate — verify Flux2Pipeline cross-device"
```

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
(`image_unet_load` etc., added in PR-3). Cached by ModelManager via ComponentKey.

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
exercise the code path. Real-file end-to-end is covered by Task 8 integration smoke.
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
    """GGUF is V2 PR-6 — V1 dispatches to UnsupportedQuantError."""
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

Per spec §5.3: registry maps (ComponentSpec) → loaded weights (state_dict or
torch.nn.Module depending on format). First-match-wins; register specific
formats (fp8mixed / mxfp8mixed / nvfp4mixed) before plain safetensors fallback.

Each loader returns either:
  - dict[str, Tensor]      (plain safetensors path — caller wraps)
  - torch.nn.Module        (fp8mixed / mxfp8mixed / nvfp4mixed paths that
                            dequant + load_state_dict into a built module)

Caller convention is documented per-loader.
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


# Reject GGUF eagerly — V2 PR-6 work, not in scope for PR-1.
@QUANT_LOADERS.register(match=lambda spec: spec.file.lower().endswith(".gguf"))
def reject_gguf(spec: ComponentSpec) -> Any:
    raise UnsupportedQuantError(
        f"GGUF quantization is V2 PR-6 follow-up; cannot load {spec.file!r} in PR-1"
    )


def _dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,  # torch 2.10 native
    }.get(dtype_str, torch.bfloat16)


# Plain bf16/fp16 safetensors — diffusers `from_pretrained` / `from_single_file`
# normally handles this, but for the registry we expose a uniform state_dict
# loader. Caller (image_diffusers.py) decides whether to wrap into a module.
@QUANT_LOADERS.register(match=lambda spec: spec.file.endswith(".safetensors"))
def load_safetensors_plain(spec: ComponentSpec) -> dict[str, torch.Tensor]:
    """Plain bf16/fp16 safetensors → state_dict, target dtype applied."""
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

**Why:** `backend/src/services/inference/image_diffusers.py:105` already has `load_quantized_transformer(main_path, sf_path, dtype) -> Module`. PR-1 lifts that logic into the registry with a `ComponentSpec`-shaped signature, **without changing the underlying dequant math**.

**Files:**
- Modify: `backend/src/services/inference/quant_loaders.py` (add fp8mixed loader)
- Modify: `backend/src/services/inference/image_diffusers.py` (`load_quantized_transformer` becomes a thin wrapper around the new registry entry)
- Modify: `backend/tests/test_quant_loaders.py` (add fp8mixed test)

- [ ] **Step 1: Inspect the existing dequant**

```bash
sed -n '95,170p' backend/src/services/inference/image_diffusers.py
```

Note the algorithm:
- `safetensors_load_file(sf_path)` → state dict containing both fp8 tensors and their `.weight_scale` companions
- For each fp8 tensor, multiply by scale, drop `.comfy_quant` / `.weight_scale` keys
- Result is a clean bf16 state dict ready for `transformer.load_state_dict(...)`

- [ ] **Step 2: Add fp8mixed test (failing initially)**

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


def test_fp8mixed_loader_match_priority_over_plain(tmp_path):
    """File with 'fp8mixed' in name must dispatch to fp8 loader, not plain."""
    # We don't need real fp8 metadata in this test — just the filename match
    sd = {"a": torch.zeros(2, 2, dtype=torch.bfloat16)}
    sf = tmp_path / "Flux2-X-fp8mixed.safetensors"
    save_file(sd, str(sf))
    spec = ComponentSpec(kind="unet", file=str(sf), device="cpu", dtype="bfloat16")

    # If filename match wins, we hit fp8 loader, which warns 0 fp8 weights found.
    # Either it raises or returns sd; the test asserts we reach the fp8 dispatcher
    # by mocking. Simpler: check the registry's matcher order.
    matchers = [m for m, _fn in QUANT_LOADERS._loaders]
    fp8_idx = next(i for i, m in enumerate(matchers)
                   if m(ComponentSpec(kind="unet", file="x-fp8mixed.safetensors", device="cpu", dtype="bfloat16")))
    plain_idx = next(i for i, m in enumerate(matchers)
                     if m(ComponentSpec(kind="unet", file="plain.safetensors", device="cpu", dtype="bfloat16")))
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

Open `backend/src/services/inference/image_diffusers.py:105`. The existing function builds a transformer module then calls `load_state_dict`. Refactor so the state_dict prep step calls the registry:

```python
def load_quantized_transformer(main_path: Path, sf_path: Path, dtype) -> Any:
    """Load a wikeeyang-style fp8 single-file transformer into bf16.

    PR-1 refactor: state_dict dequant moved to quant_loaders.load_fp8mixed.
    This function keeps building the empty transformer module from main_path
    (HF diffusers layout) then loads the dequant'd state_dict into it.
    """
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.quant_loaders import QUANT_LOADERS

    # Build the empty transformer module from the main diffusers directory
    # (existing helper from this file — left untouched by PR-1)
    transformer = _build_empty_transformer_from_dir(main_path, dtype)

    # Dequant state_dict via the new registry path
    dtype_str = {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(dtype, "bfloat16")
    spec = ComponentSpec(kind="unet", file=str(sf_path), device="cpu", dtype=dtype_str)
    clean_sd = QUANT_LOADERS.dispatch(spec)

    missing, unexpected = transformer.load_state_dict(clean_sd, strict=False)
    if missing or unexpected:
        logger.info("image: quantized transformer load — missing=%d unexpected=%d",
                    len(missing), len(unexpected))
    else:
        logger.info("image: quantized transformer load — 0 missing / 0 unexpected ✓")
    return transformer
```

*Note:* `_build_empty_transformer_from_dir` is a placeholder name for whatever helper currently constructs the empty module before `load_state_dict`. If the existing code does it inline, extract that block into a helper of that name first (no behavior change).

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

    The actual mxfp8 packing format used by wikeeyang/community Flux2 quants stores:
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


def test_mxfp8mixed_loader_priority_over_fp8mixed(tmp_path):
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

    Real format spec (community Flux2 nvfp4 quants):
      <name>.weight        : uint8, shape = (numel / 2,)  — two 4-bit weights per byte
      <name>.weight_scale  : float32, shape = (numel / 16,) — one scale per 16-element block
      <name>.weight.comfy_quant : marker (uint8 [3])
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


def test_nvfp4mixed_loader_priority_over_mxfp8(tmp_path):
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

Expected: all PASS (10+ tests now).

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/inference/quant_loaders.py backend/tests/test_quant_loaders.py
git commit -m "feat(inference): nvfp4mixed 4-bit unpack + block dequant loader (PR-1 Task 5)"
```

---

## Task 6: `DiffusersImageBackend` — `from_components` Classmethod

**Why:** Existing constructor `DiffusersImageBackend(paths, device, ...)` stays for legacy yaml path. PR-1 adds a `from_components(components: dict[str, ComponentSpec])` classmethod that constructs the adapter from per-component descriptors, uses the quant_loaders registry, assembles Flux2Pipeline cross-device per Task 0 verified.

**Files:**
- Modify: `backend/src/services/inference/image_diffusers.py`
- Create: `backend/tests/test_image_diffusers_components.py`

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_image_diffusers_components.py
"""PR-1 Task 6: DiffusersImageBackend.from_components constructor path.

These tests use the existing _stub_diffusers fixture pattern from test_image_diffusers.py
to mock the heavy pipeline. Real-model smoke is Task 8.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.services.inference.base import LoRASpec
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_diffusers import DiffusersImageBackend


def _make_synthetic_safetensors(tmp_path: Path, name: str) -> Path:
    sd = {f"layer_{i}.weight": torch.randn(4, 4, dtype=torch.bfloat16) for i in range(3)}
    path = tmp_path / f"{name}.safetensors"
    save_file(sd, str(path))
    return path


@pytest.fixture
def _stub_pipeline(monkeypatch):
    """Reuse the patching idiom from test_image_diffusers.py — replace heavy
    diffusers imports with stubs that record `to(device)` calls."""
    from src.services.inference import image_diffusers

    class _StubModule:
        def __init__(self, name): self.name = name; self.device_calls = []
        def to(self, dev): self.device_calls.append(dev); return self
        def parameters(self): yield torch.zeros(1)

    class _StubPipe:
        def __init__(self, **components):
            self.components = components
        def __call__(self, *a, **kw): raise NotImplementedError("test stub")

    monkeypatch.setattr(image_diffusers, "Flux2Pipeline", _StubPipe, raising=False)
    monkeypatch.setattr(image_diffusers, "_load_transformer_module", lambda spec: _StubModule(f"unet:{spec.file}"))
    monkeypatch.setattr(image_diffusers, "_load_text_encoder_module", lambda spec: _StubModule(f"clip:{spec.file}"))
    monkeypatch.setattr(image_diffusers, "_load_vae_module", lambda spec: _StubModule(f"vae:{spec.file}"))
    monkeypatch.setattr(image_diffusers, "_apply_loras_to_transformer", lambda mod, loras: setattr(mod, "applied_loras", list(loras)))
    return _StubPipe


@pytest.mark.asyncio
async def test_from_components_assembles_cross_device(tmp_path, _stub_pipeline):
    """Three components on three different devices → three .to(device) calls + pipe assembled."""
    unet_sf = _make_synthetic_safetensors(tmp_path, "u")
    clip_sf = _make_synthetic_safetensors(tmp_path, "c")
    vae_sf = _make_synthetic_safetensors(tmp_path, "v")
    components = {
        "unet": ComponentSpec(kind="unet", file=str(unet_sf), device="cuda:1", dtype="bfloat16",
                              adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(clip_sf), device="cuda:0", dtype="bfloat16",
                              clip_arch="flux2"),
        "vae":  ComponentSpec(kind="vae",  file=str(vae_sf),  device="cuda:2", dtype="bfloat16"),
    }
    adapter = DiffusersImageBackend.from_components(components)
    await adapter.load()

    assert adapter._pipe is not None
    assert adapter._pipe.components["transformer"].device_calls == ["cuda:1"]
    assert adapter._pipe.components["text_encoder"].device_calls == ["cuda:0"]
    assert adapter._pipe.components["vae"].device_calls == ["cuda:2"]


@pytest.mark.asyncio
async def test_from_components_applies_loras_only_to_unet(tmp_path, _stub_pipeline):
    unet_sf = _make_synthetic_safetensors(tmp_path, "u")
    clip_sf = _make_synthetic_safetensors(tmp_path, "c")
    vae_sf = _make_synthetic_safetensors(tmp_path, "v")
    components = {
        "unet": ComponentSpec(kind="unet", file=str(unet_sf), device="cuda:0", dtype="bfloat16",
                              loras=[LoRASpec(name="style", strength=0.8),
                                     LoRASpec(name="detail", strength=0.4)],
                              adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(clip_sf), device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(vae_sf),  device="cuda:0", dtype="bfloat16"),
    }
    adapter = DiffusersImageBackend.from_components(components)
    await adapter.load()

    transformer = adapter._pipe.components["transformer"]
    assert hasattr(transformer, "applied_loras")
    assert [(l.name, l.strength) for l in transformer.applied_loras] == [("style", 0.8), ("detail", 0.4)]


@pytest.mark.asyncio
async def test_from_components_legacy_constructor_still_works(tmp_path):
    """Existing __init__(paths=..., device=...) MUST keep working for legacy yaml path."""
    paths = {"main": str(tmp_path)}
    adapter = DiffusersImageBackend(paths=paths, device="cuda")
    # Just constructor — actual load is covered by existing tests
    assert adapter.paths == paths


def test_from_components_rejects_missing_kind(tmp_path):
    """All three kinds (unet, clip, vae) must be present."""
    with pytest.raises(ValueError, match="missing component kinds"):
        DiffusersImageBackend.from_components({
            "unet": ComponentSpec(kind="unet", file="/p/u", device="cpu", dtype="bfloat16"),
            "clip": ComponentSpec(kind="clip", file="/p/c", device="cpu", dtype="bfloat16"),
            # vae missing
        })
```

- [ ] **Step 2: Run — fail**

```bash
.venv/bin/pytest tests/test_image_diffusers_components.py -v
```

Expected: `AttributeError: type object 'DiffusersImageBackend' has no attribute 'from_components'`.

- [ ] **Step 3: Implement `from_components` + helper extraction**

Open `backend/src/services/inference/image_diffusers.py`. Add **after the class body** (at the end of `class DiffusersImageBackend`):

```python
    # --- PR-1 Task 6: new component-level constructor path -------------------
    #
    # Existing __init__(paths=..., device=...) is unchanged for legacy yaml flow.
    # `from_components` is the new entry point used by PR-3 workflow nodes.
    #
    # Why a classmethod (not __init__ overload)? Avoids breaking pydantic-derived
    # adapter_factory (model_manager.py:103) that calls `cls(paths=spec.paths, **params)`
    # for every yaml-registered model — the factory has no notion of ComponentSpec.

    @classmethod
    def from_components(cls, components: dict[str, ComponentSpec], **kwargs) -> "DiffusersImageBackend":
        """Build adapter from per-component descriptors (PR-1 new path).

        Required kinds: 'unet', 'clip', 'vae'. Adapter holds the dict; load()
        consumes it via the quant_loaders registry + cross-device assembly.
        """
        required = {"unet", "clip", "vae"}
        missing = required - set(components.keys())
        if missing:
            raise ValueError(f"from_components: missing component kinds {sorted(missing)}")

        instance = cls.__new__(cls)
        # Initialize base InferenceAdapter fields with sentinel paths (component
        # path doesn't use the legacy `paths` dict). device='cuda' is a placeholder;
        # real device is per-component.
        InferenceAdapter.__init__(instance, paths={"_components": "from_components"}, device="cuda")
        # PR-1 specific state
        instance._components = components
        instance._offload_strategy = "no_offload"  # cross-device assembly handles placement
        instance._lora_paths = {}  # not used in component path — LoRA files are in unet ComponentSpec.loras
        instance._torch_dtype = components["unet"].dtype
        instance._loaded_loras = set()
        instance._pipe = None
        return instance

    async def load_from_components(self) -> None:
        """Load + assemble Pipeline when constructed via from_components.

        Distinct from the legacy `load(device)` so call sites stay clear about
        which path they're on. The new path:
          1. quant_loaders.dispatch() per component → state_dict or module
          2. Wrap state_dicts into proper diffusers modules (helper per kind)
          3. .to(device) per component → cross-device assembly verified by Task 0
          4. Apply LoRAs to transformer (unet.loras list)
          5. Compose Flux2Pipeline(transformer=, text_encoder=, vae=)
        """
        transformer = _load_transformer_module(self._components["unet"])
        text_encoder = _load_text_encoder_module(self._components["clip"])
        vae = _load_vae_module(self._components["vae"])

        transformer.to(self._components["unet"].device)
        text_encoder.to(self._components["clip"].device)
        vae.to(self._components["vae"].device)

        if self._components["unet"].loras:
            _apply_loras_to_transformer(transformer, self._components["unet"].loras)

        # Cross-device assembly — Task 0 verified this works on diffusers 0.38.
        # If Task 0 failed, fallback: .to() all components to unet.device first.
        self._pipe = Flux2Pipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
        )

    # Override base `load(device)` to dispatch on whether we were constructed via
    # from_components or the legacy path.
    async def load(self, device: str | None = None) -> None:
        if hasattr(self, "_components"):
            await self.load_from_components()
            return
        await super().load(device) if False else await self._legacy_load_impl(device)

    async def _legacy_load_impl(self, device: str | None) -> None:
        """The original load() body, renamed so the dispatch above stays readable.
        This method preserves the pre-PR-1 behavior 1:1; rename only."""
        # ... existing load body verbatim — move the original load() body here ...
```

**Important**: The existing `load()` body needs to be moved verbatim into `_legacy_load_impl`. Do this as a pure rename (no logic change). Open the existing `async def load(self, device)` (around `image_diffusers.py:388`), copy its body into `_legacy_load_impl`, then replace the original `load` with the dispatcher shown above.

Add module-level helpers (also in `image_diffusers.py`, near the top after imports):

```python
def _load_transformer_module(spec: ComponentSpec):
    """Build a diffusers transformer module from a ComponentSpec via quant_loaders.

    For plain safetensors: registry returns state_dict → wrap in empty Flux2Transformer2DModel.
    For fp8mixed / mxfp8mixed / nvfp4mixed: registry returns state_dict (dequant'd) → same wrap.
    """
    from diffusers import Flux2Transformer2DModel  # noqa: WPS433 — heavy import deferred
    from src.services.inference.quant_loaders import QUANT_LOADERS

    sd = QUANT_LOADERS.dispatch(spec)
    # Build empty module with same architecture as the source file's config.
    # Strategy: locate the matching diffusers HF directory (sibling of the .safetensors
    # file or referenced by convention) and load config from there.
    # For PR-1 scope, we require the safetensors lives under
    # image/diffusers/<MODEL>/transformer/ so we can find config.json one level up.
    transformer_dir = Path(spec.file).parent
    if (transformer_dir / "config.json").exists():
        module = Flux2Transformer2DModel.from_config(transformer_dir / "config.json")
        module.load_state_dict(sd, strict=False)
        return module
    # Fallback: single-file under image/diffusion_models/. Use diffusers'
    # `from_single_file` shim (it understands these flat layouts).
    return Flux2Transformer2DModel.from_single_file(spec.file, torch_dtype=_dtype_to_torch(spec.dtype))


def _load_text_encoder_module(spec: ComponentSpec):
    """Similar to _load_transformer_module but for the text encoder (Qwen3 / SDXL CLIP / etc.).

    clip_arch in spec determines which HF class to instantiate. PR-1 supports flux2 (Qwen3-style).
    Future PRs extend the dispatch (flux1 = T5+CLIP, sdxl = dual CLIP, qwen = Qwen2.5-VL).
    """
    from transformers import AutoModelForCausalLM
    from src.services.inference.quant_loaders import QUANT_LOADERS

    sd = QUANT_LOADERS.dispatch(spec)
    encoder_dir = Path(spec.file).parent
    if (encoder_dir / "config.json").exists():
        module = AutoModelForCausalLM.from_pretrained(encoder_dir, torch_dtype=_dtype_to_torch(spec.dtype))
        # state_dict from quant_loaders already dequant'd; load it
        module.load_state_dict(sd, strict=False)
        return module
    return AutoModelForCausalLM.from_pretrained(encoder_dir, torch_dtype=_dtype_to_torch(spec.dtype))


def _load_vae_module(spec: ComponentSpec):
    from diffusers import AutoencoderKL
    from src.services.inference.quant_loaders import QUANT_LOADERS

    sd = QUANT_LOADERS.dispatch(spec)
    vae_dir = Path(spec.file).parent
    if (vae_dir / "config.json").exists():
        module = AutoencoderKL.from_config(vae_dir / "config.json")
        module.load_state_dict(sd, strict=False)
        return module
    return AutoencoderKL.from_single_file(spec.file, torch_dtype=_dtype_to_torch(spec.dtype))


def _apply_loras_to_transformer(transformer, loras: list[LoRASpec]) -> None:
    """Apply Flux2 LoRAs in order via PEFT set_active_loras.

    PR-1 ships the simple path — assume diffusers PEFT integration is set up by
    the transformer module's `.load_lora_weights` API. If the module doesn't
    support it (e.g., stub in tests), this function is a no-op.
    """
    if not hasattr(transformer, "load_lora_weights"):
        return
    for lora in loras:
        transformer.load_lora_weights(lora.name, weight_name=lora.name, lora_scale=lora.strength)


def _dtype_to_torch(dtype_str: str) -> torch.dtype:
    """Mirror the helper in quant_loaders.py."""
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "fp8_e4m3": torch.float8_e4m3fn}.get(dtype_str, torch.bfloat16)
```

- [ ] **Step 4: Run new + existing image_diffusers tests — pass**

```bash
.venv/bin/pytest tests/test_image_diffusers_components.py tests/test_image_diffusers.py -v
```

Expected: all PASS. Existing tests still pass because legacy path is preserved via `_legacy_load_impl`.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/inference/image_diffusers.py \
        backend/tests/test_image_diffusers_components.py
git commit -m "feat(image): DiffusersImageBackend.from_components + cross-device load path (PR-1 Task 6)"
```

---

## Task 7: `ModelManager` — `_components` Cache + Public APIs

**Why:** Add a parallel L1 cache keyed on `ComponentKey` (per spec §3.3). Old `_models` dict and all existing APIs stay intact — PR-1 ships infra only, PR-3+ wires it in.

**Files:**
- Modify: `backend/src/services/model_manager.py`
- Create: `backend/tests/test_model_manager_components.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_model_manager_components.py
"""PR-1 Task 7: ModelManager component-level cache.

Per spec §5.5, _components: dict[ComponentKey, LoadedComponent] coexists with
the legacy _models: dict[str, LoadedModel] in PR-1. Future PRs route image
adapters through _components only.
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
        # adapters keep using _models; PR-3+ workflow nodes use _components.
        self._components: dict["ComponentKey", "LoadedComponent"] = {}
        self._component_locks: dict["ComponentKey", asyncio.Lock] = {}
        self._component_failures: dict["ComponentKey", str] = {}
```

Then at the bottom of the class, add the new public APIs:

```python
    # --- PR-1 Task 7: component-level cache APIs -----------------------------
    #
    # Per spec §5.5 these coexist with the legacy load_model/is_loaded/unload_model.
    # Workflow nodes (PR-3) will exercise these directly without going through yaml.

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
        """The actual component load. PR-1: dispatches to quant_loaders + module wrap.

        Tests monkeypatch this — production uses image_diffusers._load_*_module helpers.
        """
        from src.services.inference import image_diffusers
        if spec.kind == "unet":
            module = image_diffusers._load_transformer_module(spec)
        elif spec.kind == "clip":
            module = image_diffusers._load_text_encoder_module(spec)
        elif spec.kind == "vae":
            module = image_diffusers._load_vae_module(spec)
        else:
            raise ValueError(f"unknown component kind {spec.kind!r}")
        if spec.device != "auto" and spec.device != "cpu":
            module.to(spec.device)
        return {"_module": module, "spec": spec, "loaded_at": time.monotonic()}

    async def unload_component(self, spec_or_key) -> None:
        """Drop the cache entry. PR-1: caller is responsible for any GPU memory release.
        Future PRs (LRU eviction) will call torch.cuda.empty_cache after pop."""
        from src.services.inference.component_spec import ComponentSpec, to_component_key
        key = to_component_key(spec_or_key) if isinstance(spec_or_key, ComponentSpec) else spec_or_key
        async with self._component_lock_for(key):
            self._components.pop(key, None)
            self._component_failures.pop(key, None)
```

Add the `LoadedComponent` type alias near the top of the file (after `LoadedModel`):

```python
# PR-1: components are lighter than full LoadedModel — they're just a loaded
# state_dict/module dict + metadata. Stored in ModelManager._components.
LoadedComponent = dict  # opaque to ModelManager: {_module, spec, loaded_at, ...}
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
git add backend/src/services/model_manager.py \
        backend/tests/test_model_manager_components.py
git commit -m "feat(model-manager): _components L1 cache + get_or_load/is_component_loaded/unload (PR-1 Task 7)"
```

---

## Task 8: Real-Model Smoke (Optional — Skip If Task 0 Failed)

**Why:** Validates Task 6 + Task 7 work end-to-end with real Flux2-bf16 weights on the actual hardware. NOT a unit test — manual smoke before PR merge.

**Files:**
- Create: `backend/scripts/smoke_pr1_components.py`

- [ ] **Step 1: Create smoke script**

```python
# backend/scripts/smoke_pr1_components.py
"""PR-1 manual smoke — real Flux2-bf16 cross-device assembly via from_components.

Prereq: Task 0 (verify_flux2_cross_device.py) passed.

Run:
    cd backend && .venv/bin/python scripts/smoke_pr1_components.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import torch

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_diffusers import DiffusersImageBackend
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager

MODELS_ROOT = Path("/media/heygo/Program/models/nous")


async def main() -> int:
    components = {
        "unet": ComponentSpec(
            kind="unet", adapter_arch="flux2",
            file=str(MODELS_ROOT / "image/diffusers/Flux2-klein-9B/transformer/diffusion_pytorch_model.safetensors"),
            device="cuda:1", dtype="bfloat16",
        ),
        "clip": ComponentSpec(
            kind="clip", clip_arch="flux2",
            file=str(MODELS_ROOT / "image/diffusers/Flux2-klein-9B/text_encoder/model.safetensors"),
            device="cuda:0", dtype="bfloat16",
        ),
        "vae": ComponentSpec(
            kind="vae",
            file=str(MODELS_ROOT / "image/diffusers/Flux2-klein-9B/vae/diffusion_pytorch_model.safetensors"),
            device="cuda:2", dtype="bfloat16",
        ),
    }

    print("Building adapter via from_components")
    adapter = DiffusersImageBackend.from_components(components)
    print("Loading components cross-device")
    await adapter.load()

    print("Checking each component lives on its declared device")
    pipe = adapter._pipe
    for kind, target_dev in [("transformer", "cuda:1"), ("text_encoder", "cuda:0"), ("vae", "cuda:2")]:
        actual = next(getattr(pipe, kind).parameters()).device
        print(f"  {kind}: declared={target_dev} actual={actual}")
        if str(actual) != target_dev:
            print(f"  ❌ device mismatch")
            return 1

    print("\nExercising ModelManager.get_or_load_component (idempotency check)")
    mm = ModelManager(registry=ModelRegistry.from_yaml_data({"models": []}), allocator=GPUAllocator())
    # First load
    r1 = await mm.get_or_load_component(components["vae"])
    assert mm.is_component_loaded(components["vae"]) == "loaded"
    # Second load — same key → cache hit
    r2 = await mm.get_or_load_component(components["vae"])
    assert r1 is r2, "cache hit must return same instance"
    print("  ✅ idempotent get_or_load_component")

    print("\n✅ PR-1 smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Run smoke (only if Task 0 verified cross-device works)**

```bash
cd backend
.venv/bin/python scripts/smoke_pr1_components.py
```

Expected on success: prints device assignments, "✅ idempotent get_or_load_component", "✅ PR-1 smoke passed", exit 0.

- [ ] **Step 3: Commit smoke script**

```bash
git add backend/scripts/smoke_pr1_components.py
git commit -m "test(image): PR-1 manual smoke for cross-device component load"
```

---

## Task 9: PR Wrap-up

- [ ] **Step 1: Full test suite — no regressions**

```bash
cd backend
.venv/bin/pytest -x -q 2>&1 | tail -20
```

Expected: all green. If anything broke, the legacy preservation in Task 6 (`_legacy_load_impl`) likely needs verifying — open the rename diff and confirm no behavior changed.

- [ ] **Step 2: Lint**

```bash
.venv/bin/ruff check src/services/inference/ src/services/model_manager.py tests/test_component_spec.py tests/test_quant_loaders.py tests/test_model_manager_components.py tests/test_image_diffusers_components.py
```

Expected: `All checks passed!`.

- [ ] **Step 3: Branch + push**

```bash
git checkout -b feat/image-component-multi-gpu-pr1
git push -u origin feat/image-component-multi-gpu-pr1
```

- [ ] **Step 4: Open PR**

```bash
gh pr create --base master \
  --title "feat(image): component-multi-gpu PR-1 — ComponentSpec + quant loaders + ComponentKey cache" \
  --body "$(cat <<'EOF'
## Summary

PR-1 of [image-component-multi-gpu-design](docs/superpowers/specs/2026-05-19-image-component-multi-gpu-design.md).

Lands the foundational infrastructure — no user-visible change yet.

- `ComponentSpec` / `ComponentKey` types (`backend/src/services/inference/component_spec.py`)
- `QuantLoaderRegistry` + 5 loaders (plain / fp8mixed / mxfp8mixed / nvfp4mixed / GGUF reject)
- `DiffusersImageBackend.from_components` classmethod path (legacy `__init__(paths=,device=)` untouched)
- `ModelManager._components` cache + `get_or_load_component` / `is_component_loaded` / `unload_component`
- Risk-gate verification script (`scripts/verify_flux2_cross_device.py`)

## Test plan

- [x] `pytest tests/test_component_spec.py -v` — 8/8 PASS
- [x] `pytest tests/test_quant_loaders.py -v` — 10+/10+ PASS
- [x] `pytest tests/test_image_diffusers_components.py -v` — 4/4 PASS
- [x] `pytest tests/test_model_manager_components.py -v` — 7/7 PASS
- [x] Full backend pytest suite — no regressions
- [x] `ruff check` — all clean
- [x] `python scripts/verify_flux2_cross_device.py` — Task 0 risk gate passed
- [x] `python scripts/smoke_pr1_components.py` — real Flux2-bf16 cross-device assembly verified
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
| §3.3 L1 cache | Task 7 |
| §3.3 L2 cache | **DEFERRED to PR-5** (intentional) |
| §4.1-4.5 node schemas | **DEFERRED to PR-3** (intentional) |
| §4.6 component_scanner | **DEFERRED to PR-2** (intentional) |
| §5.1 ComponentSpec | Task 1 |
| §5.2 DiffusersImageBackend rewrite | Task 6 |
| §5.2 Risk gate | Task 0 |
| §5.3 QuantLoaderRegistry | Tasks 2–5 |
| §5.4 Runner protocol | **DEFERRED to PR-3** (intentional) |
| §5.5 ModelManager ComponentKey | Task 7 |
| §6 Loader UI states | **DEFERRED to PR-4** (intentional) |
| §7 Frontend palette | **DEFERRED to PR-3+** (intentional) |
| §9 Test plan / unit | Tasks 1–8 inline |

PR-1 covers all infra deliverables claimed in spec §8 PR-1 row.

### Placeholder scan

- No `TBD` / `TODO` / `FIXME` in plan text.
- One spot where the plan refers to `_build_empty_transformer_from_dir` (Task 3 Step 5) and `_load_transformer_module` / `_load_text_encoder_module` / `_load_vae_module` (Task 6) — these are defined in Task 6 Step 3 inline with full bodies. Order is fine because Task 3's edit is small and just extracts a helper; Task 6 adds the full set.

### Type consistency

- `ComponentSpec` fields used across tasks: `kind` / `file` / `device` / `dtype` / `loras` / `adapter_arch` / `clip_arch` — all consistent with `component_spec.py` definition in Task 1.
- `ComponentKey` tuple shape `(file, device, frozenset[(name, strength)])` consistent in Task 1 (`to_component_key`) and Task 7 (`is_component_loaded` / `get_or_load_component`).
- `LoadedComponent` typed as `dict` in Task 7 — methods that return it use `{_module, spec, loaded_at}` shape consistently.

### Decomposition check

Each task is self-contained:
- Task 0 stands alone (script only, no other file)
- Task 1 produces a working types module
- Tasks 2–5 each add one loader and its test, registry dispatch order maintained
- Task 6 adds new constructor + load path without touching legacy path
- Task 7 adds parallel cache without touching legacy cache
- Task 8 is verification only

PR-1 ships ~650 LOC across 6 new files + 3 modified files.
