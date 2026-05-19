# Image Component Multi-GPU Loader — PR-2 Implementation Plan (rev 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the self-written `ImageSampler` (spec §5.6) that replaces `Flux2KleinPipeline.__call__` for cross-device image generation, plus `ModelArchAdapter` Protocol for future Pipeline-family extensibility, plus refactored `DiffusersImageBackend` that constructs ImageSampler in `load()` and routes `infer()` through it. **PR-2 unlocks the actual user-visible value**: image_generate runs with components on different GPUs (e.g. transformer on Pro 6000, vae on 3090).

**Architecture:** ImageSampler holds a constructed `Flux2KleinPipeline` instance (cross-device assembly DOES succeed per Task 0; only `__call__` fails). We **reuse Pipeline helper methods** (`pipe.encode_prompt`, `pipe.prepare_latents`, `pipe.vae.decode`) but **drive the denoise loop ourselves** with explicit `.to(target.device)` at each cross-component boundary — mirroring the structure of `pipeline_flux2_klein.py:831-877` but with manual tensor transfers. This avoids reimplementing Flux2-specific math (latent packing, position ids, Mistral3 chat template) while gaining cross-device control. `ModelArchAdapter` Protocol abstracts the Pipeline-class-specific bits (does it have CFG? does it have negative prompt? what's the transformer forward signature?) so future PRs can add `FluxDevArchAdapter` / `SDXLArchAdapter` in ~50 LOC each.

**Tech Stack:** Python 3.12 / pydantic v2 / diffusers 0.38+ (Flux2KleinPipeline + Flux2Transformer2DModel + AutoencoderKLFlux2) / safetensors 0.8 / torch 2.10 / pytest + pytest-asyncio.

**Spec reference:** `docs/superpowers/specs/2026-05-19-image-component-multi-gpu-design.md` rev 2 §5.2 + §5.6.

**Branch:** Work continues on `feat/image-component-multi-gpu-pr2` (already created off master post-PR-1 squash-merge).

**Out of scope for PR-2** (per spec §8):
- `component_scanner` + `/api/v1/components` endpoints → **PR-3**
- 4 new workflow loader nodes + `image_generate` rewrite → **PR-4**
- `useComponentState` hook + frontend palette → **PR-5**
- L2 image_generate output cache + `node_cache_hit` WS → **PR-6**
- Other Pipeline arch adapters (FluxDev / SDXL / Z-Image / Qwen-Image-Edit) → V2 PR

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `backend/src/services/inference/model_arch_adapter.py` | `ModelArchAdapter` Protocol + `MODEL_ARCH_REGISTRY` dict + `FluxKleinArchAdapter` implementation |
| `backend/src/services/inference/image_sampler.py` | `ImageSampler` class — holds Pipeline + drives cross-device denoise loop + cancel/progress |
| `backend/scripts/smoke_pr2_cross_device.py` | Real-model cross-device smoke (unet→cuda:1, clip→cuda:0, vae→cuda:2) |
| `backend/scripts/smoke_pr2_cancel.py` | Real-model cancel-mid-sampler smoke (cancel at step 10/25, verify NodeResult cancelled within 500ms) |
| `backend/tests/test_model_arch_adapter.py` | Adapter Protocol conformance + FluxKleinArchAdapter dispatch tests |
| `backend/tests/test_image_sampler.py` | Stub-based unit tests for ImageSampler orchestration (cancel timing, progress callback, error propagation) |
| `backend/tests/test_image_sampler_ssim.py` | Real-model SSIM regression (single-device) — pytest.mark.slow, gated |

### Modified files

| Path | Change |
|---|---|
| `backend/src/services/inference/image_diffusers.py` | (a) Add `DiffusersImageBackend.from_components(components: dict[str, ComponentSpec]) -> DiffusersImageBackend` classmethod; (b) Add `load_from_components()` method — constructs Pipeline (cross-device assembly), constructs ImageSampler, stores both; (c) `async def infer()` dispatches: if `_sampler` exists → call `_sampler.sample(req)`; else legacy `_pipe.__call__` path (preserved); (d) `_legacy_load_impl` rename of original `load()` body (Task 6 from PR-1 v1 plan, never applied; do it here now). |

### Files explicitly NOT touched in PR-2

- `backend/src/services/inference/component_spec.py` — PR-1 product, stable
- `backend/src/services/inference/quant_loaders.py` — PR-1 product, stable
- `backend/src/services/model_manager.py` — PR-1 added `_components` cache; PR-2 doesn't need to extend it
- Any workflow_executor / runner_process / node files — PR-3+ scope
- Any frontend file — PR-5+ scope

---

## Task 1: `ModelArchAdapter` Protocol + `FluxKleinArchAdapter`

**Why:** Spec §5.6.3. ImageSampler core logic is Pipeline-family agnostic; the per-arch differences (Klein has no CFG, Dev has CFG, SDXL has dual CLIP, etc.) factor out into adapters. PR-2 implements FluxKlein only; future PRs add more.

**Files:**
- Create: `backend/src/services/inference/model_arch_adapter.py`
- Create: `backend/tests/test_model_arch_adapter.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_model_arch_adapter.py
"""ModelArchAdapter Protocol conformance + FluxKlein dispatch."""
from __future__ import annotations

import pytest

from src.services.inference.model_arch_adapter import (
    ModelArchAdapter,
    MODEL_ARCH_REGISTRY,
    FluxKleinArchAdapter,
)


def test_flux_klein_adapter_in_registry():
    """FluxKlein adapter must be registered under the diffusers Pipeline class name."""
    assert "Flux2KleinPipeline" in MODEL_ARCH_REGISTRY
    adapter = MODEL_ARCH_REGISTRY["Flux2KleinPipeline"]
    assert isinstance(adapter, FluxKleinArchAdapter)


def test_flux_klein_adapter_supports_cfg_false():
    """Klein is distilled — no CFG branch."""
    adapter = FluxKleinArchAdapter()
    assert adapter.supports_cfg() is False


def test_flux_klein_adapter_supports_negative_prompt_false():
    """Klein doesn't accept negative_prompt (distilled inference is positive-prompt-only)."""
    adapter = FluxKleinArchAdapter()
    assert adapter.supports_negative_prompt() is False


def test_flux_klein_adapter_default_steps():
    """Klein default steps = 25 (distilled but supports 9-50 in practice)."""
    adapter = FluxKleinArchAdapter()
    assert adapter.default_steps() == 25


def test_flux_klein_adapter_default_guidance_scale():
    """Klein guidance is ignored at inference but registered for parameter pass-through."""
    adapter = FluxKleinArchAdapter()
    assert adapter.default_guidance_scale() == 4.0  # matches Pipeline default


def test_unknown_pipeline_class_not_in_registry():
    assert "StableDiffusionXLPipeline" not in MODEL_ARCH_REGISTRY
    assert "Flux2Pipeline" not in MODEL_ARCH_REGISTRY  # FluxDev — V2 PR


def test_protocol_can_type_check_adapter():
    """ModelArchAdapter is a Protocol; FluxKleinArchAdapter satisfies it structurally."""
    adapter: ModelArchAdapter = FluxKleinArchAdapter()  # type-check passes
    assert adapter is not None
```

- [ ] **Step 2: Run tests — fail**

```bash
cd backend
.venv/bin/pytest tests/test_model_arch_adapter.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement Protocol + FluxKleinArchAdapter**

```python
# backend/src/services/inference/model_arch_adapter.py
"""ModelArchAdapter — abstracts diffusers Pipeline family differences for ImageSampler.

PR-2 of image-component-multi-gpu spec §5.6.3. The ImageSampler's main loop is
Pipeline-family agnostic; per-arch differences (Klein has no CFG, Dev has CFG,
SDXL has dual CLIP) are isolated in adapter implementations registered by
Pipeline class name.

PR-2 ships only FluxKleinArchAdapter (matches the on-disk Flux2-Klein-9B model
verified by Task 0). Future PRs add FluxDev / SDXL / Z-Image / QwenImageEdit
adapters in ~50 LOC each.
"""
from __future__ import annotations

from typing import Protocol


class ModelArchAdapter(Protocol):
    """Per-Pipeline-class settings + behavior switches consumed by ImageSampler.

    All methods are pure (no side effects) — adapter instances are singletons
    in MODEL_ARCH_REGISTRY.
    """

    def supports_cfg(self) -> bool:
        """True if this Pipeline class uses classifier-free guidance (CFG)
        in its denoise loop. Distilled models (Klein, Z-Image-Turbo) → False.
        """
        ...

    def supports_negative_prompt(self) -> bool:
        """True if encode_prompt accepts a negative_prompt argument.
        Distilled models reject it; mainline (Dev, SDXL) accept it.
        """
        ...

    def default_steps(self) -> int:
        """Default num_inference_steps when caller didn't specify.
        Klein default = 25 (the Pipeline default at __call__ is 50, but distilled
        models converge in 9-25 — we pick 25 for safety).
        """
        ...

    def default_guidance_scale(self) -> float:
        """Default guidance_scale parameter for the Pipeline call.
        Distilled models ignore this but pipelines still expect the kwarg."""
        ...


class FluxKleinArchAdapter:
    """Flux2-Klein-9B (distilled). Matches diffusers Flux2KleinPipeline."""

    def supports_cfg(self) -> bool:
        return False

    def supports_negative_prompt(self) -> bool:
        return False

    def default_steps(self) -> int:
        return 25

    def default_guidance_scale(self) -> float:
        return 4.0  # matches Pipeline kwarg default; ignored at inference for distilled


# Registry — key is the diffusers Pipeline class name as returned by
# Pipeline.__class__.__name__ (or read from model_index.json _class_name).
# PR-2 only registers FluxKlein; future PRs add more entries.
MODEL_ARCH_REGISTRY: dict[str, ModelArchAdapter] = {
    "Flux2KleinPipeline": FluxKleinArchAdapter(),
}
```

- [ ] **Step 4: Run tests — pass**

```bash
.venv/bin/pytest tests/test_model_arch_adapter.py -v
```

Expected: 7/7 PASS.

- [ ] **Step 5: Ruff**

```bash
.venv/bin/ruff check src/services/inference/model_arch_adapter.py tests/test_model_arch_adapter.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/inference/model_arch_adapter.py backend/tests/test_model_arch_adapter.py
git commit -m "feat(inference): ModelArchAdapter Protocol + FluxKleinArchAdapter (PR-2 Task 1)"
```

---

## Task 2: `ImageSampler` Class — Cross-Device Denoise Loop

**Why:** Spec §5.6. Replace `Flux2KleinPipeline.__call__` with our own driver that uses Pipeline helper methods + explicit `.to(device)` at component boundaries.

**Strategy:** Hold the constructed Pipeline instance (cross-device assembly works per Task 0). Reuse `pipe.encode_prompt`, `pipe.prepare_latents`, `pipe.vae.decode` as helpers. Drive the **denoise inner loop ourselves** (mirrors `pipeline_flux2_klein.py:831-878`) with `.to(transformer.device)` before transformer forward and `.to(vae.device)` before vae decode.

**Files:**
- Create: `backend/src/services/inference/image_sampler.py`
- Create: `backend/tests/test_image_sampler.py`

**Reference**: `backend/.venv/lib/python3.12/site-packages/diffusers/pipelines/flux2/pipeline_flux2_klein.py`. Key methods/lines:
- `__call__` (line 613) — full inference orchestration the implementer mirrors
- `encode_prompt` (line 427) — produces `prompt_embeds`, `text_ids`
- `prepare_latents` (line 478) — produces initial random latent + `latent_ids`
- Denoise loop (lines 831-877) — `for i, t in enumerate(timesteps): noise_pred = self.transformer(...); latents = scheduler.step(noise_pred, t, latents).prev_sample`
- VAE decode (post-loop) — `image = self.vae.decode(latents / scaling_factor).sample`

- [ ] **Step 1: Write failing tests (stub-based, no real model)**

```python
# backend/tests/test_image_sampler.py
"""ImageSampler orchestration unit tests.

Stubs out Pipeline.encode_prompt / pipe.vae.decode / transformer.forward
with deterministic return values so we can test:
  - cancel flag stops the loop mid-iteration
  - progress callback fires per step
  - cross-device .to() is called at boundaries
  - errors propagate

Real-model SSIM correctness is gated by test_image_sampler_ssim.py (Task 4).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.services.inference.base import ImageRequest
from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.image_sampler import ImageSampler, SamplerCancelled
from src.services.inference.model_arch_adapter import FluxKleinArchAdapter


def _stub_pipeline(transformer_device: str, clip_device: str, vae_device: str):
    """Build a MagicMock Pipeline with components on declared devices and stubs
    for encode_prompt / vae.decode / transformer.forward."""
    pipe = MagicMock()
    # Components — each with a fake .device attribute via .parameters() generator
    pipe.transformer = MagicMock()
    pipe.transformer.parameters.return_value = iter([torch.zeros(1, device="cpu")])
    pipe.transformer.device = torch.device(transformer_device)
    pipe.text_encoder = MagicMock()
    pipe.text_encoder.parameters.return_value = iter([torch.zeros(1, device="cpu")])
    pipe.text_encoder.device = torch.device(clip_device)
    pipe.vae = MagicMock()
    pipe.vae.parameters.return_value = iter([torch.zeros(1, device="cpu")])
    pipe.vae.device = torch.device(vae_device)
    pipe.vae.config = MagicMock(scaling_factor=0.3611, shift_factor=0.1159)
    pipe.scheduler = MagicMock()
    return pipe


def _stub_request(steps=5, seed=42):
    return ImageRequest(
        request_id="test",
        prompt="cat",
        steps=steps,
        seed=seed,
        width=512,
        height=512,
    )


def test_sampler_constructs_with_pipeline_and_adapter():
    pipe = _stub_pipeline("cpu", "cpu", "cpu")
    adapter = FluxKleinArchAdapter()
    sampler = ImageSampler(pipe=pipe, arch_adapter=adapter)
    assert sampler.arch_adapter is adapter
    assert sampler.pipe is pipe


@pytest.mark.asyncio
async def test_sample_calls_encode_prompt_then_denoise_then_vae_decode(monkeypatch):
    """End-to-end orchestration: encode → loop transformer N steps → vae decode."""
    pipe = _stub_pipeline("cpu", "cpu", "cpu")
    adapter = FluxKleinArchAdapter()

    # encode_prompt returns (prompt_embeds, text_ids)
    pipe.encode_prompt = MagicMock(return_value=(
        torch.randn(1, 512, 4096),  # prompt_embeds
        torch.zeros(1, 512, 3),     # text_ids
    ))
    # prepare_latents returns (latents, latent_ids)
    pipe.prepare_latents = MagicMock(return_value=(
        torch.randn(1, 1024, 64),
        torch.zeros(1, 1024, 3),
    ))
    pipe.scheduler.timesteps = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])  # 5 steps
    pipe.transformer.return_value = MagicMock(sample=torch.randn(1, 1024, 64))
    pipe.scheduler.step = MagicMock(return_value=MagicMock(prev_sample=torch.randn(1, 1024, 64)))
    pipe.vae.decode = MagicMock(return_value=MagicMock(sample=torch.randn(1, 3, 512, 512)))
    # _pack_latents/_unpack_latents helpers used by Pipeline — stub as identity
    pipe._unpack_latents = MagicMock(side_effect=lambda x, *a, **kw: x)
    pipe._prepare_latent_image_ids = MagicMock(return_value=torch.zeros(1, 1024, 3))

    sampler = ImageSampler(pipe=pipe, arch_adapter=adapter)
    result = await sampler.sample(_stub_request(steps=5))

    pipe.encode_prompt.assert_called_once()
    assert pipe.transformer.call_count == 5  # 5 timesteps
    pipe.vae.decode.assert_called_once()
    assert result.media_type == "image/png"
    assert isinstance(result.data, bytes)


@pytest.mark.asyncio
async def test_sample_cancel_stops_loop_within_one_step():
    """Cancel flag set mid-denoise → next iteration check raises SamplerCancelled."""
    pipe = _stub_pipeline("cpu", "cpu", "cpu")
    pipe.encode_prompt = MagicMock(return_value=(torch.randn(1, 1, 1), torch.zeros(1, 1, 3)))
    pipe.prepare_latents = MagicMock(return_value=(torch.randn(1, 1, 64), torch.zeros(1, 1, 3)))
    pipe.scheduler.timesteps = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
    pipe._prepare_latent_image_ids = MagicMock(return_value=torch.zeros(1, 1, 3))

    cancel_flag = CancelFlag()

    call_count = {"n": 0}
    def _transformer_side_effect(**kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            # Set cancel after 2nd forward — should stop before 3rd
            cancel_flag.set("test_cancel")
        return MagicMock(sample=torch.randn(1, 1, 64))
    pipe.transformer.side_effect = _transformer_side_effect
    pipe.scheduler.step = MagicMock(return_value=MagicMock(prev_sample=torch.randn(1, 1, 64)))

    sampler = ImageSampler(pipe=pipe, arch_adapter=FluxKleinArchAdapter(), cancel_flag=cancel_flag)
    with pytest.raises(SamplerCancelled, match="test_cancel"):
        await sampler.sample(_stub_request(steps=5))
    assert call_count["n"] <= 3  # cancelled before doing all 5


@pytest.mark.asyncio
async def test_sample_progress_callback_fires_per_step():
    pipe = _stub_pipeline("cpu", "cpu", "cpu")
    pipe.encode_prompt = MagicMock(return_value=(torch.randn(1, 1, 1), torch.zeros(1, 1, 3)))
    pipe.prepare_latents = MagicMock(return_value=(torch.randn(1, 1, 64), torch.zeros(1, 1, 3)))
    pipe.scheduler.timesteps = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
    pipe.transformer.return_value = MagicMock(sample=torch.randn(1, 1, 64))
    pipe.scheduler.step = MagicMock(return_value=MagicMock(prev_sample=torch.randn(1, 1, 64)))
    pipe.vae.decode = MagicMock(return_value=MagicMock(sample=torch.randn(1, 3, 64, 64)))
    pipe._unpack_latents = MagicMock(side_effect=lambda x, *a, **kw: x)
    pipe._prepare_latent_image_ids = MagicMock(return_value=torch.zeros(1, 1, 3))

    progress_events = []
    async def _on_progress(step: int, total: int) -> None:
        progress_events.append((step, total))

    sampler = ImageSampler(pipe=pipe, arch_adapter=FluxKleinArchAdapter(),
                          on_progress=_on_progress)
    await sampler.sample(_stub_request(steps=5))

    # Progress fires once per denoise step (5 steps → 5 events)
    assert len(progress_events) == 5
    assert progress_events == [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]


@pytest.mark.asyncio
async def test_sample_cross_device_to_calls(monkeypatch):
    """Verify .to() is called at component boundaries: clip→transformer, transformer→vae."""
    pipe = _stub_pipeline("cuda:1", "cuda:0", "cuda:2")

    # Return tensors that record their .to() invocations
    embeds_to_calls = []
    class _RecordedTensor:
        def __init__(self, dev): self.device = torch.device(dev); self._dev = dev
        def to(self, dev):
            embeds_to_calls.append(str(dev))
            self._dev = str(dev)
            self.device = torch.device(dev)
            return self
        def size(self, *a): return 1  # for denoise loop checks
        @property
        def shape(self): return (1, 1, 1)

    pipe.encode_prompt = MagicMock(return_value=(_RecordedTensor("cuda:0"), _RecordedTensor("cuda:0")))
    pipe.prepare_latents = MagicMock(return_value=(_RecordedTensor("cuda:1"), _RecordedTensor("cuda:1")))
    pipe.scheduler.timesteps = torch.tensor([1.0])
    pipe.transformer.return_value = MagicMock(sample=_RecordedTensor("cuda:1"))
    pipe.scheduler.step = MagicMock(return_value=MagicMock(prev_sample=_RecordedTensor("cuda:1")))
    pipe.vae.decode = MagicMock(return_value=MagicMock(sample=torch.randn(1, 3, 64, 64)))
    pipe._unpack_latents = MagicMock(side_effect=lambda x, *a, **kw: x)
    pipe._prepare_latent_image_ids = MagicMock(return_value=_RecordedTensor("cuda:0"))

    sampler = ImageSampler(pipe=pipe, arch_adapter=FluxKleinArchAdapter())
    await sampler.sample(_stub_request(steps=1))

    # At least 2 cross-device transfers expected:
    # - embeds: cuda:0 → cuda:1 (before denoise)
    # - latent: cuda:1 → cuda:2 (before vae decode)
    assert "cuda:1" in embeds_to_calls or "cuda:2" in embeds_to_calls
```

- [ ] **Step 2: Run tests — fail**

```bash
.venv/bin/pytest tests/test_image_sampler.py -v
```

Expected: `ImportError: cannot import name 'ImageSampler'`.

- [ ] **Step 3: Implement `ImageSampler`**

```python
# backend/src/services/inference/image_sampler.py
"""ImageSampler — self-written denoise driver replacing Flux2KleinPipeline.__call__.

Why we need our own: diffusers Pipeline.__call__ hard-assumes same-device components
(verified by Task 0 risk gate, commit 551cd83). We reuse Pipeline HELPER METHODS
(`pipe.encode_prompt`, `pipe.prepare_latents`, `pipe.vae.decode`) — those run
on individual components and accept tensor inputs from anywhere. The denoise
inner loop (which is where same-device assumption lives) is rewritten here with
explicit `.to(target.device)` at each cross-component boundary.

Reference: pipeline_flux2_klein.py:613-892 (__call__ in diffusers).
Reference: ComfyUI samplers.py:986-1010 (outer_sample — same pattern, ours is
            scoped down to single-prompt, no CFG, distilled-only for PR-2).

Spec §5.6 — V1 supports FluxKleinArchAdapter only (matches on-disk Flux2-Klein-9B).
Future PRs extend to FluxDev / SDXL / Z-Image / QwenImageEdit by registering more
adapters in MODEL_ARCH_REGISTRY + adjusting denoise loop branches.
"""
from __future__ import annotations

import asyncio
import io
import logging
import secrets
import time
from typing import Any, Awaitable, Callable

import torch

from src.services.inference.base import ImageRequest, InferenceResult, UsageMeter
from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.model_arch_adapter import ModelArchAdapter

logger = logging.getLogger(__name__)


class SamplerCancelled(Exception):
    """Raised when CancelFlag is set during the denoise loop.
    Caller (DiffusersImageBackend.infer) maps this to NodeCancelled or a 499."""

    def __init__(self, reason: str = "cancelled"):
        super().__init__(reason)
        self.reason = reason


class ImageSampler:
    """Drives image generation across components on potentially different devices.

    Construction is cheap (just stores references). All heavy work happens in `sample()`.
    """

    def __init__(
        self,
        pipe: Any,                             # diffusers Flux2KleinPipeline (or compat)
        arch_adapter: ModelArchAdapter,
        cancel_flag: CancelFlag | None = None,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ):
        self.pipe = pipe
        self.arch_adapter = arch_adapter
        self.cancel_flag = cancel_flag or CancelFlag()
        self.on_progress = on_progress

    async def sample(self, req: ImageRequest) -> InferenceResult:
        """End-to-end orchestration:
          1. encode_prompt on text_encoder.device → embeds.to(transformer.device)
          2. prepare_latents on transformer.device (already there per Pipeline)
          3. denoise loop on transformer.device — driven by us
          4. latents.to(vae.device) → vae.decode → image
          5. encode PIL → PNG bytes → InferenceResult

        Cancel: checked at each denoise iteration; raises SamplerCancelled.
        Progress: callback fires after each denoise step (0-indexed).
        """
        t0 = time.monotonic()

        # Seed handling (matches DiffusersImageBackend.infer convention)
        seed = req.seed if req.seed is not None else secrets.randbelow(2**63)
        generator = torch.Generator(device=self.pipe.transformer.device).manual_seed(seed)

        # ----- Phase 1: encode_prompt -----------------------------------------
        # Pipeline's encode_prompt runs on self.text_encoder. It returns
        # (prompt_embeds, text_ids) — tensors on text_encoder.device.
        prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=req.prompt,
            num_images_per_prompt=1,
            max_sequence_length=512,
            device=self.pipe.text_encoder.device,
        )
        # Cross-device transfer: embeds → transformer.device
        prompt_embeds = prompt_embeds.to(self.pipe.transformer.device)
        text_ids = text_ids.to(self.pipe.transformer.device)

        # ----- Phase 2: prepare_latents ---------------------------------------
        # Pipeline's prepare_latents instantiates the random initial latent on
        # transformer.device (it reads self._execution_device which the
        # Pipeline thinks is transformer.device, since transformer is the
        # canonical "main" component).
        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        latents, latent_ids = self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=req.height,
            width=req.width,
            dtype=prompt_embeds.dtype,
            device=self.pipe.transformer.device,
            generator=generator,
        )

        # ----- Phase 3: setup scheduler ---------------------------------------
        # Klein uses a fixed sigma sequence determined by the scheduler + step count.
        # We set num_inference_steps directly; the Pipeline does this in __call__
        # via retrieve_timesteps but for distilled we can just call set_timesteps.
        steps = req.steps or self.arch_adapter.default_steps()
        self.pipe.scheduler.set_timesteps(steps, device=self.pipe.transformer.device)
        timesteps = self.pipe.scheduler.timesteps

        # ----- Phase 4: denoise loop (the part Pipeline.__call__ crashes on) --
        # Mirrors pipeline_flux2_klein.py:831-877 but each transformer call is
        # explicitly on transformer.device and cross-device transfers happen
        # outside the loop.
        for step_idx, t in enumerate(timesteps):
            self._check_cancel()

            # Pipeline broadcasts t to batch shape — replicate that
            timestep_input = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.pipe.transformer(
                hidden_states=latents,
                timestep=timestep_input / 1000,  # Pipeline divides by 1000 (line 850 in source)
                encoder_hidden_states=prompt_embeds,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=None,  # Klein is distilled, guidance ignored
                return_dict=False,
            )[0]
            # Distilled has no CFG branch — noise_pred used directly.

            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Async progress callback after each step
            if self.on_progress is not None:
                await self.on_progress(step_idx, steps)

        # ----- Phase 5: vae decode --------------------------------------------
        # Unpack latents (Pipeline packs them at prepare_latents for transformer);
        # Pipeline exposes _unpack_latents as a method.
        latents = self.pipe._unpack_latents(latents, req.height, req.width, vae_scale_factor=8)

        # Cross-device transfer: latent → vae.device
        latents = latents.to(self.pipe.vae.device, dtype=self.pipe.vae.dtype)
        # Apply VAE scaling (Pipeline does this inline at the decode call)
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        with torch.no_grad():
            image_tensor = self.pipe.vae.decode(latents, return_dict=False)[0]
        # image_tensor shape: (1, 3, H, W) in [-1, 1]
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)

        # ----- Phase 6: tensor → PIL → PNG bytes -----------------------------
        from torchvision.transforms.functional import to_pil_image
        pil = to_pil_image(image_tensor[0].cpu())
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "ImageSampler done: steps=%d size=%dx%d seed=%d latency=%dms",
            steps, req.width, req.height, seed, latency_ms,
        )

        return InferenceResult(
            media_type="image/png",
            data=png_bytes,
            metadata={
                "steps": steps,
                "width": req.width,
                "height": req.height,
                "seed": seed,
                "duration_ms": latency_ms,
            },
            usage=UsageMeter(),
        )

    def _check_cancel(self) -> None:
        if self.cancel_flag.is_set():
            reason = self.cancel_flag.reason or "cancelled"
            raise SamplerCancelled(reason)
```

**Important: the implementer should READ `pipeline_flux2_klein.py:613-892` to verify the exact call shapes** for `encode_prompt`, `prepare_latents`, `transformer.__call__`, and `vae.decode`. The plan above shows the SHAPE but specific kwargs may differ slightly. Acceptance criteria for Task 2 is the stub tests pass + Task 4 SSIM test passes (Task 4 will catch any algorithmic divergence).

- [ ] **Step 4: Run tests — pass**

```bash
.venv/bin/pytest tests/test_image_sampler.py -v
```

Expected: 5/5 PASS (4 main tests + 1 cross-device .to() test).

- [ ] **Step 5: Regression + ruff**

```bash
.venv/bin/pytest tests/test_image_sampler.py tests/test_model_arch_adapter.py tests/test_component_spec.py tests/test_quant_loaders.py 2>&1 | tail -5
.venv/bin/ruff check src/services/inference/image_sampler.py tests/test_image_sampler.py
```

Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/inference/image_sampler.py backend/tests/test_image_sampler.py
git commit -m "feat(inference): ImageSampler — cross-device denoise loop (PR-2 Task 2)"
```

---

## Task 3: `DiffusersImageBackend.from_components` + Routing

**Why:** Adapter integration. `from_components` is the new factory used by PR-4's loader nodes. `load_from_components` constructs Pipeline + ImageSampler. `infer()` routes through the sampler when present; legacy Pipeline.__call__ path preserved for old yaml model_key flows.

**Files:**
- Modify: `backend/src/services/inference/image_diffusers.py`
- Modify: `backend/tests/test_image_diffusers.py` (no new test file — extend existing)

- [ ] **Step 1: Add failing tests for the new path**

Append to `backend/tests/test_image_diffusers.py`:

```python
def test_from_components_classmethod_exists():
    """from_components is the PR-4-facing factory for component-level construction."""
    from src.services.inference.image_diffusers import DiffusersImageBackend
    assert hasattr(DiffusersImageBackend, "from_components")
    assert callable(DiffusersImageBackend.from_components)


def test_from_components_requires_all_three_kinds():
    """Must have unet + clip + vae; missing any → ValueError."""
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.image_diffusers import DiffusersImageBackend
    with pytest.raises(ValueError, match="missing component kinds"):
        DiffusersImageBackend.from_components({
            "unet": ComponentSpec(kind="unet", file="/p/u", device="cpu", dtype="bfloat16"),
            "clip": ComponentSpec(kind="clip", file="/p/c", device="cpu", dtype="bfloat16"),
            # vae missing
        })


def test_from_components_stores_components_dict():
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.image_diffusers import DiffusersImageBackend
    components = {
        "unet": ComponentSpec(kind="unet", file="/p/u", device="cpu", dtype="bfloat16"),
        "clip": ComponentSpec(kind="clip", file="/p/c", device="cpu", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/p/v", device="cpu", dtype="bfloat16"),
    }
    adapter = DiffusersImageBackend.from_components(components, pipeline_class="Flux2KleinPipeline")
    assert adapter._components == components
    assert adapter._pipeline_class == "Flux2KleinPipeline"


def test_legacy_init_path_still_works():
    """Existing __init__(paths=..., device=...) must keep working for yaml flow."""
    from src.services.inference.image_diffusers import DiffusersImageBackend
    adapter = DiffusersImageBackend(paths={"main": "/some/path"}, device="cuda")
    assert adapter.paths == {"main": "/some/path"}
    # Crucially: no _components attribute on legacy path
    assert not hasattr(adapter, "_components") or adapter._components is None
```

- [ ] **Step 2: Run — fail**

```bash
.venv/bin/pytest tests/test_image_diffusers.py::test_from_components_classmethod_exists -v
```

Expected: AttributeError or test fails.

- [ ] **Step 3: Implement `from_components` + `load_from_components` + infer routing**

Open `backend/src/services/inference/image_diffusers.py`. Add **inside the `DiffusersImageBackend` class** (at the end, after existing methods):

```python
    # ===== PR-2: component-level construction + ImageSampler routing =====
    #
    # from_components is a classmethod factory used by PR-4 workflow nodes.
    # It bypasses the yaml model_key path entirely. The legacy
    # __init__(paths=..., device=...) constructor stays untouched for yaml flow.

    @classmethod
    def from_components(
        cls,
        components: dict[str, "ComponentSpec"],
        pipeline_class: str = "Flux2KleinPipeline",
        **kwargs,
    ) -> "DiffusersImageBackend":
        """Build adapter from per-component descriptors.

        Required kinds: 'unet', 'clip', 'vae'. Adapter holds the dict; load()
        builds the cross-device Pipeline + ImageSampler.
        """
        required = {"unet", "clip", "vae"}
        missing = required - set(components.keys())
        if missing:
            raise ValueError(f"from_components: missing component kinds {sorted(missing)}")

        from src.services.inference.base import InferenceAdapter

        instance = cls.__new__(cls)
        # Bypass legacy __init__ — InferenceAdapter.__init__ wants `paths` dict but
        # the component path doesn't use it. Sentinel value keeps the base class happy.
        InferenceAdapter.__init__(instance, paths={"_from_components": "true"}, device="cuda")
        instance._components = components
        instance._pipeline_class = pipeline_class
        instance._offload_strategy = "no_offload"  # cross-device — each component on its declared GPU
        instance._lora_paths = {}  # not used in component path (loras live in unet ComponentSpec.loras)
        instance._torch_dtype = components["unet"].dtype
        instance._loaded_loras = set()
        instance._pipe = None
        instance._sampler = None  # ImageSampler — built in load_from_components
        return instance

    async def load_from_components(self) -> None:
        """Construct Pipeline (cross-device assembly) + ImageSampler.

        Called by ModelManager.get_or_load when the adapter was built via
        from_components. Pipeline assembly is verified to work cross-device
        by Task 0 risk gate; only Pipeline.__call__ would crash, which we
        sidestep by routing infer() through ImageSampler.
        """
        from src.services.inference.image_sampler import ImageSampler
        from src.services.inference.model_arch_adapter import MODEL_ARCH_REGISTRY
        from src.services.inference.quant_loaders import QUANT_LOADERS

        # Load each component's weights via the quant registry (PR-1 product).
        # Each loader returns state_dict; the diffusers model class wraps it.
        # For PR-2 minimum, we instantiate the components inside this method using
        # the standard diffusers class constructors + load_state_dict.
        from diffusers import (
            AutoencoderKLFlux2,
            Flux2KleinPipeline,
            Flux2Transformer2DModel,
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pathlib import Path

        unet_spec = self._components["unet"]
        clip_spec = self._components["clip"]
        vae_spec = self._components["vae"]

        # Resolve diffusers config sibling directories (e.g. transformer/config.json)
        # so from_config + load_state_dict can rebuild the empty module then patch.
        # In PR-2 we use from_pretrained on the parent dir for simplicity — that
        # reads the safetensors directly. For quantized formats the registry
        # dispatch returns a dequant state_dict and we call load_state_dict.
        def _load_module(spec, hf_class):
            parent_dir = Path(spec.file).parent
            try:
                # Try the "diffusers HF layout" path first — works for plain bf16
                # safetensors in image/diffusers/<MODEL>/<component>/
                module = hf_class.from_pretrained(parent_dir, torch_dtype=_torch_dtype_from(spec.dtype))
            except Exception:
                # Fall back to quant_loaders path — dequant'd state_dict + load
                sd = QUANT_LOADERS.dispatch(spec)
                module = hf_class.from_config(parent_dir / "config.json")
                module.load_state_dict(sd, strict=False)
            return module

        transformer = _load_module(unet_spec, Flux2Transformer2DModel).to(unet_spec.device)
        text_encoder = _load_module(clip_spec, AutoModelForCausalLM).to(clip_spec.device)
        vae = _load_module(vae_spec, AutoencoderKLFlux2).to(vae_spec.device)

        # Tokenizer lives in <model>/tokenizer/ (peer of text_encoder/). We
        # walk up one level from clip.file then over to tokenizer/.
        tokenizer_dir = Path(clip_spec.file).parent.parent / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        # Scheduler — load from <model>/scheduler/
        scheduler_dir = Path(unet_spec.file).parent.parent / "scheduler"
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_dir)

        # Apply LoRAs on the transformer (unet spec only)
        if unet_spec.loras:
            self._apply_loras(unet_spec.loras)  # existing helper, see image_diffusers.py

        # Construct Pipeline — Task 0 verified this assembly works cross-device.
        # We don't call self._pipe(...) anywhere; ImageSampler uses its helpers.
        self._pipe = Flux2KleinPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )

        # Construct sampler — held alongside _pipe, used by infer()
        adapter = MODEL_ARCH_REGISTRY.get(self._pipeline_class)
        if adapter is None:
            raise RuntimeError(
                f"No ModelArchAdapter registered for {self._pipeline_class!r}. "
                f"Known: {sorted(MODEL_ARCH_REGISTRY)}"
            )
        from src.services.inference.image_sampler import ImageSampler
        self._sampler = ImageSampler(pipe=self._pipe, arch_adapter=adapter)

    # Override the base load() — dispatch on which constructor was used
    async def load(self, device: str | None = None) -> None:
        if hasattr(self, "_components") and self._components is not None:
            await self.load_from_components()
            return
        # Legacy path: rename the original load body to _legacy_load_impl below
        await self._legacy_load_impl(device)

    # The original load() body is moved here verbatim (no logic change) so
    # the dispatcher above stays readable. To do the rename:
    #   1. Cut the original `async def load(self, device: str | None = None)`
    #      body (everything between def line and the next method).
    #   2. Paste it into a new `async def _legacy_load_impl(self, device)`.
    #   3. Add the dispatcher above as the new `load`.

    # Override infer() — route to ImageSampler when present
    async def infer(
        self, req: InferenceRequest, cancel_flag: CancelFlag | None = None
    ) -> InferenceResult:
        if self._sampler is not None:
            # Component path: wire cancel_flag into the sampler before sample()
            if cancel_flag is not None:
                self._sampler.cancel_flag = cancel_flag
            return await self._sampler.sample(req)
        # Legacy path — existing infer body unchanged
        return await self._legacy_infer_impl(req, cancel_flag)

    # Same rename pattern as load: move existing infer body to _legacy_infer_impl.


def _torch_dtype_from(dtype_str: str):
    """Local helper — same mapping as quant_loaders._dtype_str_to_torch."""
    import torch
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,
    }.get(dtype_str, torch.bfloat16)
```

**Important notes for the implementer:**

1. The `_legacy_load_impl` and `_legacy_infer_impl` renames are **pure code motion** — cut the existing `async def load(self, device)` body and paste verbatim into the new method, then write the new `load` dispatcher above. Same for `infer`.

2. `_load_module` is intentionally lenient — it tries `from_pretrained` (works for plain bf16) and falls back to `from_config + load_state_dict` (works for quantized). PR-4 will tighten this.

3. Tokenizer + scheduler paths assume the diffusers HF layout (sibling subdirs of `transformer/`). PR-3's `component_scanner` will canonicalize this; for PR-2 we trust the assumption.

4. The `_apply_loras` helper already exists in `image_diffusers.py` — verify by grep before calling.

- [ ] **Step 4: Run tests — pass**

```bash
.venv/bin/pytest tests/test_image_diffusers.py -v 2>&1 | tail -20
```

Expected: 37 existing + 4 new tests PASS (41 total). If any existing test regressed, the `_legacy_load_impl` rename broke something — open the diff carefully.

- [ ] **Step 5: Ruff**

```bash
.venv/bin/ruff check src/services/inference/image_diffusers.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/inference/image_diffusers.py backend/tests/test_image_diffusers.py
git commit -m "feat(image): DiffusersImageBackend.from_components + ImageSampler routing (PR-2 Task 3)"
```

---

## Task 4: SSIM Regression Test (Real Model, Single-Device)

**Why:** Core acceptance criterion of spec §8 PR-2: "ImageSampler.sample vs Flux2KleinPipeline.__call__ single-device output SSIM > 0.99 on same prompt + seed". This catches algorithmic divergence — wrong scheduler config, wrong CFG handling, wrong VAE scaling — before they ship.

**Files:**
- Create: `backend/tests/test_image_sampler_ssim.py`

This test is **slow** (~60s on Pro 6000 + cuda:1, two Flux2 inferences). Marked `@pytest.mark.slow` so it's opt-in via `pytest -m slow`.

- [ ] **Step 1: Write the SSIM test**

```python
# backend/tests/test_image_sampler_ssim.py
"""SSIM regression — ImageSampler must produce nearly identical output to
diffusers Flux2KleinPipeline.__call__ when both run on the same single device.

Why this matters: if our self-written denoise loop diverges from Pipeline's,
the spec §8 PR-2 success criterion fails. SSIM > 0.99 confirms math parity.

Cost: ~60s on Pro 6000 (two 25-step Flux2 inferences back-to-back).
Run with: pytest tests/test_image_sampler_ssim.py -m slow -v
Skipped by default in normal test runs.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

MODELS_ROOT = Path("/media/heygo/Program/models/nous")
FLUX2_KLEIN_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B"

skip_reason = "real Flux2-Klein-9B model required + GPU; gated by NOUS_SSIM_TEST=1"
pytestmark = pytest.mark.skipif(
    os.environ.get("NOUS_SSIM_TEST") != "1" or not FLUX2_KLEIN_DIR.exists(),
    reason=skip_reason,
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_image_sampler_matches_pipeline_ssim_single_device():
    """Same prompt + seed + steps + single device → SSIM > 0.99 between
    our ImageSampler.sample and the stock Pipeline.__call__ output."""
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    from src.services.inference.base import ImageRequest, LoRASpec
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.image_diffusers import DiffusersImageBackend

    PROMPT = "a small grey kitten sitting on a wooden table, soft natural lighting"
    SEED = 4242
    STEPS = 25
    DEVICE = "cuda:1"  # Pro 6000 — assumes Task 0 / PR #111 confirmed cuda:1 = Pro 6000 under PCI_BUS_ID

    # ===== Baseline: stock Pipeline.__call__ single-device =====
    baseline_pipe = Flux2KleinPipeline.from_pretrained(
        FLUX2_KLEIN_DIR, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    baseline_out = baseline_pipe(
        prompt=PROMPT,
        num_inference_steps=STEPS,
        height=512, width=512,
        generator=gen,
    )
    baseline_img: Image.Image = baseline_out.images[0]
    # Free baseline pipeline to make room for the sampler's pipeline
    del baseline_pipe
    torch.cuda.empty_cache()

    # ===== Our: ImageSampler via DiffusersImageBackend.from_components, same DEVICE =====
    components = {
        "unet": ComponentSpec(
            kind="unet", adapter_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "transformer/diffusion_pytorch_model.safetensors"),
            device=DEVICE, dtype="bfloat16",
        ),
        "clip": ComponentSpec(
            kind="clip", clip_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "text_encoder/model.safetensors"),
            device=DEVICE, dtype="bfloat16",
        ),
        "vae": ComponentSpec(
            kind="vae",
            file=str(FLUX2_KLEIN_DIR / "vae/diffusion_pytorch_model.safetensors"),
            device=DEVICE, dtype="bfloat16",
        ),
    }
    adapter = DiffusersImageBackend.from_components(components, pipeline_class="Flux2KleinPipeline")
    await adapter.load()
    req = ImageRequest(
        request_id="ssim-test",
        prompt=PROMPT,
        seed=SEED,
        steps=STEPS,
        width=512, height=512,
    )
    sampler_result = await adapter.infer(req)
    sampler_img = Image.open(__import__("io").BytesIO(sampler_result.data))

    # ===== Compare =====
    # Both images should be same size
    assert baseline_img.size == sampler_img.size == (512, 512)

    # Compute SSIM with skimage
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    a = np.array(baseline_img.convert("RGB"))
    b = np.array(sampler_img.convert("RGB"))
    score = ssim(a, b, channel_axis=2, data_range=255)
    print(f"SSIM = {score:.4f}")

    # Spec §8 PR-2 acceptance criterion
    assert score > 0.99, f"ImageSampler output diverges from Pipeline: SSIM={score:.4f}"
```

- [ ] **Step 2: Add `slow` marker to pyproject.toml**

Check `backend/pyproject.toml` for an `[tool.pytest.ini_options]` block with `markers`. If not present, add:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

If the section exists, just add the `slow` marker to its list.

- [ ] **Step 3: Verify the test skips by default**

```bash
cd backend
.venv/bin/pytest tests/test_image_sampler_ssim.py -v
```

Expected: 1 skipped (because `NOUS_SSIM_TEST` is unset).

- [ ] **Step 4: Run the test for real (manual, requires GPU + model)**

```bash
NOUS_SSIM_TEST=1 .venv/bin/pytest tests/test_image_sampler_ssim.py -v -s -m slow
```

Expected: PASS with `SSIM = 0.99XX` printed (X depends on numerical noise). If FAIL (`AssertionError: ImageSampler output diverges`), the ImageSampler algorithm has a bug — likely in scheduler setup, VAE scaling, or transformer kwargs. **STOP** and report back; do NOT lower the threshold.

If `skimage` is not installed in the venv: `.venv/bin/uv pip install scikit-image` first.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_image_sampler_ssim.py backend/pyproject.toml
git commit -m "test(image): SSIM regression — ImageSampler vs Pipeline single-device (PR-2 Task 4)"
```

---

## Task 5: Real-Model Smoke — Cross-Device + Cancel

**Why:** Verify the actual cross-device claim of PR-2 (transformer on cuda:1, vae on cuda:2, etc.) produces an image without crash, AND that cancel-mid-sampler completes within 500ms (spec success criterion).

**Files:**
- Create: `backend/scripts/smoke_pr2_cross_device.py`
- Create: `backend/scripts/smoke_pr2_cancel.py`

- [ ] **Step 1: Write cross-device smoke script**

```python
# backend/scripts/smoke_pr2_cross_device.py
"""PR-2 manual smoke — cross-device Flux2-Klein-9B via ImageSampler.

Assumes hardware: cuda:0=3090, cuda:1=Pro 6000 96GB, cuda:2=3090 (PR #111 PCI_BUS_ID).

Run:
    cd backend && NOUS_DISABLE_RUNNER_SPAWN=1 .venv/bin/python scripts/smoke_pr2_cross_device.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from src.services.inference.base import ImageRequest
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_diffusers import DiffusersImageBackend

MODELS_ROOT = Path("/media/heygo/Program/models/nous")
FLUX2_KLEIN_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B"


async def main() -> int:
    print(f"torch={torch.__version__} cuda={torch.version.cuda} count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 1024**3:.1f}GB")

    components = {
        "unet": ComponentSpec(
            kind="unet", adapter_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "transformer/diffusion_pytorch_model.safetensors"),
            device="cuda:1", dtype="bfloat16",  # Pro 6000 (largest VRAM)
        ),
        "clip": ComponentSpec(
            kind="clip", clip_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "text_encoder/model.safetensors"),
            device="cuda:0", dtype="bfloat16",  # 3090 #1
        ),
        "vae": ComponentSpec(
            kind="vae",
            file=str(FLUX2_KLEIN_DIR / "vae/diffusion_pytorch_model.safetensors"),
            device="cuda:2", dtype="bfloat16",  # 3090 #2
        ),
    }

    print("Building adapter via from_components")
    adapter = DiffusersImageBackend.from_components(components, pipeline_class="Flux2KleinPipeline")
    print("Loading cross-device")
    await adapter.load()
    print(f"  transformer.device={next(adapter._pipe.transformer.parameters()).device}")
    print(f"  text_encoder.device={next(adapter._pipe.text_encoder.parameters()).device}")
    print(f"  vae.device={next(adapter._pipe.vae.parameters()).device}")

    print("Running 25-step inference at 512x512")
    t0 = time.monotonic()
    req = ImageRequest(
        request_id="smoke",
        prompt="a colorful hot air balloon over a green valley at sunset",
        seed=12345,
        steps=25,
        width=512, height=512,
    )
    result = await adapter.infer(req)
    elapsed = time.monotonic() - t0
    print(f"Inference: {elapsed:.1f}s, output size={len(result.data) // 1024}KB")

    # Save the image
    out_path = Path("/tmp/pr2_cross_device.png")
    out_path.write_bytes(result.data)
    print(f"Image saved to {out_path}")

    print("\n✅ Cross-device PR-2 smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Write cancel smoke script**

```python
# backend/scripts/smoke_pr2_cancel.py
"""PR-2 manual smoke — cancel ImageSampler mid-denoise, verify ≤ 500ms.

Schedules an async cancel at the start, fires it from a sibling task at
~step 10 of 25, asserts SamplerCancelled raised within 500ms of the set call.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import pytest

from src.services.inference.base import ImageRequest
from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_diffusers import DiffusersImageBackend
from src.services.inference.image_sampler import SamplerCancelled

FLUX2_KLEIN_DIR = Path("/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B")


async def main() -> int:
    components = {
        "unet": ComponentSpec(kind="unet", adapter_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "transformer/diffusion_pytorch_model.safetensors"),
            device="cuda:1", dtype="bfloat16"),
        "clip": ComponentSpec(kind="clip", clip_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "text_encoder/model.safetensors"),
            device="cuda:1", dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae",
            file=str(FLUX2_KLEIN_DIR / "vae/diffusion_pytorch_model.safetensors"),
            device="cuda:1", dtype="bfloat16"),
    }

    adapter = DiffusersImageBackend.from_components(components, pipeline_class="Flux2KleinPipeline")
    await adapter.load()

    cancel_flag = CancelFlag()
    cancel_time = {"set_at": None}

    async def _cancel_after_seconds(secs: float):
        await asyncio.sleep(secs)
        cancel_time["set_at"] = time.monotonic()
        cancel_flag.set("smoke_test_cancel_at_10s")

    req = ImageRequest(request_id="smoke-cancel", prompt="cancel test",
                       seed=1, steps=25, width=512, height=512)

    # Step ≈ 1.5s on Pro 6000 → step 10 ≈ 15s. Schedule cancel at 15s.
    cancel_task = asyncio.create_task(_cancel_after_seconds(15.0))

    t0 = time.monotonic()
    try:
        await adapter.infer(req, cancel_flag=cancel_flag)
        print("❌ FAILED: infer returned without cancellation")
        return 1
    except SamplerCancelled as e:
        cancelled_at = time.monotonic()
        latency = (cancelled_at - cancel_time["set_at"]) * 1000
        print(f"SamplerCancelled raised {latency:.0f}ms after cancel.set()")
        print(f"Total elapsed: {cancelled_at - t0:.1f}s")
        assert latency <= 500, f"cancel latency {latency:.0f}ms exceeds 500ms target"
        print("✅ Cancel-mid-sampler smoke passed (≤500ms)")
        return 0
    finally:
        cancel_task.cancel()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 3: Run cross-device smoke**

```bash
cd backend
NOUS_DISABLE_RUNNER_SPAWN=1 .venv/bin/python scripts/smoke_pr2_cross_device.py
```

Expected: prints device assignments, runs inference in ~40-60s, saves PNG to `/tmp/pr2_cross_device.png`, prints "✅ Cross-device PR-2 smoke passed".

If it fails with cross-device tensor error: ImageSampler has a missing `.to()` boundary — fix and re-run.

- [ ] **Step 4: Run cancel smoke**

```bash
NOUS_DISABLE_RUNNER_SPAWN=1 .venv/bin/python scripts/smoke_pr2_cancel.py
```

Expected: prints cancel latency, exits 0 if ≤ 500ms. If the latency exceeds 500ms, ImageSampler's `_check_cancel` is being called too rarely (should be per-step).

- [ ] **Step 5: Commit the scripts (regardless of result — they're test artifacts)**

```bash
git add backend/scripts/smoke_pr2_cross_device.py backend/scripts/smoke_pr2_cancel.py
git commit -m "test(image): PR-2 manual smokes — cross-device + cancel mid-sampler"
```

---

## Task 6: PR Wrap-up

- [ ] **Step 1: Full test suite — no regressions**

```bash
cd backend
.venv/bin/pytest -x -q 2>&1 | tail -10
```

Expected: all green except the SSIM test which is skipped (NOUS_SSIM_TEST unset).

- [ ] **Step 2: Ruff full**

```bash
.venv/bin/ruff check src/ tests/ 2>&1 | tail -5
```

Expected: `All checks passed!`.

- [ ] **Step 3: Push branch**

```bash
git push -u origin feat/image-component-multi-gpu-pr2
```

- [ ] **Step 4: Open PR**

```bash
gh pr create --base master \
  --title "feat(image): component-multi-gpu PR-2 — self-written ImageSampler + cross-device denoise loop" \
  --body "$(cat <<'EOF'
## Summary

PR-2 of [image-component-multi-gpu-design](docs/superpowers/specs/2026-05-19-image-component-multi-gpu-design.md). The big one — unlocks the user-visible "components on different GPUs" capability.

- `ModelArchAdapter` Protocol + `FluxKleinArchAdapter` first implementation (spec §5.6.3)
- `ImageSampler` class — replaces `Flux2KleinPipeline.__call__` with cross-device denoise loop (spec §5.6)
- `DiffusersImageBackend.from_components` classmethod + `load_from_components` builds Pipeline (cross-device assembly verified by Task 0 in PR-1) + constructs ImageSampler
- `DiffusersImageBackend.infer()` routes through ImageSampler when constructed via `from_components`; legacy yaml path preserved
- SSIM regression test (gated, opt-in via `NOUS_SSIM_TEST=1`) verifies algorithmic parity with Pipeline single-device
- Cross-device + cancel smoke scripts confirm real-model behavior

## What's NOT in this PR (per spec §8)

- `component_scanner` + `/api/v1/components` endpoints → PR-3
- 4 new workflow loader nodes + `image_generate` rewrite → PR-4
- `useComponentState` hook + frontend palette → PR-5
- L2 image_generate output cache → PR-6
- Other Pipeline arch adapters (Dev / SDXL / Z-Image / QwenImageEdit) → V2

## Test plan

- [x] `pytest tests/test_model_arch_adapter.py -v` — 7/7 PASS
- [x] `pytest tests/test_image_sampler.py -v` — 5/5 PASS (stub-based unit)
- [x] `pytest tests/test_image_diffusers.py -v` — 41/41 PASS (37 legacy + 4 from_components)
- [x] Full backend suite `pytest -q` — no regressions
- [x] `ruff check src/ tests/` — clean
- [x] `python scripts/smoke_pr2_cross_device.py` — Flux2-Klein cross-device (unet→cuda:1, clip→cuda:0, vae→cuda:2) runs ≤ 60s, saves image
- [x] `python scripts/smoke_pr2_cancel.py` — cancel at step 10/25 raises SamplerCancelled within 500ms
- [x] `NOUS_SSIM_TEST=1 pytest tests/test_image_sampler_ssim.py -m slow -v` — SSIM > 0.99 vs Pipeline baseline
EOF
)"
```

- [ ] **Step 5: Merge (no auto-merge — verify CI green first)**

```bash
# Wait for CI to settle, then:
gh pr merge --squash --delete-branch
```

---

## Self-Review Checklist

### Spec coverage

| Spec section | Tasks covering it |
|---|---|
| §5.6.1 ImageSampler interface | Task 2 |
| §5.6.2 Cross-device .to() hooks | Task 2 (denoise loop) |
| §5.6.3 ModelArchAdapter | Task 1 |
| §5.6.4 Cancel + progress | Task 2 (`_check_cancel`, `on_progress`) |
| §5.6.5 Non-goals | Plan header (mentioned explicitly) |
| §5.6.6 Reference implementations | Task 2 mentions pipeline_flux2_klein.py:613-892 |
| §5.2 DiffusersImageBackend rewrite | Task 3 |
| §8 PR-2 SSIM regression test | Task 4 |
| §8 PR-2 cancel-mid-sampler 500ms | Task 5 (smoke_pr2_cancel.py) |
| §8 PR-2 cross-device smoke | Task 5 (smoke_pr2_cross_device.py) |

All spec §5.6 deliverables covered.

### Placeholder scan

- No `TBD` / `TODO` / `FIXME` in plan text.
- Two intentional "verify by reading source" instructions in Task 2 + Task 3 — these are not placeholders; they're "READ THE REFERENCE FILE before pasting" directives. Code blocks contain real Python with concrete kwargs.

### Type consistency

- `ImageRequest`, `InferenceResult`, `CancelFlag`, `ComponentSpec` — all defined in existing modules (`base.py`, `cancel_flag.py`, `component_spec.py`); the plan imports them by their actual paths.
- `SamplerCancelled` introduced in Task 2 and consumed in Task 5 — name consistent.
- `MODEL_ARCH_REGISTRY` introduced in Task 1, consumed in Task 3 — name consistent.
- `from_components(components, pipeline_class="Flux2KleinPipeline")` signature consistent across Tasks 3, 4, 5.

### Decomposition check

Each task is self-contained:
- Task 1 lands the adapter Protocol — no other dependencies.
- Task 2 lands ImageSampler — depends on Task 1's adapter.
- Task 3 wires it into DiffusersImageBackend — depends on Tasks 1+2.
- Task 4 is the SSIM gate — depends on Tasks 1+2+3 (full e2e path).
- Task 5 is real-model smoke — same dependencies as Task 4.
- Task 6 is wrap-up.

PR-2 ships ~800 LOC across 5 new files + 1 modified file. Each task is independently revertable. The plan is 6 tasks, smaller than PR-1's 7, but Task 2 is heavier (~250 LOC).
