"""ImageSampler — self-written denoise driver replacing Flux2KleinPipeline.__call__.

Why we need our own: diffusers Pipeline.__call__ hard-assumes same-device components
(verified by Task 0 risk gate, commit 551cd83). We reuse Pipeline HELPER METHODS
(`pipe.encode_prompt`, `pipe.prepare_latents`, `pipe.vae.decode`) — those run on
individual components and accept tensor inputs from anywhere. The denoise inner
loop (where the same-device assumption lives) is rewritten here with explicit
`.to(target.device)` at each cross-component boundary.

Reference: backend/.venv/lib/python3.12/site-packages/diffusers/pipelines/flux2/
  pipeline_flux2_klein.py:613-925 (Flux2KleinPipeline.__call__).

Notable Pipeline-source facts that drive the implementation:
  - encode_prompt(prompt, device, num_images_per_prompt, max_sequence_length,
    text_encoder_out_layers) — returns (prompt_embeds, text_ids).      (line 427)
  - prepare_latents(batch_size, num_latents_channels, height, width, dtype,
    device, generator, latents=None) — returns (packed_latents, latent_ids).
    Note kwarg is `num_latents_channels` (NOT num_channels_latents).   (line 478)
  - num_latents_channels = transformer.config.in_channels // 4.        (line 787)
  - timesteps via retrieve_timesteps(scheduler, num_steps, device, sigmas, mu).
    mu = compute_empirical_mu(image_seq_len, num_steps).               (line 815)
  - Inner loop: timestep = t.expand(B).to(latents.dtype); then
        with transformer.cache_context("cond"):
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids, img_ids=latent_ids,
                joint_attention_kwargs=None, return_dict=False,
            )[0]
        noise_pred = noise_pred[:, :latents.size(1):]   ← slice off any extra tokens
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                                                                       (line 831-877)
  - Post-loop: latents = _unpack_latents_with_ids(latents, latent_ids,
        latent_height//2, latent_width//2) → (B, C, H, W).              (line 905)
  - VAE inverse transform uses BatchNorm running stats — NOT scaling_factor /
    shift_factor:
        mean = vae.bn.running_mean.view(1,-1,1,1).to(latents.device, dtype)
        std  = sqrt(vae.bn.running_var.view(1,-1,1,1) + vae.config.batch_norm_eps)
        latents = latents * std + mean
        latents = _unpatchify_latents(latents)
        image = vae.decode(latents, return_dict=False)[0]               (line 907-916)

V1 supports FluxKleinArchAdapter only (matches on-disk Flux2-Klein-9B). Future
PRs add other adapters in ~50 LOC each (spec §5.6.3).
"""
from __future__ import annotations

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


# Lazy diffusers helper bindings — imported on first use so test collection
# doesn't pull in the whole diffusers package (which can conflict with the
# CUDA_VISIBLE_DEVICES="" sandbox in tests/conftest.py). Tests that drive the
# sampler with stubs monkeypatch `retrieve_timesteps` directly on this module,
# bypassing the real diffusers import altogether.
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:  # noqa: D401
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu as _impl,
    )
    return _impl(image_seq_len, num_steps)


def retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, **kw):  # noqa: D401
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        retrieve_timesteps as _impl,
    )
    return _impl(scheduler, num_inference_steps, device, sigmas=sigmas, **kw)


class SamplerCancelled(Exception):
    """Raised when CancelFlag is set during the denoise loop.
    Caller (DiffusersImageBackend.infer) maps this to NodeCancelled or a 499."""

    def __init__(self, reason: str = "cancelled"):
        super().__init__(reason)
        self.reason = reason


class SamplerError(Exception):
    """Raised when a sampling phase (encode/denoise/decode) fails.

    Wraps the underlying exception with phase context so the caller
    (DiffusersImageBackend.infer) can record where the failure happened
    in NodeResult.error without parsing tracebacks.
    """

    def __init__(self, phase: str, step: int | None, cause: Exception):
        self.phase = phase
        self.step = step  # set only during 'denoise' phase
        self.cause = cause
        super().__init__(f"{phase}{f' step {step}' if step is not None else ''} failed: {type(cause).__name__}: {cause}")


class ImageSampler:
    """Drives image generation across components on potentially different devices.

    Construction is cheap (stores references only). All work happens in `sample()`.
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
          2. prepare_latents on transformer.device
          3. denoise loop on transformer.device — driven by us
          4. latents.to(vae.device) → vae.decode → image
          5. encode PIL → PNG bytes → InferenceResult

        Cancel: checked at each denoise iteration; raises SamplerCancelled.
        Progress: callback fires after each denoise step (0-indexed).

        Phase failures wrap into SamplerError(phase=..., step=..., cause=...)
        — caller can record `error.phase` in NodeResult for post-mortem.

        ASYNC NOTE: this method is `async def` but the inner denoise loop runs
        synchronous CUDA kernels in the event-loop thread. Each step yields only
        at `await self.on_progress(...)`. For runner subprocess (Lane K),
        CancelFlag uses threading.Event so cross-thread cancel works regardless
        of event-loop responsiveness. If a future caller invokes this from a
        FastAPI request handler that needs the event loop responsive during
        sampling, wrap in `asyncio.to_thread(...)`.
        """
        t0 = time.monotonic()

        transformer_device = self.pipe.transformer.device
        text_encoder_device = self.pipe.text_encoder.device
        vae_device = self.pipe.vae.device

        # Seed handling. Generator stays on transformer.device because
        # prepare_latents calls randn_tensor(device=device) where device is the
        # transformer device.
        seed = req.seed if req.seed is not None else secrets.randbelow(2**63)
        # CUDA Generators only work on the device they were created on; for CPU
        # we use the default CPU generator. Match the transformer device.
        gen_device = transformer_device if transformer_device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # ----- Phase 1: encode_prompt ----------------------------------------
        # Pipeline source line 427-459. encode_prompt runs on text_encoder.device.
        # Returns (prompt_embeds, text_ids) — tensors on text_encoder.device.
        try:
            prompt_embeds, text_ids = self.pipe.encode_prompt(
                prompt=req.prompt,
                device=text_encoder_device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            # Cross-device transfer: embeds + ids → transformer.device.
            prompt_embeds = prompt_embeds.to(transformer_device)
            text_ids = text_ids.to(transformer_device)
        except SamplerCancelled:
            raise  # don't wrap cancels
        except Exception as e:
            raise SamplerError(phase="encode_prompt", step=None, cause=e) from e

        # ----- Phase 2: prepare_latents --------------------------------------
        # Pipeline source line 478-509 + caller line 787-797.
        # num_latents_channels = in_channels // 4 (Flux2 packs 2x2 patches → 4x ch).
        try:
            num_latents_channels = self.pipe.transformer.config.in_channels // 4
            latents, latent_ids = self.pipe.prepare_latents(
                batch_size=1,
                num_latents_channels=num_latents_channels,
                height=req.height,
                width=req.width,
                dtype=prompt_embeds.dtype,
                device=transformer_device,
                generator=generator,
            )
        except SamplerCancelled:
            raise
        except Exception as e:
            raise SamplerError(phase="prepare_latents", step=None, cause=e) from e

        # ----- Phase 3: timesteps via diffusers helper -----------------------
        # Pipeline source line 811-822. We reuse retrieve_timesteps + the
        # empirical-mu schedule so the Klein sigma schedule matches byte-for-byte.
        steps = req.steps or self.arch_adapter.default_steps()
        import numpy as np  # local import — only needed here
        sigmas = np.linspace(1.0, 1 / steps, steps)
        if hasattr(self.pipe.scheduler.config, "use_flow_sigmas") and self.pipe.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler,
            steps,
            transformer_device,
            sigmas=sigmas,
            mu=mu,
        )
        if hasattr(self.pipe.scheduler, "set_begin_index"):
            # Pre-set the scheduler's step counter to 0. Avoids a device→host sync
            # on the first scheduler.step() call (which would otherwise call
            # torch.searchsorted on the timesteps tensor — that triggers a CUDA →
            # CPU sync). Pipeline source pipeline_flux2_klein.py:829.
            self.pipe.scheduler.set_begin_index(0)

        # ----- Phase 4: denoise loop -----------------------------------------
        # Pipeline source line 831-877. Klein is distilled → no CFG branch.
        for step_idx, t in enumerate(timesteps):
            self._check_cancel()

            try:
                # Broadcast t to batch dim (ONNX/CoreML-compatible).
                timestep_input = t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(self.pipe.transformer.dtype)

                with self.pipe.transformer.cache_context("cond"):
                    noise_pred = self.pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_input / 1000,
                        guidance=None,                       # Klein is distilled
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                # Slice off any trailing image-conditioning tokens. Trailing colon
                # mirrors diffusers Pipeline source pipeline_flux2_klein.py:858 byte-for-byte
                # (preserves auditability); equivalent to `[:, : latents.size(1)]`.
                noise_pred = noise_pred[:, : latents.size(1) :]

                # scheduler.step → next x_t. Preserve dtype across the step.
                latents_dtype = latents.dtype
                latents = self.pipe.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
            except SamplerCancelled:
                raise
            except Exception as e:
                raise SamplerError(phase="denoise", step=step_idx, cause=e) from e

            if self.on_progress is not None:
                await self.on_progress(step_idx, num_inference_steps)

        # ----- Phase 5: unpack + VAE decode ----------------------------------
        # Pipeline source line 902-916. Pre-computed latent grid avoids
        # DtoH sync from torch.max().item() inside _unpack_latents_with_ids.
        try:
            vae_scale_factor = getattr(self.pipe, "vae_scale_factor", 8)
            latent_height = 2 * (int(req.height) // (vae_scale_factor * 2))
            latent_width = 2 * (int(req.width) // (vae_scale_factor * 2))
            latents = self.pipe._unpack_latents_with_ids(
                latents, latent_ids, latent_height // 2, latent_width // 2
            )

            # Cross-device transfer: latents → vae.device.
            latents = latents.to(vae_device, dtype=self.pipe.vae.dtype)

            # Flux2 VAE inverse transform uses BatchNorm running stats (NOT the
            # scaling_factor / shift_factor that Flux-Dev / SDXL use).
            bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(
                latents.device, latents.dtype
            )
            bn_std = torch.sqrt(
                self.pipe.vae.bn.running_var.view(1, -1, 1, 1)
                + self.pipe.vae.config.batch_norm_eps
            ).to(latents.device, latents.dtype)
            latents = latents * bn_std + bn_mean
            latents = self.pipe._unpatchify_latents(latents)

            with torch.no_grad():
                image_tensor = self.pipe.vae.decode(latents, return_dict=False)[0]
            # image_tensor: (1, 3, H, W) in [-1, 1].
            image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        except SamplerCancelled:
            raise
        except Exception as e:
            raise SamplerError(phase="vae_decode", step=None, cause=e) from e

        # ----- Phase 6: tensor → PIL → PNG bytes -----------------------------
        # Avoid torchvision dep — convert via numpy.
        from PIL import Image
        arr = (image_tensor[0].float().cpu().numpy() * 255.0).round().astype("uint8")
        # CHW → HWC
        arr = arr.transpose(1, 2, 0)
        pil = Image.fromarray(arr, mode="RGB")
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
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )

    def _check_cancel(self) -> None:
        if self.cancel_flag.is_set():
            reason = self.cancel_flag.reason or "cancelled"
            raise SamplerCancelled(reason)
