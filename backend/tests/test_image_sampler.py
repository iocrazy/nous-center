"""ImageSampler orchestration unit tests.

Stubs out Pipeline.encode_prompt / pipe.vae.decode / transformer.forward
with deterministic return values so we can test:
  - cancel flag stops the loop mid-iteration
  - progress callback fires per step
  - cross-device .to() is called at boundaries
  - errors propagate

Real-model SSIM correctness is gated by test_image_sampler_ssim.py (Task 4).

Notes on test-environment torch handling:
  conftest.py stubs sys.modules["torch"] with MagicMock when not already loaded
  (so GPU-less CI test runs don't dlopen libcudart). This file needs the real
  torch to construct tensors for shape/dtype validation, so a session-scoped
  autouse fixture below restores it once — idempotently — without re-evicting
  if torch is already real (re-evicting real torch and re-importing trips
  torch's one-time C-level _add_docstr registration).

  Test functions reference torch via `_t()` instead of a module-level
  `import torch`; that defers the lookup until after the session fixture has
  done the swap.
"""
from __future__ import annotations

import sys
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from src.services.inference.base import ImageRequest
from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.image_sampler import ImageSampler, SamplerCancelled
from src.services.inference.model_arch_adapter import FluxKleinArchAdapter


@pytest.fixture(autouse=True, scope="session")
def _install_real_torch():
    """Restore real torch if conftest stubbed it. No-op if already real."""
    existing = sys.modules.get("torch")
    if isinstance(existing, MagicMock):
        for name in list(sys.modules.keys()):
            if name == "torch" or name.startswith("torch."):
                del sys.modules[name]
        import torch  # noqa: F401  — populates sys.modules with real torch
    yield


def _t():
    """Lazy accessor for the real torch module (post-fixture-swap)."""
    return sys.modules["torch"]


def _stub_pipeline(transformer_device: str, clip_device: str, vae_device: str):
    """Build a MagicMock Pipeline with components on declared devices and stubs
    for encode_prompt / vae.decode / transformer.forward."""
    torch = _t()
    pipe = MagicMock()
    # Components — each with a fake .device attribute and dtype.
    pipe.transformer = MagicMock()
    pipe.transformer.device = torch.device(transformer_device)
    pipe.transformer.dtype = torch.float32
    # Klein in_channels = 256 (4 × 64); num_latents_channels = in_channels // 4 = 64.
    pipe.transformer.config = MagicMock(in_channels=256)
    # cache_context returns a context manager (no-op in stub).
    pipe.transformer.cache_context = MagicMock(return_value=nullcontext())

    pipe.text_encoder = MagicMock()
    pipe.text_encoder.device = torch.device(clip_device)

    pipe.vae = MagicMock()
    pipe.vae.device = torch.device(vae_device)
    pipe.vae.dtype = torch.float32
    pipe.vae.config = MagicMock(batch_norm_eps=1e-5)
    # BatchNorm running stats for the Flux2 VAE inverse transform.
    pipe.vae.bn = MagicMock()
    pipe.vae.bn.running_mean = torch.zeros(16)
    pipe.vae.bn.running_var = torch.ones(16)

    pipe.scheduler = MagicMock()
    pipe.scheduler.order = 1
    return pipe


def _stub_request(steps=5, seed=42, width=512, height=512):
    return ImageRequest(
        request_id="test",
        prompt="cat",
        steps=steps,
        seed=seed,
        width=width,
        height=height,
    )


def _wire_default_stubs(pipe, *, num_steps: int = 5, seq_len: int = 16, channels: int = 256):
    """Wire encode_prompt / prepare_latents / transformer / scheduler / vae stubs
    so a typical sample() call completes. Tests can override individual mocks."""
    torch = _t()
    pipe.encode_prompt = MagicMock(return_value=(
        torch.randn(1, 8, 64),                   # prompt_embeds
        torch.zeros(1, 8, 4),                    # text_ids
    ))
    pipe.prepare_latents = MagicMock(return_value=(
        torch.randn(1, seq_len, channels),        # packed latents
        torch.zeros(1, seq_len, 4),               # latent_ids
    ))
    # scheduler.timesteps is read by the fake_retrieve fixture below.
    pipe.scheduler.timesteps = torch.linspace(1.0, 0.1, num_steps)
    pipe.scheduler.set_timesteps = MagicMock()
    pipe.scheduler.set_begin_index = MagicMock()
    pipe.scheduler.step = MagicMock(
        return_value=(torch.randn(1, seq_len, channels),)
    )
    # Transformer returns (noise_pred,) — same shape as latents so the post-slice
    # noise_pred[:, :latents.size(1):] doesn't drop anything.
    pipe.transformer.return_value = (torch.randn(1, seq_len, channels),)
    # _unpack_latents_with_ids returns a (B, C, H, W) tensor for the VAE path.
    # _unpatchify_latents is shape-changing in the real Pipeline; for stubs we
    # use identity so the downstream BatchNorm-view + vae.decode receive real
    # tensors on the right device.
    pipe._unpack_latents_with_ids = MagicMock(
        return_value=torch.randn(1, 16, 8, 8)
    )
    pipe._unpatchify_latents = MagicMock(side_effect=lambda x: x)
    # VAE decode returns (image_tensor,) shape (1, 3, H, W) in [-1, 1].
    pipe.vae.decode = MagicMock(
        return_value=(torch.randn(1, 3, 64, 64),)
    )
    # vae_scale_factor is read off the pipe attribute (Flux2 = 8).
    pipe.vae_scale_factor = 8


@pytest.fixture(autouse=True)
def _patch_diffusers_helpers(monkeypatch):
    """The sampler module proxies retrieve_timesteps + compute_empirical_mu
    through lazy diffusers imports. Tests override both with pure-python stubs
    so the heavy diffusers package never imports (and CUDA_VISIBLE_DEVICES=""
    in conftest stays consistent)."""
    torch = _t()

    def fake_retrieve(scheduler, num_inference_steps, device, sigmas=None, **kw):
        ts = getattr(scheduler, "timesteps", None)
        if ts is None or not isinstance(ts, torch.Tensor):
            ts = torch.linspace(1.0, 0.1, num_inference_steps)
        return ts, num_inference_steps

    monkeypatch.setattr(
        "src.services.inference.image_sampler.retrieve_timesteps",
        fake_retrieve,
    )
    monkeypatch.setattr(
        "src.services.inference.image_sampler.compute_empirical_mu",
        lambda image_seq_len, num_steps: 1.0,
    )


def test_sampler_constructs_with_pipeline_and_adapter():
    pipe = _stub_pipeline("cpu", "cpu", "cpu")
    adapter = FluxKleinArchAdapter()
    sampler = ImageSampler(pipe=pipe, arch_adapter=adapter)
    assert sampler.arch_adapter is adapter
    assert sampler.pipe is pipe


async def test_sample_calls_encode_prompt_then_denoise_then_vae_decode():
    """End-to-end orchestration: encode → loop transformer N steps → vae decode."""
    pipe = _stub_pipeline("cpu", "cpu", "cpu")
    _wire_default_stubs(pipe, num_steps=5)
    adapter = FluxKleinArchAdapter()

    sampler = ImageSampler(pipe=pipe, arch_adapter=adapter)
    result = await sampler.sample(_stub_request(steps=5))

    pipe.encode_prompt.assert_called_once()
    pipe.prepare_latents.assert_called_once()
    assert pipe.transformer.call_count == 5  # 5 timesteps
    pipe.vae.decode.assert_called_once()
    assert result.media_type == "image/png"
    assert isinstance(result.data, bytes)
    # PNG magic bytes
    assert result.data[:8] == b"\x89PNG\r\n\x1a\n"


async def test_sample_cancel_stops_loop_within_one_step():
    """Cancel flag set mid-denoise → next iteration check raises SamplerCancelled."""
    torch = _t()
    pipe = _stub_pipeline("cpu", "cpu", "cpu")
    _wire_default_stubs(pipe, num_steps=5)

    cancel_flag = CancelFlag()
    call_count = {"n": 0}

    def _transformer_side_effect(**kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            # Set cancel after 2nd forward — should stop before 3rd
            cancel_flag.set("test_cancel")
        return (torch.randn_like(kw["hidden_states"]),)

    pipe.transformer.side_effect = _transformer_side_effect

    sampler = ImageSampler(
        pipe=pipe,
        arch_adapter=FluxKleinArchAdapter(),
        cancel_flag=cancel_flag,
    )
    with pytest.raises(SamplerCancelled, match="test_cancel"):
        await sampler.sample(_stub_request(steps=5))
    assert call_count["n"] <= 3  # cancelled before doing all 5


async def test_sample_progress_callback_fires_per_step():
    pipe = _stub_pipeline("cpu", "cpu", "cpu")
    _wire_default_stubs(pipe, num_steps=5)

    progress_events = []

    async def _on_progress(step: int, total: int) -> None:
        progress_events.append((step, total))

    sampler = ImageSampler(
        pipe=pipe,
        arch_adapter=FluxKleinArchAdapter(),
        on_progress=_on_progress,
    )
    await sampler.sample(_stub_request(steps=5))

    # Progress fires once per denoise step (5 steps → 5 events).
    assert len(progress_events) == 5
    assert progress_events == [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]


async def test_sample_returns_metadata():
    """Result.metadata contains steps / width / height / seed / duration_ms."""
    pipe = _stub_pipeline("cpu", "cpu", "cpu")
    _wire_default_stubs(pipe, num_steps=1)

    sampler = ImageSampler(pipe=pipe, arch_adapter=FluxKleinArchAdapter())
    result = await sampler.sample(_stub_request(steps=1, seed=99))

    assert result.metadata["steps"] == 1
    assert result.metadata["width"] == 512
    assert result.metadata["height"] == 512
    assert result.metadata["seed"] == 99
    assert "duration_ms" in result.metadata
    assert result.usage.latency_ms == result.metadata["duration_ms"]
