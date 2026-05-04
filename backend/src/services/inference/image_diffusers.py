"""DiffusersImageBackend — Flux.2 / diffusers image adapter.

Composes a 3-component pipeline (DiT transformer + Qwen3 text encoder + VAE)
out of single-file safetensors via the diffusers from_single_file API. The
spike (commit 6dba08e) confirmed diffusers 0.38.0.dev0 ships Flux2Pipeline
at diffusers.pipelines.flux2.pipeline_flux2 — no custom subclass needed.

Production strategy on a single 24GB 3090: enable_model_cpu_offload keeps
the active block on GPU and offloads the rest to CPU. Total weights are
~25GB which doesn't fit naïvely on one card; offload trades latency for
memory.

LoRA switching follows the diffusers issue #7842 family fix: disable
offload → set_adapters → re-enable offload. Doing set_adapters while the
model is offloaded triggers cross-device tensor errors.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

from src.services.inference.base import (
    ImageRequest,
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UsageMeter,
)

logger = logging.getLogger(__name__)


class DiffusersImageBackend(InferenceAdapter):
    """Adapter for diffusers-based image models (Flux.2 first).

    paths:
      transformer:  DiT safetensors (e.g. Flux2-Klein-9B-True-v2-bf16.safetensors)
      text_encoder: Qwen3 fp8 safetensors (or HF dir for tokenizer companion)
      vae:          VAE safetensors

    params:
      offload_strategy: "single_card_offload" (default) | "no_offload"
      lora_paths:       dict[str, str] mapping LoRA display name → safetensors
                        path. Populated by PR-5 LoRA scanner; empty by default.
      torch_dtype:      "bfloat16" (default) | "float16"
    """

    modality = MediaModality.IMAGE
    estimated_vram_mb = 24_000  # ~25GB compose; 24GB card with cpu_offload

    def __init__(
        self,
        paths: dict[str, str],
        device: str = "cuda",
        offload_strategy: str = "single_card_offload",
        lora_paths: dict[str, str] | None = None,
        torch_dtype: str = "bfloat16",
        **kwargs: Any,
    ):
        super().__init__(paths=paths, device=device)
        self._offload_strategy = offload_strategy
        self._lora_paths: dict[str, str] = dict(lora_paths or {})
        self._torch_dtype = torch_dtype
        self._loaded_loras: set[str] = set()
        self._pipe: Any = None  # diffusers Flux2Pipeline | FluxPipeline

    def _resolve_path(self, key: str) -> Path:
        """Resolve a paths[key] entry to an absolute on-disk path.

        Yaml stores paths relative to LOCAL_MODELS_PATH (matching the
        vLLM/TTS convention). Diffusers refuses anything that isn't an
        absolute path or HF hub id, so we must absolutize before handing
        the path off to from_single_file / from_pretrained.

        If the absolute candidate doesn't exist, fall back to the raw
        value — that lets the spec point at HF hub ids ("org/model")
        without LOCAL_MODELS_PATH prefixing them into nonsense.
        """
        from src.config import get_settings
        raw = self.paths.get(key)
        if not raw:
            raise ValueError(
                f"DiffusersImageBackend requires paths[{key!r}] but spec.paths={list(self.paths)}"
            )
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        absolutized = Path(get_settings().LOCAL_MODELS_PATH) / candidate
        if absolutized.exists():
            return absolutized
        # Last-resort: maybe an HF hub id; let diffusers reject it itself.
        return candidate

    def _gpu_index(self) -> int:
        if ":" in self.device:
            return int(self.device.split(":")[-1])
        return 0

    async def load(self, device: str | None = None) -> None:
        """Build a diffusers pipeline using the layout indicated by self.paths.

        Two supported layouts:

          1) Diffusers full-layout dir (recommended):
             paths = {"main": "<dir>"}
             where <dir> contains model_index.json + scheduler/ +
             transformer/ + text_encoder/ + vae/ + tokenizer/.
             Loaded via DiffusionPipeline.from_pretrained which reads
             every component's local config.json — zero HF hub access.
             Custom pipeline classes (e.g. ErnieImagePipeline) are
             auto-resolved via trust_remote_code.

          2) Three-component single-file compose (Flux2 style):
             paths = {"transformer": "<file>", "text_encoder": "<dir>",
                      "vae": "<file>"}
             from_single_file pulls architecture metadata from the HF
             hub (since safetensors only stores tensor names, not the
             class topology). Requires HF_TOKEN for gated repos.
        """
        if device:
            self.device = device

        # Imports are lazy: diffusers/transformers add ~3s import cost
        # and are only present when the `image` extra is installed.
        import torch
        dtype = getattr(torch, self._torch_dtype)

        if "main" in self.paths:
            await self._load_from_pretrained(dtype)
        else:
            await self._load_from_single_file_compose(dtype)

        if self._offload_strategy == "single_card_offload":
            self._pipe.enable_model_cpu_offload(gpu_id=self._gpu_index())
            logger.info(
                "image: enabled model_cpu_offload on gpu_id=%d",
                self._gpu_index(),
            )

        self._model = self._pipe

    async def _load_from_pretrained(self, dtype) -> None:
        from diffusers import DiffusionPipeline

        path = self._resolve_path("main")
        logger.info("image: loading diffusers pipeline from %s", path)
        # trust_remote_code=True so model dirs that ship a custom pipeline
        # class (ErnieImagePipeline, etc.) load it from the dir instead of
        # falling back to a stock pipeline that doesn't fit the model.
        self._pipe = DiffusionPipeline.from_pretrained(
            str(path),
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    async def _load_from_single_file_compose(self, dtype) -> None:
        from transformers import AutoModel, AutoTokenizer
        import diffusers
        from diffusers import AutoencoderKL, FluxTransformer2DModel

        transformer_path = self._resolve_path("transformer")
        encoder_path = self._resolve_path("text_encoder")
        vae_path = self._resolve_path("vae")

        logger.info("Flux2: loading DiT %s", transformer_path)
        transformer = FluxTransformer2DModel.from_single_file(
            str(transformer_path),
            torch_dtype=dtype,
        )

        logger.info("Flux2: loading text encoder %s", encoder_path)
        encoder_dir = encoder_path if encoder_path.is_dir() else encoder_path.parent
        tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
        encoder = AutoModel.from_pretrained(
            str(encoder_dir),
            torch_dtype=dtype,
        )

        logger.info("Flux2: loading VAE %s", vae_path)
        vae = AutoencoderKL.from_single_file(str(vae_path), torch_dtype=dtype)

        pipeline_cls = getattr(diffusers, "Flux2Pipeline", None) or diffusers.FluxPipeline
        self._pipe = pipeline_cls(
            transformer=transformer,
            text_encoder=encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=None,
        )

    def unload(self) -> None:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._pipe = None
        self._loaded_loras.clear()
        self._model = None

    def _is_offloaded(self) -> bool:
        # diffusers exposes _is_offloaded on the pipeline once cpu_offload runs.
        return bool(getattr(self._pipe, "_is_offloaded", False))

    def _apply_loras(self, loras: list) -> None:
        """Switch the active LoRA adapter set with the offload-safe ordering.

        Order: disable offload → load any new LoRA weights → set_adapters →
        re-enable offload. set_adapters under an active CPU-offload hook
        crosses devices and corrupts tensors (diffusers issue #7842 family).
        """
        was_offloaded = self._is_offloaded()
        if was_offloaded:
            self._pipe.disable_model_cpu_offload()

        if loras:
            for spec in loras:
                if spec.name in self._loaded_loras:
                    continue
                lora_path = self._lora_paths.get(spec.name)
                if not lora_path:
                    raise ValueError(
                        f"LoRA {spec.name!r} not in registered lora_paths "
                        f"(have: {sorted(self._lora_paths)})"
                    )
                self._pipe.load_lora_weights(lora_path, adapter_name=spec.name)
                self._loaded_loras.add(spec.name)

            self._pipe.set_adapters(
                [s.name for s in loras],
                adapter_weights=[s.strength for s in loras],
            )
        elif self._loaded_loras:
            # No LoRAs requested AND we have previously-loaded ones to clear.
            # Fresh-state pipelines (never had a LoRA) can't take set_adapters([])
            # — diffusers raises KeyError because _component_adapter_weights
            # is empty. Only call clear when there's actually something to clear.
            self._pipe.set_adapters([])

        if was_offloaded and self._offload_strategy == "single_card_offload":
            self._pipe.enable_model_cpu_offload(gpu_id=self._gpu_index())

    async def infer(self, req: InferenceRequest) -> InferenceResult:
        if not isinstance(req, ImageRequest):
            raise TypeError(
                f"DiffusersImageBackend expects ImageRequest, got {type(req).__name__}"
            )
        if self._pipe is None:
            raise RuntimeError("DiffusersImageBackend.load() must be called before infer()")

        import torch

        self._apply_loras(req.loras)

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.seed)

        t0 = time.monotonic()
        out = self._pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            generator=generator,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        image = out.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return InferenceResult(
            media_type="image/png",
            data=png_bytes,
            metadata={
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "seed": req.seed,
                "loras": [{"name": s.name, "strength": s.strength} for s in req.loras],
            },
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )
