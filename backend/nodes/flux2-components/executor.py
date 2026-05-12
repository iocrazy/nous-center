"""Flux2 / Diffusers component-node executors (V1' Lane C / P3.2).

Five Loader nodes that produce typed bundle dicts:

    Load Checkpoint   ─→ MODEL + CLIP + VAE
    Load Diffusion    ─→ MODEL
    Load CLIP         ─→ CLIP
    Load VAE          ─→ VAE
    Load LoRA(MODEL)  ─→ MODEL (with LoRA appended)

The "Load" nodes are intentionally cheap — they only stash a yaml model_id +
a bundle kind marker. Real adapter materialization happens inside the
downstream sampling nodes (PR-C3: EncodePrompt / KSampler / VAEDecode) via
``model_manager.get_loaded_adapter(model_id)``. This matches nous-center's
load-on-demand + TTL eviction model rather than ComfyUI's load-and-pin.

Future PR-C4 will upgrade the model_key widget to a scanner-driven dropdown
so newly-added yaml specs (or auto-detected dirs) appear without an
options-list edit here.
"""
from __future__ import annotations

import asyncio
import io
from typing import Any


_DEFAULT_MODEL_KEY = "flux2-klein-9b-true-v2-fp8mixed"


def _bundle_model(model_key: str) -> dict[str, Any]:
    return {"_type": "flux2_model", "model_id": model_key, "loras": []}


def _bundle_clip(model_key: str) -> dict[str, Any]:
    return {"_type": "flux2_clip", "model_id": model_key}


def _bundle_vae(model_key: str) -> dict[str, Any]:
    return {"_type": "flux2_vae", "model_id": model_key}


def _read_model_key(data: dict) -> str:
    return data.get("model_key") or _DEFAULT_MODEL_KEY


async def exec_load_checkpoint(data: dict, inputs: dict) -> dict:
    """Single-spec triplet emit. ComfyUI's CheckpointLoaderSimple analog —
    one spec yields MODEL + CLIP + VAE so a casual workflow doesn't have
    to wire three Loaders just to get going."""
    key = _read_model_key(data)
    return {
        "model": _bundle_model(key),
        "clip": _bundle_clip(key),
        "vae": _bundle_vae(key),
    }


async def exec_load_diffusion_model(data: dict, inputs: dict) -> dict:
    """Just the MODEL — meant to pair with separate LoadCLIP + LoadVAE in
    swap-component workflows."""
    return {"model": _bundle_model(_read_model_key(data))}


async def exec_load_clip(data: dict, inputs: dict) -> dict:
    return {"clip": _bundle_clip(_read_model_key(data))}


async def exec_load_vae(data: dict, inputs: dict) -> dict:
    return {"vae": _bundle_vae(_read_model_key(data))}


async def exec_load_lora(data: dict, inputs: dict) -> dict:
    """Append a LoRA spec onto the upstream MODEL bundle and emit a new
    MODEL. Chaining multiple LoadLoRA nodes accumulates the stack — same
    semantics as the integrated image_generate.loras list.

    `lora_name` is the LoRA scanner's display name (matched against
    DiffusersImageBackend._lora_paths). An empty name is a no-op that
    passes the bundle through unchanged, which matches ComfyUI's "disabled
    LoRA loader" behavior and lets users park an unconfigured node on the
    canvas without breaking the run.
    """
    upstream = inputs.get("model")
    if not isinstance(upstream, dict) or upstream.get("_type") != "flux2_model":
        raise RuntimeError(
            "Load LoRA 节点的 MODEL 输入未连接,或上游不是 flux2_model 类型"
        )

    name = (data.get("lora_name") or "").strip()
    strength = float(data.get("strength", 1.0))

    out = dict(upstream)
    out["loras"] = list(upstream.get("loras") or [])
    if name:
        out["loras"].append({"name": name, "strength": strength})
    return {"model": out}


# --- Sampling nodes (V1' Lane C / P3.3) ------------------------------------
# EncodePrompt / KSampler / VAEDecode are the consumers of the Loader bundles.
# These nodes are where Lane C's sampling helpers from PR #82
# (encode_prompt / sample / vae_decode) finally touch the GPU.


def _require(bundle: Any, expected_type: str, port_label: str) -> dict:
    if not isinstance(bundle, dict) or bundle.get("_type") != expected_type:
        raise RuntimeError(
            f"{port_label} 端口未连接,或上游不是 {expected_type} 类型"
        )
    return bundle


async def _acquire_adapter(model_id: str) -> Any:
    from src.services import workflow_executor as we
    if we._model_manager is None:
        raise RuntimeError("ModelManager 未初始化")
    return await we._model_manager.get_loaded_adapter(model_id)


async def exec_encode_prompt(data: dict, inputs: dict) -> dict:
    """CLIP + text → CONDITIONING.

    Reuses the adapter that owns the CLIP bundle's model_id, since in
    nous-center the text encoder + tokenizer always travel with the
    diffusers pipeline (a CLIP bundle here is conceptually 'the
    text_encoder slice of model_id's pipeline')."""
    from src.services.inference.image_diffusers import encode_prompt

    clip = _require(inputs.get("clip"), "flux2_clip", "CLIP")
    text = inputs.get("text") or data.get("text") or ""
    if not text:
        raise RuntimeError("EncodePrompt 节点缺少 text 输入")

    adapter = await _acquire_adapter(clip["model_id"])
    cond = await asyncio.to_thread(encode_prompt, adapter.pipe, text)
    return {
        "conditioning": {
            "_type": "flux2_conditioning",
            "model_id": clip["model_id"],
            "prompt_embeds": cond["prompt_embeds"],
            "text_ids": cond["text_ids"],
        }
    }


async def exec_ksampler(data: dict, inputs: dict) -> dict:
    """MODEL + CONDITIONING → LATENT.

    Applies the MODEL bundle's accumulated .loras stack (built up by
    LoadLoRA chains) via `adapter.set_active_loras` right before sampling,
    so the offload/lora-load ordering invariants stay in one place.
    """
    from src.services.inference.base import LoRASpec
    from src.services.inference.image_diffusers import sample

    model = _require(inputs.get("model"), "flux2_model", "MODEL")
    cond = _require(inputs.get("conditioning"), "flux2_conditioning", "CONDITIONING")

    if model["model_id"] != cond["model_id"]:
        # Cross-model conditioning is a category error — embeds shape +
        # tokenizer vocab won't match. Stop early with a useful message.
        raise RuntimeError(
            f"KSampler MODEL ({model['model_id']!r}) 与 CONDITIONING "
            f"({cond['model_id']!r}) model_id 不一致"
        )

    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 25))
    cfg = float(data.get("cfg_scale", 4.0))
    seed = data.get("seed")

    adapter = await _acquire_adapter(model["model_id"])

    loras = [
        LoRASpec(name=spec["name"], strength=float(spec.get("strength", 1.0)))
        for spec in (model.get("loras") or [])
        if spec.get("name")
    ]
    # set_active_loras is sync (touches pipe / peft); keep it on the loop
    # since LoRA apply is fast (<100ms) and avoiding to_thread keeps the
    # offload-disable+enable cycle's "I'm holding the pipe" semantics
    # observable in a single coroutine context.
    adapter.set_active_loras(loras)

    import torch
    generator = None
    if seed is not None:
        generator = torch.Generator(device=adapter.device).manual_seed(int(seed))

    latents = await asyncio.to_thread(
        sample,
        adapter.pipe,
        {"prompt_embeds": cond["prompt_embeds"]},
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )
    return {
        "latent": {
            "_type": "flux2_latent",
            "model_id": model["model_id"],
            "tensor": latents,
        }
    }


async def exec_vae_decode(data: dict, inputs: dict) -> dict:
    """VAE + LATENT → IMAGE (signed URL).

    Output schema mirrors image_generate so existing ImageOutputNode
    consumers see a familiar shape — image_url is the signed URL,
    media_type stays "image/png", width/height carry image dims.
    """
    from src.services.image_output_storage import write_image
    from src.services.inference.image_diffusers import vae_decode

    vae = _require(inputs.get("vae"), "flux2_vae", "VAE")
    latent = _require(inputs.get("latent"), "flux2_latent", "LATENT")
    if vae["model_id"] != latent["model_id"]:
        raise RuntimeError(
            f"VAEDecode VAE ({vae['model_id']!r}) 与 LATENT "
            f"({latent['model_id']!r}) model_id 不一致"
        )

    adapter = await _acquire_adapter(vae["model_id"])
    image = await asyncio.to_thread(vae_decode, adapter.pipe, latent["tensor"])

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    ttl = int(data.get("url_ttl_seconds", 3600))
    record = write_image(buf.getvalue(), ext="png", ttl_seconds=ttl)
    if record["url"] is None:
        raise RuntimeError(
            "VAEDecode 需要 ADMIN_SESSION_SECRET 才能签名输出 URL — "
            "请在 backend/.env 配置后重启 backend"
        )
    return {
        "image_url": record["url"],
        "media_type": "image/png",
        "width": image.width,
        "height": image.height,
        "image_uuid": record["uuid"],
        "image_expires": record["expires"],
    }


EXECUTORS = {
    "flux2_load_checkpoint": exec_load_checkpoint,
    "flux2_load_diffusion_model": exec_load_diffusion_model,
    "flux2_load_clip": exec_load_clip,
    "flux2_load_vae": exec_load_vae,
    "flux2_load_lora": exec_load_lora,
    "flux2_encode_prompt": exec_encode_prompt,
    "flux2_ksampler": exec_ksampler,
    "flux2_vae_decode": exec_vae_decode,
}
