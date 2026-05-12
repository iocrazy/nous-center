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


EXECUTORS = {
    "flux2_load_checkpoint": exec_load_checkpoint,
    "flux2_load_diffusion_model": exec_load_diffusion_model,
    "flux2_load_clip": exec_load_clip,
    "flux2_load_vae": exec_load_vae,
    "flux2_load_lora": exec_load_lora,
}
