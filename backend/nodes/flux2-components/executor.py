"""Flux2 / Diffusers component-node executors — 细粒度图收敛后(spec 2026-05-21 rev 2)。

收敛后的执行模型:细粒度图是线性链(Load* → Encode → KSampler → VAE Decode)。
Load* / Encode / KSampler 是 **inline 描述符产出节点**(主进程 event loop,不碰
GPU),只产/累积**嵌套 plain dict 描述符**(无张量)。末端 ``flux2_vae_decode``
是 **dispatch 节点**(不在本 EXECUTORS),由 workflow_executor 派发到 image runner;
runner ``_build_request`` 把嵌套 latent 摊平成 ImageRequest,
``get_or_load_image_adapter`` 把整模型装到工作流所选的**单张卡**,
``ImageSampler.sample()`` 一把跑完 encode→denoise→decode。

描述符形态:
    model        {_type:flux2_model, spec:{kind:unet,file,device,dtype,adapter_arch}, loras:[]}
    clip         {_type:flux2_clip, type:<arch>, encoders:[{kind:clip,file,dtype}, ...]}
    vae          {_type:flux2_vae, spec:{kind:vae,file,dtype}}
    conditioning {_type:flux2_conditioning, clip:<clip>, text, negative}
    latent       {_type:flux2_latent, model:<model>, conditioning:<cond>, width,height,steps,cfg_scale,seed}

Load CLIP 本 PR 单编码器(file + weight_dtype);多编码器 UI(clip_stack)+ gated
执行 = PR-3。Load Checkpoint 暂留 model_key 旧形态(PR-1 Task 6 改 resolver)。
"""
from __future__ import annotations

from typing import Any

_DEFAULT_MODEL_KEY = "flux2-klein-9b-true-v2-fp8mixed"
_DEFAULT_DTYPE = "default"
_AUTO = "auto"


# --- Load Checkpoint(暂留 model_key 旧形态,Task 6 改 resolver)---------------


def _bundle_model(model_key: str) -> dict[str, Any]:
    return {"_type": "flux2_model", "model_id": model_key, "loras": []}


def _bundle_clip(model_key: str) -> dict[str, Any]:
    return {"_type": "flux2_clip", "model_id": model_key}


def _bundle_vae(model_key: str) -> dict[str, Any]:
    return {"_type": "flux2_vae", "model_id": model_key}


def _read_model_key(data: dict) -> str:
    return data.get("model_key") or _DEFAULT_MODEL_KEY


async def exec_load_checkpoint(data: dict, inputs: dict) -> dict:
    """单 spec 三件 emit(ComfyUI CheckpointLoaderSimple 类比)。PR-1 Task 6 会
    改成 model_key→三组件文件 resolver(三件同 device);本 PR 暂留旧形态。"""
    key = _read_model_key(data)
    return {
        "model": _bundle_model(key),
        "clip": _bundle_clip(key),
        "vae": _bundle_vae(key),
    }


# --- 细粒度 loader / 中间节点 → 嵌套描述符(inline, 无 GPU)-------------------


def _spec_unet(data: dict) -> dict:
    return {
        "kind": "unet",
        "file": data["file"],
        "device": data.get("device") or _AUTO,
        "dtype": data.get("weight_dtype") or _DEFAULT_DTYPE,
        "adapter_arch": data.get("adapter_arch") or "flux2",
    }


async def exec_load_diffusion_model(data: dict, inputs: dict) -> dict:
    """MODEL —— transformer 组件描述符 + device(整张图跑哪张卡)。"""
    return {"model": {"_type": "flux2_model", "spec": _spec_unet(data), "loras": []}}


async def exec_load_clip(data: dict, inputs: dict) -> dict:
    """CLIP —— 单编码器(file + weight_dtype)。多编码器(clip_stack)= PR-3。
    无 device:跟随上游 transformer 的卡(整模型单卡)。"""
    enc = {"kind": "clip", "file": data["file"], "dtype": data.get("weight_dtype") or _DEFAULT_DTYPE}
    return {"clip": {"_type": "flux2_clip", "type": data.get("type") or "flux2", "encoders": [enc]}}


async def exec_load_vae(data: dict, inputs: dict) -> dict:
    """VAE —— 组件描述符。无 device:跟随 transformer 卡。"""
    spec = {"kind": "vae", "file": data["file"], "dtype": data.get("weight_dtype") or _DEFAULT_DTYPE}
    return {"vae": {"_type": "flux2_vae", "spec": spec}}


async def exec_load_lora(data: dict, inputs: dict) -> dict:
    """串联:上游 MODEL → append 一条 LoRA(带 abs path)→ 新 MODEL。空 lora_name
    透传(ComfyUI 禁用 loader 语义)。LoRA 跟随上游 transformer 卡。"""
    upstream = inputs.get("model")
    if not isinstance(upstream, dict) or upstream.get("_type") != "flux2_model":
        raise RuntimeError("Load LoRA 的 MODEL 输入未连接,或上游不是 flux2_model")
    name = (data.get("lora_name") or "").strip()
    out = dict(upstream)
    out["loras"] = list(upstream.get("loras") or [])
    if name:
        out["loras"].append({
            "name": name,
            "path": data.get("lora_path") or None,
            "strength": float(data.get("strength", 1.0)),
        })
    return {"model": out}


async def exec_encode_prompt(data: dict, inputs: dict) -> dict:
    """CLIP + text → CONDITIONING 描述符。不在主进程 encode —— 真编码在 runner
    的 ImageSampler 内(末端 VAE Decode 派发触发)。"""
    clip = inputs.get("clip")
    if not isinstance(clip, dict) or clip.get("_type") != "flux2_clip":
        raise RuntimeError("Encode Prompt 的 CLIP 端口未连接,或上游不是 flux2_clip")
    text = inputs.get("text") or data.get("text") or ""
    return {"conditioning": {
        "_type": "flux2_conditioning", "clip": clip,
        "text": text, "negative": data.get("negative_prompt", "") or "",
    }}


async def exec_ksampler(data: dict, inputs: dict) -> dict:
    """MODEL + CONDITIONING → LATENT 描述符(采样参数 + 嵌套上游计划)。不在主进程
    sample —— 真采样在 runner 的 ImageSampler 内。"""
    model = inputs.get("model")
    if not isinstance(model, dict) or model.get("_type") != "flux2_model":
        raise RuntimeError("KSampler 的 MODEL 端口未连接,或上游不是 flux2_model")
    cond = inputs.get("conditioning")
    if not isinstance(cond, dict) or cond.get("_type") != "flux2_conditioning":
        raise RuntimeError("KSampler 的 CONDITIONING 端口未连接,或上游不是 flux2_conditioning")
    raw_seed = data.get("seed")
    seed = int(raw_seed) if raw_seed not in (None, "") else None
    return {"latent": {
        "_type": "flux2_latent", "model": model, "conditioning": cond,
        "width": int(data.get("width", 1024)), "height": int(data.get("height", 1024)),
        "steps": int(data.get("steps", 25)), "cfg_scale": float(data.get("cfg_scale", 4.0)),
        "seed": seed,
    }}


# flux2_vae_decode 不在此 —— 它走 dispatch(node_routing.DISPATCH_NODE_TYPES),
# 由 workflow_executor 派发到 image runner;runner _build_request 摊平嵌套 latent
# 成 ImageRequest,get_or_load_image_adapter + ImageSampler 在所选卡整模型执行。
EXECUTORS = {
    "flux2_load_checkpoint": exec_load_checkpoint,
    "flux2_load_diffusion_model": exec_load_diffusion_model,
    "flux2_load_clip": exec_load_clip,
    "flux2_load_vae": exec_load_vae,
    "flux2_load_lora": exec_load_lora,
    "flux2_encode_prompt": exec_encode_prompt,
    "flux2_ksampler": exec_ksampler,
}
