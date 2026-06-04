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

_DEFAULT_MODEL_KEY = "flux2-klein-9b-true-v2-fp8mixed"
# bfloat16 默认(非 "default"):"default"→不给 torch_dtype→diffusers 加载成 fp32
# (4× 慢 + 2× 显存)。"default" 选项保留作高级原生精度。实测 default 27s vs bf16 6.5s。
_DEFAULT_DTYPE = "bfloat16"
_AUTO = "auto"


# --- Load Checkpoint(便捷单卡入口:model_key → 三组件文件,三件同 device)-----


def _first_safetensors(repo_dir, sub: str) -> str:
    """HF-layout 整模型 <dir>/<sub>/ 首个 .safetensors(分片取首片,loader 向上找 index)。"""
    from pathlib import Path  # noqa: PLC0415
    hits = sorted(Path(repo_dir).joinpath(sub).glob("*.safetensors"))
    if not hits:
        raise RuntimeError(f"Load Checkpoint: 整模型目录缺 {sub}/(无 .safetensors): {repo_dir}")
    return str(hits[0])


async def exec_load_checkpoint(data: dict, inputs: dict) -> dict:
    """整模型(HF-layout diffusers 目录)→ MODEL+CLIP+VAE 三组件(ComfyUI DiffusersLoader 类比)。

    data['file'] = `diffusers/<model>/` 目录(component_select role=checkpoint 选的整模型)。
    解析 <dir>/{transformer,text_encoder,vae} 各首文件,三件同 device/dtype。
    注:adapter_arch 暂固定 flux2(引擎当前只支持 Flux2;ERNIE 等多架构是 future)。
    """
    repo = data.get("file")
    if not repo:
        raise RuntimeError("Load Checkpoint: 未选整模型(file 空)—— 在 diffusers/ 放 HF-layout 模型")
    device = data.get("device") or _AUTO
    dtype = data.get("weight_dtype") or _DEFAULT_DTYPE
    offload = data.get("offload") or "none"  # 三件同卡同 offload(便捷节点)
    return {
        "model": {"_type": "flux2_model", "spec": {
            "kind": "diffusion_models", "file": _first_safetensors(repo, "transformer"),
            "device": device, "dtype": dtype, "adapter_arch": "flux2"}, "loras": [], "offload": offload},
        "clip": {"_type": "flux2_clip", "type": "flux2", "device": device, "offload": offload,
                 "encoders": [{"kind": "clip", "file": _first_safetensors(repo, "text_encoder"), "dtype": dtype}]},
        "vae": {"_type": "flux2_vae", "spec": {"kind": "vae", "file": _first_safetensors(repo, "vae"),
                                               "dtype": dtype, "device": device, "offload": offload}},
    }


# --- 细粒度 loader / 中间节点 → 嵌套描述符(inline, 无 GPU)-------------------


def _spec_unet(data: dict) -> dict:
    return {
        "kind": "diffusion_models",
        "file": data["file"],
        "device": data.get("device") or _AUTO,
        "dtype": data.get("weight_dtype") or _DEFAULT_DTYPE,
        "adapter_arch": data.get("adapter_arch") or "flux2",
    }


async def exec_load_diffusion_model(data: dict, inputs: dict) -> dict:
    """MODEL —— transformer 组件描述符 + device(整张图跑哪张卡)+ offload(权重 stash,PR-D)。"""
    return {
        "model": {"_type": "flux2_model", "spec": _spec_unet(data), "loras": [],
                  "offload": data.get("offload") or "none"},
    }


async def exec_load_clip(data: dict, inputs: dict) -> dict:
    """CLIP —— 动态多编码器(clip_stack:每条 file + weight_dtype)+ type(架构)。
    逐组件选卡(2026-06-04):node 级 device/offload(套用所有 encoder);device=auto 跟随
    transformer 卡(零回归)。多编码器执行 gated(runner _build_request 拦,见 spec §4.3)。
    兜底旧单 file 格式(PR-1/PR-2 期存的 workflow)。"""
    clips = data.get("clips")
    if not clips and data.get("file"):  # back-compat:PR-1/PR-2 单 file
        clips = [{"file": data["file"], "weight_dtype": data.get("weight_dtype")}]
    encoders = [
        {"kind": "clip", "file": c["file"], "dtype": c.get("weight_dtype") or _DEFAULT_DTYPE}
        for c in (clips or []) if c.get("file")
    ]
    return {"clip": {"_type": "flux2_clip", "type": data.get("type") or "flux2", "encoders": encoders,
                     "device": data.get("device") or _AUTO, "offload": data.get("offload") or "none"}}


async def exec_load_vae(data: dict, inputs: dict) -> dict:
    """VAE —— 组件描述符。逐组件选卡(2026-06-04):device=auto 跟随 transformer 卡(零回归),
    显式选卡则落该卡;offload 同 Diffusion Model。"""
    spec = {"kind": "vae", "file": data["file"], "dtype": data.get("weight_dtype") or _DEFAULT_DTYPE,
            "device": data.get("device") or _AUTO, "offload": data.get("offload") or "none"}
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


# 架构兼容表:DiT(diffusion model)的 adapter_arch → 它能配的 CLIP type 集合。
# 端口类型校验只到 MODEL/CLIP/VAE 类别,管不了「架构语义」—— anima DiT 配 flux2 CLIP/VAE
# 在前端连得上、inline 节点也全过,直到 runner 真加载模型才甩 PyTorch size-mismatch 堆栈
# (AutoencoderKLQwenImage / unexpected weight keys),用户看不懂。这里在主进程派发前拦,
# 给人话错误(node.yaml 注释的意图:catch category mismatches at draw time)。
_ARCH_CLIP_COMPAT = {
    "anima": {"anima", "qwen"},      # Anima 2B DiT 自带 qwen3 text encoder
    "flux2": {"flux2", "flux1"},     # Flux2 family
    "flux1": {"flux1", "flux2"},
}


def _check_arch_compat(unet_arch: str, clip_type: str) -> None:
    allowed = _ARCH_CLIP_COMPAT.get(unet_arch)
    if allowed is not None and clip_type not in allowed:
        raise RuntimeError(
            f"架构不匹配:Diffusion Model 架构是 '{unet_arch}',但 CLIP 架构是 '{clip_type}'。"
            f"'{unet_arch}' 需配 {sorted(allowed)} 之一的 CLIP/text encoder"
            + ("。Anima 用 qwen_3_06b_base.safetensors(架构选 anima/qwen)+ "
               "qwen_image_vae.safetensors" if unet_arch == "anima" else "")
            + "。请在 Load CLIP / Load VAE 选对应架构的组件。"
        )


async def exec_ksampler(data: dict, inputs: dict) -> dict:
    """MODEL + CONDITIONING → LATENT 描述符(采样参数 + 嵌套上游计划)。不在主进程
    sample —— 真采样在 runner 的 ImageSampler 内。"""
    model = inputs.get("model")
    if not isinstance(model, dict) or model.get("_type") != "flux2_model":
        raise RuntimeError("KSampler 的 MODEL 端口未连接,或上游不是 flux2_model")
    cond = inputs.get("conditioning")
    if not isinstance(cond, dict) or cond.get("_type") != "flux2_conditioning":
        raise RuntimeError("KSampler 的 CONDITIONING 端口未连接,或上游不是 flux2_conditioning")
    # round-2026-06-01:DiT 架构 vs CLIP 架构一致性检查(派发前拦,人话错误)。
    unet_arch = (model.get("spec") or {}).get("adapter_arch") or "flux2"
    clip_type = (cond.get("clip") or {}).get("type") or "flux2"
    _check_arch_compat(unet_arch, clip_type)
    raw_seed = data.get("seed")
    seed = int(raw_seed) if raw_seed not in (None, "") else None
    return {"latent": {
        "_type": "flux2_latent", "model": model, "conditioning": cond,
        # round5:空串 widget(default: "")→ 默认值,不 int("")/float("") 崩
        "width": int(data.get("width") or 1024), "height": int(data.get("height") or 1024),
        "steps": int(data.get("steps") or 25), "cfg_scale": float(data.get("cfg_scale") or 4.0),
        "sampler_name": data.get("sampler_name") or "euler",
        "scheduler": data.get("scheduler") or "normal",
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
