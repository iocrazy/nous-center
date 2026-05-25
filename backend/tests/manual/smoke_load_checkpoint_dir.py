"""Arc 8 T6 — Load Checkpoint(整模型 diffusers 目录)经**生产路径**出图。standalone,需 GPU。

验收敛后的 Load Checkpoint:节点选 `diffusers/<model>/` 目录(component_select role=checkpoint)
→ exec_load_checkpoint 解析 <dir>/{transformer,text_encoder,vae} 首片 → 三描述符
→ (runner _build_request 摊平)→ get_or_load_image_adapter → _modular_repo_from_components
上溯到同一 HF repo → ModularImageBackend.from_pretrained → 出图。验**正确狐狸图**(非噪声)。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_load_checkpoint_dir.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

# 与生产(main.py / .env)一致 —— 否则 torch 默认 FASTEST_FIRST 会把 Pro 6000 排到
# cuda:0,跟工作流/nvidia-smi 的 PCI 索引错位(uv 不 load .env,standalone 必须自设)。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = Path("/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")  # PCI_BUS_ID 下 cuda:1 = Pro 6000
OUT_DIR = Path(__file__).parent / "_smoke_out"


def _load_exec_load_checkpoint():
    pkg = Path(__file__).resolve().parents[2] / "nodes" / "flux2-components"
    spec = importlib.util.spec_from_file_location("_ckpt_exec_smoke", pkg / "executor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXECUTORS["flux2_load_checkpoint"]


def _flatten(model_d: dict, clip_d: dict, vae_d: dict, device: str):
    """复刻 runner_process._build_request 的摊平:三描述符 → ComponentSpec(同卡)。"""
    from src.services.inference.component_spec import ComponentSpec
    unet_spec = dict(model_d["spec"]); unet_spec["device"] = device
    clip_spec = dict(clip_d["encoders"][0]); clip_spec["device"] = device
    vae_spec = dict(vae_d["spec"]); vae_spec["device"] = device
    return {
        "diffusion_models": ComponentSpec(loras=model_d.get("loras") or [], **unet_spec),
        "clip": ComponentSpec(**clip_spec),
        "vae": ComponentSpec(**vae_spec),
    }


async def main() -> None:
    os.environ["NOUS_IMAGE_ENGINE"] = "modular"
    if not (REPO / "model_index.json").exists():
        raise SystemExit(f"整模型目录缺 model_index.json: {REPO}")

    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.base import ImageRequest
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    # 1) 走真正的 Load Checkpoint executor:目录 → 三描述符
    exec_load_checkpoint = _load_exec_load_checkpoint()
    out = await exec_load_checkpoint({"file": str(REPO), "device": DEVICE, "weight_dtype": "bfloat16"}, {})
    print(f"[load-ckpt] dir={REPO.name}")
    print(f"[load-ckpt]   transformer={out['model']['spec']['file']}")
    print(f"[load-ckpt]   text_encoder={out['clip']['encoders'][0]['file']}")
    print(f"[load-ckpt]   vae={out['vae']['spec']['file']}")

    # 2) runner 摊平 → ComponentSpec
    components = _flatten(out["model"], out["clip"], out["vae"], DEVICE)

    class _EmptyRegistry(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
    print(f"[load-ckpt] adapter={type(adapter).__name__}")

    req = ImageRequest(
        request_id="load-ckpt-dir",
        prompt="a photo of a red fox sitting in autumn leaves, sharp focus, detailed",
        negative_prompt="", width=1024, height=1024, steps=20, cfg_scale=4.0, seed=42,
        components=components, pipeline_class="Flux2KleinPipeline",
    )
    res = await adapter.infer(req)
    OUT_DIR.mkdir(exist_ok=True)
    out_png = OUT_DIR / "smoke_load_checkpoint_dir.png"
    out_png.write_bytes(res.data)
    print(f"[load-ckpt] saved → {out_png} ({len(res.data)} bytes) meta={res.metadata}")
    print("[load-ckpt] OK — Load Checkpoint(整模型目录)经生产路径出图")


if __name__ == "__main__":
    asyncio.run(main())
