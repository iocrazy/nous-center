"""Ideogram-4 引擎 smoke —— ModularImageBackend 的 Ideogram4Pipeline 分支真机出图。standalone,需 GPU。

验证:① _build_ideogram4_pipe 整模型 from_pretrained ② call_kwargs 的
guidance_schedule=None 互斥处理 ③ 结构化 JSON caption 文字渲染正确(肉眼看出图)。

用法(落 Pro 6000;bf16 峰值 ~58G,载入 ~22s + 20 步 ~22s):
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_ideogram4.py
出图存 _smoke_out/ideogram4.png。首跑当 golden 备份(cp 成 ideogram4_golden.png),
改引擎/升 diffusers 后重生成再 SSIM 比对(同 smoke_image_ab golden 流程)。
"""
import os

# Pro 6000 在 PCI 序是 cuda:1;torch 默认 FASTEST_FIRST 会把它排 cuda:0(#278 坑)。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import json
import time
from pathlib import Path

REPO = os.environ.get(
    "SMOKE_MODEL", "/media/heygo/Program/models/nous/image/diffusers/Ideogram-4-bf16")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT = Path(__file__).parent / "_smoke_out"
OUT.mkdir(exist_ok=True)

PROMPT = json.dumps({
    "high_level_description": (
        "A minimal poster on deep blue background with giant white 3D letters "
        "spelling 'NOUS', subtitle text 'image engine smoke' at the bottom."),
}, ensure_ascii=False)


async def main() -> None:
    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    be = ModularImageBackend(repo=REPO, device=DEVICE, pipeline_class="Ideogram4Pipeline")
    t0 = time.time()
    resp = await be.infer(ImageRequest(
        request_id="smoke-ideo4", prompt=PROMPT,
        steps=20, width=1024, height=1024, cfg_scale=7.0, seed=42,
    ))
    print(f"[ideogram4] infer {time.time()-t0:.1f}s; media_type={resp.media_type}")
    assert resp.media_type == "image/png", f"期望 PNG,得到 {resp.media_type}"
    (OUT / "ideogram4.png").write_bytes(resp.data)
    print(f"saved {OUT/'ideogram4.png'}")


if __name__ == "__main__":
    asyncio.run(main())
