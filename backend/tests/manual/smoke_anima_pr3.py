"""PR-anima-3 真模型 smoke:AnimaTextEncoder 加载 qwen3-0.6b base + encode。

需要:
  - /media/heygo/Program/models/nous/image/text_encoders/qwen_3_06b_base.safetensors
  - qwen25_tokenizer 配置目录(从 ComfyUI 拷,或下面 SMOKE_QWEN_TOKENIZER 环境变量指定)
  - (可选)t5_tokenizer 配置目录,SMOKE_T5_TOKENIZER 指定;走 t5xxl 桥接路径

用法:
    cd backend
    SMOKE_QWEN_TOKENIZER=/home/heygo/sites/ComfyUI/comfy/text_encoders/qwen25_tokenizer \
    SMOKE_T5_TOKENIZER=/home/heygo/sites/ComfyUI/comfy/text_encoders/t5_tokenizer \
    SMOKE_DEVICE=cuda:1 \
    uv run python tests/manual/smoke_anima_pr3.py

GPU 跑约 5-10s。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

QWEN_WEIGHTS = "/media/heygo/Program/models/nous/image/text_encoders/qwen_3_06b_base.safetensors"
QWEN_TOKENIZER = os.environ.get(
    "SMOKE_QWEN_TOKENIZER",
    "/home/heygo/sites/ComfyUI/comfy/text_encoders/qwen25_tokenizer",
)
T5_TOKENIZER = os.environ.get(
    "SMOKE_T5_TOKENIZER",
    "/home/heygo/sites/ComfyUI/comfy/text_encoders/t5_tokenizer",
)
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")


def main() -> None:
    import torch  # noqa: PLC0415

    from src.services.inference.arch_anima import AnimaTextEncoder  # noqa: PLC0415

    if not Path(QWEN_WEIGHTS).exists():
        print(f"[anima-pr3] !! qwen weights missing: {QWEN_WEIGHTS} — skipping")
        return
    if not Path(QWEN_TOKENIZER).exists():
        print(f"[anima-pr3] !! qwen tokenizer dir missing: {QWEN_TOKENIZER}")
        print("[anima-pr3]    set SMOKE_QWEN_TOKENIZER=/path/to/qwen25_tokenizer")
        return

    t5_dir = T5_TOKENIZER if Path(T5_TOKENIZER).exists() else None
    print(f"[anima-pr3] qwen={QWEN_WEIGHTS}")
    print(f"[anima-pr3] qwen_tokenizer={QWEN_TOKENIZER}")
    print(f"[anima-pr3] t5_tokenizer={t5_dir or '(disabled)'}")
    print(f"[anima-pr3] device={DEVICE}")

    te = AnimaTextEncoder(
        qwen_weights_path=QWEN_WEIGHTS,
        qwen_tokenizer_dir=QWEN_TOKENIZER,
        t5_tokenizer_dir=t5_dir,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    print("[anima-pr3] loading qwen3-0.6b base + tokenizers...")
    te.load()
    assert te.is_loaded
    print("[anima-pr3]   ✓ loaded")

    prompt = "a photo of a red fox in autumn leaves"
    out = te.encode(prompt)
    context = out["context"]
    t5_ids = out["t5xxl_ids"]
    t5_w = out["t5xxl_weights"]
    print(f"[anima-pr3]   ✓ encode('{prompt[:40]}...') →")
    print(f"      context = {tuple(context.shape)}  ({context.dtype}, on {context.device})")
    if t5_ids is not None:
        print(f"      t5xxl_ids = {tuple(t5_ids.shape)} (on {t5_ids.device})")
        print(f"      t5xxl_weights = {tuple(t5_w.shape)}")
    else:
        print("      t5xxl_ids = None (t5 tokenizer disabled)")

    assert context.ndim == 3 and context.shape[0] == 1
    assert torch.isfinite(context).all(), "context has nan/inf"
    if t5_ids is not None:
        assert t5_ids.ndim == 2 and t5_ids.shape[0] == 1

    print(f"[anima-pr3] peak VRAM on {DEVICE}: "
          f"{torch.cuda.max_memory_allocated(torch.device(DEVICE)) / 1024**2:.0f} MiB")
    print("[anima-pr3] verdict = PASS")


if __name__ == "__main__":
    main()
