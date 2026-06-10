"""PR-3 probe:GGUF 文本编码器装配(transformers 原生 from_pretrained(gguf_file=))。

验 build_bridged_text_encoder 的 .gguf 分支:Qwen3-4b-Z-Image-Engineer-V4-Q8_0.gguf → qwen3 编码器,
forward 出 hidden states 形状与普通 qwen_3_4b.safetensors 一致(同架构,不同微调权重)。
真模型/GPU,非 CI。CUDA_DEVICE_ORDER 必须 import torch 前设(cuda:1=Pro6000 非 3090)。
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.services.inference.component_spec import ComponentSpec  # noqa: E402
from src.services.inference.image_modular import build_bridged_text_encoder  # noqa: E402

DEV = os.environ.get("SMOKE_DEVICE", "cuda:1")
REPO = "configs/image_arch/z-image"
TE = "/media/heygo/Program/models/nous/image/text_encoders"
GGUF = f"{TE}/Qwen3-4b-Z-Image-Engineer-V4-Q8_0.gguf"
PLAIN = f"{TE}/qwen_3_4b.safetensors"


def spec(path: str) -> ComponentSpec:
    return ComponentSpec(kind="clip", file=path, device=DEV, dtype="bfloat16")


def main() -> int:
    print(f"== 装配 GGUF 编码器:{os.path.basename(GGUF)} ==")
    enc_g = build_bridged_text_encoder(spec(GGUF), REPO, DEV)
    print("  类:", type(enc_g).__name__, "| dtype:", next(enc_g.parameters()).dtype,
          "| device:", next(enc_g.parameters()).device)
    print(f"== 装配普通编码器:{os.path.basename(PLAIN)} ==")
    enc_p = build_bridged_text_encoder(spec(PLAIN), REPO, DEV)
    print("  类:", type(enc_p).__name__)

    # forward 短 prompt(用 GGUF/plain 各自模型),比 hidden state 形状
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(f"{REPO}/tokenizer")
    ids = tok("a portrait of a woman", return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        hg = enc_g(ids, output_hidden_states=True).hidden_states[-1]
        hp = enc_p(ids, output_hidden_states=True).hidden_states[-1]
    print(f"  GGUF hidden: {tuple(hg.shape)} | plain hidden: {tuple(hp.shape)}")
    same_shape = hg.shape == hp.shape
    # 权重不同微调 → 数值应不同(证实真用了 GGUF 权重,不是 silently 退普通)
    diff = (hg.float() - hp.float()).abs().mean().item()
    print(f"  形状一致: {same_shape} | hidden 均值差(应>0,证实不同权重): {diff:.4f}")
    ok = same_shape and diff > 1e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
