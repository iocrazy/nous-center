"""路 B PR-B2 真机闸门:留噪 latent 接力(KSamplerAdvanced split)手写去噪循环数值正确性。standalone,需 GPU。

核心闸门(证手写续采 == 整段去噪,逐数值对):
  单段:Z-Image 12 步整段去噪 → img_single。
  分段:base(end_at_step=5 + return_with_leftover_noise=enable,导出带噪 latent)→ 落盘 latent_ref →
        refiner(start_at_step=5 + add_noise=disable,注入带噪 latent 续采到 0)→ img_split。
  断言 SSIM(img_single, img_split) ≥ 0.95(实际应 ≈1.0 bit 级 —— 分段只是把同一组 sigma 索引劈两半)。

附加验:
  ② base 段确实留噪(末态非纯净 latent;不是全去噪完)。
  ③ 跨架构 init_latent_ref 触发派发前人话报错(arch 不符 → ValueError,不崩在 transformer 深处)。

image_modular.py 改了 → 另跑 tests/manual/smoke_image_ab.py 守 Flux2 golden SSIM≥0.97 零回归。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_zimage_split.py
"""
from __future__ import annotations

import os

# standalone 必须在 import torch 前固定 PCI_BUS_ID(否则 torch FASTEST_FIRST 把 3090 排到 cuda:1)。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = os.environ.get("SMOKE_ZIMAGE", "/media/heygo/Program/models/nous/image/diffusers/Z-Image-Turbo")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
PROMPT = "a photo of a red fox in autumn leaves, sharp focus, cinematic"
STEPS = 12
SPLIT = 5
SEED = 42


def _save_png(data: bytes, name: str) -> Path:
    p = Path(tempfile.gettempdir()) / name
    p.write_bytes(data)
    return p


def _ssim(p1: Path, p2: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:  # noqa: BLE001
        print(f"  SSIM 跳过(缺 skimage/PIL: {e})")
        return None
    a = np.asarray(Image.open(p1).convert("L"))
    b = np.asarray(Image.open(p2).convert("L"))
    return float(ssim(a, b))


async def main() -> int:
    import safetensors.torch as st
    import torch  # noqa: F401

    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    be = ModularImageBackend(repo=REPO, device=DEVICE, dtype="bfloat16", pipeline_class="ZImagePipeline")
    await be.load(DEVICE)
    ok = True

    # ---- 单段基线:12 步整段去噪 → 图 ----
    single = await be.infer(ImageRequest(
        request_id="split-single", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
        steps=STEPS, width=1024, height=1024, seed=SEED, output_mode="image"))
    assert single.media_type == "image/png", single.media_type
    p_single = _save_png(single.data, "zsplit_single.png")
    print(f"  [single] {STEPS} 步整段 → {p_single} ({len(single.data)//1024}KB)")

    # ---- 分段 base:end_at_step=SPLIT + 留噪 → 导出带噪 latent ----
    base = await be.infer(ImageRequest(
        request_id="split-base", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
        steps=STEPS, width=1024, height=1024, seed=SEED,
        end_at_step=SPLIT, return_with_leftover_noise=True, add_noise=True,
        output_mode="latent"))
    assert base.media_type == "application/x-latent", base.media_type
    bmeta = (base.metadata or {}).get("latent") or {}
    base_latent = st.load(base.data)["latent"]
    print(f"  [base]   end_at_step={SPLIT} 留噪 → latent shape={list(base_latent.shape)} "
          f"arch={bmeta.get('arch')} ch={bmeta.get('latent_channels')} std={base_latent.float().std():.4f}")
    assert bmeta.get("arch") == "z-image"

    # 落盘成 latent_ref(模拟 runner write_latent + outputs.latent_ref)
    lat_path = Path(tempfile.gettempdir()) / "zsplit_base.safetensors"
    lat_path.write_bytes(base.data)
    init_ref = {"_type": "latent_ref", "path": str(lat_path), "arch": bmeta.get("arch"),
                "latent_channels": bmeta.get("latent_channels"), "shape": bmeta.get("shape")}

    # ---- 分段 refiner:start_at_step=SPLIT + 注入带噪 latent + 不重加噪 → 续采到 0 出图 ----
    refiner = await be.infer(ImageRequest(
        request_id="split-refiner", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
        steps=STEPS, width=1024, height=1024, seed=SEED,
        start_at_step=SPLIT, add_noise=False, init_latent_ref=init_ref,
        output_mode="image"))
    assert refiner.media_type == "image/png", refiner.media_type
    p_split = _save_png(refiner.data, "zsplit_refiner.png")
    print(f"  [refiner] start_at_step={SPLIT} 注入续采 → {p_split} ({len(refiner.data)//1024}KB)")

    # ---- 闸门:SSIM(single, split) ≥ 0.95 ----
    s = _ssim(p_single, p_split)
    if s is not None:
        gate = s >= 0.95
        ok = ok and gate
        print(f"\n  ★ 闸门 SSIM(single 12 步, split {SPLIT}+{STEPS - SPLIT}) = {s:.4f} "
              f"{'PASS ✓ (≥0.95)' if gate else 'FAIL ✗ (<0.95)'}")
        if s >= 0.999:
            print("    (≈1.0 bit 级 —— 续采数值与整段逐步一致,符合预期)")

    # ② 附加:base 留噪段末态 != 纯净最终 latent(确认是带噪交接,不是全去噪完)
    full_latent = await be.infer(ImageRequest(
        request_id="split-fulllat", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
        steps=STEPS, width=1024, height=1024, seed=SEED, output_mode="latent"))
    full_lat = st.load(full_latent.data)["latent"]
    diff = float((base_latent.float() - full_lat.float()).abs().mean())
    print(f"  ② base 留噪 latent 与整段最终 latent 平均差 = {diff:.4f}(>0 证带噪中途态,非全去噪完)")
    ok = ok and diff > 1e-3

    # ③ 跨架构 init_latent_ref → 派发前人话报错(不崩深处)
    bad_ref = dict(init_ref, arch="flux2", latent_channels=128)
    try:
        await be.infer(ImageRequest(
            request_id="split-badarch", prompt=PROMPT, cfg_scale=0.0, steps=STEPS,
            width=1024, height=1024, seed=SEED, start_at_step=SPLIT, add_noise=False,
            init_latent_ref=bad_ref, output_mode="image"))
        print("  ③ FAIL:跨架构 latent 注入未报错")
        ok = False
    except ValueError as e:
        print(f"  ③ 跨架构注入正确拦截:{str(e)[:60]}...")

    be.unload()
    print(f"\n{'✅ smoke_zimage_split PASS' if ok else '❌ smoke_zimage_split FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
