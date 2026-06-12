"""B1 真机:batch 出图(num_images>1,标准 pipe 经 num_images_per_prompt 一次出 N 张)。

验:① num_images=3 → InferenceResult.data(首张)+ extra_images(2 张)= 共 3 张;② 3 张各为合法
非损坏 PNG;③ 3 张互不相同(SSIM<0.99 两两 → 真出了不同变体,非复制);④ num_images=1 仍单图
(extra_images 空,零回归)。Z-Image euler 走标准 pipe(非段路)。⑤ 段路纯生成 batch(euler_ancestral
num_images=3)。⑥ 段路 img2img 续采 batch(init_latent + add_noise=enable + num_images=3 → 3 张变体);
⑦ 段路留噪续采(add_noise=disable)单 latent + num_images=3 → 守卫退回 1 张。

用法:cd backend && SMOKE_DEVICE=cuda:2 uv run python tests/manual/smoke_image_batch.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = os.environ.get("SMOKE_ZIMAGE", "/media/heygo/Program/models/nous/image/diffusers/Z-Image-Turbo")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:2")
PROMPT = "a photo of a red fox in autumn leaves, sharp focus"
STEPS, SEED = 12, 42
SIZE = int(os.environ.get("SMOKE_SIZE", "512"))  # batch×VAE decode 吃显存,3090 用 512 稳


def _valid_png(b: bytes) -> bool:
    try:
        from PIL import Image
        Image.open(io.BytesIO(b)).verify()
        return len(b) > 10000
    except Exception:  # noqa: BLE001
        return False


def _ssim(a: bytes, b: bytes):
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
        ia = np.asarray(Image.open(io.BytesIO(a)).convert("L"))
        ib = np.asarray(Image.open(io.BytesIO(b)).convert("L"))
        return float(ssim(ia, ib))
    except Exception:  # noqa: BLE001
        return None


async def main() -> int:
    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    be = ModularImageBackend(repo=REPO, device=DEVICE, dtype="bfloat16", pipeline_class="ZImagePipeline")
    await be.load(DEVICE)
    ok = True

    # ① num_images=3
    r = await be.infer(ImageRequest(request_id="batch-3", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                                    steps=STEPS, width=SIZE, height=SIZE, seed=SEED, num_images=3))
    blobs = [r.data, *r.extra_images]
    print(f"  num_images=3 → data + extra_images = {len(blobs)} 张 (image_count={r.usage.image_count})")
    ok = ok and len(blobs) == 3
    for i, b in enumerate(blobs):
        v = _valid_png(b)
        print(f"    [{i}] {len(b)//1024}KB  {'✓合法' if v else 'FAIL'}")
        ok = ok and v
        Path(tempfile.gettempdir(), f"batch_{i}.png").write_bytes(b)

    # ③ 两两互异
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            s = _ssim(blobs[i], blobs[j])
            if s is not None:
                print(f"    SSIM({i},{j})={s:.4f} → {'异 ✓' if s < 0.99 else 'FAIL 相同'}")
                ok = ok and s < 0.99

    # ④ num_images=1 零回归(单图,extra 空)
    r1 = await be.infer(ImageRequest(request_id="batch-1", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                                     steps=STEPS, width=SIZE, height=SIZE, seed=SEED, num_images=1))
    solo = _valid_png(r1.data) and not r1.extra_images
    print(f"  ④ num_images=1 → 单图 + extra_images 空: {solo}")
    ok = ok and solo

    # ⑤ 段路 batch(非 euler 采样器走手写分段循环 prepare_latents(N)):euler_ancestral num_images=3 → 3 张互异。
    rs = await be.infer(ImageRequest(request_id="batch-seg", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                                     steps=STEPS, width=SIZE, height=SIZE, seed=SEED, num_images=3,
                                     sampler_name="euler_ancestral"))
    seg_blobs = [rs.data, *rs.extra_images]
    seg_ok = len(seg_blobs) == 3 and all(_valid_png(b) for b in seg_blobs)
    print(f"  ⑤ 段路 euler_ancestral num_images=3 → {len(seg_blobs)} 张 (image_count={rs.usage.image_count}) {'✓' if seg_ok else 'FAIL'}")
    ok = ok and seg_ok
    if len(seg_blobs) == 3:
        s = _ssim(seg_blobs[0], seg_blobs[1])
        if s is not None:
            print(f"    SSIM(0,1)={s:.4f} → {'异 ✓' if s < 0.99 else 'FAIL 相同'}")
            ok = ok and s < 0.99

    # ⑥ 段路 batch 续采(init_latent img2img,add_noise=enable):单 init latent 复制 N 份 + N 个独立噪声
    #    → N 张变体。先导出一个 latent_ref(output_mode=latent),再喂回作 img2img 续采 num_images=3。
    exp = await be.infer(ImageRequest(request_id="export-lat", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                                      steps=STEPS, width=SIZE, height=SIZE, seed=SEED,
                                      sampler_name="euler_ancestral", output_mode="latent"))
    lat_path = Path(tempfile.gettempdir(), "seg_batch_init.safetensors")
    lat_path.write_bytes(exp.data)
    lat_meta = exp.metadata["latent"]
    print(f"  ⑥ 导出 latent_ref shape={lat_meta['shape']} arch={lat_meta['arch']}")
    ref = {"path": str(lat_path), "arch": lat_meta["arch"], "latent_channels": lat_meta["latent_channels"]}
    ri = await be.infer(ImageRequest(request_id="seg-i2i-batch", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                                     steps=STEPS, width=SIZE, height=SIZE, seed=SEED, num_images=3,
                                     sampler_name="euler_ancestral", init_latent_ref=ref,
                                     add_noise=True, start_at_step=6))
    i2i_blobs = [ri.data, *ri.extra_images]
    i2i_ok = len(i2i_blobs) == 3 and all(_valid_png(b) for b in i2i_blobs)
    print(f"     段路 img2img num_images=3 → {len(i2i_blobs)} 张 (image_count={ri.usage.image_count}) {'✓' if i2i_ok else 'FAIL'}")
    ok = ok and i2i_ok
    for i in range(len(i2i_blobs)):
        for j in range(i + 1, len(i2i_blobs)):
            s = _ssim(i2i_blobs[i], i2i_blobs[j])
            if s is not None:
                print(f"     SSIM({i},{j})={s:.4f} → {'异 ✓' if s < 0.99 else 'FAIL 相同'}")
                ok = ok and s < 0.99

    # ⑦ 段路留噪续采(add_noise=disable)单 latent + num_images=3 → 退回 1 张(无加噪无变体源,守卫)。
    rr = await be.infer(ImageRequest(request_id="seg-relay-batch", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                                     steps=STEPS, width=SIZE, height=SIZE, seed=SEED, num_images=3,
                                     sampler_name="euler_ancestral", init_latent_ref=ref,
                                     add_noise=False, start_at_step=6))
    relay_ok = _valid_png(rr.data) and not rr.extra_images
    print(f"  ⑦ 段路留噪续采单 latent num_images=3 → 退回 1 张(extra 空): {relay_ok}")
    ok = ok and relay_ok

    be.unload()
    print(f"\n{'✅ smoke_image_batch PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
