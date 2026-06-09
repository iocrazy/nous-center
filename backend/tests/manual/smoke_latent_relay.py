"""路 B PR-B1 真机冒烟:VAE Decode output_mode=latent 导出真 latent + 落盘 roundtrip。standalone,需 GPU。

验:① latent 模式返 media_type=application/x-latent + 可反序列化的真 latent 张量(shape/通道与 metadata 一致);
② 同 seed 两次 latent 模式字节级一致(确定性);③ safetensors 落盘→读回 bit-exact;④ image 模式仍出正常图
(默认路径零回归)。Z-Image latent = 16ch(复用 Flux1 latent 空间)。

注:本 smoke 验「latent 导出+落盘」这一层;latent→sample_from_latent→出图的完整接力由 PR-B2 验。
image_modular.py 改了 → 另跑 tests/manual/smoke_image_ab.py 守 Flux2 golden SSIM≥0.97。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_latent_relay.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = os.environ.get("SMOKE_ZIMAGE", "/media/heygo/Program/models/nous/image/diffusers/Z-Image-Turbo")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
PROMPT = "a photo of a red fox in autumn leaves, sharp focus"


async def main() -> int:
    import safetensors.torch as st
    import torch  # noqa: F401

    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    be = ModularImageBackend(repo=REPO, device=DEVICE, dtype="bfloat16", pipeline_class="ZImagePipeline")
    await be.load(DEVICE)

    def _req(mode: str) -> ImageRequest:
        return ImageRequest(request_id=f"latrelay-{mode}", prompt=PROMPT, cfg_scale=4.0,
                            negative_prompt="", steps=8, width=1024, height=1024, seed=42, output_mode=mode)

    # ① latent 模式
    r1 = await be.infer(_req("latent"))
    assert r1.media_type == "application/x-latent", f"media_type={r1.media_type}"
    meta = (r1.metadata or {}).get("latent") or {}
    t = st.load(r1.data)["latent"]
    print(f"  [latent] media_type={r1.media_type} shape={list(t.shape)} arch={meta.get('arch')} "
          f"channels={meta.get('latent_channels')} bytes={len(r1.data)//1024}KB")
    assert list(t.shape) == meta.get("shape"), "返回张量 shape 与 metadata 不一致"
    assert meta.get("arch") == "z-image"

    # ② 确定性:同 seed 再来一次,字节一致
    r2 = await be.infer(_req("latent"))
    same = r1.data == r2.data
    print(f"  [determinism] 同 seed 两次 latent 字节一致: {same}")

    # ③ 落盘 roundtrip bit-exact
    os.environ.setdefault("NOUS_IMAGE_OUTPUTS", str(Path(__file__).parent / "_smoke_out" / "images"))
    from src.services.latent_storage import write_latent
    rec = write_latent(r1.data)
    roundtrip = Path(rec["path"]).read_bytes() == r1.data
    print(f"  [roundtrip] 落盘读回 bit-exact: {roundtrip} → {rec['path']}")

    # ④ image 模式仍出图(默认路径零回归)
    r3 = await be.infer(_req("image"))
    out = Path(__file__).parent / "_smoke_out" / "smoke_latent_relay_image.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r3.data)
    img_ok = r3.media_type == "image/png" and len(r3.data) > 1000
    print(f"  [image] image 模式出图 {r3.media_type} {len(r3.data)//1024}KB → {out.name}")

    ok = same and roundtrip and img_ok and list(t.shape) == meta.get("shape")
    print("PASS — latent 导出 + 落盘 roundtrip + image 零回归" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
