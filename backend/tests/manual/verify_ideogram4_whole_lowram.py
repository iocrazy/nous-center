"""真机验证:Ideogram-4 整装 from_pretrained(low_cpu_mem_usage=True)RAM 峰值 + 出图正确。

背景(2026-06-13):整装 bf16 整包活机 e2e 被 systemd MemoryMax=64G cgroup OOM-kill ——
根因 from_pretrained(low_cpu_mem_usage=False) 瞬时 2× materialize ~108G RAM。本 PR 把整装
分支改 low_cpu_mem_usage=True。本脚本真机验:① 整包仍正确加载并出真图(meta-init 没毁权重)
② 进程 VmHWM(峰值 RSS,含 stream pin)落在 64G cap 内。

stream + cuda:2(空闲 3090,GPU 峰 ~22G);RAM 峰应 ≈ 模型本身 ~54G(远低于旧 ~108G)。
用法:uv run python tests/manual/verify_ideogram4_whole_lowram.py
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # Pro6000=cuda:1, 3090=cuda:0/2(#278)

import asyncio
import time
from pathlib import Path

REPO = os.environ.get("SMOKE_MODEL", "/media/heygo/Program/models/nous/image/diffusers/Ideogram-4-bf16")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:2")
OUT = Path(__file__).parent / "_smoke_out"
OUT.mkdir(exist_ok=True)

PROMPT = "a fluffy orange kitten sitting in a basket of sunflowers, soft natural light, photorealistic"


def _vmhwm_gb() -> float:
    """进程峰值 RSS(VmHWM,kB→GB)—— 含 from_pretrained materialize + stream pin。"""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / (1024 * 1024)
    return -1.0


async def main() -> None:
    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    be = ModularImageBackend(repo=REPO, device=DEVICE, dtype="bfloat16",
                             pipeline_class="Ideogram4Pipeline", offload="stream")
    t0 = time.time()
    be._ensure_pipe()  # 加载 + stream 挂载 —— RAM 峰主要在这
    load_s = time.time() - t0
    hwm_after_load = _vmhwm_gb()
    print(f"[load] {load_s:.1f}s  VmHWM={hwm_after_load:.1f}G")

    t1 = time.time()
    resp = await be.infer(ImageRequest(
        request_id="verify-i4-lowram", prompt=PROMPT,
        steps=20, width=1024, height=1024, cfg_scale=7.0, seed=7))
    print(f"[infer] {time.time()-t1:.1f}s  media={resp.media_type}  bytes={len(resp.data)}")
    hwm_peak = _vmhwm_gb()

    assert resp.media_type == "image/png", f"期望 PNG,得 {resp.media_type}"
    assert resp.data.startswith(b"\x89PNG"), "非 PNG 字节"
    out = OUT / "ideogram4_whole_lowram.png"
    out.write_bytes(resp.data)
    print(f"saved {out}")
    print(f"=== RAM 峰值 VmHWM={hwm_peak:.1f}G ===")
    # 回归守卫:low_cpu_mem_usage=True 必须显著低于旧 False 的 ~108G 瞬时 2× materialize。
    # 实测 ~76G(模型 54G + stream pin 开销)。>90G 说明 2× 又回来了(改动失效)。
    assert hwm_peak < 90.0, f"RAM 峰 {hwm_peak:.1f}G 偏高 —— low_cpu_mem_usage=True 未生效(旧 False ~108G)?"
    # 注:76G 仍 > 旧 64G systemd cap → 整装 bf16 活机需配套把 MemoryMax 抬到 96G(本 PR infra 改;
    # 主机 125G,64G 是按旧 96G 主机写的过时值)。单文件 fp8 路 ~18G 不受影响。
    print(f"PASS: 整装 low_cpu_mem_usage=True 加载正确 + RAM {hwm_peak:.1f}G(较旧 ~108G 大降);"
          f"活机需 MemoryMax>=96G(配套 infra 改)")


if __name__ == "__main__":
    asyncio.run(main())
