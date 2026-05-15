"""F2 GPU-free 探针 —— runner 重启前确认 GPU 显存已回落（spec 4.2）。

spec 4.2 F2：runner crash（OOM / native fault）后，死进程的 CUDA context 回收
是**异步**的 —— nvidia-smi 可能要几秒才反映显存释放。RunnerSupervisor 的
RESTART_BACKOFF `[5,15,60,300]` 是「防 crash 风暴」的退避，**不保证** context
已清。所以重启前必须额外过这道 gate：轮询 nvidia-smi，直到该 group 的所有 GPU
的 free_mb 回落到基线，才 re-fork runner + re-preload resident 模型。

本模块产出注入给 RunnerSupervisor 的 `gpu_free_probe` 回调（Lane C 已留注入点，
`_default_gpu_free_probe` 是它的骨架，本模块换成真实现）。

无 GPU 环境（CUDA_VISIBLE_DEVICES='' 测试 / nvidia-smi 不可用）：poll_gpu_stats
返回空 → 探针保守返回 True，不阻塞重启（与 Lane C 骨架行为一致）。
"""
from __future__ import annotations

import logging
from typing import Callable

from src.services.gpu_monitor import poll_gpu_stats

logger = logging.getLogger(__name__)

# 默认基线：每卡 total 显存的 80%。空载的 GPU 应有 >=80% free；低于此
# 说明上一个进程的 CUDA context 还没回收干净。
_DEFAULT_BASELINE_FRACTION = 0.8


def make_gpu_free_probe(
    baseline_free_mb: int | None = None,
) -> Callable[[list[int]], bool]:
    """构造一个 GPU-free 探针 —— 传给 RunnerSupervisor 的 gpu_free_probe 参数。

    Parameters
    ----------
    baseline_free_mb:
        每张目标 GPU 的 free_mb 必须 >= 此值才算「free」。None → 对每张卡用
        `total_mb * 0.8` 动态算（不同显存的卡各用自己的基线）。

    Returns
    -------
    `probe(gpus: list[int]) -> bool`：传入该 group 的 GPU index 列表，全部回落
    到基线返回 True，否则 False。nvidia-smi 不可用 / 目标 GPU 缺失 → 保守 True
    （宁可早重启，也不无限等一个查不到状态的 GPU 把重启循环卡死）。
    """

    def _probe(gpus: list[int]) -> bool:
        stats = poll_gpu_stats()
        if not stats:
            # nvidia-smi 不可用 / 无 GPU —— 保守放行（不阻塞重启）。
            return True
        by_index = {s["index"]: s for s in stats}
        for gpu in gpus:
            gpu_stat = by_index.get(gpu)
            if gpu_stat is None:
                # 目标 GPU 不在 stats 里 —— 不可判定，保守放行。
                logger.warning(
                    "GPU-free probe: GPU %d not in nvidia-smi output, skipping gate",
                    gpu,
                )
                continue
            if baseline_free_mb is not None:
                threshold = baseline_free_mb
            else:
                threshold = int(gpu_stat["total_mb"] * _DEFAULT_BASELINE_FRACTION)
            if gpu_stat["free_mb"] < threshold:
                logger.info(
                    "GPU-free probe: GPU %d free=%dMB < baseline=%dMB, not yet free",
                    gpu, gpu_stat["free_mb"], threshold,
                )
                return False
        return True

    return _probe
