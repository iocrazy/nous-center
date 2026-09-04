"""子进程型推理适配器(vLLM / SGLang)共用的落卡逻辑。

**适配器不做放置决策** —— 决定"这个模型落哪些卡"的地方只有
`ModelManager._resolve_placement` 一处(2026-09-03 审查)。这里只负责把 manager 传下来的
`device` / `gpus` 翻译成:钉哪几张物理卡、tp 是多少、按哪组卡算显存预算。

三条不变式:
  1. **绝不存在"不钉卡"的启动分支**。老代码在 tp>1 时不设 `CUDA_VISIBLE_DEVICES`,
     子进程继承父进程环境看到全部卡,于是 vLLM 自己按 index 顺序抓前 tp 张 ——
     在 3090 + PRO 6000 + 3090 的机器上就是异构混跑(2026-09-02 事故的形状)。
  2. **tp > 1 必须伴随一个显式的多卡组**。要 tp>1 却没组 → 退单卡(tp=1)并 log error,
     绝不拿"全部可见卡"顶上。
  3. **预算按最终真正钉的那组卡算**。此前回退分支会把 `gpu_idx` 改成"全局最空闲卡"
     再按它算 utilization,而钉卡时 `device` 优先 —— 按 96G 的 Pro 6000 算出的比例
     用在 24G 的 3090 上,启动即 OOM,日志还说"退回 cuda:1"误导排障。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaunchPlacement:
    """一次子进程启动的落卡结论。"""

    cards: list[int]        # 要钉进 CUDA_VISIBLE_DEVICES 的物理 index(可能为空 = 无 GPU 信息)
    tp: int                 # --tensor-parallel-size / --tp(1 = 不传)
    total_gb: float         # 组内最小 total(预算分母)
    free_gb: float          # 组内最小 free(clamp 用)

    @property
    def primary(self) -> int:
        return self.cards[0] if self.cards else 0

    @property
    def visible_devices(self) -> str | None:
        return ",".join(str(i) for i in self.cards) if self.cards else None


def estimate_model_size_gb(model_dir: Path) -> float:
    """模型权重体积(GB)—— safetensors 优先,退回 .bin。目录不存在 → 0.0。

    manager 与适配器共用同一把尺子:manager 拿它决定放置(单卡装不下才考虑组),
    适配器拿它算 kv buffer / max_model_len。两边算出不同的数会让"决定"和"执行"对不上。
    """
    try:
        if not model_dir.exists():
            return 0.0
        total = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
        if total == 0:
            total = sum(f.stat().st_size for f in model_dir.glob("*.bin"))
        return total / (1024 ** 3)
    except OSError:
        return 0.0


def resolve_launch_placement(
    *,
    device: str | None,
    gpus: list[int] | None,
    requested_tp: int | None,
    stats: list[dict] | None = None,
    label: str = "vLLM",
) -> LaunchPlacement:
    """把 manager 给的 device/gpus 翻译成最终落卡 + tp + 预算。

    优先级:`gpus`(多卡组)> `device="cuda:N"` > 最空闲的一张卡。
    `requested_tp` 只能**收窄**组(取组内前 N 张),不能超过组内卡数;没有组时
    `requested_tp > 1` 一律降级成 1(不变式 2)。
    """
    from src.gpu.topology import group_budget_gb  # noqa: PLC0415

    if stats is None:
        from src.services.gpu_monitor import poll_gpu_stats  # noqa: PLC0415

        stats = poll_gpu_stats()

    cards: list[int] = []
    if device == "cpu":
        # manager 明确判了 CPU —— 一张卡都不给(下面 apply_visible_devices 会钉成 "")。
        return LaunchPlacement(cards=[], tp=1, total_gb=0.0, free_gb=0.0)
    if gpus:
        cards = [int(i) for i in gpus]
        # 显式 tensor_parallel_size 只能**收窄**组(取前 N 张);requested_tp=1 就是
        # "组里只用第一张"= 退化成单卡。永远不能超过组内卡数。
        if requested_tp and 0 < requested_tp < len(cards):
            cards = cards[:requested_tp]
    elif device and ":" in device:
        try:
            cards = [int(device.split(":")[-1])]
        except ValueError:
            cards = []
    if not cards and stats:
        # manager 没给 device(外部服务/重连路径)→ 兜底钉最空闲的一张,仍然是"钉住"。
        cards = [max((int(s["index"]) for s in stats), key=lambda i: _free(stats, i))]

    tp = len(cards) if len(cards) > 1 else 1
    if requested_tp and requested_tp > 1 and tp <= 1:
        logger.error(
            "%s: 配置要求 tensor_parallel_size=%d,但没有可用的 GPU 组 —— 退回单卡 "
            "(tp=1) 并钉住 %s。绝不拿'全部可见卡'顶上(会混异构卡)。"
            "如需张量并行,请在 hardware.yaml 声明多卡组并给模型配 gpus:[...]",
            label, requested_tp, cards or "(无 GPU)",
        )
    total_gb, free_gb = group_budget_gb(cards, stats)
    return LaunchPlacement(cards=cards, tp=tp, total_gb=total_gb, free_gb=free_gb)


def apply_visible_devices(env: dict, placement: LaunchPlacement, cmd: list[str], label: str) -> None:
    """把落卡结论写进子进程环境 + 打启动日志。

    **任何情况下都会写 `CUDA_VISIBLE_DEVICES`**(不变式 1)。没有卡可钉时写空串 ——
    "让子进程继承父进程环境、看到全部 GPU" 正是 2026-09-02 事故的形状,宁可让它
    起不来也不能放它去摸所有卡。
    """
    visible = placement.visible_devices
    if visible is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
        logger.error(
            "%s: 没有可用的 GPU 落卡结论 —— 钉 CUDA_VISIBLE_DEVICES=\"\"(绝不放它看全部卡)。"
            "命令:%s", label, " ".join(cmd),
        )
        return
    env["CUDA_VISIBLE_DEVICES"] = visible
    if placement.tp > 1:
        logger.info("Starting %s on GPU group %s (TP=%d): %s",
                    label, placement.cards, placement.tp, " ".join(cmd))
    else:
        logger.info("Starting %s on GPU %s: %s", label, visible, " ".join(cmd))


def _free(stats: list[dict], idx: int) -> float:
    for s in stats:
        if int(s["index"]) == idx:
            return float(s.get("free_mb") or 0)
    return 0.0
