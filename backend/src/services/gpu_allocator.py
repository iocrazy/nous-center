from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from src.gpu.detector import detect_gpus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUGroup:
    """A scheduling unit: one runner owns exactly one GPUGroup.

    Parsed from hardware.yaml (spec §3.2). Multi-GPU groups (nvlink=True)
    are NVLink-paired cards used together for tensor-parallel models.
    """

    id: str
    gpus: list[int]
    nvlink: bool
    role: str          # image / llm / tts
    vram_gb: int       # group total VRAM (sum across cards), not per-card


class GPUAllocator:
    def __init__(
        self,
        poll_fn: Callable[[], list[dict]] | None = None,
        hardware_config: dict | None = None,
        preferred_gpus: list[int] | None = None,
    ):
        if poll_fn is None:
            from src.services.gpu_monitor import poll_gpu_stats
            poll_fn = poll_gpu_stats
        self._poll = poll_fn
        # runner 子进程的 group 卡(gpus=[N])。get_best_gpu 优先在这些卡里选 ——
        # 否则 image runner「分配 gpu0」却把模型装到全局最空的 gpu1(撞常驻 LLM 那张卡),
        # 破坏 per-group 隔离。None = 不约束(主进程 allocator / 测试)。
        self._preferred_gpus = list(preferred_gpus) if preferred_gpus else None

        if hardware_config is None:
            from src.config import load_hardware_config
            hardware_config = load_hardware_config()
        self._groups: list[GPUGroup] = self._build_groups(hardware_config)

    # ------------------------------------------------------------------
    # Group topology (Lane A)
    # ------------------------------------------------------------------

    def _build_groups(self, hardware_config: dict) -> list[GPUGroup]:
        """Parse hardware.yaml groups[]. fail-soft: empty config →
        one single-card group per detected GPU; no GPUs → []."""
        raw = hardware_config.get("groups") or []
        if raw:
            groups: list[GPUGroup] = []
            for entry in raw:
                groups.append(
                    GPUGroup(
                        id=entry["id"],
                        gpus=list(entry["gpus"]),
                        nvlink=bool(entry.get("nvlink", False)),
                        role=entry.get("role", "image"),
                        vram_gb=int(entry.get("vram_gb", 0)),
                    )
                )
            return groups

        # Fallback: hardware.yaml missing / empty. Build one single-card
        # group per detected GPU so the rest of V1.5 still has a topology.
        detected = detect_gpus()
        if not detected:
            logger.warning(
                "hardware.yaml has no groups and no GPUs detected — "
                "GPUAllocator.groups() will be empty."
            )
            return []
        logger.warning(
            "hardware.yaml has no groups — falling back to %d detect-based "
            "single-card group(s).", len(detected),
        )
        return [
            GPUGroup(
                id=f"gpu{g.index}",
                gpus=[g.index],
                nvlink=False,
                role="image",
                vram_gb=int(g.vram_total_gb),
            )
            for g in detected
        ]

    def groups(self) -> list[GPUGroup]:
        """All GPU groups parsed from hardware.yaml (or detect fallback)."""
        return list(self._groups)

    def runner_count(self) -> int:
        """How many GPU Runner subprocesses to spawn = number of groups
        (spec §3.2: runner count is not hard-coded)."""
        return len(self._groups)

    def group_by_id(self, group_id: str) -> GPUGroup | None:
        for g in self._groups:
            if g.id == group_id:
                return g
        return None

    def group_for_role(self, role: str) -> GPUGroup | None:
        """First group whose role matches. Used by node dispatch to find
        the GPU group for an image / llm / tts node."""
        for g in self._groups:
            if g.role == role:
                return g
        return None

    def nvlink_groups(self) -> list[GPUGroup]:
        """Groups with nvlink=True. tensor-parallel model spec validation
        requires the model land on one of these (spec §1.3)."""
        return [g for g in self._groups if g.nvlink]

    def llm_group_gpus(self) -> list[int]:
        """GPU indices of the role:llm group — the data source that
        replaces vLLM's self-detection (spec §3.2). Empty if no llm group.
        Lane E consumes this; Lane A only provides it."""
        g = self.group_for_role("llm")
        return list(g.gpus) if g is not None else []

    # ------------------------------------------------------------------
    # Free-VRAM probing (unchanged — model_manager.py:319 + tests depend
    # on this surface; do not break it)
    # ------------------------------------------------------------------

    def get_best_gpu(self, required_vram_mb: float) -> int:
        """装得下的卡里选最空的。preferred_gpus(runner group 卡)优先 —— 先在 group
        内找装得下的;group 内都装不下才 fallback 到全局(spill 出 group,打 warning),
        既守 per-group 隔离又不让大模型因卡太小而无处可装。"""
        stats = self._poll()
        if not stats:
            return -1

        def _pick(pool: list[dict]) -> int:
            cands = [(s["index"], s["free_mb"]) for s in pool if s["free_mb"] >= required_vram_mb]
            if not cands:
                return -1
            cands.sort(key=lambda x: x[1], reverse=True)
            return cands[0][0]

        if self._preferred_gpus is not None:
            in_group = [s for s in stats if s["index"] in self._preferred_gpus]
            pick = _pick(in_group)
            if pick >= 0:
                return pick
            # group 内装不下 → spill 到全局(明确告警,这是隔离被打破的信号)。
            pick = _pick(stats)
            if pick >= 0:
                logger.warning(
                    "get_best_gpu: group %s 内无卡可装 %.0fMB,spill 到 gpu %d(隔离被打破)",
                    self._preferred_gpus, required_vram_mb, pick,
                )
            return pick

        return _pick(stats)

    def get_free_mb(self, gpu_index: int) -> int:
        stats = self._poll()
        for s in stats:
            if s["index"] == gpu_index:
                return s["free_mb"]
        return 0
