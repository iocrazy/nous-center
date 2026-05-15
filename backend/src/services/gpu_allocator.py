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
    ):
        if poll_fn is None:
            from src.services.gpu_monitor import poll_gpu_stats
            poll_fn = poll_gpu_stats
        self._poll = poll_fn

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
        stats = self._poll()
        if not stats:
            return -1
        candidates = [(s["index"], s["free_mb"]) for s in stats if s["free_mb"] >= required_vram_mb]
        if not candidates:
            return -1
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_free_mb(self, gpu_index: int) -> int:
        stats = self._poll()
        for s in stats:
            if s["index"] == gpu_index:
                return s["free_mb"]
        return 0
