from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    model_type: str
    adapter_class: str
    path: str
    vram_mb: int
    params: dict[str, Any] = field(default_factory=dict)
    resident: bool = False
    ttl_seconds: int = 300
    gpu: int | list[int] | None = None


class ModelRegistry:
    def __init__(self, config_path: str):
        self._config_path = config_path
        self._specs: dict[str, ModelSpec] = {}
        self._load(config_path)

    def reload(self) -> int:
        """Hot-reload config from disk. Returns number of new specs added."""
        old_ids = set(self._specs.keys())
        self._load(self._config_path)
        new_ids = set(self._specs.keys()) - old_ids
        if new_ids:
            logger.info("Registry reloaded, new models: %s", new_ids)
        return len(new_ids)

    def _load(self, config_path: str) -> None:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        for entry in data.get("models", []):
            spec = ModelSpec(
                id=entry["id"],
                model_type=entry["type"],
                adapter_class=entry["adapter"],
                path=entry.get("path", ""),
                vram_mb=entry.get("vram_mb", 0),
                params=entry.get("params", {}),
                resident=entry.get("resident", False),
                ttl_seconds=entry.get("ttl_seconds", 300),
                gpu=entry.get("gpu"),
            )
            self._specs[spec.id] = spec
            logger.debug("Loaded model spec: %s (%s)", spec.id, spec.model_type)

    @property
    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def get(self, model_id: str) -> ModelSpec | None:
        return self._specs.get(model_id)

    def list_by_type(self, model_type: str) -> list[ModelSpec]:
        return [s for s in self._specs.values() if s.model_type == model_type]
