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
                ttl_seconds=entry.get("ttl_seconds", 3600 if entry["type"] == "llm" else 300),
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

    def add_from_scan(self, model_id: str) -> ModelSpec | None:
        """Synthesize a ModelSpec from the on-disk scanner output.

        Used as a load-time fallback when `model_id` wasn't in models.yaml
        at startup. Auto-detected LLM/VL entries fill `adapter` (always
        VLLMAdapter); image/video do not — those return None here so the
        caller raises ValueError instead of silently registering an
        unloadable spec.

        Imported locally to avoid a startup-time circular import between
        registry and model_scanner.
        """
        from src.services.model_scanner import scan_models

        configs = scan_models()
        cfg = configs.get(model_id)
        if cfg is None:
            return None
        adapter = cfg.get("adapter")
        if not adapter:
            return None
        spec = ModelSpec(
            id=model_id,
            model_type=cfg.get("type", "llm"),
            adapter_class=adapter,
            path=cfg.get("local_path", ""),
            vram_mb=int(round(cfg.get("vram_gb", 0) * 1024)),
            params=cfg.get("params", {}),
            resident=cfg.get("resident", False),
            ttl_seconds=cfg.get("ttl_seconds", 3600 if cfg.get("type") == "llm" else 300),
            gpu=cfg.get("gpu"),
        )
        self._specs[spec.id] = spec
        logger.info(
            "Auto-registered model spec from scan: %s (%s, %s)",
            spec.id, spec.model_type, spec.adapter_class,
        )
        return spec
