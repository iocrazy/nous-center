from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ModelSpec(BaseModel):
    """Static model registry entry (yaml-loaded or scanner-synthesized).

    Frozen — instances are stored in registry dict + adapter instantiation
    closures, must be hashable / immutable.
    """

    id: str
    model_type: str
    adapter_class: str
    paths: dict[str, str]  # multi-component paths; single-component uses {"main": "..."}
    vram_mb: int
    params: dict[str, Any] = Field(default_factory=dict)
    resident: bool = False
    ttl_seconds: int = 300
    gpu: int | list[int] | None = None
    preload_order: int | None = None

    model_config = ConfigDict(frozen=True)


def _coerce_paths(entry: dict[str, Any]) -> dict[str, str]:
    """Read `paths` block from a yaml entry; legacy `path` falls into paths['main']."""
    paths = entry.get("paths") or {}
    if not paths and entry.get("path"):
        paths = {"main": entry["path"]}
    return paths


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
        # 模型定义走 collect_model_entries(models.d/*.yaml + models.yaml,单一来源;
        # spec 2026-06-20 一模型一文件)—— 与 load_model_configs 同一入口,口径不分叉。
        # 运行时覆盖单一来源(spec 2026-06-16「数据加载统一」):registry 历史上直读 yaml,
        # 绕过 runtime_overrides.json overlay → overlay 的 gpu/resident 对 vLLM 落卡不生效
        # (落卡读 spec.gpu)。这里套用 overlay,使 overlay 成为运行时覆盖的唯一来源。
        # 局部 import 避免 registry↔config 启动期循环(同 add_from_scan)。
        from pathlib import Path as _Path  # noqa: PLC0415
        from src.config import collect_model_entries, load_runtime_overrides  # noqa: PLC0415
        overrides = load_runtime_overrides()
        for entry in collect_model_entries(_Path(config_path)):
            ov = overrides.get(entry["id"]) or {}
            spec = ModelSpec(
                id=entry["id"],
                model_type=entry["type"],
                adapter_class=entry["adapter"],
                paths=_coerce_paths(entry),
                vram_mb=entry.get("vram_mb", 0),
                params=entry.get("params", {}),
                resident=ov.get("resident", entry.get("resident", False)),
                ttl_seconds=entry.get("ttl_seconds", 3600 if entry["type"] == "llm" else 300),
                gpu=ov.get("gpu", entry.get("gpu")),
                preload_order=entry.get("preload_order"),
            )
            self._specs[spec.id] = spec
            logger.debug("Loaded model spec: %s (%s)", spec.id, spec.model_type)

    def set_resident(self, model_id: str, resident: bool) -> bool:
        """翻内存 spec 的常驻位。ModelSpec 是 frozen → model_copy 换新对象。

        PATCH /engines/{name}/resident 持久化走 DB override(重启后 _load 叠加),
        这里补运行期同步 —— 否则 /health 的 resident 统计和 preload_residents
        名单读的是启动时快照,与 UI 显示的 override 脱节直到重启。
        """
        spec = self._specs.get(model_id)
        if spec is None:
            return False
        self._specs[model_id] = spec.model_copy(update={"resident": resident})
        return True

    @property
    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def get(self, model_id: str) -> ModelSpec | None:
        return self._specs.get(model_id)

    def list_by_type(self, model_type: str) -> list[ModelSpec]:
        return [s for s in self._specs.values() if s.model_type == model_type]

    def add_from_scan(self, model_id: str) -> ModelSpec | None:
        """Synthesize a ModelSpec from the on-disk scanner output.

        Used as a load-time fallback when `model_id` wasn't in the model
        configs (`configs/models.d/*.yaml` + legacy models.yaml) at startup.
        Auto-detected LLM/VL entries fill `adapter` (always
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
        # Scanner emits `local_path` for single-component models; image/video
        # emit `paths` dict directly when known.
        paths = cfg.get("paths") or {}
        if not paths and cfg.get("local_path"):
            paths = {"main": cfg["local_path"]}
        # scan_models() 不并 overlay → 同 _load 套用,保持运行时覆盖单一来源(2026-06-16 统一)。
        from src.config import load_runtime_overrides  # noqa: PLC0415
        ov = load_runtime_overrides().get(model_id) or {}
        spec = ModelSpec(
            id=model_id,
            model_type=cfg.get("type", "llm"),
            adapter_class=adapter,
            paths=paths,
            vram_mb=int(round(cfg.get("vram_gb", 0) * 1024)),
            params=cfg.get("params", {}),
            resident=ov.get("resident", cfg.get("resident", False)),
            ttl_seconds=cfg.get("ttl_seconds", 3600 if cfg.get("type") == "llm" else 300),
            gpu=ov.get("gpu", cfg.get("gpu")),
            preload_order=cfg.get("preload_order"),
        )
        self._specs[spec.id] = spec
        logger.info(
            "Auto-registered model spec from scan: %s (%s, %s)",
            spec.id, spec.model_type, spec.adapter_class,
        )
        return spec
