from __future__ import annotations

import asyncio
import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.services.inference.base import InferenceAdapter
from src.services.inference.registry import ModelRegistry, ModelSpec
from src.services.gpu_allocator import GPUAllocator

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    spec: ModelSpec
    adapter: InferenceAdapter
    gpu_index: int
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()


class ModelManager:
    """Unified model lifecycle manager: load, unload, evict, reference-count."""

    def __init__(self, registry: ModelRegistry, allocator: GPUAllocator) -> None:
        self._registry = registry
        self._allocator = allocator
        self._models: dict[str, LoadedModel] = {}
        self._references: dict[str, set[str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        return self._locks[model_id]

    def _instantiate_adapter(self, spec: ModelSpec) -> InferenceAdapter:
        """Dynamically import and instantiate adapter from spec.adapter_class dotted path."""
        dotted = spec.adapter_class
        module_path, _, class_name = dotted.rpartition(".")
        if not module_path:
            raise ImportError(
                f"adapter_class '{dotted}' must be a fully-qualified dotted path"
            )
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(spec.path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_loaded(self, model_id: str) -> bool:
        entry = self._models.get(model_id)
        return entry is not None and entry.adapter.is_loaded

    def get_adapter(self, model_id: str) -> InferenceAdapter | None:
        entry = self._models.get(model_id)
        if entry is None:
            return None
        entry.touch()
        return entry.adapter

    def get_references(self, model_id: str) -> set[str]:
        return set(self._references.get(model_id, set()))

    def add_reference(self, model_id: str, ref_id: str) -> None:
        self._references.setdefault(model_id, set()).add(ref_id)

    def remove_reference(self, model_id: str, ref_id: str) -> None:
        refs = self._references.get(model_id)
        if refs:
            refs.discard(ref_id)

    async def load_model(
        self,
        model_id: str,
        adapter_factory: Callable[[ModelSpec], InferenceAdapter] | None = None,
    ) -> None:
        """Load *model_id* onto the best available GPU.

        Parameters
        ----------
        adapter_factory:
            Optional callable ``(ModelSpec) -> InferenceAdapter``.  When
            omitted the adapter is instantiated via dynamic import from
            ``spec.adapter_class``.
        """
        spec = self._registry.get(model_id)
        if spec is None:
            raise ValueError(f"Unknown model: {model_id!r}")

        async with self._lock_for(model_id):
            if self.is_loaded(model_id):
                self._models[model_id].touch()
                return

            # Determine device
            gpu_index = self._allocator.get_best_gpu(spec.vram_mb)
            device = f"cuda:{gpu_index}" if gpu_index >= 0 else "cpu"

            # Build adapter
            if adapter_factory is not None:
                adapter = adapter_factory(spec)
            else:
                adapter = self._instantiate_adapter(spec)

            await adapter.load(device)
            self._models[model_id] = LoadedModel(
                spec=spec,
                adapter=adapter,
                gpu_index=gpu_index,
            )
            self._references.setdefault(model_id, set())
            logger.info("Loaded model %r on %s", model_id, device)

    async def unload_model(self, model_id: str, force: bool = False) -> None:
        """Unload *model_id*.

        If the model is *resident* or has active references it will NOT be
        unloaded unless *force=True*.
        """
        async with self._lock_for(model_id):
            entry = self._models.get(model_id)
            if entry is None:
                return

            if not force:
                if entry.spec.resident:
                    logger.debug("Skipping unload of resident model %r", model_id)
                    return
                refs = self._references.get(model_id, set())
                if refs:
                    logger.debug(
                        "Skipping unload of referenced model %r (refs=%s)",
                        model_id,
                        refs,
                    )
                    return

            entry.adapter.unload()
            del self._models[model_id]
            logger.info("Unloaded model %r", model_id)

    @property
    def loaded_model_ids(self) -> list[str]:
        return [mid for mid, entry in self._models.items() if entry.adapter.is_loaded]

    def get_status(self) -> dict:
        """Return current model manager status."""
        return {
            "loaded": self.loaded_model_ids,
            "references": {k: list(v) for k, v in self._references.items() if v},
            "last_used": {
                mid: entry.last_used for mid, entry in self._models.items()
            },
        }

    def get_model_dependencies(self, workflow: dict) -> list[dict]:
        """Extract model dependencies from workflow nodes."""
        deps: list[dict] = []
        seen: set[str] = set()
        for node in workflow.get("nodes", []):
            node_type = node.get("type", "")
            data = node.get("data", {})
            model_key: str | None = None
            if node_type == "tts_engine":
                model_key = data.get("engine")
            elif node_type == "llm":
                model_key = data.get("model_key")
            if model_key and model_key not in seen:
                spec = self._registry.get(model_key)
                if spec is not None:
                    seen.add(model_key)
                    deps.append({"key": model_key, "type": spec.model_type})
        return deps

    async def check_idle_models(self) -> None:
        """Unload models that have been idle too long with no references."""
        now = time.monotonic()
        to_unload: list[str] = []
        for mid, entry in list(self._models.items()):
            if entry.spec.resident:
                continue
            if self._references.get(mid):
                continue
            ttl = entry.spec.ttl_seconds
            if ttl <= 0:
                continue
            if now - entry.last_used > ttl:
                to_unload.append(mid)
        for mid in to_unload:
            logger.info("TTL expired: unloading %s", mid)
            await self.unload_model(mid)

    async def evict_lru(self, gpu_index: int | None = None) -> str | None:
        """Evict the least-recently-used non-resident, non-referenced model.

        Parameters
        ----------
        gpu_index:
            When provided, only consider models loaded on that GPU.

        Returns
        -------
        The evicted model_id, or ``None`` if nothing was evicted.
        """
        candidates = [
            entry
            for mid, entry in self._models.items()
            if not entry.spec.resident
            and not self._references.get(mid)
            and (gpu_index is None or entry.gpu_index == gpu_index)
        ]

        if not candidates:
            return None

        lru = min(candidates, key=lambda e: e.last_used)
        model_id = lru.spec.id
        await self.unload_model(model_id, force=True)
        logger.info("Evicted LRU model %r from gpu %s", model_id, lru.gpu_index)
        return model_id
