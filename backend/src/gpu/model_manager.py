import threading
from typing import Any

from src.gpu.vram_tracker import VRAMTracker


class ModelManager:
    def __init__(
        self,
        model_configs: dict[str, dict],
        gpu_count: int = 2,
        vram_per_gpu_gb: float = 24.0,
    ):
        self._lock = threading.Lock()
        self._configs = model_configs
        self._tracker = VRAMTracker(gpu_count, vram_per_gpu_gb)
        self._loaded: dict[str, Any] = {}  # model_name -> model instance

    def get_model_config(self, model_name: str) -> dict | None:
        return self._configs.get(model_name)

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self._loaded

    def can_load(self, model_name: str) -> bool:
        config = self._configs.get(model_name)
        if config is None:
            return False

        raw_gpu = config.get("gpu")
        if raw_gpu is None:
            # Detector picks at load time; track as GPU 0 for accounting.
            raw_gpu = 0
        gpus = raw_gpu if isinstance(raw_gpu, list) else [raw_gpu]
        vram = config["vram_gb"]

        if config.get("exclusive"):
            # Exclusive models need all GPUs completely free
            for gpu in gpus:
                if self._tracker.get_free(gpu) < self._tracker._vram_total:
                    return False
            return True

        vram_per_gpu = vram / len(gpus)
        return all(self._tracker.get_free(gpu) >= vram_per_gpu for gpu in gpus)

    def register_loaded(self, model_name: str, instance: Any) -> bool:
        config = self._configs.get(model_name)
        if config is None:
            return False

        raw_gpu = config.get("gpu")
        if raw_gpu is None:
            # Detector picks at load time; track as GPU 0 for accounting.
            raw_gpu = 0
        gpus = raw_gpu if isinstance(raw_gpu, list) else [raw_gpu]
        vram_per_gpu = config["vram_gb"] / len(gpus)

        with self._lock:
            for gpu in gpus:
                if not self._tracker.allocate(gpu, model_name, vram_per_gpu):
                    # Rollback
                    for g in gpus:
                        self._tracker.release(g, model_name)
                    return False
            self._loaded[model_name] = instance
            return True

    def unload(self, model_name: str) -> None:
        config = self._configs.get(model_name)
        if config is None or model_name not in self._loaded:
            return

        raw_gpu = config.get("gpu")
        if raw_gpu is None:
            # Detector picks at load time; track as GPU 0 for accounting.
            raw_gpu = 0
        gpus = raw_gpu if isinstance(raw_gpu, list) else [raw_gpu]
        with self._lock:
            for gpu in gpus:
                self._tracker.release(gpu, model_name)
            instance = self._loaded.pop(model_name, None)
            del instance

    def unload_all(self) -> list[str]:
        unloaded = list(self._loaded.keys())
        for name in unloaded:
            self.unload(name)
        return unloaded

    def get_instance(self, model_name: str) -> Any | None:
        return self._loaded.get(model_name)

    def gpu_status(self) -> dict[int, dict]:
        loaded = self._tracker.get_loaded_models()
        return {
            gpu: {
                "free_gb": self._tracker.get_free(gpu),
                "models": models,
            }
            for gpu, models in loaded.items()
        }
