import threading


class VRAMTracker:
    def __init__(self, gpu_count: int = 2, vram_per_gpu_gb: float = 24.0):
        self._lock = threading.Lock()
        self._gpu_count = gpu_count
        self._vram_total = vram_per_gpu_gb
        # {gpu_id: {model_name: vram_gb}}
        self._allocated: dict[int, dict[str, float]] = {
            i: {} for i in range(gpu_count)
        }

    def get_free(self, gpu: int) -> float:
        with self._lock:
            used = sum(self._allocated[gpu].values())
            return self._vram_total - used

    def allocate(self, gpu: int, model_name: str, vram_gb: float) -> bool:
        with self._lock:
            used = sum(self._allocated[gpu].values())
            if used + vram_gb > self._vram_total:
                return False
            self._allocated[gpu][model_name] = vram_gb
            return True

    def release(self, gpu: int, model_name: str) -> None:
        with self._lock:
            self._allocated[gpu].pop(model_name, None)

    def release_all(self) -> None:
        with self._lock:
            for gpu in self._allocated:
                self._allocated[gpu].clear()

    def get_loaded_models(self) -> dict[int, list[tuple[str, float]]]:
        with self._lock:
            return {
                gpu: [(name, vram) for name, vram in models.items()]
                for gpu, models in self._allocated.items()
            }
