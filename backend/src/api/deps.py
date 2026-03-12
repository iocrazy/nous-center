from functools import lru_cache

from src.config import load_model_configs
from src.gpu.detector import get_gpus
from src.gpu.model_manager import ModelManager
from src.storage.nas import StorageService


@lru_cache
def get_model_manager() -> ModelManager:
    configs = load_model_configs()
    gpus = get_gpus()
    gpu_count = max(len(gpus), 1)
    vram_per_gpu = gpus[0].vram_total_gb if gpus else 0.0
    return ModelManager(configs, gpu_count=gpu_count, vram_per_gpu_gb=vram_per_gpu)


@lru_cache
def get_storage() -> StorageService:
    return StorageService()
