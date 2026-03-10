from functools import lru_cache

from src.config import load_model_configs
from src.gpu.model_manager import ModelManager
from src.storage.nas import StorageService


@lru_cache
def get_model_manager() -> ModelManager:
    configs = load_model_configs()
    return ModelManager(configs, gpu_count=2, vram_per_gpu_gb=24.0)


@lru_cache
def get_storage() -> StorageService:
    return StorageService()
