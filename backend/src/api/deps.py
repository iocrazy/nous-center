from functools import lru_cache

from src.storage.nas import StorageService


@lru_cache
def get_storage() -> StorageService:
    return StorageService()
