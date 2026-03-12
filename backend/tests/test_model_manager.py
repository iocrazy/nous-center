import pytest
from unittest.mock import MagicMock, patch
from src.gpu.model_manager import ModelManager


@pytest.fixture
def model_configs():
    return {
        "sdxl": {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "type": "image",
            "gpu": 0,
            "vram_gb": 10,
            "resident": True,
        },
        "cosyvoice2": {
            "name": "CosyVoice2-0.5B",
            "type": "tts",
            "gpu": 1,
            "vram_gb": 3,
            "resident": True,
        },
        "wan21": {
            "name": "Wan2.1",
            "type": "video",
            "gpu": [0, 1],
            "vram_gb": 40,
            "resident": False,
            "exclusive": True,
        },
    }


def test_manager_init(model_configs):
    manager = ModelManager(model_configs, gpu_count=2, vram_per_gpu_gb=24)
    assert manager.is_loaded("sdxl") is False
    status = manager.gpu_status()
    assert status[0]["free_gb"] == 24.0


def test_can_load(model_configs):
    manager = ModelManager(model_configs, gpu_count=2, vram_per_gpu_gb=24)
    assert manager.can_load("sdxl") is True
    # wan21 is exclusive and GPUs are empty (free == total), so it CAN load
    assert manager.can_load("wan21") is True


def test_can_load_exclusive_blocked(model_configs):
    """Exclusive model cannot load when a GPU is partially occupied."""
    manager = ModelManager(model_configs, gpu_count=2, vram_per_gpu_gb=24)
    # Load sdxl on GPU 0 so it's no longer fully free
    manager.register_loaded("sdxl", MagicMock())
    assert manager.can_load("wan21") is False


def test_get_model_config(model_configs):
    manager = ModelManager(model_configs, gpu_count=2, vram_per_gpu_gb=24)
    config = manager.get_model_config("sdxl")
    assert config["type"] == "image"
    assert manager.get_model_config("nonexistent") is None
