"""Lane J test infrastructure: hardware.yaml topology fixture (spec §5.6).

Provides 2gpu / 3gpu hardware.yaml content dicts + a write_hardware_yaml
helper + pytest fixtures that point src.config.load_hardware_config at a
temp file.

2gpu  = spec §1.4 plan A: single group llm-tp (current 2x3090 deployment).
3gpu  = spec §3.2 hardware.3gpu.yaml (Pro 6000 future layout).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

HARDWARE_2GPU: dict[str, Any] = {
    "groups": [
        {
            "id": "llm-tp",
            "gpus": [0, 1],
            "nvlink": True,
            "role": "llm",  # image/TTS nodes also run here, time-share with LLM
            "vram_gb": 48,
        },
    ],
}

HARDWARE_3GPU: dict[str, Any] = {
    "groups": [
        {"id": "image", "gpus": [2], "nvlink": False, "role": "image", "vram_gb": 96},
        {"id": "llm-tp", "gpus": [0, 1], "nvlink": True, "role": "llm", "vram_gb": 48},
        {"id": "tts", "gpus": [3], "nvlink": False, "role": "tts", "vram_gb": 24},
    ],
}


def write_hardware_yaml(dir_path: Path, content: dict[str, Any]) -> Path:
    """Write a hardware.yaml file under dir_path and return its path."""
    path = Path(dir_path) / "hardware.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


def _clear_loader_cache() -> None:
    """Clear load_hardware_config's lru_cache so subsequent calls see new paths."""
    from src import config as _config

    if hasattr(_config.load_hardware_config, "cache_clear"):
        _config.load_hardware_config.cache_clear()


@pytest.fixture
def hardware_2gpu(tmp_path):
    """Write a 2gpu hardware.yaml under tmp_path, return its path.

    load_hardware_config(path=...) accepts an explicit path arg so the
    consumer can pass this fixture's return value directly. We also
    clear the lru_cache before+after to keep tests hermetic.
    """
    _clear_loader_cache()
    path = write_hardware_yaml(tmp_path, HARDWARE_2GPU)
    yield path
    _clear_loader_cache()


@pytest.fixture
def hardware_3gpu(tmp_path):
    """Write a 3gpu hardware.yaml under tmp_path, return its path."""
    _clear_loader_cache()
    path = write_hardware_yaml(tmp_path, HARDWARE_3GPU)
    yield path
    _clear_loader_cache()
