import os
from pathlib import Path

from src.config import Settings, load_model_configs, _resolve_path


def test_settings_defaults():
    settings = Settings(
        REDIS_URL="redis://localhost:6379/0",
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/db",
    )
    assert settings.REDIS_URL == "redis://localhost:6379/0"
    assert settings.NAS_OUTPUTS_PATH == "/mnt/nas/outputs"
    assert settings.VLLM_BASE_URL == "http://localhost:8100"


def test_load_model_configs():
    configs = load_model_configs("configs/models.yaml")
    assert "sdxl" in configs
    assert configs["sdxl"]["type"] == "image"
    assert configs["wan21"]["exclusive"] is True


def test_resolve_path_is_relative_to_backend():
    """Paths must resolve relative to backend/ dir, not cwd."""
    resolved = _resolve_path("configs/models.yaml")
    assert "backend" in str(resolved), f"Expected path relative to backend/, got {resolved}"


def test_load_model_configs_from_any_cwd(tmp_path, monkeypatch):
    """load_model_configs works even when cwd is not backend/."""
    monkeypatch.chdir(tmp_path)
    configs = load_model_configs()
    assert isinstance(configs, dict)
