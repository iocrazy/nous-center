from src.config import Settings, load_model_configs


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
