
from src.config import Settings, load_model_configs, _resolve_path


def test_settings_defaults():
    # Use _env_file=None to prevent .env from overriding defaults
    settings = Settings(
        _env_file=None,
        REDIS_URL="redis://localhost:6379/0",
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/db",
    )
    assert settings.REDIS_URL == "redis://localhost:6379/0"
    assert settings.NAS_OUTPUTS_PATH == "/mnt/nas/outputs"
    assert settings.VLLM_BASE_URL == "http://localhost:8100"


def test_load_model_configs():
    configs = load_model_configs("configs/models.yaml")
    assert "cosyvoice2" in configs
    assert configs["cosyvoice2"]["type"] == "tts"
    assert "qwen3_tts_base" in configs


def test_resolve_path_is_relative_to_backend():
    """Paths must resolve relative to backend/ dir, not cwd."""
    resolved = _resolve_path("configs/models.yaml")
    assert "backend" in str(resolved), f"Expected path relative to backend/, got {resolved}"


def test_load_model_configs_from_any_cwd(tmp_path, monkeypatch):
    """load_model_configs works even when cwd is not backend/."""
    monkeypatch.chdir(tmp_path)
    configs = load_model_configs()
    assert isinstance(configs, dict)


def test_load_settings_yaml_non_dict_returns_empty(tmp_path, monkeypatch):
    """round4 #5:settings.yaml 手改成 YAML 列表/标量 → 降级空 dict,不让 Settings(**) 崩。"""
    import src.config as cfg
    p = tmp_path / "settings.yaml"
    p.write_text("- a\n- b\n")  # YAML 列表
    monkeypatch.setattr(cfg, "SETTINGS_YAML_PATH", p)
    assert cfg._load_settings_yaml() == {}
    # 标量
    p.write_text("just a string\n")
    assert cfg._load_settings_yaml() == {}
    # 正常 dict 仍工作
    p.write_text("FOO: bar\n")
    assert cfg._load_settings_yaml() == {"FOO": "bar"}
