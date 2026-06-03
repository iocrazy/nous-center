
import pytest

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


def test_runtime_override_roundtrip(tmp_path, monkeypatch):
    """set_runtime_override 写 → load_runtime_overrides 读回(gitignore 的 overlay)。"""
    import src.config as cfg
    monkeypatch.setattr(cfg, "_BACKEND_DIR", tmp_path)
    assert cfg.load_runtime_overrides() == {}  # 文件不存在 → 空
    cfg.set_runtime_override("m1", "resident", True)
    cfg.set_runtime_override("m2", "gpu", 1)
    assert cfg.load_runtime_overrides() == {"m1": {"resident": True}, "m2": {"gpu": 1}}
    # 同 model 再写不同 key,合并不覆盖
    cfg.set_runtime_override("m1", "gpu", 0)
    assert cfg.load_runtime_overrides()["m1"] == {"resident": True, "gpu": 0}


def test_runtime_override_rejects_non_overridable_key(tmp_path, monkeypatch):
    """只白名单 resident/gpu,别让随意键污染 overlay。"""
    import src.config as cfg
    monkeypatch.setattr(cfg, "_BACKEND_DIR", tmp_path)
    with pytest.raises(ValueError):
        cfg.set_runtime_override("m1", "type", "evil")


def test_runtime_override_corrupt_file_is_soft(tmp_path, monkeypatch):
    """坏 JSON 不该拖垮模型加载 → 降级空 dict。"""
    import src.config as cfg
    monkeypatch.setattr(cfg, "_BACKEND_DIR", tmp_path)
    p = tmp_path / "configs"
    p.mkdir()
    (p / "runtime_overrides.json").write_text("{not json")
    assert cfg.load_runtime_overrides() == {}


def test_runtime_override_overlays_model_configs(monkeypatch):
    """overlay 优先叠加到 load_model_configs 的结果(只对已存在的 model 生效)。"""
    import src.config as cfg
    monkeypatch.setattr(
        cfg, "load_runtime_overrides",
        lambda: {"m1": {"resident": True}, "ghost": {"resident": True}},
    )
    cfgs = {"m1": {"resident": False}, "m2": {"resident": False}}
    cfg._apply_runtime_overrides(cfgs)
    assert cfgs["m1"]["resident"] is True   # overlay 生效
    assert cfgs["m2"]["resident"] is False  # 未覆盖的不动
    assert "ghost" not in cfgs              # overlay 里有但 cfgs 没有 → 不凭空建条目
