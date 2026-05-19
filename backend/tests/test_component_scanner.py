"""component_scanner: model_paths config + role glob + quant detection."""
from __future__ import annotations

from src.services.component_scanner import load_model_paths_config, ROLE_DIRS


def test_load_model_paths_config_returns_role_dirs():
    cfg = load_model_paths_config()
    assert "unet" in cfg
    assert "clip" in cfg
    assert "vae" in cfg
    assert "loras" in cfg
    for role, patterns in cfg.items():
        assert isinstance(patterns, list)
        assert all(isinstance(p, str) for p in patterns)


def test_role_dirs_constant_matches_config_keys():
    cfg = load_model_paths_config()
    assert set(ROLE_DIRS) == set(cfg.keys())
