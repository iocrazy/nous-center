import pytest
import yaml
from src.services.inference.registry import ModelRegistry

@pytest.fixture
def registry_yaml(tmp_path):
    config = {
        "models": [
            {"id": "test-tts", "type": "tts", "adapter": "src.workers.tts_engines.cosyvoice2.CosyVoice2Engine", "path": "/models/tts/test", "vram_mb": 2000},
            {"id": "test-llm", "type": "llm", "adapter": "src.services.inference.llm_vllm.VLLMAdapter", "path": "/models/llm/test", "vram_mb": 0, "params": {"vllm_base_url": "http://localhost:8100"}},
        ]
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.dump(config))
    return path

def test_load_registry(registry_yaml):
    reg = ModelRegistry(str(registry_yaml))
    assert len(reg.specs) == 2

def test_get_spec_by_id(registry_yaml):
    reg = ModelRegistry(str(registry_yaml))
    spec = reg.get("test-tts")
    assert spec is not None
    assert spec.model_type == "tts"
    assert spec.vram_mb == 2000

def test_get_spec_missing(registry_yaml):
    reg = ModelRegistry(str(registry_yaml))
    assert reg.get("nonexistent") is None

def test_list_by_type(registry_yaml):
    reg = ModelRegistry(str(registry_yaml))
    assert len(reg.list_by_type("tts")) == 1

def test_spec_params(registry_yaml):
    reg = ModelRegistry(str(registry_yaml))
    spec = reg.get("test-llm")
    assert spec.params["vllm_base_url"] == "http://localhost:8100"


# ── spec 2026-06-16「数据加载统一」:registry 套用 runtime_overrides overlay ──

def test_registry_applies_overlay_gpu_and_resident(tmp_path, monkeypatch):
    """registry._load 套用 overlay 的 gpu/resident(overlay 优先于 yaml)——
    使 overlay 成为运行时覆盖单一来源,overlay gpu 由此驱动 vLLM 落卡(修旧 gap)。"""
    import src.config as cfg
    monkeypatch.setattr(cfg, "_BACKEND_DIR", tmp_path)
    config = {"models": [
        {"id": "m1", "type": "llm",
         "adapter": "src.services.inference.llm_vllm.VLLMAdapter",
         "path": "/m/x", "vram_mb": 0, "gpu": 1, "resident": False},
    ]}
    ypath = tmp_path / "models.yaml"
    ypath.write_text(yaml.dump(config))
    cfg.set_runtime_override("m1", "gpu", 2)
    cfg.set_runtime_override("m1", "resident", True)

    reg = ModelRegistry(str(ypath))
    spec = reg.get("m1")
    assert spec.gpu == 2, "overlay gpu 应优先于 yaml"
    assert spec.resident is True, "overlay resident 应优先于 yaml"


def test_registry_falls_back_to_yaml_without_overlay(tmp_path, monkeypatch):
    """无 overlay → 用 yaml 值(零回归)。"""
    import src.config as cfg
    monkeypatch.setattr(cfg, "_BACKEND_DIR", tmp_path)
    config = {"models": [
        {"id": "m1", "type": "llm",
         "adapter": "src.services.inference.llm_vllm.VLLMAdapter",
         "path": "/m/x", "vram_mb": 0, "gpu": 1, "resident": True},
    ]}
    ypath = tmp_path / "models.yaml"
    ypath.write_text(yaml.dump(config))

    reg = ModelRegistry(str(ypath))
    spec = reg.get("m1")
    assert spec.gpu == 1 and spec.resident is True


def test_registry_reload_picks_up_overlay_change(tmp_path, monkeypatch):
    """写 overlay 后 reload → spec.gpu 刷新(set_gpu 端点正是这条路径让落卡立即换卡)。"""
    import src.config as cfg
    monkeypatch.setattr(cfg, "_BACKEND_DIR", tmp_path)
    config = {"models": [
        {"id": "m1", "type": "llm",
         "adapter": "src.services.inference.llm_vllm.VLLMAdapter",
         "path": "/m/x", "vram_mb": 0},
    ]}
    ypath = tmp_path / "models.yaml"
    ypath.write_text(yaml.dump(config))

    reg = ModelRegistry(str(ypath))
    assert reg.get("m1").gpu is None  # yaml 未设 + 无 overlay → auto
    cfg.set_runtime_override("m1", "gpu", 2)
    reg.reload()
    assert reg.get("m1").gpu == 2, "reload 后应从 overlay 取到 gpu"
