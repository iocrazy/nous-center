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
