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


# ── spec 2026-06-16「数据加载统一」:registry 套用运行时覆盖(DB→缓存,registry 读缓存)──
# 用 set_cache_for_test 直灌覆盖缓存(等价 DB hydrate 后的状态),免起 DB。

def _yaml(tmp_path, entry):
    ypath = tmp_path / "models.yaml"
    ypath.write_text(yaml.dump({"models": [entry]}))
    return str(ypath)


def test_registry_applies_overlay_gpu_and_resident(tmp_path):
    """registry._load 套用覆盖的 gpu/resident(覆盖优先于 yaml)——
    使覆盖成为运行时设置单一来源,gpu 由此驱动 vLLM 落卡(修旧 gap)。"""
    from src.services import runtime_override_store
    ypath = _yaml(tmp_path, {
        "id": "m1", "type": "llm", "adapter": "src.services.inference.llm_vllm.VLLMAdapter",
        "path": "/m/x", "vram_mb": 0, "gpu": 1, "resident": False})
    runtime_override_store.set_cache_for_test({"m1": {"gpu": 2, "resident": True}})
    try:
        spec = ModelRegistry(ypath).get("m1")
        assert spec.gpu == 2, "覆盖 gpu 应优先于 yaml"
        assert spec.resident is True, "覆盖 resident 应优先于 yaml"
    finally:
        runtime_override_store.reset_cache()


def test_registry_falls_back_to_yaml_without_overlay(tmp_path):
    """无覆盖 → 用 yaml 值(零回归)。"""
    from src.services import runtime_override_store
    runtime_override_store.reset_cache()
    ypath = _yaml(tmp_path, {
        "id": "m1", "type": "llm", "adapter": "src.services.inference.llm_vllm.VLLMAdapter",
        "path": "/m/x", "vram_mb": 0, "gpu": 1, "resident": True})
    spec = ModelRegistry(ypath).get("m1")
    assert spec.gpu == 1 and spec.resident is True


def test_registry_reload_picks_up_overlay_change(tmp_path):
    """覆盖变更后 reload → spec.gpu 刷新(set_gpu 端点正是这条路径让落卡立即换卡)。"""
    from src.services import runtime_override_store
    runtime_override_store.reset_cache()
    ypath = _yaml(tmp_path, {
        "id": "m1", "type": "llm", "adapter": "src.services.inference.llm_vllm.VLLMAdapter",
        "path": "/m/x", "vram_mb": 0})
    reg = ModelRegistry(ypath)
    assert reg.get("m1").gpu is None  # yaml 未设 + 无覆盖 → auto
    try:
        runtime_override_store.set_cache_for_test({"m1": {"gpu": 2}})
        reg.reload()
        assert reg.get("m1").gpu == 2, "reload 后应从覆盖取到 gpu"
    finally:
        runtime_override_store.reset_cache()
