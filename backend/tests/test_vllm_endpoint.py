"""Lane E: vLLM base-URL 查找真相源测试（纯内存，无子进程、无 GPU）。"""
import pytest

from src.services.inference.vllm_endpoint import (
    VLLMNoEndpoint,
    VLLMNotLoaded,
    get_vllm_base_url,
)


class _FakeAdapter:
    def __init__(self, is_loaded: bool, base_url: str | None):
        self.is_loaded = is_loaded
        self.base_url = base_url


class _FakeModelManager:
    def __init__(self, adapters: dict):
        self._adapters = adapters

    def get_adapter(self, engine_name: str):
        return self._adapters.get(engine_name)


def test_returns_base_url_when_loaded():
    """adapter 已加载且有 base_url → 返回 base_url。"""
    mgr = _FakeModelManager({"qwen": _FakeAdapter(True, "http://localhost:8123")})
    assert get_vllm_base_url(mgr, "qwen") == "http://localhost:8123"


def test_raises_not_loaded_when_adapter_missing():
    """engine_name 没有对应 adapter → VLLMNotLoaded。"""
    mgr = _FakeModelManager({})
    with pytest.raises(VLLMNotLoaded, match="qwen"):
        get_vllm_base_url(mgr, "qwen")


def test_raises_not_loaded_when_adapter_not_loaded():
    """adapter 存在但 is_loaded=False → VLLMNotLoaded。"""
    mgr = _FakeModelManager({"qwen": _FakeAdapter(False, "http://localhost:8123")})
    with pytest.raises(VLLMNotLoaded, match="qwen"):
        get_vllm_base_url(mgr, "qwen")


def test_raises_no_endpoint_when_base_url_empty():
    """adapter 已加载但 base_url 为空 → VLLMNoEndpoint。"""
    mgr = _FakeModelManager({"qwen": _FakeAdapter(True, None)})
    with pytest.raises(VLLMNoEndpoint, match="qwen"):
        get_vllm_base_url(mgr, "qwen")


def test_raises_not_loaded_when_model_manager_none():
    """model_manager 本身为 None（app.state 未初始化）→ VLLMNotLoaded。"""
    with pytest.raises(VLLMNotLoaded):
        get_vllm_base_url(None, "qwen")
