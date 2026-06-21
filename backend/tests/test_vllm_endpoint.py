"""Lane E: vLLM base-URL 查找真相源测试（纯内存，无子进程、无 GPU）。"""
import pytest

from src.services.inference.vllm_endpoint import (
    VLLMNoEndpoint,
    VLLMNotLoaded,
    ensure_vllm_base_url,
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


# ---------- ensure_vllm_base_url:按需懒加载 ----------

class _LoadableAdapter:
    def __init__(self, base_url: str | None = "http://localhost:8123"):
        self.is_loaded = False
        self.base_url = base_url


class _AutoloadMgr:
    """get_adapter + async load_model(flip is_loaded)。fail=True 模拟加载失败。"""
    def __init__(self, adapters: dict, fail: bool = False):
        self._adapters = adapters
        self.fail = fail
        self.load_calls = 0

    def get_adapter(self, name: str):
        return self._adapters.get(name)

    async def load_model(self, name: str):
        self.load_calls += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        a = self._adapters.get(name)
        if a is not None:
            a.is_loaded = True


@pytest.mark.asyncio
async def test_ensure_autoloads_when_not_loaded():
    """未加载 → 自动 load_model 一次 → 返回 base_url(发现即能调)。"""
    mgr = _AutoloadMgr({"qwen3_asr": _LoadableAdapter("http://localhost:9001")})
    url = await ensure_vllm_base_url(mgr, "qwen3_asr")
    assert url == "http://localhost:9001"
    assert mgr.load_calls == 1


@pytest.mark.asyncio
async def test_ensure_no_load_when_already_loaded():
    """已加载 → 不再触发 load_model(零额外开销)。"""
    a = _LoadableAdapter("http://localhost:9002")
    a.is_loaded = True
    mgr = _AutoloadMgr({"qwen": a})
    url = await ensure_vllm_base_url(mgr, "qwen")
    assert url == "http://localhost:9002"
    assert mgr.load_calls == 0


@pytest.mark.asyncio
async def test_ensure_raises_not_loaded_when_autoload_fails():
    """自动加载失败(OOM/未知模型)→ VLLMNotLoaded(调用方仍 503,不比原来差)。"""
    mgr = _AutoloadMgr({"qwen": _LoadableAdapter()}, fail=True)
    with pytest.raises(VLLMNotLoaded, match="自动加载失败"):
        await ensure_vllm_base_url(mgr, "qwen")


@pytest.mark.asyncio
async def test_ensure_raises_not_loaded_when_model_manager_none():
    with pytest.raises(VLLMNotLoaded):
        await ensure_vllm_base_url(None, "qwen")
