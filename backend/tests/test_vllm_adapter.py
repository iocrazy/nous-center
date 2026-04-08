import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.inference.llm_vllm import VLLMAdapter
from src.services.inference.base import InferenceAdapter, InferenceResult

@pytest.fixture
def adapter(tmp_path):
    return VLLMAdapter(model_path=str(tmp_path), device="cpu", vllm_base_url="http://localhost:8100")

async def test_adapter_is_inference_adapter(adapter):
    assert isinstance(adapter, InferenceAdapter)
    assert adapter.model_type == "llm"

async def test_load_checks_vllm_health(adapter):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch.object(adapter._client, "get", new_callable=AsyncMock, return_value=mock_resp):
        await adapter.load("cpu")
    assert adapter.is_loaded

async def test_load_fails_if_vllm_down(adapter):
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    with patch.object(adapter._client, "get", new_callable=AsyncMock, return_value=mock_resp):
        await adapter.load("cpu")
    assert not adapter.is_loaded

async def test_infer_returns_result(adapter):
    adapter._model = True
    fake_body = {"choices": [{"message": {"content": "hi"}}]}
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(fake_body).encode()
    mock_resp.status_code = 200
    with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await adapter.infer({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
    assert isinstance(result, InferenceResult)
    assert result.content_type == "application/json"
    assert b"hi" in result.data
