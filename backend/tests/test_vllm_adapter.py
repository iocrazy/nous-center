import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.inference.llm_vllm import VLLMAdapter
from src.services.inference.base import InferenceAdapter, InferenceResult


@pytest.fixture
def adapter(tmp_path):
    return VLLMAdapter(model_path=str(tmp_path), device="cpu", vllm_port=19999)


async def test_adapter_is_inference_adapter(adapter):
    assert isinstance(adapter, InferenceAdapter)
    assert adapter.model_type == "llm"


async def test_load_connects_to_existing_vllm(adapter):
    """If vLLM is already running, load() connects without spawning subprocess."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch.object(adapter._client, "get", new_callable=AsyncMock, return_value=mock_resp):
        await adapter.load("cpu")
    assert adapter.is_loaded
    assert not adapter._managed  # didn't spawn subprocess


async def test_load_fails_if_no_vllm_and_bad_model(adapter):
    """If vLLM is not running and model path is invalid, load() raises RuntimeError."""
    auto_result = {
        "port": 19999, "tp": 1, "max_model_len": 4096,
        "utilization": 0.85, "quantization": None, "dtype": None,
        "max_num_seqs": 32, "gpu_idx": 0, "model_size_gb": 0.0,
    }
    # Health check fails (no existing vLLM)
    with patch.object(adapter, "_health_check", new_callable=AsyncMock, return_value=False), \
         patch.object(adapter, "_auto_configure", return_value=auto_result):
        # subprocess will fail because model_path is a temp dir with no model
        with pytest.raises(RuntimeError, match="vLLM failed to start"):
            await adapter.load("cpu")
    assert not adapter.is_loaded


async def test_unload_kills_subprocess(adapter):
    """unload() on a managed adapter kills the subprocess."""
    adapter._model = True
    adapter._managed = True
    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = MagicMock()
    adapter._process = mock_proc
    adapter.unload()
    mock_proc.terminate.assert_called_once()
    assert not adapter.is_loaded
    assert adapter._process is None


async def test_unload_external_doesnt_kill(adapter):
    """unload() on an external (non-managed) adapter just disconnects."""
    adapter._model = True
    adapter._managed = False
    adapter._process = None
    adapter.unload()
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
