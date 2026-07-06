import json
import pytest

# Skip this entire file when vllm is not installed (main deps don't include it;
# use `uv sync --extra inference` to enable). test_load_fails_if_no_vllm_and_bad_model
# spawns a real subprocess via subprocess.Popen which, without vllm, enters a state
# that can hang the NVIDIA driver / X session on dual-GPU hosts — so avoid running
# it at all in non-inference environments. When vllm IS installed (inference host),
# this test runs normally and validates the adapter's subprocess lifecycle.
pytest.importorskip("vllm")

from unittest.mock import AsyncMock, patch, MagicMock
from src.services.inference.llm_vllm import VLLMAdapter
from src.services.inference.base import (
    InferenceAdapter,
    InferenceResult,
    MediaModality,
    Message,
    TextRequest,
)


@pytest.fixture
def adapter(tmp_path):
    return VLLMAdapter(paths={"main": str(tmp_path)}, device="cpu", vllm_port=19999)


async def test_adapter_is_inference_adapter(adapter):
    assert isinstance(adapter, InferenceAdapter)
    assert adapter.modality == MediaModality.TEXT


async def test_load_connects_to_existing_vllm(adapter):
    """If vLLM is already running, load() connects without spawning subprocess."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch.object(adapter._client, "get", new_callable=AsyncMock, return_value=mock_resp):
        await adapter.load("cpu")
    assert adapter.is_loaded
    assert not adapter._managed  # didn't spawn subprocess


async def test_load_backfills_configured_max_model_len_on_reconnect(tmp_path):
    """重连存活 vLLM 时,yaml 配的 max_model_len 必须回填到 self.max_model_len —— 否则
    _clamp_max_tokens 退回 4096 砍长输出(bug hunt round2 #2)。"""
    a = VLLMAdapter(paths={"main": str(tmp_path)}, device="cpu", vllm_port=19999,
                    max_model_len=131072)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value={"data": [{"max_model_len": 200000}]})
    with patch.object(a._client, "get", new_callable=AsyncMock, return_value=mock_resp):
        await a.load("cpu")
    assert a.max_model_len == 131072  # yaml 配的优先,不退 4096
    assert a._clamp_max_tokens(100000) > 4096  # clamp 用真值


async def test_load_fetches_remote_max_model_len_when_unconfigured(tmp_path):
    """yaml 没配时,从运行中 vLLM 的 /v1/models 读 max_model_len(而非退 4096)。"""
    a = VLLMAdapter(paths={"main": str(tmp_path)}, device="cpu", vllm_port=19999)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value={"data": [{"max_model_len": 200000}]})
    with patch.object(a._client, "get", new_callable=AsyncMock, return_value=mock_resp):
        await a.load("cpu")
    assert a.max_model_len == 200000  # 从 /v1/models 读到


async def test_load_fails_if_no_vllm_and_bad_model(adapter):
    """If vLLM is not running and model path is invalid, load() raises RuntimeError."""
    auto_result = {
        "port": 19999, "tp": 1, "max_model_len": 4096,
        "utilization": 0.85, "quantization": None, "dtype": None,
        "max_num_seqs": 32, "gpu_idx": 0, "model_size_gb": 0.0,
        # gpu_total_gb / gpu_free_gb are consumed by resolve_vram_utilization
        # and clamp_util_to_free on the load path — must be present.
        "gpu_total_gb": 24.0, "gpu_free_gb": 20.0,
    }
    # Health check fails (no existing vLLM)
    with patch.object(adapter, "_health_check", new_callable=AsyncMock, return_value=False), \
         patch.object(adapter, "_auto_configure", return_value=auto_result):
        # subprocess will fail because model_path is a temp dir with no model
        with pytest.raises(RuntimeError, match="vLLM failed to start"):
            await adapter.load("cpu")
    assert not adapter.is_loaded


async def test_unload_kills_subprocess(adapter):
    """unload() on a managed adapter signals the subprocess's process group via
    safe_killpg (broadcast-guarded) and clears state. The adapter kills by
    process GROUP (os.killpg), never proc.terminate() — assert the real path."""
    import src.services.safe_signal as _ss
    adapter._model = True
    adapter._managed = True
    mock_proc = MagicMock()
    mock_proc.pid = 424242  # real int so the broadcast guard (pid<=1) evaluates
    mock_proc.wait = MagicMock()
    adapter._process = mock_proc

    sent: list = []
    with patch.object(_ss, "safe_killpg",
                      side_effect=lambda pid, sig, **kw: sent.append((pid, sig)) or True):
        adapter.unload()

    assert sent and sent[0][0] == 424242  # signalled our child's pid
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
    fake_body = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(fake_body).encode()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_body
    with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        req = TextRequest(
            request_id="r1",
            messages=[Message(role="user", content="hi")],
            model="test",
        )
        result = await adapter.infer(req)
    assert isinstance(result, InferenceResult)
    assert result.media_type == "application/json"
    assert b"hi" in result.data
    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 2

