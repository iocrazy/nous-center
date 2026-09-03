import json
import pytest

# 这个文件不需要装 vllm:VLLMAdapter 本身不 import vllm,只是拼命令行起子进程。
# 以前这里 `pytest.importorskip("vllm")`,逻辑是"没装 vllm 就跳过,装了就真起子进程
# 验证生命周期" —— 在装了 vllm 的生产机上这恰恰是灾难(2026-09-02:真起的 vllm 把
# CUDA_VISIBLE_DEVICES 覆盖成 "0",多 worker 并发初始化 CUDA,把 RTX 3090 驱动跑挂)。
# 现在 test_load_fails_if_no_vllm_and_bad_model 用假 Popen,全文件零真子进程;
# conftest 另有 Popen 层护栏,谁再真起 vllm.entrypoints 会直接 AssertionError。
import io
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
    """If vLLM is not running and the spawned server dies, load() raises RuntimeError.

    子进程是**假的**(Popen 被换成一个已退出、returncode=1 的桩):这个用例验证的是
    adapter 对"子进程起不来"的处理路径(读尾部日志 → 杀进程 → 抛 RuntimeError),
    不需要、也**绝不能**真起 vLLM —— 见文件头注释。
    """
    auto_result = {
        "port": 19999, "tp": 1, "max_model_len": 4096,
        "utilization": 0.85, "quantization": None, "dtype": None,
        "max_num_seqs": 32, "gpu_idx": 0, "model_size_gb": 0.0,
        # gpu_total_gb / gpu_free_gb are consumed by resolve_vram_utilization
        # and clamp_util_to_free on the load path — must be present.
        "gpu_total_gb": 24.0, "gpu_free_gb": 20.0,
    }
    fake_proc = MagicMock()
    fake_proc.pid = 424242
    fake_proc.returncode = 1
    fake_proc.poll = MagicMock(return_value=1)          # 已退出
    fake_proc.stdout = io.StringIO("ValueError: no model found\n")  # 抽干线程读到 EOF
    fake_proc.wait = MagicMock(return_value=1)

    import src.services.inference.llm_vllm as _mod
    # Health check fails (no existing vLLM); Popen 换成桩;_kill_process 不去信号一个假 pid
    with patch.object(adapter, "_health_check", new_callable=AsyncMock, return_value=False), \
         patch.object(adapter, "_auto_configure", return_value=auto_result), \
         patch.object(_mod.subprocess, "Popen", return_value=fake_proc) as popen, \
         patch.object(adapter, "_kill_process"):
        with pytest.raises(RuntimeError, match="vLLM failed to start"):
            await adapter.load("cpu")
    assert not adapter.is_loaded

    # 顺带钉住两个事实,免得以后有人把它改回真起进程还不自知:
    # 1) 起的确实是 vllm 的 OpenAI server 入口;
    # 2) device="cpu" 会被映射成 CUDA_VISIBLE_DEVICES="0" —— 这正是 2026-09-02
    #    事故里子进程真摸到 GPU 的原因(conftest 设的 "" 被它覆盖)。
    assert popen.call_count == 1
    argv = popen.call_args.args[0]
    assert "vllm.entrypoints.openai.api_server" in argv
    assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"


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

