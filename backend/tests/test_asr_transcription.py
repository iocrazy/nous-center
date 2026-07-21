"""ASR 转写端点单测(spec 2026-07-20-moss-asr-sglang-serving)。

后端已切 MOSS-Transcribe-Diarize SGLang 微服务:这里 mock MOSS 的 HTTP 响应(不起真
服务/不碰 GPU),锁 `_asr_moss_transcribe` 的 verbose_json 解析 + 503 语义,以及端点的
segments 门控与秒数计量。真机 e2e 见 spec §4 PR-2 验收。
"""
import io
import wave

import httpx
import pytest

from fastapi import HTTPException

from src.api.routes.openai_compat import (
    _MOSS_DEFAULT_PROMPT,
    _asr_moss_transcribe,
    _wav16k_seconds,
)


# --- MOSS 微服务响应替身 --------------------------------------------------


class _FakeResp:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.text = text

    def json(self):
        return self._json


class _FakeClient:
    """最小 httpx.AsyncClient 替身:记录请求参数、回放预置响应或抛错。"""

    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc
        self.last_call = None

    async def post(self, url, **kwargs):
        self.last_call = (url, kwargs)
        if self._exc is not None:
            raise self._exc
        return self._resp


# --- _asr_moss_transcribe:verbose_json 解析 -------------------------------


@pytest.mark.asyncio
async def test_moss_transcribe_multi_speaker():
    # ① 多段多说话人:text 剥掉 [Sxx]/时间戳顺序拼接;segments 抠出 speaker + 秒 float。
    body = {
        "task": "transcribe", "duration": 6.0, "language": "zh",
        "text": "[0.26][S01]你好。[3.24][3.5][S02]再见。[6.0]",
        "segments": [
            {"id": 0, "start": 0.26, "end": 3.24, "text": "[S01]你好。"},
            {"id": 1, "start": 3.5, "end": 6.0, "text": "[S02]再见。"},
        ],
    }
    client = _FakeClient(resp=_FakeResp(200, body))
    text, language, segments = await _asr_moss_transcribe(client, b"wav", None, 6)
    assert text == "你好。再见。"
    assert language == "zh"
    assert segments == [
        {"start": 0.26, "end": 3.24, "speaker": "S01", "text": "你好。"},
        {"start": 3.5, "end": 6.0, "speaker": "S02", "text": "再见。"},
    ]


@pytest.mark.asyncio
async def test_moss_transcribe_missing_speaker_prefix():
    # ④ 防御:段缺 [Sxx] 前缀 → speaker=None,文本原样;顶层无 language → None(不假设有)。
    body = {"segments": [{"start": 0.0, "end": 2.0, "text": "没有前缀的文本"}]}
    client = _FakeClient(resp=_FakeResp(200, body))
    text, language, segments = await _asr_moss_transcribe(client, b"wav", None, 2)
    assert language is None
    assert segments == [
        {"start": 0.0, "end": 2.0, "speaker": None, "text": "没有前缀的文本"},
    ]
    assert text == "没有前缀的文本"


@pytest.mark.asyncio
async def test_moss_transcribe_request_shape():
    # multipart 固定项 + max_new_tokens 按时长动态(min(65536,max(5120,secs*32+512)))。
    client = _FakeClient(resp=_FakeResp(200, {"segments": []}))
    await _asr_moss_transcribe(client, b"wavbytes", None, 6)
    url, kwargs = client.last_call
    assert url.endswith("/v1/audio/transcriptions")
    assert kwargs["data"]["model"] == "moss-transcribe-diarize"
    assert kwargs["data"]["response_format"] == "verbose_json"
    assert kwargs["files"]["file"][1] == b"wavbytes"
    # 6s → max(5120, 6*32+512=704) = 5120(下限保短音频)
    assert kwargs["data"]["max_new_tokens"] == "5120"
    # 中等时长走公式;长音频封顶 65536
    for secs, expected in [(460, 460 * 32 + 512), (3000, 65536)]:
        await _asr_moss_transcribe(client, b"w", None, secs)
        assert client.last_call[1]["data"]["max_new_tokens"] == str(expected)


# --- _asr_moss_transcribe:context → 热词 prompt --------------------------


@pytest.mark.asyncio
async def test_moss_transcribe_context_appends_hotword_prompt():
    # context 非空:multipart 含 prompt,以默认转写指令开头、以热词后缀结尾(保输出契约 + 生效热词)。
    client = _FakeClient(resp=_FakeResp(200, {"segments": []}))
    await _asr_moss_transcribe(client, b"wav", "  瑞幸咖啡  ", 4)
    prompt = client.last_call[1]["data"]["prompt"]
    assert prompt.startswith(_MOSS_DEFAULT_PROMPT)
    assert prompt.endswith("热词提示:瑞幸咖啡")  # context 两端空白已 strip


@pytest.mark.asyncio
async def test_moss_transcribe_no_context_omits_prompt():
    # context 空/纯空白:不传 prompt,吃服务端默认指令。
    for ctx in (None, "", "   "):
        client = _FakeClient(resp=_FakeResp(200, {"segments": []}))
        await _asr_moss_transcribe(client, b"wav", ctx, 4)
        assert "prompt" not in client.last_call[1]["data"]


# --- _asr_moss_transcribe:503 语义(唯一 ASR 主路,不静默降级) ----------


@pytest.mark.asyncio
async def test_moss_transcribe_unreachable_503():
    # ③ 连接失败 → HTTPException 503。
    client = _FakeClient(exc=httpx.ConnectError("connection refused"))
    with pytest.raises(HTTPException) as ei:
        await _asr_moss_transcribe(client, b"wav", None, 4)
    assert ei.value.status_code == 503


@pytest.mark.asyncio
async def test_moss_transcribe_non_200_503():
    # 非 200 → 503(不把上游错误当结果)。
    client = _FakeClient(resp=_FakeResp(500, {}, "internal error"))
    with pytest.raises(HTTPException) as ei:
        await _asr_moss_transcribe(client, b"wav", None, 4)
    assert ei.value.status_code == 503


# --- 端点:segments 门控 + 秒数计量 ---------------------------------------


def _make_wav16k(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # s16le
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


@pytest.mark.asyncio
async def test_endpoint_segments_gated_by_timestamps(
    api_client, bearer_headers, mock_vllm, monkeypatch,
):
    # ② timestamps=true 带 segments、=false 不带;⑤ usage.seconds = 自算音频秒(4)。
    import src.api.routes.openai_compat as oc

    wav = _make_wav16k(4.0)

    async def _fake_ffmpeg(raw):
        return wav

    async def _fake_moss(client, w, ctx, secs):
        return ("你好世界", "zh", [
            {"start": 0.0, "end": 4.0, "speaker": "S01", "text": "你好世界"},
        ])

    monkeypatch.setattr(oc, "_ffmpeg_to_wav16k", _fake_ffmpeg)
    monkeypatch.setattr(oc, "_asr_moss_transcribe", _fake_moss)

    resp = await api_client.post(
        "/v1/audio/transcriptions",
        headers=bearer_headers,
        files={"file": ("a.wav", wav, "audio/wav")},
        data={"model": "qwen3.5", "timestamps": "true"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == "你好世界"
    assert body["language"] == "zh"
    assert body["usage"] == {"type": "duration", "seconds": 4}
    assert body["segments"][0]["speaker"] == "S01"

    resp2 = await api_client.post(
        "/v1/audio/transcriptions",
        headers=bearer_headers,
        files={"file": ("a.wav", wav, "audio/wav")},
        data={"model": "qwen3.5"},
    )
    assert resp2.status_code == 200, resp2.text
    body2 = resp2.json()
    assert body2["text"] == "你好世界"
    assert "segments" not in body2


@pytest.mark.asyncio
async def test_endpoint_moss_unreachable_returns_503(
    api_client, bearer_headers, mock_vllm, monkeypatch,
):
    # ③ 端点透传:MOSS 不可达 → 503(不静默降级)。
    import src.api.routes.openai_compat as oc

    wav = _make_wav16k(2.0)

    async def _fake_ffmpeg(raw):
        return wav

    async def _fake_moss(client, w, ctx, secs):
        raise HTTPException(503, detail="MOSS ASR 服务不可达")

    monkeypatch.setattr(oc, "_ffmpeg_to_wav16k", _fake_ffmpeg)
    monkeypatch.setattr(oc, "_asr_moss_transcribe", _fake_moss)

    resp = await api_client.post(
        "/v1/audio/transcriptions",
        headers=bearer_headers,
        files={"file": ("a.wav", wav, "audio/wav")},
        data={"model": "qwen3.5"},
    )
    assert resp.status_code == 503, resp.text


# --- 计量口径不变(既有断言保住) ----------------------------------------


def test_wav16k_seconds():
    assert _wav16k_seconds(_make_wav16k(4.0)) == 4
    assert _wav16k_seconds(_make_wav16k(0.3)) == 1  # 至少 1 秒计费
    # 损坏/非 wav 字节 → 退化估算不崩(至少 1)
    assert _wav16k_seconds(b"not a wav") >= 1
