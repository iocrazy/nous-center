"""A5:vLLM 流式中途断连(httpx 错)必须 yield 结构化 error 事件,不裸冒泡。
不依赖 vllm 包(llm_vllm 模块可独立 import,vllm 只在 load() 里 lazy import)。"""
import httpx
import pytest

from src.services.inference.base import MediaModality, Message, TextRequest
from src.services.inference.llm_vllm import VLLMAdapter


class _FakeResp:
    status_code = 200
    def __init__(self, lines_then_raise):
        self._lines = lines_then_raise
    async def aiter_lines(self):
        for item in self._lines:
            if isinstance(item, Exception):
                raise item
            yield item


class _FakeStreamCM:
    def __init__(self, resp):
        self._resp = resp
    async def __aenter__(self):
        return self._resp
    async def __aexit__(self, *a):
        return False


@pytest.mark.asyncio
async def test_stream_interrupt_yields_error_event(monkeypatch):
    ad = VLLMAdapter(paths={"main": "/x"}, device="cpu", vllm_port=19999)
    ad._base_url = "http://localhost:19999"
    # 先吐一个 delta,再中途抛 ReadError(vLLM 崩溃/网络 reset)
    delta_line = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
    lines = [delta_line, httpx.ReadError("connection reset mid-stream")]
    monkeypatch.setattr(ad._client, "stream",
                        lambda *a, **k: _FakeStreamCM(_FakeResp(lines)))

    req = TextRequest(request_id="t1", modality=MediaModality.TEXT, messages=[Message(role="user", content="hi")])
    events = [e async for e in ad.infer_stream(req)]
    types = [e.type for e in events]
    assert "delta" in types, types
    assert types[-1] == "error", f"中途断流应以 error 事件收尾,实际 {types}"
    assert "stream interrupted" in events[-1].payload.get("error", "")
    # 不该抛异常(能正常收集完 = 没裸冒泡)
