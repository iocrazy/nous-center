"""round9 PR-D · vLLM/SGLang adapter infer 鲁棒性。

不依赖真 vllm/sglang(只 mock httpx client),所以**不**在 CI 的 --ignore 名单里 ——
跑在普通 backend job。覆盖两个回归:

- 200-but-body-error:服务端返 HTTP 200 但 body 是 OpenAI 错误体
  {"object":"error",...}。旧实现只判 status_code → 当成功、下游静默吐空。
- stream_options 强制:_build_payload 展开 **req.extra,调用方在 extra 里塞
  stream_options 会盖掉 include_usage;旧 setdefault 不纠正 → usage chunk 不发、
  计费拿空。现在强制合并 include_usage=True。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.inference.base import Message, TextRequest
from src.services.inference.llm_sglang import SGLangAdapter
from src.services.inference.llm_vllm import VLLMAdapter


def _adapters(tmp_path):
    return [
        VLLMAdapter(paths={"main": str(tmp_path)}, device="cpu", vllm_port=19991),
        SGLangAdapter(paths={"main": str(tmp_path)}, device="cpu", sglang_port=19992),
    ]


def _req(**extra):
    return TextRequest(
        request_id="r1",
        messages=[Message(role="user", content="hi")],
        model="test",
        extra=extra or {},
    )


@pytest.mark.parametrize("err_body", [
    {"object": "error", "message": "context length exceeded"},
    {"error": {"message": "bad request"}},
])
@pytest.mark.asyncio
async def test_infer_raises_on_200_body_error(tmp_path, err_body):
    for adapter in _adapters(tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(err_body).encode()
        mock_resp.json.return_value = err_body
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(RuntimeError):
                await adapter.infer(_req())


@pytest.mark.asyncio
async def test_infer_stream_forces_include_usage_over_extra(tmp_path):
    """调用方在 extra 里关掉 include_usage,适配器必须强制打开(否则计费丢 usage)。"""
    for adapter in _adapters(tmp_path):
        captured = {}

        class _FakeStreamCtx:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False
            async def aiter_lines(self):
                yield "data: [DONE]"
            async def aread(self):
                return b""

        def _fake_stream(method, url, **kwargs):
            return _FakeStreamCtx(**kwargs)

        with patch.object(adapter._client, "stream", _fake_stream):
            req = _req(stream_options={"include_usage": False})
            async for _ in adapter.infer_stream(req):
                pass

        assert captured["json"]["stream_options"]["include_usage"] is True
