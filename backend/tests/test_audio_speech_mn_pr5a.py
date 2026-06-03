"""legacy rip PR-5a:/v1/audio/speech 从 legacy verify_bearer_token → M:N verify_bearer_token_any。

旧版只认 legacy 1:1 key(M:N key 实际 403);改后 M:N 有效 key 即可(handler 用 req.model 取 engine,
不用 instance)。CI 安全(不触真 TTS 引擎 —— 用不存在的 model,验「auth 过了→到 409 引擎未加载」)。
"""
from __future__ import annotations

import secrets as _secrets

import bcrypt
import pytest

from src.models.instance_api_key import InstanceApiKey


async def _mn_key(session_factory) -> str:
    """造一个 M:N key(instance_id=None,不绑单一 instance)。speech 不解析服务,有效 key 即可。"""
    raw = f"sk-mn-{_secrets.token_hex(8)}"
    async with session_factory() as s:
        s.add(InstanceApiKey(
            instance_id=None, label="mn",
            key_hash=bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode(),
            key_prefix=raw[:10], is_active=True,
        ))
        await s.commit()
    return raw


@pytest.mark.asyncio
async def test_speech_accepts_mn_key(api_client):
    """M:N key 过 auth → 到引擎检查 → 409 未加载(不是 401/403)。证明 PR-5a 迁移生效。"""
    raw = await _mn_key(api_client.app.state.async_session_factory)
    resp = await api_client.post(
        "/v1/audio/speech",
        json={"model": "nonexistent-tts", "input": "hi", "voice": "default"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert resp.status_code == 409, resp.text  # 引擎未加载 = auth 已过


@pytest.mark.asyncio
async def test_speech_rejects_no_auth(api_client):
    """无 Authorization → 拒(missing-header 经 app 校验包装成 400),不放行。"""
    resp = await api_client.post(
        "/v1/audio/speech",
        json={"model": "x", "input": "hi", "voice": "default"},
    )
    assert resp.status_code in (400, 401, 422), resp.text


@pytest.mark.asyncio
async def test_speech_rejects_invalid_key(api_client):
    """乱 key → 401。"""
    resp = await api_client.post(
        "/v1/audio/speech",
        json={"model": "x", "input": "hi", "voice": "default"},
        headers={"Authorization": "Bearer sk-bogus-nope"},
    )
    assert resp.status_code == 401, resp.text
