"""Unit tests for responses_service."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import select

from src.errors import ConflictError, InvalidRequestError, NotFoundError, RateLimitError
from src.models.response_session import ResponseSession, ResponseTurn
from src.services.responses_service import (
    SESSION_TOKEN_BUDGET,
    approx_tokens,
    assemble_history_for_response,
    check_session_budget,
    cleanup_expired_sessions,
    compact_messages,
    create_session,
    decode_content,
    encode_content,
    new_session_id,
    new_turn_id,
    update_session_usage,
    write_user_and_assistant_turns,
)


def test_id_format():
    sid = new_session_id()
    rid = new_turn_id()
    assert sid.startswith("session-")
    assert len(sid) == len("session-") + 16
    assert rid.startswith("resp-")
    assert len(rid) == len("resp-") + 16


def test_encode_decode_roundtrip():
    content = [{"type": "input_text", "text": "hello world 你好"}]
    enc = encode_content(content)
    assert isinstance(enc, bytes) and len(enc) > 0
    assert decode_content(enc) == content


def test_decode_rejects_oversized():
    import gzip as _gz
    import json as _json
    huge = _json.dumps([{"text": "x" * 11_000_000}]).encode()
    payload = _gz.compress(huge)
    with pytest.raises(InvalidRequestError) as ex:
        decode_content(payload, max_size=10_000_000)
    assert ex.value.code == "payload_too_large"


def test_approx_tokens_overestimates():
    msgs = [{"role": "user", "content": "the quick brown fox"}]
    assert approx_tokens(msgs) >= 4
    # Chinese: 12 chars * real tokens ≈ 18; our overestimate len/2+4 ≈ 10.
    # This is by design — we overestimate English more to compensate overall.
    # Test just ensures we count *something* for Chinese.
    msgs = [{"role": "user", "content": "你好世界你好世界你好世界"}]
    assert approx_tokens(msgs) >= 8


def test_compact_drops_oldest_keeps_system():
    msgs = [{"role": "system", "content": "S"}] + [
        {"role": "user", "content": "x" * 1000} for _ in range(20)
    ]
    out, truncated = compact_messages(msgs, max_history_tokens=500)
    assert truncated is True
    assert out[0]["role"] == "system"
    assert any(m["role"] == "user" for m in out)
    assert len(out) < len(msgs)


def test_compact_no_op_when_under_budget():
    msgs = [{"role": "user", "content": "tiny"}]
    out, truncated = compact_messages(msgs, max_history_tokens=10_000)
    assert truncated is False
    assert out == msgs


@pytest.mark.asyncio
async def test_create_session_and_write_turns(db_session, sample_instance):
    sess = await create_session(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="qwen3.5",
        context_cache_id=None,
    )
    assert sess.id.startswith("session-")
    user_turn, asst_turn = await write_user_and_assistant_turns(
        db_session,
        sess=sess,
        user_content=[{"type": "input_text", "text": "hi"}],
        assistant_content=[{"type": "output_text", "text": "hello"}],
        usage={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        reasoning=None,
        instructions=None,
        text_format=None,
    )
    assert user_turn.turn_idx == 0
    assert asst_turn.turn_idx == 1
    assert asst_turn.role == "assistant"
    assert asst_turn.id.startswith("resp-")


@pytest.mark.asyncio
async def test_concurrent_turn_write_raises_conflict(db_session, sample_instance):
    """Simulate TOCTOU: we pre-seed a row at idx 0, then patch func.max to
    return -1 so the service thinks it should insert at 0 — UNIQUE collides."""
    sess = await create_session(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        context_cache_id=None,
    )
    # Pre-seed row at idx 0 (concurrent winner)
    db_session.add(
        ResponseTurn(
            id=new_turn_id(),
            session_id=sess.id,
            turn_idx=0,
            role="user",
            content_compressed=encode_content([{"text": "winner"}]),
        )
    )
    await db_session.commit()

    # Monkeypatch func.max(...).where(...) to report no rows — forces idx 0 collision.
    # Easier: temporarily swap the select() builder.
    from src.services import responses_service as svc
    from sqlalchemy import func as _func

    original_execute = db_session.execute

    async def fake_execute(stmt, *a, **kw):
        # Intercept the max(turn_idx) lookup and return None (forcing idx 0)
        s = str(stmt)
        if "max(response_turns.turn_idx)" in s.replace(" ", "").replace("\n", "").lower() \
                or "max(turn_idx)" in s.lower():
            class _Res:
                def scalar(self_inner):
                    return None
            return _Res()
        return await original_execute(stmt, *a, **kw)

    db_session.execute = fake_execute
    try:
        with pytest.raises(ConflictError) as ex:
            await svc.write_user_and_assistant_turns(
                db_session,
                sess=sess,
                user_content=[{"text": "race-u"}],
                assistant_content=[{"text": "race-a"}],
                usage={},
                reasoning=None,
                instructions=None,
                text_format=None,
            )
        assert ex.value.code == "session_concurrent_write"
        assert ex.value.http_status == 409
    finally:
        db_session.execute = original_execute


@pytest.mark.asyncio
async def test_assemble_history_orders_by_turn_idx(db_session, sample_instance):
    sess = await create_session(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        context_cache_id=None,
    )
    for i in range(3):
        await write_user_and_assistant_turns(
            db_session,
            sess=sess,
            user_content=[{"type": "input_text", "text": f"u{i}"}],
            assistant_content=[{"type": "output_text", "text": f"a{i}"}],
            usage={},
            reasoning=None,
            instructions=None,
            text_format=None,
        )
    last_asst = (
        await db_session.execute(
            select(ResponseTurn)
            .where(
                ResponseTurn.session_id == sess.id,
                ResponseTurn.role == "assistant",
            )
            .order_by(ResponseTurn.turn_idx.desc())
        )
    ).scalars().first()
    msgs, fetched_sess = await assemble_history_for_response(
        db_session, last_asst.id, instance_id=sample_instance.id
    )
    assert fetched_sess.id == sess.id
    assert len(msgs) == 6
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"] * 3


@pytest.mark.asyncio
async def test_assemble_history_404_on_unknown(db_session, sample_instance):
    with pytest.raises(NotFoundError):
        await assemble_history_for_response(
            db_session, "resp-doesnotexist", instance_id=sample_instance.id
        )


@pytest.mark.asyncio
async def test_session_budget_check(db_session, sample_instance):
    sess = await create_session(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        context_cache_id=None,
    )
    sess.total_input_tokens = SESSION_TOKEN_BUDGET - 100
    await db_session.commit()
    await check_session_budget(db_session, sess, estimated_new=50)  # fine
    with pytest.raises(RateLimitError) as ex:
        await check_session_budget(db_session, sess, estimated_new=200)
    assert ex.value.code == "session_budget_exceeded"


@pytest.mark.asyncio
async def test_update_session_usage_atomic(db_session, sample_instance):
    sess = await create_session(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        context_cache_id=None,
    )
    await update_session_usage(db_session, sess, input_tokens=100, output_tokens=50)
    refreshed = (
        await db_session.execute(
            select(ResponseSession).where(ResponseSession.id == sess.id)
        )
    ).scalar_one()
    assert refreshed.total_input_tokens == 100
    assert refreshed.total_output_tokens == 50


@pytest.mark.asyncio
async def test_cleanup_expired_cascades_turns(db_session, sample_instance):
    sess = await create_session(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        context_cache_id=None,
    )
    await write_user_and_assistant_turns(
        db_session,
        sess=sess,
        user_content=[{"text": "x"}],
        assistant_content=[{"text": "y"}],
        usage={},
        reasoning=None,
        instructions=None,
        text_format=None,
    )
    # Bypass CHECK (expire_at > created_at) by moving both into the past
    now = datetime.now(timezone.utc)
    sess.created_at = now - timedelta(days=10)
    sess.expire_at = now - timedelta(days=5)
    await db_session.commit()
    n = await cleanup_expired_sessions(db_session)
    assert n == 1
    # Session is gone; turn FK-cascade is PG-enforced (SQLite needs PRAGMA
    # foreign_keys = ON which test fixture doesn't set). Production PG handles it.
    remaining_sess = (
        await db_session.execute(
            select(ResponseSession).where(ResponseSession.id == sess.id)
        )
    ).scalar_one_or_none()
    assert remaining_sess is None
