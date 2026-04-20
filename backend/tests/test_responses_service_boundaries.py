"""Extra boundary tests for responses_service that the original suite doesn't cover."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from src.errors import (
    NotFoundError,
    PermissionError as NousPermissionError,
)
from src.services.responses_service import (
    create_session,
    fetch_session_for_turn,
    write_partial_assistant_turn,
    write_user_and_assistant_turns,
    assemble_history_for_response,
    compact_messages,
    decode_content,
    encode_content,
)


@pytest.mark.asyncio
async def test_fetch_session_returns_404_when_turn_missing(db_session, sample_instance):
    with pytest.raises(NotFoundError) as ex:
        await fetch_session_for_turn(db_session, "resp-doesnotexist", sample_instance.id)
    assert ex.value.code == "previous_response_not_found"


@pytest.mark.asyncio
async def test_fetch_session_404_when_session_expired(db_session, sample_instance):
    sess = await create_session(
        db_session, instance_id=sample_instance.id, api_key_id=None,
        model="m", context_cache_id=None,
    )
    user, asst = await write_user_and_assistant_turns(
        db_session, sess=sess,
        user_content=[{"text": "x"}],
        assistant_content=[{"text": "y"}],
        usage={}, reasoning=None, instructions=None, text_format=None,
    )
    # Force-expire (and backdate created_at to satisfy CHECK constraint)
    now = datetime.now(timezone.utc)
    sess.created_at = now - timedelta(days=10)
    sess.expire_at = now - timedelta(seconds=5)
    await db_session.commit()
    with pytest.raises(NotFoundError) as ex:
        await fetch_session_for_turn(db_session, asst.id, sample_instance.id)
    assert ex.value.code == "previous_response_not_found"


@pytest.mark.asyncio
async def test_fetch_session_403_when_other_instance(db_session, sample_instance, other_instance):
    sess = await create_session(
        db_session, instance_id=sample_instance.id, api_key_id=None,
        model="m", context_cache_id=None,
    )
    _user, asst = await write_user_and_assistant_turns(
        db_session, sess=sess,
        user_content=[{"text": "x"}],
        assistant_content=[{"text": "y"}],
        usage={}, reasoning=None, instructions=None, text_format=None,
    )
    with pytest.raises(NousPermissionError) as ex:
        await fetch_session_for_turn(db_session, asst.id, other_instance.id)
    assert ex.value.code == "previous_response_wrong_instance"


@pytest.mark.asyncio
async def test_assemble_history_skips_non_assistant_user_rows(db_session, sample_instance):
    """If the schema ever lets through a 'system' / 'tool' row, it should be skipped."""
    sess = await create_session(
        db_session, instance_id=sample_instance.id, api_key_id=None,
        model="m", context_cache_id=None,
    )
    # Insert a 'tool' row directly to simulate junk
    from src.models.response_session import ResponseTurn
    from src.services.responses_service import new_turn_id
    db_session.add(ResponseTurn(
        id=new_turn_id(),
        session_id=sess.id,
        turn_idx=0,
        role="tool",  # not user/assistant
        content_compressed=encode_content([{"text": "ignored"}]),
    ))
    await db_session.commit()
    # Add a normal pair after
    _u, asst = await write_user_and_assistant_turns(
        db_session, sess=sess,
        user_content=[{"text": "hi"}],
        assistant_content=[{"text": "hello"}],
        usage={}, reasoning=None, instructions=None, text_format=None,
    )
    msgs, _ = await assemble_history_for_response(
        db_session, asst.id, instance_id=sample_instance.id
    )
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]  # tool row dropped


@pytest.mark.asyncio
async def test_partial_write_swallows_when_completed_already_landed(db_session, sample_instance):
    """If completed-path persisted at idx 0, partial path should swallow IntegrityError."""
    sess = await create_session(
        db_session, instance_id=sample_instance.id, api_key_id=None,
        model="m", context_cache_id=None,
    )
    await write_user_and_assistant_turns(
        db_session, sess=sess,
        user_content=[{"text": "u"}], assistant_content=[{"text": "a"}],
        usage={}, reasoning=None, instructions=None, text_format=None,
    )

    # Force max(turn_idx) lookup to return None so partial tries idx 0 → conflict
    original = db_session.execute

    async def fake_execute(stmt, *a, **kw):
        s = str(stmt).lower().replace(" ", "").replace("\n", "")
        if "max(response_turns.turn_idx)" in s or "max(turn_idx)" in s:
            class _R:
                def scalar(self_inner): return None
            return _R()
        return await original(stmt, *a, **kw)
    db_session.execute = fake_execute
    try:
        u_turn, a_turn = await write_partial_assistant_turn(
            db_session, sess=sess,
            user_content=[{"text": "race"}],
            partial_text="dropped",
            status="incomplete",
            incomplete_reason="connection_closed",
            instructions=None, text_format=None,
        )
        assert u_turn is None and a_turn is None
    finally:
        db_session.execute = original


def test_compact_force_keeps_last_when_everything_blown():
    """Even when even the last turn exceeds budget, compactor keeps it (don't drop everything)."""
    msgs = [{"role": "user", "content": "x" * 5000}]
    out, truncated = compact_messages(msgs, max_history_tokens=10)
    assert truncated is True
    assert len(out) == 1  # safety: keep the last turn


def test_decode_rejects_empty_payload():
    with pytest.raises(Exception):
        decode_content(b"", max_size=100)


def test_encode_decode_unicode_emoji_safe():
    content = [{"type": "input_text", "text": "🦀 안녕하세요 لمرحبا"}]
    assert decode_content(encode_content(content)) == content
