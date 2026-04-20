"""Service layer for Responses API (Step 4).

Handles: id generation, gzip codec, token estimation, compaction,
history assembly, atomic turn writes with concurrent-write 409 mapping,
session budget enforcement, expired-session cleanup.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import secrets
from datetime import datetime, timezone, timedelta

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.errors import (
    ConflictError,
    InvalidRequestError,
    NotFoundError,
    PermissionError as NousPermissionError,
    RateLimitError,
)
from src.models.response_session import ResponseSession, ResponseTurn

logger = logging.getLogger(__name__)

SESSION_TOKEN_BUDGET = 200_000


# ---------- ID generation ---------- #

def new_session_id() -> str:
    # token_urlsafe(12) produces 16 chars
    return f"session-{secrets.token_urlsafe(12)}"


def new_turn_id() -> str:
    return f"resp-{secrets.token_urlsafe(12)}"


# ---------- Content codec (gzip over JSON) ---------- #

def encode_content(content: list[dict]) -> bytes:
    return gzip.compress(json.dumps(content, ensure_ascii=False).encode("utf-8"))


def decode_content(data: bytes, max_size: int = 10_000_000) -> list[dict]:
    """Decompress with bounded size; Py3.12 gzip.decompress has no max_length
    kwarg, so we use GzipFile + bounded read."""
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        out = gz.read(max_size + 1)
    if len(out) > max_size:
        raise InvalidRequestError(
            "decompressed payload too large",
            code="payload_too_large",
        )
    return json.loads(out.decode("utf-8"))


# ---------- Token estimation + compaction ---------- #

def approx_tokens(messages: list[dict]) -> int:
    """Conservative OVER-estimate. BPE tokenizers vary; len/2+4 overshoots
    for English, ~matches Chinese. Prefer compacting too aggressively vs
    crashing on context_length_exceeded."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c) // 2 + 4
        elif isinstance(c, list):
            for item in c:
                t = item.get("text", "")
                if isinstance(t, str):
                    total += len(t) // 2 + 4
                else:
                    total += 200  # image / other placeholder
    return total


def compact_messages(
    messages: list[dict],
    *,
    max_history_tokens: int,
    keep_system: bool = True,
) -> tuple[list[dict], bool]:
    """Drop oldest non-system turns until estimated token count fits.
    Returns (compacted, was_truncated)."""
    if approx_tokens(messages) <= max_history_tokens:
        return messages, False
    system_msgs = (
        [m for m in messages if m.get("role") == "system"] if keep_system else []
    )
    rest = [m for m in messages if m.get("role") != "system"]
    while rest and approx_tokens(system_msgs + rest) > max_history_tokens:
        rest.pop(0)
    if not rest and messages:
        rest = messages[-1:]  # force-keep last turn as safety
    return system_msgs + rest, True


# ---------- Session + turn writes ---------- #

async def create_session(
    session: AsyncSession,
    *,
    instance_id: int,
    api_key_id: int | None,
    model: str,
    context_cache_id: str | None = None,
    ttl_seconds: int = 72 * 3600,
) -> ResponseSession:
    if not (60 <= ttl_seconds <= 604800):
        raise InvalidRequestError(
            "ttl_seconds out of range [60, 604800]",
            code="invalid_ttl",
        )
    now = datetime.now(timezone.utc)
    sess = ResponseSession(
        id=new_session_id(),
        instance_id=instance_id,
        api_key_id=api_key_id,
        model=model,
        context_cache_id=context_cache_id,
        total_input_tokens=0,
        total_output_tokens=0,
        expire_at=now + timedelta(seconds=ttl_seconds),
        created_at=now,
    )
    session.add(sess)
    await session.commit()
    await session.refresh(sess)
    return sess


async def write_user_and_assistant_turns(
    session: AsyncSession,
    *,
    sess: ResponseSession,
    user_content: list[dict],
    assistant_content: list[dict],
    usage: dict,
    reasoning: dict | None,
    instructions: str | None,
    text_format: dict | None,
    status: str = "completed",
    incomplete_reason: str | None = None,
) -> tuple[ResponseTurn, ResponseTurn]:
    """Insert user + assistant turn pair atomically. Maps any IntegrityError
    on the UNIQUE(session_id, turn_idx) constraint to ConflictError(409).
    """
    last_idx = (
        await session.execute(
            select(func.max(ResponseTurn.turn_idx)).where(
                ResponseTurn.session_id == sess.id
            )
        )
    ).scalar()
    last_idx = -1 if last_idx is None else last_idx

    user_turn = ResponseTurn(
        id=new_turn_id(),
        session_id=sess.id,
        turn_idx=last_idx + 1,
        role="user",
        content_compressed=encode_content(user_content),
    )
    asst_turn = ResponseTurn(
        id=new_turn_id(),
        session_id=sess.id,
        turn_idx=last_idx + 2,
        role="assistant",
        content_compressed=encode_content(assistant_content),
        usage_json=usage or None,
        reasoning_json=reasoning,
        instructions=instructions,
        text_format=text_format,
        status=status,
        incomplete_reason=incomplete_reason,
    )
    session.add_all([user_turn, asst_turn])
    try:
        await session.commit()
    except IntegrityError:
        # Only constraint that can fire: UNIQUE(session_id, turn_idx).
        # Don't filter on error-message substring (SQLite vs PG differ).
        await session.rollback()
        raise ConflictError(
            "concurrent write to the same session; refetch and retry",
            code="session_concurrent_write",
        )
    return user_turn, asst_turn


async def write_partial_assistant_turn(
    session: AsyncSession,
    *,
    sess: ResponseSession,
    user_content: list[dict],
    partial_text: str,
    status: str,
    incomplete_reason: str,
    instructions: str | None,
    text_format: dict | None,
) -> tuple[ResponseTurn | None, ResponseTurn | None]:
    """Write user + (possibly partial) assistant turn when stream is
    interrupted. Called from the background partial-write worker.

    If the UNIQUE constraint fires (a completed persist already landed),
    swallow — the completed path owns the row. Return (None, None) in that case.
    """
    partial_content = [
        {"type": "output_text", "text": partial_text, "annotations": []}
    ] if partial_text else []
    try:
        return await write_user_and_assistant_turns(
            session,
            sess=sess,
            user_content=user_content,
            assistant_content=partial_content,
            usage={},
            reasoning=None,
            instructions=instructions,
            text_format=text_format,
            status=status,
            incomplete_reason=incomplete_reason,
        )
    except ConflictError:
        logger.info("partial write skipped; completed path already persisted")
        return None, None


# ---------- History assembly ---------- #

async def fetch_session_for_turn(
    session: AsyncSession, turn_id: str, instance_id: int,
) -> ResponseSession:
    """Lookup the session containing a given turn, with permission + TTL checks."""
    turn = await session.get(ResponseTurn, turn_id)
    if turn is None:
        raise NotFoundError(
            "previous response not found",
            code="previous_response_not_found",
        )
    sess = await session.get(ResponseSession, turn.session_id)
    if sess is None:
        raise NotFoundError(
            "previous response not found (session missing)",
            code="previous_response_not_found",
        )
    expire_at = sess.expire_at
    if expire_at.tzinfo is None:
        expire_at = expire_at.replace(tzinfo=timezone.utc)
    if expire_at < datetime.now(timezone.utc):
        raise NotFoundError(
            "previous response expired",
            code="previous_response_not_found",
        )
    if sess.instance_id != instance_id:
        raise NousPermissionError(
            "previous response belongs to another instance",
            code="previous_response_wrong_instance",
        )
    return sess


async def assemble_history_for_response(
    session: AsyncSession, prev_resp_id: str, instance_id: int,
) -> tuple[list[dict], ResponseSession]:
    """Walk the session of prev_resp_id, return (messages, session).
    Returns only user/assistant turns (skips system / tool rows as defense)."""
    sess = await fetch_session_for_turn(session, prev_resp_id, instance_id)
    stmt = (
        select(ResponseTurn)
        .where(ResponseTurn.session_id == sess.id)
        .order_by(ResponseTurn.turn_idx.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    messages: list[dict] = []
    for r in rows:
        if r.role not in ("user", "assistant"):
            continue
        messages.append(
            {"role": r.role, "content": decode_content(r.content_compressed)}
        )
    return messages, sess


# ---------- Session budget ---------- #

async def check_session_budget(
    session: AsyncSession, sess: ResponseSession, *, estimated_new: int,
) -> None:
    projected = (sess.total_input_tokens or 0) + (sess.total_output_tokens or 0) + estimated_new
    if projected > SESSION_TOKEN_BUDGET:
        raise RateLimitError(
            f"Session token budget exceeded ({SESSION_TOKEN_BUDGET})",
            code="session_budget_exceeded",
        )


async def update_session_usage(
    session: AsyncSession,
    sess: ResponseSession,
    *,
    input_tokens: int,
    output_tokens: int,
) -> None:
    stmt = (
        update(ResponseSession)
        .where(ResponseSession.id == sess.id)
        .values(
            total_input_tokens=ResponseSession.total_input_tokens + input_tokens,
            total_output_tokens=ResponseSession.total_output_tokens + output_tokens,
        )
        .execution_options(synchronize_session="fetch")
    )
    await session.execute(stmt)
    await session.commit()


# ---------- Cleanup ---------- #

async def cleanup_expired_sessions(session: AsyncSession) -> int:
    """DELETE WHERE expire_at < now. FK CASCADE handles turns."""
    now = datetime.now(timezone.utc)
    stmt = (
        delete(ResponseSession)
        .where(ResponseSession.expire_at < now)
        .execution_options(synchronize_session="fetch")
    )
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount or 0


# ---------- agent binding ---------- #

def assert_agent_matches_session(
    sess: ResponseSession, request_agent: str | None
) -> str | None:
    """Validate request's agent against session's bound agent_id.

    Returns the effective agent_id to use, or raises 400 on mismatch.
    Called by continuation requests (with previous_response_id).
    """
    from fastapi import HTTPException

    if request_agent is None:
        return sess.agent_id  # 从 session 恢复（可能为 None）
    if sess.agent_id is None:
        raise HTTPException(
            400,
            {
                "error": "agent_session_mismatch",
                "message": f"session has no agent binding; got {request_agent!r}",
            },
        )
    if request_agent != sess.agent_id:
        raise HTTPException(
            400,
            {
                "error": "agent_session_mismatch",
                "message": (
                    f"session bound to {sess.agent_id!r}, got {request_agent!r}"
                ),
            },
        )
    return sess.agent_id
