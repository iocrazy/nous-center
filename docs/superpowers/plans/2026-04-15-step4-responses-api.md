# Step 4: Responses API (server-side conversation chain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/responses` (with optional streaming), `GET /v1/responses/{id}`, `GET /v1/responses` (cursor-paginated list), `DELETE /v1/responses/{id}`. Backed by event-sourcing (response_sessions + response_turns) in PG. Implements auto-compaction, session token budget, semantic stop reasons, and OpenAI-Responses-compatible SSE event protocol.

**Architecture:** Two new tables. `previous_response_id` stays in the public API (= `resp-xxx` turn id) but internally maps to a session via JOIN. Each turn stores gzip-compressed content (small per-row); chain assembly is a single SQL ORDER BY turn_idx. Streaming uses semantic events (response.created / output_text.delta / completed / error) and a background worker for partial-content persistence on client disconnect.

**Tech Stack:** SQLAlchemy 2.0 async + asyncpg, FastAPI, httpx, gzip stdlib, vLLM 0.19. Reuses Step 1 (errors, RequestIdMiddleware, SSE wrapper concept), Step 2 (thinking mapping), Step 3 (Context Cache helper).

**Spec:** `docs/superpowers/specs/2026-04-15-step4-responses-api-design.md`

---

## File Structure

**Create:**
- `backend/src/models/response_session.py` — both `ResponseSession` + `ResponseTurn` (single file; they're tightly coupled)
- `backend/src/services/responses_service.py` — assemble_history, write_response_turn, compaction, budget, cleanup
- `backend/src/api/routes/responses.py` — 4 endpoints + SSE envelope wrapper + lifespan worker hooks
- `backend/tests/test_responses_service.py` — service-layer unit tests
- `backend/tests/test_api_responses.py` — endpoint integration tests

**Modify:**
- `backend/src/errors.py` — add `ConflictError` subclass (Step 1 reverse-dependency)
- `backend/src/services/context_cache_service.py` — add `resolve_for_request` helper (extract dup logic)
- `backend/src/api/routes/openai_compat.py` — refactor existing context_id block to call `resolve_for_request`
- `backend/src/api/main.py` — register router; import model; init partial-write queue + worker in lifespan
- `backend/tests/conftest.py` — register new model

**Do not touch:** Step 1/2/3 routes, Context Cache schema, frontend.

---

## Task 1: Add ConflictError to Step 1 errors.py + test

**Files:**
- Modify: `backend/src/errors.py`
- Modify: `backend/tests/test_errors.py`

- [ ] **Step 1: Write failing test**
```python
# Append to backend/tests/test_errors.py
from src.errors import ConflictError

def test_conflict_error_is_409_invalid_request_type():
    e = ConflictError("dup write", code="session_concurrent_write")
    assert e.http_status == 409
    assert e.type == "invalid_request_error"  # matches Step 1's _HTTP_STATUS_TO_ERROR[409]
    assert e.to_dict()["error"]["code"] == "session_concurrent_write"
```

- [ ] **Step 2: Run, confirm fails (ImportError)**

- [ ] **Step 3: Implement**
Append to `backend/src/errors.py`:
```python
class ConflictError(NousError):
    type = "invalid_request_error"  # matches main.py:_HTTP_STATUS_TO_ERROR[409]
    http_status = 409
```

- [ ] **Step 4: Run tests, confirm pass**
- [ ] **Step 5: Commit** — `feat(errors): add ConflictError(409) for concurrent-write conflicts`

---

## Task 2: ResponseSession + ResponseTurn models + table migration

**Files:**
- Create: `backend/src/models/response_session.py`
- Modify: `backend/src/api/main.py` (model import for create_all)
- Modify: `backend/tests/conftest.py` (model import for create_all)

- [ ] **Step 1: Write the model**
```python
# backend/src/models/response_session.py
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from sqlalchemy import (
    JSON, BigInteger, CheckConstraint, Column, DateTime, ForeignKey,
    Index, Integer, LargeBinary, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from src.models.database import Base

JsonColumn = JSON().with_variant(JSONB(), "postgresql")


def _expires_at_default():
    return datetime.now(timezone.utc) + timedelta(seconds=72 * 3600)


class ResponseSession(Base):
    __tablename__ = "response_sessions"

    id = Column(String(64), primary_key=True)  # session-{token_urlsafe(12)}
    instance_id = Column(BigInteger,
        ForeignKey("service_instances.id", ondelete="CASCADE"),
        nullable=False, index=True)
    api_key_id = Column(BigInteger, nullable=True)
    model = Column(String(128), nullable=False)
    context_cache_id = Column(String(64), nullable=True)
    total_input_tokens = Column(BigInteger, nullable=False, default=0)
    total_output_tokens = Column(BigInteger, nullable=False, default=0)
    expire_at = Column(DateTime(timezone=True), nullable=False,
        default=_expires_at_default, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        # Keep CHECK simple; 7-day cap enforced at API layer (PG interval syntax
        # would break SQLite test fixture).
        CheckConstraint("expire_at > created_at",
                        name="response_session_expire_at_check"),
        Index("ix_response_sessions_instance_created",
              "instance_id", "created_at"),
    )


class ResponseTurn(Base):
    __tablename__ = "response_turns"

    id = Column(String(64), primary_key=True)  # resp-{token_urlsafe(12)}
    session_id = Column(String(64),
        ForeignKey("response_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True)
    turn_idx = Column(Integer, nullable=False)
    role = Column(String(20), nullable=False)
    content_compressed = Column(LargeBinary, nullable=False)
    usage_json = Column(JsonColumn, nullable=True)
    reasoning_json = Column(JsonColumn, nullable=True)
    instructions = Column(Text, nullable=True)
    text_format = Column(JsonColumn, nullable=True)
    status = Column(String(20), nullable=False, default="completed")
    incomplete_reason = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("session_id", "turn_idx",
                         name="uq_response_turn_session_idx"),
        Index("ix_response_turns_session_idx", "session_id", "turn_idx"),
    )
```

- [ ] **Step 2: Add explicit imports** (mirroring Step 3 pattern)
- `backend/src/api/main.py` (after `import src.models.context_cache`):
  ```python
  import src.models.response_session  # noqa: F401
  ```
- `backend/tests/conftest.py` (after `import src.models.context_cache`):
  ```python
  import src.models.response_session  # noqa: F401 — register model
  ```

- [ ] **Step 3: Apply migration to running PG**
```bash
cd backend
.venv/bin/python - <<'PY'
import asyncio
from src.models.database import Base, create_engine
from src.models import response_session  # noqa
async def go():
    eng = create_engine()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await eng.dispose()
asyncio.run(go())
PY
docker exec nous-center-postgres-1 psql -U nous_heygo -d nous_center \
  -c "\d response_sessions" -c "\d response_turns"
```
Expect: both tables present with FK + unique + indexes.

- [ ] **Step 4: Commit** — `feat(models): add ResponseSession + ResponseTurn for event-sourced responses`

---

## Task 3: responses_service.py — core logic + unit tests

**Files:**
- Create: `backend/src/services/responses_service.py`
- Create: `backend/tests/test_responses_service.py`

The service layer encapsulates: id generation, content codec, token estimation, compaction, history assembly, write turn (with concurrent-write 409), session budget check + update, cleanup. Tested in isolation with the SQLite db_session fixture.

- [ ] **Step 1: Write failing tests**
```python
# backend/tests/test_responses_service.py
import pytest, asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy import select

from src.errors import ConflictError, InvalidRequestError, NotFoundError
from src.models.response_session import ResponseSession, ResponseTurn
from src.services.responses_service import (
    new_session_id, new_turn_id, encode_content, decode_content,
    approx_tokens, compact_messages,
    create_session, write_user_and_assistant_turns, write_partial_assistant_turn,
    assemble_history_for_response, fetch_session_for_turn,
    update_session_usage, check_session_budget,
    cleanup_expired_sessions,
    SESSION_TOKEN_BUDGET,
)


def test_id_format():
    sid = new_session_id()
    rid = new_turn_id()
    assert sid.startswith("session-") and len(sid) == len("session-") + 16
    assert rid.startswith("resp-") and len(rid) == len("resp-") + 16


def test_encode_decode_roundtrip():
    content = [{"type": "input_text", "text": "hello world 你好"}]
    enc = encode_content(content)
    assert isinstance(enc, bytes) and len(enc) > 0
    assert decode_content(enc) == content


def test_decode_rejects_oversized():
    # Forge a payload that decompresses huge
    import gzip, json
    huge = json.dumps([{"text": "x" * 11_000_000}]).encode()
    payload = gzip.compress(huge)
    with pytest.raises(InvalidRequestError) as ex:
        decode_content(payload, max_size=10_000_000)
    assert ex.value.code == "payload_too_large"


def test_approx_tokens_overestimates():
    # English: should never under-estimate badly
    msgs = [{"role": "user", "content": "the quick brown fox"}]
    assert approx_tokens(msgs) >= 4  # 4 tokens minimum
    # Chinese: over-estimate is safer than under
    msgs = [{"role": "user", "content": "你好世界你好世界你好世界"}]
    assert approx_tokens(msgs) >= 12


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
    sess = await create_session(db_session,
        instance_id=sample_instance.id, api_key_id=None,
        model="qwen3.5", context_cache_id=None)
    assert sess.id.startswith("session-")
    user_turn, asst_turn = await write_user_and_assistant_turns(
        db_session, sess=sess,
        user_content=[{"type": "input_text", "text": "hi"}],
        assistant_content=[{"type": "output_text", "text": "hello"}],
        usage={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        reasoning=None, instructions=None, text_format=None,
    )
    assert user_turn.turn_idx == 0
    assert asst_turn.turn_idx == 1
    assert asst_turn.role == "assistant"


@pytest.mark.asyncio
async def test_concurrent_turn_write_raises_conflict(
    db_session, sample_instance, monkeypatch,
):
    """Simulate the TOCTOU race: SELECT max(turn_idx) returns N; before our
    INSERT lands, another writer takes idx N+1. Our commit then violates
    UNIQUE and the service maps to ConflictError."""
    sess = await create_session(db_session,
        instance_id=sample_instance.id, api_key_id=None, model="m",
        context_cache_id=None)
    # Pre-seed turns at idx 0 and 1 to be the "concurrent winner"
    db_session.add(ResponseTurn(
        id=new_turn_id(), session_id=sess.id, turn_idx=0,
        role="user", content_compressed=encode_content([{"text": "x"}]),
    ))
    db_session.add(ResponseTurn(
        id=new_turn_id(), session_id=sess.id, turn_idx=1,
        role="assistant", content_compressed=encode_content([{"text": "y"}]),
    ))
    await db_session.commit()
    # Monkeypatch the SELECT max() to return -1 so write_user_and_assistant_turns
    # tries to insert at idx 0,1 — which already exist -> IntegrityError -> ConflictError
    from src.services import responses_service as svc
    real_func = svc.write_user_and_assistant_turns
    # Easier: just call write_user_and_assistant_turns; max() returns 1 -> tries 2,3 OK.
    # To force collision, we manually reset by inserting at idx 2 too:
    db_session.add(ResponseTurn(
        id=new_turn_id(), session_id=sess.id, turn_idx=2,
        role="user", content_compressed=encode_content([{"text": "z"}]),
    ))
    await db_session.commit()
    # Now max = 2; our service computes idx 3,4 — fine, no race here.
    # To genuinely trigger: patch func.max to return a stale value
    from unittest.mock import patch
    with patch("src.services.responses_service.func.max",
               lambda col: type("M", (), {"label": lambda *a, **k: 0})()):
        # crude — alt: patch select() directly. Real test harness should
        # use respx-like SQL injection. For now, accept that this single
        # test verifies the *mapping* (IntegrityError -> ConflictError);
        # the real race is covered by the E2E test in test_api_responses.py.
        with pytest.raises(ConflictError) as ex:
            await svc.write_user_and_assistant_turns(
                db_session, sess=sess,
                user_content=[{"text": "race-user"}],
                assistant_content=[{"text": "race-assistant"}],
                usage={}, reasoning=None, instructions=None, text_format=None,
            )
    assert ex.value.code == "session_concurrent_write"
    assert ex.value.http_status == 409


@pytest.mark.asyncio
async def test_assemble_history_orders_by_turn_idx(db_session, sample_instance):
    sess = await create_session(db_session,
        instance_id=sample_instance.id, api_key_id=None, model="m",
        context_cache_id=None)
    # Write 3 turn pairs
    for i in range(3):
        await write_user_and_assistant_turns(db_session, sess=sess,
            user_content=[{"type": "input_text", "text": f"u{i}"}],
            assistant_content=[{"type": "output_text", "text": f"a{i}"}],
            usage={}, reasoning=None, instructions=None, text_format=None)
    # Get the last assistant turn id
    last_asst = (await db_session.execute(
        select(ResponseTurn).where(
            ResponseTurn.session_id == sess.id,
            ResponseTurn.role == "assistant",
        ).order_by(ResponseTurn.turn_idx.desc())
    )).scalars().first()

    msgs, fetched_sess = await assemble_history_for_response(
        db_session, last_asst.id, instance_id=sample_instance.id)
    assert fetched_sess.id == sess.id
    assert len(msgs) == 6  # 3 user + 3 assistant
    # Order check
    roles = [m["role"] for m in msgs]
    assert roles == ["user","assistant","user","assistant","user","assistant"]


@pytest.mark.asyncio
async def test_assemble_history_404_on_unknown(db_session, sample_instance):
    with pytest.raises(NotFoundError):
        await assemble_history_for_response(
            db_session, "resp-doesnotexist", instance_id=sample_instance.id)


@pytest.mark.asyncio
async def test_session_budget_check(db_session, sample_instance):
    sess = await create_session(db_session,
        instance_id=sample_instance.id, api_key_id=None, model="m",
        context_cache_id=None)
    sess.total_input_tokens = SESSION_TOKEN_BUDGET - 100
    await db_session.commit()
    # 50 more is fine
    await check_session_budget(db_session, sess, estimated_new=50)
    # 200 more exceeds
    with pytest.raises(Exception) as ex:
        await check_session_budget(db_session, sess, estimated_new=200)
    assert "session_budget_exceeded" in str(ex.value).lower() or \
        getattr(ex.value, "code", "") == "session_budget_exceeded"


@pytest.mark.asyncio
async def test_update_session_usage_atomic(db_session, sample_instance):
    sess = await create_session(db_session,
        instance_id=sample_instance.id, api_key_id=None, model="m",
        context_cache_id=None)
    await update_session_usage(db_session, sess, input_tokens=100, output_tokens=50)
    refreshed = (await db_session.execute(
        select(ResponseSession).where(ResponseSession.id == sess.id))).scalar_one()
    assert refreshed.total_input_tokens == 100
    assert refreshed.total_output_tokens == 50


@pytest.mark.asyncio
async def test_cleanup_expired_cascades_turns(db_session, sample_instance):
    sess = await create_session(db_session,
        instance_id=sample_instance.id, api_key_id=None, model="m",
        context_cache_id=None)
    await write_user_and_assistant_turns(db_session, sess=sess,
        user_content=[{"text": "x"}], assistant_content=[{"text": "y"}],
        usage={}, reasoning=None, instructions=None, text_format=None)
    sess.expire_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    await db_session.commit()
    n = await cleanup_expired_sessions(db_session)
    assert n == 1
    # turns are cascade-deleted
    remaining = (await db_session.execute(
        select(ResponseTurn).where(ResponseTurn.session_id == sess.id)
    )).scalars().all()
    assert remaining == []
```

- [ ] **Step 2: Run tests, confirm fail (module missing)**

- [ ] **Step 3: Implement service**
Implement `backend/src/services/responses_service.py` with all functions imported above. Key details:
- `SESSION_TOKEN_BUDGET = 200_000`
- `encode_content` = `gzip.compress(json.dumps(...).encode())`
- `decode_content`: use `gzip.GzipFile + read(max_size+1)` (NOT `gzip.decompress(max_length=)` — that kwarg doesn't exist in py3.12)
- `write_user_and_assistant_turns`:
  ```python
  last_idx = (await session.execute(
      select(func.max(ResponseTurn.turn_idx)).where(
          ResponseTurn.session_id == sess.id))).scalar() or -1
  user_turn = ResponseTurn(id=new_turn_id(), session_id=sess.id,
      turn_idx=last_idx + 1, role="user",
      content_compressed=encode_content(user_content))
  asst_turn = ResponseTurn(id=new_turn_id(), session_id=sess.id,
      turn_idx=last_idx + 2, role="assistant",
      content_compressed=encode_content(assistant_content),
      usage_json=usage, reasoning_json=reasoning,
      instructions=instructions, text_format=text_format,
      status="completed", incomplete_reason=None)
  session.add_all([user_turn, asst_turn])
  try:
      await session.commit()
  except IntegrityError:
      # Any integrity violation on this insert path is a concurrent-write race
      # (only constraint that can fire here is UNIQUE(session_id, turn_idx)).
      # SQLite + PG produce different error message strings, so don't filter
      # on substring — just map.
      await session.rollback()
      raise ConflictError(
          "concurrent write to the same session; refetch and retry",
          code="session_concurrent_write")
  return user_turn, asst_turn
  ```
- `write_partial_assistant_turn(sess, accumulated_text, status, reason)` — writes a single assistant turn with the streamed text-so-far + status + incomplete_reason. Used by the partial-write worker.
- `update_session_usage` — atomic UPDATE with `synchronize_session="fetch"` to keep SQLite test happy (Step 3 pattern)
- `cleanup_expired_sessions` — DELETE WHERE expire_at < now; FK CASCADE handles turns

- [ ] **Step 4: Run tests, confirm pass**
- [ ] **Step 5: Commit** — `feat(responses): service layer (encode/decode, compact, assemble, write, budget, cleanup)`

---

## Task 4: Refactor — extract resolve_for_request into context_cache_service

**Files:**
- Modify: `backend/src/services/context_cache_service.py` (add `resolve_for_request`)
- Modify: `backend/src/api/routes/openai_compat.py` (replace inline block with helper call)

This is preparatory cleanup so Task 6 (POST /v1/responses) can call the same helper without duplicating the 50-line context_id lookup.

- [ ] **Step 1: Add `resolve_for_request` to `context_cache_service.py`**
```python
async def resolve_for_request(
    session: AsyncSession,
    *,
    context_id: str | None,
    instance_id: int,
    engine_name: str,
) -> tuple[list[dict] | None, int | None]:
    """Common cache lookup. Returns (messages, ttl) or (None, None).

    Raises NotFoundError / PermissionError / InvalidRequestError on validation failures.
    Used by both /v1/chat/completions and /v1/responses.
    """
    from src.errors import (
        InvalidRequestError, NotFoundError,
        PermissionError as NousPermissionError,
    )
    if not context_id:
        return None, None
    cache = await fetch_active_cache(session, context_id, instance_id)
    if cache is None:
        other = await fetch_cache_any_instance(session, context_id)
        if other is not None and other.instance_id != instance_id:
            raise NousPermissionError(
                "Cache belongs to another instance",
                code="context_wrong_instance")
        raise NotFoundError(
            "Context cache not found or expired",
            code="context_not_found")
    if cache.model != engine_name:
        raise InvalidRequestError(
            f"Cache was created for '{cache.model}', not '{engine_name}'",
            code="context_model_mismatch", param="model")
    return list(cache.messages_json), cache.ttl_seconds
```

- [ ] **Step 2: Refactor `openai_compat.py` to call helper**
Find the `if context_id:` block (~lines 132-179). Replace inline lookup + error mapping with:
```python
if context_id:
    from src.services.context_cache_service import resolve_for_request
    from src.models.database import create_session_factory as _csf
    sf = _csf()
    async with sf() as cache_session:
        cached_messages, cached_ttl = await resolve_for_request(
            cache_session, context_id=context_id,
            instance_id=instance.id, engine_name=engine_name)
    if cached_messages:
        body["messages"] = cached_messages + list(body.get("messages", []))
        # ... existing _bump task code unchanged
```

- [ ] **Step 3: Run existing context cache test suite — confirm no regression**
```bash
.venv/bin/pytest backend/tests/test_context_cache_service.py backend/tests/test_api_context_cache.py -v
```
All Step 3 tests should still pass.

- [ ] **Step 4: Commit** — `refactor(context-cache): extract resolve_for_request shared helper`

---

## Task 5: SSE wrapper + lifespan partial-write worker

**Files:**
- Create: `backend/src/api/routes/responses.py` — SSE wrapper + worker queue + worker function (endpoints come in Task 6)
- Modify: `backend/src/api/main.py` — initialize queue + spawn worker in lifespan

- [ ] **Step 1: Implement wrapper + worker queue (no endpoints yet)**

Create `backend/src/api/routes/responses.py` with **complete imports needed for Tasks 5-7**:
```python
from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import verify_bearer_token
from src.errors import (
    APIError, ConflictError, InvalidRequestError, NotFoundError,
    NousError, PermissionError as NousPermissionError,
)
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.response_session import ResponseSession, ResponseTurn
from src.models.service_instance import ServiceInstance
from src.services.context_cache_service import resolve_for_request
from src.services.responses_service import (
    SESSION_TOKEN_BUDGET,
    approx_tokens, assemble_history_for_response,
    check_session_budget, compact_messages, create_session,
    decode_content, update_session_usage,
    write_partial_assistant_turn, write_user_and_assistant_turns,
)


def _to_utc(dt: datetime | None) -> datetime | None:
    """SQLite stores tz-naive; PG returns tz-aware. Normalize to UTC-aware
    for safe comparisons. Returns None unchanged."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

logger = logging.getLogger(__name__)
router = APIRouter(tags=["responses"])

# Module-level queue for partial-write tasks (survives request cancellation).
# Initialized in main.py lifespan; until then, drops silently.
_partial_write_queue: asyncio.Queue | None = None


def _set_queue(q: asyncio.Queue | None) -> None:
    global _partial_write_queue
    _partial_write_queue = q


def schedule_partial_write(persist_fn, *args) -> None:
    if _partial_write_queue is None:
        logger.warning("partial_write_queue not initialized; drop")
        return
    try:
        _partial_write_queue.put_nowait((persist_fn, args))
    except asyncio.QueueFull:
        logger.error("partial-write queue full; drop")


async def partial_write_worker():
    """Drains partial-write requests serially. Started in lifespan."""
    assert _partial_write_queue is not None
    while True:
        item = await _partial_write_queue.get()
        if item is None:  # shutdown sentinel
            break
        persist_fn, args = item
        try:
            await persist_fn(*args)
        except Exception:
            logger.exception("partial-write worker failed")
        finally:
            _partial_write_queue.task_done()


def _sse_format(evt: str, payload: dict) -> str:
    return f"event: {evt}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def responses_sse_envelope(inner, persist_partial_fn, request_id: str | None):
    """Wrap an inner async-iter of (evt_type, payload_dict) tuples.
    Always emits exactly one `data: [DONE]\\n\\n`. Injects request_id into each event.

    Inner MUST yield response.created AFTER first vLLM byte (not at start).
    On client disconnect, accumulated text is enqueued for background persistence.
    """
    accumulated_text = ""
    cancelled = False
    try:
        async for evt_type, payload in inner:
            payload = dict(payload)
            if request_id and "request_id" not in payload:
                payload["request_id"] = request_id
            if evt_type == "response.output_text.delta":
                accumulated_text += payload.get("delta", "")
            yield _sse_format(evt_type, payload)
    except asyncio.CancelledError:
        cancelled = True
        # schedule_partial_write is sync (put_nowait) — safe under cancellation
        schedule_partial_write(
            persist_partial_fn,
            accumulated_text,
            "incomplete",
            "connection_closed",
        )
        raise
    except NousError as e:
        err_payload = {"type": "error", "error": e.to_dict()["error"]}
        if request_id:
            err_payload["request_id"] = request_id
        yield _sse_format("error", err_payload)
    except Exception:
        logger.exception("responses stream failure")
        err = APIError("Internal server error", code="internal_error")
        err_payload = {"type": "error", "error": err.to_dict()["error"]}
        if request_id:
            err_payload["request_id"] = request_id
        yield _sse_format("error", err_payload)
    finally:
        # Don't emit DONE on cancellation (socket already gone; would error on yield)
        if not cancelled:
            yield "data: [DONE]\n\n"
```

- [ ] **Step 2: Wire into `main.py` lifespan**
```python
# In lifespan, after existing cache_cleanup_task spawn:
from src.api.routes import responses as responses_routes
responses_routes._set_queue(asyncio.Queue(maxsize=1000))
partial_worker = asyncio.create_task(responses_routes.partial_write_worker())

try:
    yield
finally:
    cache_cleanup_task.cancel()
    try:
        await cache_cleanup_task
    except asyncio.CancelledError:
        pass
    # Drain partial-write worker
    if responses_routes._partial_write_queue is not None:
        await responses_routes._partial_write_queue.put(None)
        try:
            await asyncio.wait_for(partial_worker, timeout=5.0)
        except asyncio.TimeoutError:
            partial_worker.cancel()
```

- [ ] **Step 3: Smoke test the wrapper standalone**
```bash
.venv/bin/python -c "
import asyncio
from src.api.routes import responses as r
async def fake_inner():
    yield ('response.created', {'response_id': 'resp-x'})
    yield ('response.output_text.delta', {'delta': 'hello', 'item_id': 'msg-x', 'output_index': 0, 'content_index': 0})
    yield ('response.completed', {'response': {'id': 'resp-x'}})
async def fake_persist(*args): print('persist:', args)
async def main():
    async for chunk in r.responses_sse_envelope(fake_inner(), fake_persist, 'req-test'):
        print(repr(chunk))
asyncio.run(main())
"
```
Expected: 3 SSE-formatted strings + `data: [DONE]\n\n` at end.

- [ ] **Step 4: Commit** — `feat(responses): SSE envelope + partial-write background worker`

---

## Task 6: POST /v1/responses

**⚠️ Split note:** This is the largest task. Break into commits:
- **Task 6a**: Pydantic schema + helpers + non-streaming path → ship + test
- **Task 6b**: streaming inner generator + persist_partial → ship + test

Two commits prevent monolithic 350-line diff. 6a is independently shippable; users without streaming get value first. 6b layers on top.

---

### Task 6a: POST /v1/responses (non-streaming only)

**Files:**
- Modify: `backend/src/api/routes/responses.py` — add the endpoint + inner generator
- Modify: `backend/src/api/main.py` — register router (`app.include_router(responses_routes.router)`)

This is the biggest task. Has two execution paths (sync / streaming) but they share input normalization, history assembly, compaction, budget check, vLLM call setup.

- [ ] **Step 1: Pydantic request schema**
```python
# At top of responses.py
from pydantic import BaseModel, Field
from typing import Any

class _ThinkingCfg(BaseModel):
    type: str = "auto"

class _ReasoningCfg(BaseModel):
    effort: str = "medium"

class _TextFormatCfg(BaseModel):
    type: str = "text"
    schema: dict | None = Field(default=None, alias="json_schema")

class _TextCfg(BaseModel):
    format: _TextFormatCfg = Field(default_factory=_TextFormatCfg)

class CreateResponseRequest(BaseModel):
    model: str
    input: str | list[Any]
    previous_response_id: str | None = None
    context_id: str | None = None
    instructions: str | None = None
    thinking: _ThinkingCfg = Field(default_factory=_ThinkingCfg)
    reasoning: _ReasoningCfg = Field(default_factory=_ReasoningCfg)
    store: bool = True
    expire_at: int | None = None  # UTC unix seconds; default = now+72h, max +7d
    stream: bool = False
    text: _TextCfg = Field(default_factory=_TextCfg)
```

- [ ] **Step 2: Helper functions**
```python
def normalize_input(input_field) -> list[dict]:
    """str -> [{role:user, content:[{type:input_text,text:...}]}]; list passes through."""
    if isinstance(input_field, str):
        return [{"role": "user", "content": [{"type": "input_text", "text": input_field}]}]
    if isinstance(input_field, list):
        if input_field and all(
            isinstance(it, dict) and it.get("type", "").startswith("input_")
            for it in input_field
        ):
            return [{"role": "user", "content": input_field}]
        return input_field
    from src.errors import InvalidRequestError
    raise InvalidRequestError("input must be string or array",
                              param="input", code="invalid_input")


def resolve_image(item: dict) -> dict:
    """input_image -> OpenAI chat-format image_url message content."""
    if item.get("file_id"):
        from src.errors import NousError
        # Step 5 (Files API) reservation
        err = NousError("file_id input not supported until Step 5 (Files API)",
                        code="image_file_id_not_implemented")
        err.http_status = 501
        raise err
    if item.get("image_url"):
        return {
            "type": "image_url",
            "image_url": {
                "url": item["image_url"],
                "detail": item.get("detail", "auto"),
            },
        }
    from src.errors import InvalidRequestError
    raise InvalidRequestError("input_image requires image_url or file_id",
                              param="input_image", code="invalid_image_input")


def transform_inputs_to_chat_messages(inputs: list[dict]) -> list[dict]:
    """Convert input items to chat/completions message format vLLM understands."""
    out = []
    for msg in inputs:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            transformed = []
            for item in content:
                t = item.get("type", "")
                if t == "input_text":
                    transformed.append({"type": "text", "text": item.get("text", "")})
                elif t == "input_image":
                    transformed.append(resolve_image(item))
                elif t == "output_text":
                    # Replaying a prior assistant turn (e.g. chain bootstrap).
                    # vLLM expects 'text' content type, not 'output_text'.
                    transformed.append({"type": "text", "text": item.get("text", "")})
                else:
                    # Unknown type — pass through; vLLM will reject if bad
                    transformed.append(item)
            out.append({"role": role, "content": transformed})
    return out
```

- [ ] **Step 3: The endpoint**
```python
@router.post("/v1/responses")
async def create_response(
    req: CreateResponseRequest,
    request: Request,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, api_key = auth
    if instance.source_type != "model":
        raise InvalidRequestError("Responses only on model-type instances",
                                   code="not_a_model_instance")
    engine_name = instance.source_name or str(instance.source_id)

    # Adapter resolution (same pattern as /v1/chat/completions and Step 3)
    model_mgr = getattr(request.app.state, "model_manager", None)
    if model_mgr is None:
        raise APIError("Model manager unavailable", code="model_manager_missing")
    adapter = model_mgr.get_adapter(engine_name)
    if adapter is None or not adapter.is_loaded:
        raise APIError(f"Model '{engine_name}' is not loaded",
                        code="model_not_loaded")
    base_url = getattr(adapter, "base_url", None)
    if not base_url:
        raise APIError("Model has no inference endpoint",
                        code="no_inference_endpoint")
    max_model_len = getattr(adapter, "max_model_len", 4096) or 4096

    # Step 3 helper (Task 4)
    cached_messages, _ttl = await resolve_for_request(
        session, context_id=req.context_id,
        instance_id=instance.id, engine_name=engine_name)

    # Walk previous chain
    previous_messages: list[dict] = []
    sess: ResponseSession | None = None
    if req.previous_response_id:
        previous_messages, sess = await assemble_history_for_response(
            session, req.previous_response_id, instance_id=instance.id)
        if sess.model != engine_name:
            raise InvalidRequestError(
                f"Previous response was for '{sess.model}', not '{engine_name}'",
                code="previous_response_model_mismatch", param="model")
        # Doc-convention warning if both context_id + previous_response_id
        if req.context_id:
            logger.warning(
                "both context_id (%s) and previous_response_id (%s) provided; "
                "first turn already had cache, skip-merge per doc convention",
                req.context_id, req.previous_response_id)
            cached_messages = None  # don't double-prepend

    # Normalize new input
    new_input_messages = transform_inputs_to_chat_messages(
        normalize_input(req.input))

    # Assemble per MESSAGES_ORDER constant: context → chain → instructions → input.
    # NOTE: instructions go BEFORE new_input but AFTER the cache+chain prefix
    # (insert at the boundary, not at position 0). This way previous system from
    # cache stays at the head; per-request instructions are a recent override.
    messages: list[dict] = []
    if cached_messages:
        messages.extend(cached_messages)
    messages.extend(previous_messages)
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})
    messages.extend(new_input_messages)

    # Compaction
    compacted, history_truncated = compact_messages(
        messages, max_history_tokens=max_model_len - 2048)
    if approx_tokens(compacted) > max_model_len - 2048:
        raise InvalidRequestError(
            f"input alone exceeds max_history_tokens",
            code="input_too_long_for_model", param="input")
    messages = compacted

    # Budget check
    estimated_input = approx_tokens(messages)
    if sess is not None:
        await check_session_budget(session, sess, estimated_new=estimated_input)

    # Create new session if no previous chain
    if sess is None:
        sess = await create_session(session,
            instance_id=instance.id, api_key_id=api_key.id,
            model=engine_name, context_cache_id=req.context_id)

    # Build vLLM body
    vllm_body = {
        "model": "",
        "messages": messages,
        "max_tokens": 2048,  # TODO: read from req.max_output_tokens once added
        "stream": req.stream,
    }
    # Step 2: thinking mapping. _maybe_inject_thinking mutates body in-place
    # and returns None, so it must be called on the actual vllm_body, not a
    # throwaway dict. Set the input field on vllm_body first.
    vllm_body["thinking"] = req.thinking.model_dump()
    from src.api.routes.openai_compat import _maybe_inject_thinking
    _maybe_inject_thinking(vllm_body, engine_name)
    # _maybe_inject_thinking pops `thinking` key after processing.
    if req.text.format and req.text.format.type == "json_schema" and req.text.format.schema:
        vllm_body["response_format"] = {
            "type": "json_schema",
            "json_schema": req.text.format.schema,
        }
    elif req.text.format.type == "json_object":
        vllm_body["response_format"] = {"type": "json_object"}

    request_id = getattr(request.state, "request_id", None)

    if not req.stream:
        # ---- Non-streaming path ----
        async with httpx.AsyncClient(timeout=300, proxy=None) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/v1/chat/completions", json=vllm_body)
        if resp.status_code != 200:
            err_text = resp.text[:500]
            if 400 <= resp.status_code < 500:
                raise InvalidRequestError(err_text, code="upstream_bad_request")
            raise APIError("vLLM error", code="upstream_error")
        data = resp.json()
        choice = data["choices"][0]
        assistant_text = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason", "stop")
        usage = data.get("usage", {})

        # Persist user + assistant turns
        user_content = new_input_messages[-1]["content"] if new_input_messages else []
        asst_content = [{"type": "output_text", "text": assistant_text, "annotations": []}]
        _, asst_turn = await write_user_and_assistant_turns(
            session, sess=sess,
            user_content=user_content, assistant_content=asst_content,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "input_tokens_details": {
                    "cached_tokens": usage.get("prompt_tokens_details", {})
                                          .get("cached_tokens", 0)
                },
            },
            reasoning=None,
            instructions=req.instructions,
            text_format=req.text.model_dump(),
        )
        # Update session usage
        await update_session_usage(session, sess,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0))
        # Record llm_usage (Step 1 pattern)
        from src.services.usage_service import record_llm_usage
        await record_llm_usage(model=engine_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            duration_ms=0,  # timer not added; future improvement
            instance_id=instance.id, api_key_id=api_key.id)

        # Map finish_reason -> status / incomplete_details
        status = "completed"
        incomplete_details = None
        if finish_reason == "length":
            status = "incomplete"
            incomplete_details = {"reason": "max_output_tokens"}
        elif history_truncated:
            # finish was natural but history was clipped
            status = "completed"
            # Still flag at top level

        return {
            "id": asst_turn.id,
            "object": "response",
            "status": status,
            "incomplete_details": incomplete_details,
            "created_at": int(asst_turn.created_at.timestamp()),
            "model": engine_name,
            "previous_response_id": req.previous_response_id,
            "instructions": req.instructions,
            "store": req.store,
            "expire_at": int(sess.expire_at.timestamp()),
            "output": [{
                "type": "message",
                "id": f"msg-{asst_turn.id[5:]}",
                "role": "assistant",
                "content": asst_content,
            }],
            "usage": asst_turn.usage_json,
            "history_truncated": history_truncated,
        }

    # ↑↑↑ Task 6a stops here. If req.stream=True, return 400 not_implemented
    # (or just skip the if-not-stream guard and add the streaming branch in 6b).
    if req.stream:
        raise InvalidRequestError("streaming not yet implemented in 6a",
                                   code="streaming_pending")

```

- [ ] **Step 4a: Commit Task 6a**
- [ ] **Step 5a: Run integration test (non-streaming POST)**

---

### Task 6b: POST /v1/responses streaming branch

Add the `if req.stream:` branch BELOW the non-streaming `return`. Append to the same `create_response` function:

```python
    # ---- Streaming path ----
    # Inner generator: do vLLM streaming, yield (evt_type, payload).
    accumulated_text = ""
    latest_usage = {}
    completed_persist_done = False  # sentinel to prevent partial-write race

    async def inner():
        nonlocal accumulated_text, latest_usage, completed_persist_done
        first_byte = False
        async with httpx.AsyncClient(timeout=300, proxy=None) as client:
            async with client.stream(
                "POST", f"{base_url.rstrip('/')}/v1/chat/completions",
                json=vllm_body,
            ) as resp:
                if resp.status_code != 200:
                    err_text = (await resp.aread()).decode(errors="replace")[:500]
                    if 400 <= resp.status_code < 500:
                        raise InvalidRequestError(err_text, code="upstream_bad_request")
                    raise APIError("vLLM error", code="upstream_error")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if not first_byte:
                        first_byte = True
                        # Emit response.created NOW (after first byte from vLLM)
                        yield ("response.created", {
                            "type": "response.created",
                            "response": {
                                "id": "resp-pending",  # filled with real id at completion
                                "status": "in_progress",
                                "model": engine_name,
                            },
                        })
                        # Then output_item / content_part
                        yield ("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {"id": "msg-pending", "type": "message",
                                     "role": "assistant", "content": []},
                        })
                        yield ("response.content_part.added", {
                            "type": "response.content_part.added",
                            "item_id": "msg-pending", "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        })
                    if delta:
                        accumulated_text += delta
                        yield ("response.output_text.delta", {
                            "type": "response.output_text.delta",
                            "item_id": "msg-pending",
                            "output_index": 0, "content_index": 0,
                            "delta": delta,
                        })
                    # Capture usage when present; persist ONLY on finish_reason.
                    # vLLM may send usage in one chunk and finish_reason later (or both
                    # together). We must not persist twice or persist before stream ends.
                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        latest_usage = chunk_usage  # remember for final persist
                    finish = chunk.get("choices", [{}])[0].get("finish_reason")
                    if not finish:
                        continue
                    usage = latest_usage  # use accumulated usage at completion time
                    if True:  # was: if finish or usage
                        asst_content = [{
                            "type": "output_text",
                            "text": accumulated_text,
                            "annotations": [],
                        }]
                        user_content = new_input_messages[-1]["content"] \
                            if new_input_messages else []
                        usage_data = {
                            "input_tokens": (usage or {}).get("prompt_tokens", 0),
                            "output_tokens": (usage or {}).get("completion_tokens", 0),
                            "total_tokens": (usage or {}).get("total_tokens", 0),
                        }
                        # New session needed? Open a new session for write
                        from src.models.database import create_session_factory as _csf
                        async with _csf()() as wsess:
                            # Re-fetch session in this DB session
                            from sqlalchemy import select as _sel
                            real_sess = (await wsess.execute(
                                _sel(ResponseSession).where(
                                    ResponseSession.id == sess.id))).scalar_one()
                            _, asst_turn = await write_user_and_assistant_turns(
                                wsess, sess=real_sess,
                                user_content=user_content,
                                assistant_content=asst_content,
                                usage=usage_data, reasoning=None,
                                instructions=req.instructions,
                                text_format=req.text.model_dump())
                            await update_session_usage(wsess, real_sess,
                                input_tokens=usage_data["input_tokens"],
                                output_tokens=usage_data["output_tokens"])
                        from src.services.usage_service import record_llm_usage
                        await record_llm_usage(model=engine_name,
                            prompt_tokens=usage_data["input_tokens"],
                            completion_tokens=usage_data["output_tokens"],
                            duration_ms=0,
                            instance_id=instance.id, api_key_id=api_key.id)
                        # Emit completed event
                        status = "incomplete" if finish == "length" else "completed"
                        incomplete_details = {"reason": "max_output_tokens"} \
                            if finish == "length" else None
                        yield ("response.completed", {
                            "type": "response.completed",
                            "response": {
                                "id": asst_turn.id,
                                "status": status,
                                "incomplete_details": incomplete_details,
                                "model": engine_name,
                                "output": [{
                                    "type": "message",
                                    "id": f"msg-{asst_turn.id[5:]}",
                                    "role": "assistant",
                                    "content": asst_content,
                                }],
                                "usage": usage_data,
                                "history_truncated": history_truncated,
                            },
                        })
                        completed_persist_done = True  # block persist_partial race
                        return

    # Persist-partial callback for client disconnect.
    # CRITICAL: must check completed_persist_done sentinel — if the completed
    # event fired before cancellation, we already wrote the turn pair; another
    # write would collide on UNIQUE(session_id, turn_idx).
    async def persist_partial(text, status, reason):
        if completed_persist_done:
            logger.info("persist_partial skipped (completed already persisted)")
            return
        from src.models.database import create_session_factory as _csf
        async with _csf()() as wsess:
            from sqlalchemy import select as _sel
            real_sess = (await wsess.execute(
                _sel(ResponseSession).where(
                    ResponseSession.id == sess.id))).scalar_one()
            user_content = new_input_messages[-1]["content"] \
                if new_input_messages else []
            await write_partial_assistant_turn(wsess, sess=real_sess,
                user_content=user_content, partial_text=text,
                status=status, incomplete_reason=reason,
                instructions=req.instructions, text_format=req.text.model_dump())
            # Also record llm_usage with whatever we have (may be 0 input/output)
            from src.services.usage_service import record_llm_usage
            await record_llm_usage(model=engine_name,
                prompt_tokens=latest_usage.get("prompt_tokens", 0),
                completion_tokens=latest_usage.get("completion_tokens", 0),
                duration_ms=0, instance_id=instance.id, api_key_id=api_key.id)

    return StreamingResponse(
        responses_sse_envelope(inner(), persist_partial, request_id),
        media_type="text/event-stream",
    )
```

- [ ] **Step 4b: Register router in main.py** (alongside other `app.include_router` calls)

- [ ] **Step 5b: Manual smoke (both paths)**
```bash
KEY=<api-key-for-qwen-instance>

# Non-streaming
curl --noproxy '*' -X POST http://localhost:8000/v1/responses \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","input":"你好","store":true}' | jq .

# Streaming
curl --noproxy '*' -N -X POST http://localhost:8000/v1/responses \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","input":"讲个笑话","stream":true,"thinking":{"type":"disabled"}}'
```

- [ ] **Step 6b: Commit Task 6b** — `feat(responses): POST /v1/responses streaming with semantic SSE events`

---

## Task 7: GET / DELETE / LIST endpoints

**Files:**
- Modify: `backend/src/api/routes/responses.py`

- [ ] **Step 1: GET /v1/responses/{id}**
```python
@router.get("/v1/responses/{response_id}")
async def get_response(
    response_id: str,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    turn = await session.get(ResponseTurn, response_id)
    if turn is None or turn.role != "assistant":
        raise NotFoundError("response not found", code="response_not_found")
    sess = await session.get(ResponseSession, turn.session_id)
    if sess is None or _to_utc(sess.expire_at) < datetime.now(timezone.utc):
        raise NotFoundError("response not found or expired", code="response_not_found")
    if sess.instance_id != instance.id:
        raise NousPermissionError("response belongs to another instance",
                                   code="response_wrong_instance")
    asst_content = decode_content(turn.content_compressed)
    return {
        "id": turn.id,
        "object": "response",
        "status": turn.status,
        "incomplete_details": {"reason": turn.incomplete_reason}
            if turn.incomplete_reason else None,
        "created_at": int(turn.created_at.timestamp()),
        "model": sess.model,
        "previous_response_id": None,  # could compute from prev turn pair
        "instructions": turn.instructions,
        "store": True,
        "expire_at": int(sess.expire_at.timestamp()),
        "output": [{
            "type": "message",
            "id": f"msg-{turn.id[5:]}",
            "role": "assistant",
            "content": asst_content,
        }],
        "usage": turn.usage_json,
    }
```

- [ ] **Step 2: GET /v1/responses (cursor list)**
```python
@router.get("/v1/responses")
async def list_responses(
    limit: int = 20, after: str | None = None, model: str | None = None,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    from sqlalchemy import select, tuple_  # CRITICAL: tuple_ for row-constructor cmp
    instance, _ = auth
    now = datetime.now(timezone.utc)
    q = (select(ResponseTurn, ResponseSession)
         .join(ResponseSession, ResponseTurn.session_id == ResponseSession.id)
         .where(
             ResponseSession.instance_id == instance.id,
             # NOTE: PG side, no _to_utc needed in WHERE; SQL handles tz-aware compare.
             # Tests on SQLite require running pytest with PG, not aiosqlite.
             ResponseSession.expire_at > now,
             ResponseTurn.role == "assistant",
         ))
    if model:
        q = q.where(ResponseSession.model == model)
    if after:
        anchor = await session.get(ResponseTurn, after)
        if anchor is None:
            raise InvalidRequestError("invalid cursor",
                param="after", code="invalid_cursor")
        q = q.where(
            tuple_(ResponseTurn.created_at, ResponseTurn.id)
            < tuple_(anchor.created_at, anchor.id))
    q = q.order_by(
        ResponseTurn.created_at.desc(), ResponseTurn.id.desc()
    ).limit(min(limit, 100) + 1)
    rows = (await session.execute(q)).all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    data = [{
        "id": turn.id,
        "object": "response",
        "status": turn.status,
        "model": sess.model,
        "created_at": int(turn.created_at.timestamp()),
        "expire_at": int(sess.expire_at.timestamp()),
        "usage": turn.usage_json,
    } for turn, sess in rows]
    return {
        "data": data,
        "has_more": has_more,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }
```

- [ ] **Step 3: DELETE /v1/responses/{id}** (cascades full session)
```python
@router.delete("/v1/responses/{response_id}", status_code=204)
async def delete_response(
    response_id: str,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    turn = await session.get(ResponseTurn, response_id)
    if turn is None:
        return Response(status_code=204)  # idempotent
    sess = await session.get(ResponseSession, turn.session_id)
    if sess is None:
        return Response(status_code=204)
    if sess.instance_id != instance.id:
        raise NousPermissionError("response belongs to another instance",
                                   code="response_wrong_instance")
    # CASCADE deletes all turns
    await session.delete(sess)
    await session.commit()
    return Response(status_code=204)
```

- [ ] **Step 4: Integration tests**
```python
# backend/tests/test_api_responses.py
# Cover all 4 endpoints + error cases.
# Use sample_instance, sample_api_key fixtures from conftest.
# Mock model_manager so adapter.is_loaded == True; mock httpx vLLM with respx.
# (or use a local httpx.MockTransport)
```

- [ ] **Step 5: Run tests, confirm pass**
- [ ] **Step 6: Commit** — `feat(responses): GET/LIST/DELETE endpoints + integration tests`

---

## Task 8: Cleanup loop + E2E verification

**Files:**
- Modify: `backend/src/api/main.py` (add cleanup loop alongside cache cleanup)

- [ ] **Step 1: Add cleanup loop to lifespan**
```python
async def response_cleanup_loop(interval_seconds: int = 3600):
    from src.services.responses_service import cleanup_expired_sessions
    from src.models.database import create_session_factory as _csf
    sf = _csf()
    while True:
        try:
            async with sf() as s:
                n = await cleanup_expired_sessions(s)
                if n:
                    logger.info("response cleanup: %d expired sessions", n)
        except Exception:
            logger.exception("response cleanup error")
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break

response_cleanup_task = asyncio.create_task(response_cleanup_loop())

try:
    yield
finally:
    cache_cleanup_task.cancel()
    response_cleanup_task.cancel()
    for t in (cache_cleanup_task, response_cleanup_task):
        try: await t
        except asyncio.CancelledError: pass
    # ... existing partial-write worker shutdown
```

- [ ] **Step 2: E2E verification** (run all 13 cases from spec)

```bash
fuser -k 8000/tcp; sleep 1
cd backend && nohup .venv/bin/python -m uvicorn src.api.main:create_app \
  --factory --port 8000 --reload --reload-exclude .venv \
  > /tmp/nous-backend.log 2>&1 &
```

Then run cases 1-13 from spec §验证. Each must match expected output.

For case 11 (streaming interrupt):
```bash
# Start a streaming request, kill curl after 1s, then GET the response id from logs
# Inspect response_turns: status='incomplete', incomplete_reason='connection_closed'
```

For case 12 (input too long):
```bash
# Need a small max_model_len model; can mock by overriding adapter.max_model_len
```

For case 13 (concurrent write):
```bash
# Race two POSTs with same previous_response_id; one should 200, one 409
RESP1=$(curl ... )
seq 1 2 | xargs -P 2 -I {} curl ... -d "{...,\"previous_response_id\":\"$RESP1\"}"
```

- [ ] **Step 3: Commit** — `feat(responses): cleanup loop + verification complete`

- [ ] **Step 4: Push branch**
```bash
git push
```

---

## Notes for Implementers

- **Reverse-dependency:** Task 1 adds `ConflictError` to Step 1's errors.py. Implementer must do this BEFORE Task 3 (service raises ConflictError on UNIQUE collision).
- **vLLM streaming**: vLLM emits final usage chunk with `choices[0].finish_reason == "stop"` (not in a separate event). The inner generator looks for either `usage` field present OR `finish_reason` set.
- **Test isolation**: `_silence_db_log_handler` from Step 1 conftest auto-applies; tests that trigger 500s won't pollute logs.db.
- **Performance note**: Each turn's gzip+JSON adds ~1ms per assemble. For 100-turn chains, that's 100ms wall-clock. Acceptable for now; profile before optimizing.
- **`text_format.schema` reserved word**: Pydantic v2 allows `Field(alias="schema")` but be careful of `BaseModel.schema()` shadowing in older code. We use Pydantic v2 throughout, so this is fine.
- **Vision testing**: If qwen3.5-VL is loaded, case 5 (multimodal) works; otherwise test with `respx` mock on httpx.
- **Background worker queue size 1000**: at 100 streams/sec with avg 10s connection, expect <100 simultaneous partial-writes. Plenty of headroom.
- **`max_tokens` default**: hardcoded to 2048 in vllm_body. Future: read from req.max_output_tokens (Ark field).
