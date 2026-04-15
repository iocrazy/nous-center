# Step 3: Context Cache API (common_prefix) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/context/create`, integrate `extra_body.context_id` (and top-level `context_id`) into `/v1/chat/completions`, plus `GET / DELETE /v1/context/{id}` for management. Backed by a new PG table `context_caches`. Cleans expired entries via a background loop.

**Architecture:** vLLM already has prefix caching in GPU. We expose explicit lifecycle (create → use → expire) on top, persist metadata in PG, and pre-warm vLLM's prefix cache at create time so the first user-visible chat request gets cache hits. Cache scope is per `instance_id`; same-instance keys can share. Use resets TTL.

**Tech Stack:** SQLAlchemy 2.0 async + asyncpg, FastAPI, httpx, vLLM 0.19.

**Spec:** `docs/superpowers/specs/2026-04-15-step3-context-cache-api-design.md`

---

## File Structure

**Create:**
- `backend/src/models/context_cache.py` — SQLAlchemy model
- `backend/src/services/context_cache_service.py` — service layer (lookup, validate, hit, cleanup)
- `backend/src/services/snowflake.py` *(if not present)* — id generator (probably reuse existing helper, see Task 1 step 1)
- `backend/src/api/routes/context_cache.py` — 4 endpoints
- `backend/tests/test_context_cache_service.py` — unit tests
- `backend/tests/test_api_context_cache.py` — integration tests

**Modify:**
- `backend/src/api/routes/openai_compat.py` — read `context_id`, prepend cache messages, validate model, schedule hit-count update
- `backend/src/api/main.py` — register router, launch cleanup task in `lifespan`
- `backend/src/models/database.py` — *no change*; `Base.metadata.create_all` already auto-creates new tables on startup if needed (verify: see existing engine create patterns)

**Notes:**
- Existing snowflake helper: `grep -rn 'def snowflake_id\|def gen_id\|snowflake' backend/src/` — reuse it. If none exists, implement a tiny one in `services/snowflake.py` (we already have `BigInteger` IDs on other tables, so a generator must exist).
- For the new `context_caches` table to materialize, you need either an Alembic migration or to call `Base.metadata.create_all`. The PG migration script in Step 0 used `create_all` directly. Add the same call (or rely on existing startup logic; verify in `main.py`).

---

## Task 1: ContextCache model + table migration

**Files:**
- Create: `backend/src/models/context_cache.py`
- Possibly modify: `backend/src/models/database.py` if model auto-import is needed
- Test: `backend/tests/test_context_cache_service.py` (basic schema check)

- [ ] **Step 1: Find existing snowflake / id helper**

```bash
grep -rn 'snowflake\|def gen_id\|secrets\.token' backend/src/ | head -10
```
Reuse it. If none, write a tiny generator in `backend/src/services/snowflake.py`:
```python
import time, threading
_lock = threading.Lock()
_seq = 0
_last_ms = 0
def snowflake_id() -> int:
    global _seq, _last_ms
    with _lock:
        ms = int(time.time() * 1000)
        if ms == _last_ms:
            _seq = (_seq + 1) & 0xFFF
        else:
            _seq = 0
            _last_ms = ms
        return (ms << 12) | _seq
```

- [ ] **Step 2: Create model**

```python
# backend/src/models/context_cache.py
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from sqlalchemy import (
    BigInteger, Column, DateTime, ForeignKey, Integer, String, Text,
    CheckConstraint, Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from src.models.database import Base


def _expires_at_default():
    return datetime.now(timezone.utc) + timedelta(seconds=86400)


class ContextCache(Base):
    __tablename__ = "context_caches"

    id = Column(String(64), primary_key=True)
    instance_id = Column(BigInteger,
                         ForeignKey("service_instances.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    api_key_id = Column(BigInteger, nullable=True)  # audit only, not FK to allow key deletion
    model = Column(String(128), nullable=False)
    mode = Column(String(32), nullable=False, default="common_prefix")
    messages_json = Column(JSONB, nullable=False)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    ttl_seconds = Column(Integer, nullable=False, default=86400)
    expires_at = Column(DateTime(timezone=True), nullable=False,
                        default=_expires_at_default)
    hit_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc))
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint("ttl_seconds >= 60 AND ttl_seconds <= 604800",
                        name="context_cache_ttl_range"),
        Index("ix_context_caches_expires_at", "expires_at"),
        Index("ix_context_caches_instance_expires", "instance_id", "expires_at"),
    )
```

- [ ] **Step 3: Make sure model is imported on startup so create_all sees it**

If main.py does explicit imports for create_all, add `from src.models import context_cache` in the appropriate spot. Verify by reading main.py around the lifespan / create_all call.

- [ ] **Step 4: Manual migration (one-time apply against running PG)**

```bash
cd backend
.venv/bin/python - <<'PY'
import asyncio
from src.models.database import Base, create_engine
from src.models import context_cache  # noqa: import for metadata
async def go():
    eng = create_engine()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await eng.dispose()
asyncio.run(go())
PY
```

Verify in PG:
```bash
docker exec nous-center-postgres-1 psql -U nous_heygo -d nous_center \
  -c "\d context_caches"
```

- [ ] **Step 5: Commit**
```
feat(models): add ContextCache table for prefix cache lifecycle
```

---

## Task 2: Service layer + unit tests

**Files:**
- Create: `backend/src/services/context_cache_service.py`
- Test: `backend/tests/test_context_cache_service.py`

- [ ] **Step 1: Write failing tests first**

```python
# backend/tests/test_context_cache_service.py
import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from src.models.context_cache import ContextCache
from src.services.context_cache_service import (
    create_cache_row, fetch_active_cache,
    increment_hit_and_extend, delete_cache,
    cleanup_expired,
)

@pytest.mark.asyncio
async def test_create_and_fetch_roundtrip(db_session, sample_instance):
    row = await create_cache_row(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="qwen3.5-35b",
        messages=[{"role":"system","content":"hi"}],
        prompt_tokens=42,
        ttl_seconds=3600,
    )
    assert row.id.startswith("ctx-")
    fetched = await fetch_active_cache(db_session, row.id, sample_instance.id)
    assert fetched is not None
    assert fetched.prompt_tokens == 42

@pytest.mark.asyncio
async def test_fetch_expired_returns_none(db_session, sample_instance):
    # create, manually expire
    row = await create_cache_row(db_session, instance_id=sample_instance.id,
        api_key_id=None, model="m", messages=[{"role":"system","content":"x"}],
        prompt_tokens=1, ttl_seconds=3600)
    row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    await db_session.commit()
    assert await fetch_active_cache(db_session, row.id, sample_instance.id) is None

@pytest.mark.asyncio
async def test_wrong_instance_returns_none(db_session, sample_instance, other_instance):
    row = await create_cache_row(db_session, instance_id=sample_instance.id,
        api_key_id=None, model="m", messages=[{"role":"system","content":"x"}],
        prompt_tokens=1, ttl_seconds=3600)
    assert await fetch_active_cache(db_session, row.id, other_instance.id) is None

@pytest.mark.asyncio
async def test_hit_extends_and_counts(db_session, sample_instance):
    row = await create_cache_row(db_session, instance_id=sample_instance.id,
        api_key_id=None, model="m", messages=[{"role":"system","content":"x"}],
        prompt_tokens=1, ttl_seconds=3600)
    original_exp = row.expires_at
    await increment_hit_and_extend(db_session, row.id, ttl_seconds=3600)
    refreshed = (await db_session.execute(
        select(ContextCache).where(ContextCache.id == row.id)
    )).scalar_one()
    assert refreshed.hit_count == 1
    assert refreshed.last_used_at is not None
    assert refreshed.expires_at > original_exp

@pytest.mark.asyncio
async def test_cleanup_deletes_only_expired(db_session, sample_instance):
    fresh = await create_cache_row(db_session, instance_id=sample_instance.id,
        api_key_id=None, model="m", messages=[{"role":"system","content":"x"}],
        prompt_tokens=1, ttl_seconds=3600)
    stale = await create_cache_row(db_session, instance_id=sample_instance.id,
        api_key_id=None, model="m", messages=[{"role":"system","content":"x"}],
        prompt_tokens=1, ttl_seconds=3600)
    stale.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    await db_session.commit()
    n = await cleanup_expired(db_session)
    assert n == 1
    assert await fetch_active_cache(db_session, fresh.id, sample_instance.id) is not None
    assert await fetch_active_cache(db_session, stale.id, sample_instance.id) is None
```

You'll need fixtures `sample_instance` and `other_instance` in `conftest.py` if absent (insert `ServiceInstance` rows). Reuse `db_session` if it exists; otherwise add a session fixture using `nous_center` test DB or transaction rollback pattern.

- [ ] **Step 2: Run tests, confirm they fail (module does not exist)**

- [ ] **Step 3: Implement service**

```python
# backend/src/services/context_cache_service.py
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.context_cache import ContextCache

logger = logging.getLogger(__name__)


def _new_cache_id() -> str:
    # 16-char URL-safe token; collision probability with PG PK constraint is fine.
    return f"ctx-{secrets.token_urlsafe(12)[:16]}"


async def create_cache_row(
    session: AsyncSession,
    *,
    instance_id: int,
    api_key_id: int | None,
    model: str,
    messages: list[dict],
    prompt_tokens: int,
    ttl_seconds: int = 86400,
    mode: str = "common_prefix",
) -> ContextCache:
    if not isinstance(messages, list) or not messages:
        from src.errors import InvalidRequestError
        raise InvalidRequestError("messages must be a non-empty list",
                                  param="messages", code="invalid_messages")
    if not (60 <= ttl_seconds <= 604800):
        from src.errors import InvalidRequestError
        raise InvalidRequestError("ttl out of range [60, 604800]",
                                  param="ttl", code="invalid_ttl")

    now = datetime.now(timezone.utc)
    row = ContextCache(
        id=_new_cache_id(),
        instance_id=instance_id,
        api_key_id=api_key_id,
        model=model,
        mode=mode,
        messages_json=messages,
        prompt_tokens=prompt_tokens,
        ttl_seconds=ttl_seconds,
        expires_at=now + timedelta(seconds=ttl_seconds),
        hit_count=0,
        created_at=now,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def fetch_active_cache(
    session: AsyncSession,
    cache_id: str,
    instance_id: int,
) -> ContextCache | None:
    """Returns row only if exists, not expired, and belongs to instance."""
    now = datetime.now(timezone.utc)
    stmt = select(ContextCache).where(
        ContextCache.id == cache_id,
        ContextCache.instance_id == instance_id,
        ContextCache.expires_at > now,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def fetch_cache_any_instance(
    session: AsyncSession,
    cache_id: str,
) -> ContextCache | None:
    """For permission error reporting (returns row even if instance differs)."""
    return await session.get(ContextCache, cache_id)


async def increment_hit_and_extend(
    session: AsyncSession,
    cache_id: str,
    ttl_seconds: int,
) -> None:
    """Atomic UPDATE: hit_count +=1, expires_at = now+ttl, last_used_at = now."""
    now = datetime.now(timezone.utc)
    new_exp = now + timedelta(seconds=ttl_seconds)
    stmt = (
        update(ContextCache)
        .where(ContextCache.id == cache_id)
        .values(
            hit_count=ContextCache.hit_count + 1,
            expires_at=new_exp,
            last_used_at=now,
        )
    )
    await session.execute(stmt)
    await session.commit()


async def delete_cache(
    session: AsyncSession,
    cache_id: str,
    instance_id: int,
) -> bool:
    """Delete only if owned by instance. Returns True if a row was deleted."""
    stmt = delete(ContextCache).where(
        ContextCache.id == cache_id,
        ContextCache.instance_id == instance_id,
    )
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount > 0


async def cleanup_expired(session: AsyncSession) -> int:
    """Delete all expired rows. Returns count."""
    now = datetime.now(timezone.utc)
    result = await session.execute(
        delete(ContextCache).where(ContextCache.expires_at < now)
    )
    await session.commit()
    return result.rowcount
```

- [ ] **Step 4: Run tests, confirm pass**
- [ ] **Step 5: Commit** — `feat(context-cache): service layer with TTL extend + cleanup`

---

## Task 3: POST /v1/context/create endpoint

**Files:**
- Create: `backend/src/api/routes/context_cache.py`
- Modify: `backend/src/api/main.py` (register router)
- Test: `backend/tests/test_api_context_cache.py`

- [ ] **Step 1: Write failing tests** (using TestClient + a mock vLLM via httpx mock or live engine)

```python
# backend/tests/test_api_context_cache.py
def test_create_returns_id_and_metadata(client_with_real_engine, sample_api_key):
    """Requires a loaded model; mark slow / skip if not loaded."""
    resp = client_with_real_engine.post("/v1/context/create",
        headers={"Authorization": f"Bearer {sample_api_key}"},
        json={"model": "qwen3.5-35b",
              "messages": [{"role":"system","content":"hi from test"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"].startswith("ctx-")
    assert body["mode"] == "common_prefix"
    assert body["usage"]["prompt_tokens"] > 0

def test_create_rejects_empty_messages(client, sample_api_key):
    resp = client.post("/v1/context/create",
        headers={"Authorization": f"Bearer {sample_api_key}"},
        json={"model": "qwen3.5-35b", "messages": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_messages"

def test_create_rejects_invalid_ttl(client, sample_api_key):
    resp = client.post("/v1/context/create",
        headers={"Authorization": f"Bearer {sample_api_key}"},
        json={"model": "qwen3.5-35b",
              "messages": [{"role":"system","content":"x"}],
              "ttl": 30})  # too low
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_ttl"
```

The `client_with_real_engine` integration test can be marked with `@pytest.mark.skipif(not _engine_loaded("qwen3.5-35b"))` to gate on an actually loaded model.

- [ ] **Step 2: Implement endpoint**

```python
# backend/src/api/routes/context_cache.py
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import verify_bearer_token
from src.errors import APIError, NotFoundError, PermissionError as NousPermissionError
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.context_cache_service import (
    create_cache_row, fetch_active_cache, fetch_cache_any_instance,
    delete_cache,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["context-cache"])


class CreateContextRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    ttl: int | None = Field(default=86400, ge=60, le=604800)


@router.post("/v1/context/create")
async def create_context(
    req: CreateContextRequest,
    request: Request,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, api_key = auth
    if instance.source_type != "model":
        from src.errors import InvalidRequestError
        raise InvalidRequestError(
            "Context Cache only supported on model-type instances",
            code="not_a_model_instance",
        )
    engine_name = instance.source_name or str(instance.source_id)

    # Resolve adapter
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

    # Pre-warm: send messages through vLLM with max_tokens=1, capture prompt_tokens
    warm_body = {
        "model": "",
        "messages": req.messages,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=60, proxy=None) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/v1/chat/completions", json=warm_body
            )
            resp.raise_for_status()
            warm_data = resp.json()
            prompt_tokens = warm_data.get("usage", {}).get("prompt_tokens", 0)
    except httpx.HTTPError as e:
        raise APIError(f"vLLM pre-warm failed: {e}", code="warm_failed")

    # Persist
    row = await create_cache_row(
        session,
        instance_id=instance.id,
        api_key_id=api_key.id,
        model=engine_name,
        messages=req.messages,
        prompt_tokens=prompt_tokens,
        ttl_seconds=req.ttl or 86400,
    )

    return {
        "id": row.id,
        "model": row.model,
        "mode": row.mode,
        "ttl": row.ttl_seconds,
        "expires_at": row.expires_at.isoformat(),
        "usage": {
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": row.prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


@router.get("/v1/context/{cache_id}")
async def get_context(
    cache_id: str,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    row = await fetch_active_cache(session, cache_id, instance.id)
    if row is None:
        # Check if exists in another instance for clearer error
        other = await fetch_cache_any_instance(session, cache_id)
        if other is not None and other.instance_id != instance.id:
            raise NousPermissionError("Cache belongs to another instance",
                                       code="context_wrong_instance")
        raise NotFoundError("Context cache not found or expired",
                            code="context_not_found")

    preview = []
    for m in (row.messages_json or [])[:5]:
        c = m.get("content", "")
        if isinstance(c, str) and len(c) > 200:
            c = c[:200] + "..."
        preview.append({"role": m.get("role"), "content": c})

    return {
        "id": row.id,
        "model": row.model,
        "mode": row.mode,
        "ttl": row.ttl_seconds,
        "expires_at": row.expires_at.isoformat(),
        "created_at": row.created_at.isoformat(),
        "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
        "hit_count": row.hit_count,
        "prompt_tokens": row.prompt_tokens,
        "messages_preview": preview,
    }


@router.delete("/v1/context/{cache_id}", status_code=204)
async def delete_context(
    cache_id: str,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    other = await fetch_cache_any_instance(session, cache_id)
    if other is not None and other.instance_id != instance.id:
        raise NousPermissionError("Cache belongs to another instance",
                                   code="context_wrong_instance")
    await delete_cache(session, cache_id, instance.id)  # idempotent
    return Response(status_code=204)
```

- [ ] **Step 3: Register router in `main.py`**

Find where other routers are included (e.g. `app.include_router(openai_compat.router)`) and add:
```python
from src.api.routes import context_cache as context_cache_routes
app.include_router(context_cache_routes.router)
```

- [ ] **Step 4: Run tests, confirm pass**
- [ ] **Step 5: Commit** — `feat(api): POST/GET/DELETE /v1/context endpoints`

---

## Task 4: Integrate context_id into /v1/chat/completions

**Files:**
- Modify: `backend/src/api/routes/openai_compat.py`
- Test: `backend/tests/test_api_context_cache.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_api_context_cache.py (append)
def test_chat_with_context_id_prepends_messages(client_with_real_engine, sample_api_key):
    # First create a cache
    create_resp = client_with_real_engine.post("/v1/context/create",
        headers={"Authorization": f"Bearer {sample_api_key}"},
        json={"model": "qwen3.5-35b",
              "messages": [{"role":"system","content":"You only answer in JSON."}]})
    ctx_id = create_resp.json()["id"]

    # Use the cache in chat
    resp = client_with_real_engine.post("/v1/chat/completions",
        headers={"Authorization": f"Bearer {sample_api_key}"},
        json={"model": "qwen3.5-35b",
              "context_id": ctx_id,
              "messages": [{"role":"user","content":"What is 1+1?"}]})
    assert resp.status_code == 200
    body = resp.json()
    # Assert cached_tokens > 0 (prefix from cache should be hit)
    cached = body["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    assert cached > 0

def test_chat_with_unknown_context_id_404(client, sample_api_key):
    resp = client.post("/v1/chat/completions",
        headers={"Authorization": f"Bearer {sample_api_key}"},
        json={"model": "qwen3.5-35b",
              "context_id": "ctx-doesnotexist",
              "messages": [{"role":"user","content":"hi"}]})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "context_not_found"

def test_chat_with_mismatched_model_400(client_with_real_engine, sample_api_key):
    create_resp = client_with_real_engine.post("/v1/context/create",
        headers={"Authorization": f"Bearer {sample_api_key}"},
        json={"model": "qwen3.5-35b",
              "messages": [{"role":"system","content":"x"}]})
    ctx_id = create_resp.json()["id"]
    resp = client_with_real_engine.post("/v1/chat/completions",
        headers={"Authorization": f"Bearer {sample_api_key}"},
        json={"model": "gemma-4-26b",
              "context_id": ctx_id,
              "messages": [{"role":"user","content":"hi"}]})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "context_model_mismatch"
```

- [ ] **Step 2: Implement integration in `openai_compat.py`**

In `chat_completions`, after parsing body and before the `is_stream` branch:

```python
# Resolve context_id (top-level or extra_body)
context_id = body.pop("context_id", None)
if not context_id and isinstance(body.get("extra_body"), dict):
    context_id = body["extra_body"].pop("context_id", None)
    if not body["extra_body"]:
        body.pop("extra_body", None)

if context_id:
    from src.models.database import create_session_factory
    from src.services.context_cache_service import (
        fetch_active_cache, fetch_cache_any_instance, increment_hit_and_extend,
    )
    from src.errors import (
        InvalidRequestError, NotFoundError as NF,
        PermissionError as NP,
    )
    sf = create_session_factory()
    async with sf() as cache_session:
        cache = await fetch_active_cache(cache_session, context_id, instance.id)
        if cache is None:
            other = await fetch_cache_any_instance(cache_session, context_id)
            if other is not None and other.instance_id != instance.id:
                raise NP("Cache belongs to another instance",
                         code="context_wrong_instance")
            raise NF("Context cache not found or expired",
                    code="context_not_found")
        if cache.model != engine_name:
            raise InvalidRequestError(
                f"Cache was created for '{cache.model}', not '{engine_name}'",
                code="context_model_mismatch", param="model",
            )
        # Prepend cached messages
        body["messages"] = list(cache.messages_json) + list(body.get("messages", []))

    # Schedule hit-count update fire-and-forget (won't block chat latency)
    async def _bump():
        try:
            sf2 = create_session_factory()
            async with sf2() as s2:
                await increment_hit_and_extend(s2, context_id, cache.ttl_seconds)
        except Exception:
            logger.exception("hit_count update failed for %s", context_id)
    import asyncio as _asyncio
    _asyncio.create_task(_bump())
```

This goes between body parsing and the `_maybe_inject_thinking` call (or after; order doesn't matter — both modify body before stream/non-stream branching).

- [ ] **Step 3: Run tests, confirm pass**
- [ ] **Step 4: Commit** — `feat(openai): accept context_id, prepend cached messages, schedule hit_count`

---

## Task 5: Background cleanup task

**Files:**
- Modify: `backend/src/api/main.py` (lifespan startup)
- Test: covered by `test_cleanup_deletes_only_expired` in Task 2 + manual log check

- [ ] **Step 1: Add cleanup loop to lifespan**

In `main.py`'s `lifespan`:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing startup ...
    cleanup_task = asyncio.create_task(_context_cache_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        # ... existing shutdown ...

async def _context_cache_cleanup_loop(interval_seconds: int = 3600):
    from src.services.context_cache_service import cleanup_expired
    from src.models.database import create_session_factory
    sf = create_session_factory()
    while True:
        try:
            async with sf() as session:
                n = await cleanup_expired(session)
                if n:
                    logger.info("context cache cleanup: %d expired rows deleted", n)
        except Exception:
            logger.exception("context cache cleanup error")
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break
```

- [ ] **Step 2: Verify by running backend, watch logs (insert a stale row manually if needed)**
```bash
docker exec nous-center-postgres-1 psql -U nous_heygo -d nous_center \
  -c "INSERT INTO context_caches (id, instance_id, model, messages_json, prompt_tokens, ttl_seconds, expires_at) VALUES ('ctx-staletest', <SOME_INSTANCE_ID>, 'm', '[]', 0, 60, now() - interval '1 hour');"
```
Then (if needed) reduce `interval_seconds` to 30 temporarily, restart, watch logs for "1 expired rows deleted". Revert the interval before commit.

- [ ] **Step 3: Commit** — `feat(api): launch context cache cleanup loop in lifespan`

---

## Task 6: End-to-end verification

**Files:** none, pure verification.

- [ ] **Step 1: Restart backend**

```bash
fuser -k 8000/tcp 2>/dev/null
cd backend && nohup .venv/bin/python -m uvicorn src.api.main:create_app \
  --factory --port 8000 --reload --reload-exclude .venv \
  > /tmp/nous-backend.log 2>&1 &
```

- [ ] **Step 2: Run all 6 verification cases from spec §Verification**

Copy from `docs/superpowers/specs/2026-04-15-step3-context-cache-api-design.md` "验证" section. You'll need a real API key bound to an instance pointing to a loaded LLM (e.g. qwen3.5-35b).

- [ ] **Step 3: Confirm hit_count increments**

After running case 2 a few times:
```bash
docker exec nous-center-postgres-1 psql -U nous_heygo -d nous_center \
  -c "SELECT id, hit_count, expires_at FROM context_caches WHERE id='<CTX_ID>';"
```

- [ ] **Step 4: Confirm logs.db is not polluted by tests**

```bash
sqlite3 backend/data/archive/logs.db \
  "SELECT count(*) FROM app_logs WHERE message LIKE '%context%' AND timestamp > datetime('now','-10 minutes');"
# Should be 0 for tests run via pytest (fixture detaches DbLogHandler).
# Manual curl invocations DO log naturally; that's expected.
```

- [ ] **Step 5: Push branch**

```bash
git push
```

---

## Notes for Implementers

- **PG migration:** Step 0 used `Base.metadata.create_all` directly. We follow the same pattern for `context_caches` — no Alembic for now. Run the script in Task 1 Step 4 once.
- **Async sessions:** every place that uses `AsyncSession` must `await session.execute(...)` and `await session.commit()`. Don't mix sync.
- **Test fixtures:** `db_session`, `sample_instance`, `other_instance`, `sample_api_key` may not exist in `conftest.py`. Add them following existing patterns (look at `test_api_engines.py` or `test_api_monitor.py`).
- **vLLM warm-up cost:** ~500ms per create. For large prompts (>5KB) this can hit a few seconds. Acceptable for an explicit cache-warming endpoint.
- **`asyncio.create_task(_bump())` lifecycle:** the task is fire-and-forget; if the request handler returns before the task completes, FastAPI / asyncio keeps the task alive on the running loop. Failures get logged via `logger.exception`. Acceptable trade-off for not blocking chat latency.
- **`extra_body.context_id` extraction:** the OpenAI Python SDK serializes `extra_body={"context_id":"ctx-xxx"}` as a top-level field in the request body. So checking `body.pop("context_id")` first covers both top-level and SDK-style. The fallback to `body["extra_body"]` is for clients that explicitly send a nested `extra_body` object.
- **Don't forget to `import asyncio` at the top of `openai_compat.py`** if not already imported (it is — verified earlier).
- **CheckConstraint syntax:** PostgreSQL 17 accepts the constraint string as-is. If running this against SQLite for some reason (we don't anymore), the `Index` and `CheckConstraint` work too.
