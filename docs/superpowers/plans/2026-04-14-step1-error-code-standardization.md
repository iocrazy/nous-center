# Step 1: OpenAI-Style Error Standardization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify nous-center error responses across `/v1/*` and `/api/v1/*` into OpenAI-style `{error: {message, type, code, param, request_id}}` payload; add `X-Request-Id` header propagation; preserve generic 500 messages while logging full tracebacks to `app_logs`.

**Architecture:** Zero-touch on existing routes. A `RequestIdMiddleware` injects a UUID into `request.state`. Four global FastAPI exception handlers (`NousError` / `HTTPException` / `RequestValidationError` / `Exception`) convert any raise into the OpenAI payload and attach the request id. Streaming endpoints use SSE `data: {error: ...}` + `data: [DONE]`. Frontend `apiFetch` is rewritten to throw a typed `NousApiError`.

**Tech Stack:** FastAPI, Pydantic v2, stdlib `logging` (already wired to `DbLogHandler` writing to `logs.db`), TypeScript/React on the frontend.

**Spec:** `docs/superpowers/specs/2026-04-14-step1-error-code-standardization-design.md`

---

## File Structure

**Create:**
- `backend/src/errors.py` — `NousError` base + 6 subclasses, `.to_dict()` serializer
- `backend/src/api/middleware/__init__.py` — re-exports
- `backend/src/api/middleware/request_id.py` — `RequestIdMiddleware`
- `frontend/src/api/errors.ts` — `NousApiError` class
- `backend/tests/test_errors.py` — unit tests for NousError serialization
- `backend/tests/test_api_errors.py` — integration tests for all 4 handlers + RequestID header
- `backend/tests/conftest_logs.py` — fixture to isolate tests from the real `logs.db`

**Modify:**
- `backend/src/api/main.py` — register `RequestIdMiddleware` and 4 exception handlers (around line 246 where other middleware is added)
- `backend/src/api/routes/openai_compat.py` — catch exceptions inside SSE generator, yield OpenAI-style error chunk + `[DONE]`
- `frontend/src/api/client.ts` — rewrite `apiFetch` to read `X-Request-Id` header and throw `NousApiError`

**Do not touch:** Existing route handlers (`raise HTTPException(...)` auto-converts). `logs.db` schema.

---

## Task 1: NousError classes + unit tests

**Files:**
- Create: `backend/src/errors.py`
- Test: `backend/tests/test_errors.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_errors.py
from src.errors import (
    NousError, InvalidRequestError, AuthenticationError,
    PermissionError as NousPermissionError, NotFoundError,
    RateLimitError, APIError,
)

def test_default_fields():
    e = APIError("boom")
    assert e.type == "api_error" and e.http_status == 500
    assert e.to_dict() == {"error": {"message": "boom", "type": "api_error"}}

def test_with_all_fields():
    e = InvalidRequestError("bad model", code="model_not_found",
                            param="model", request_id="req-123")
    assert e.http_status == 400
    assert e.to_dict() == {"error": {
        "message": "bad model", "type": "invalid_request_error",
        "code": "model_not_found", "param": "model", "request_id": "req-123",
    }}

def test_http_status_per_subclass():
    assert AuthenticationError("").http_status == 401
    assert NousPermissionError("").http_status == 403
    assert NotFoundError("").http_status == 404
    assert RateLimitError("").http_status == 429
```

- [ ] **Step 2: Run tests, confirm they fail (module doesn't exist)**

- [ ] **Step 3: Implement `backend/src/errors.py`**

```python
class NousError(Exception):
    type: str = "api_error"
    http_status: int = 500

    def __init__(self, message: str, *, code: str | None = None,
                 param: str | None = None, request_id: str | None = None):
        self.message = message
        self.code = code
        self.param = param
        self.request_id = request_id
        super().__init__(message)

    def to_dict(self) -> dict:
        err: dict = {"message": self.message, "type": self.type}
        if self.code: err["code"] = self.code
        if self.param: err["param"] = self.param
        if self.request_id: err["request_id"] = self.request_id
        return {"error": err}

class InvalidRequestError(NousError):  type = "invalid_request_error"; http_status = 400
class AuthenticationError(NousError):  type = "authentication_error"; http_status = 401
class PermissionError(NousError):      type = "permission_error";     http_status = 403
class NotFoundError(NousError):        type = "not_found_error";      http_status = 404
class RateLimitError(NousError):       type = "rate_limit_error";     http_status = 429
class APIError(NousError):             type = "api_error";            http_status = 500
```

- [ ] **Step 4: Run tests, confirm pass**
- [ ] **Step 5: Commit** — `feat(errors): add NousError base + 6 OpenAI-style subclasses`

---

## Task 2: RequestIdMiddleware + unit test

**Files:**
- Create: `backend/src/api/middleware/__init__.py` (empty or re-exports)
- Create: `backend/src/api/middleware/request_id.py`
- Test: `backend/tests/test_api_errors.py` (start of this file; shared test module)

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_api_errors.py
from fastapi.testclient import TestClient
from src.api.main import create_app

def _client():
    return TestClient(create_app())

def test_request_id_generated_when_absent():
    r = _client().get("/api/v1/engines")
    assert "x-request-id" in r.headers
    assert len(r.headers["x-request-id"]) >= 8

def test_request_id_echoed_when_provided():
    r = _client().get("/api/v1/engines", headers={"X-Request-Id": "my-trace-abc"})
    assert r.headers["x-request-id"] == "my-trace-abc"
```

- [ ] **Step 2: Run tests, confirm they fail**

- [ ] **Step 3: Implement middleware**

```python
# backend/src/api/middleware/request_id.py
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response
```

```python
# backend/src/api/middleware/__init__.py
from .request_id import RequestIdMiddleware
__all__ = ["RequestIdMiddleware"]
```

- [ ] **Step 4: Register middleware in `backend/src/api/main.py`**

**⚠️ Middleware ordering:** Starlette's `add_middleware()` is LIFO — **last added runs first**. We want `RequestIdMiddleware` to run FIRST so `request.state.request_id` is populated before `AuditMiddleware` and `RequestLoggingMiddleware` inspect it. Therefore **add it LAST** in the `add_middleware` calls:

```python
# In create_app(), AFTER the existing app.add_middleware(AuditMiddleware)
# and app.add_middleware(RequestLoggingMiddleware) calls:
from src.api.middleware import RequestIdMiddleware
app.add_middleware(RequestIdMiddleware)  # LIFO — last added = first to run
```

**CORS `expose_headers`:** the existing `CORSMiddleware` registration (around line 232) has an `expose_headers` arg. Append `"X-Request-Id"` to that list so browser JS can read it via `resp.headers.get('x-request-id')`:
```python
# Modify the existing CORSMiddleware add_middleware call:
app.add_middleware(
    CORSMiddleware,
    ...existing args...,
    expose_headers=[..., "X-Request-Id"],  # add if missing
)
```
If the current registration has no `expose_headers`, add `expose_headers=["X-Request-Id"]`.

**Optional enhancement:** `RequestLoggingMiddleware` currently logs request rows without request_id. If straightforward, extend its schema to include `request_id` (new column in `request_logs`) in a follow-up PR — **out of scope for this Task**. Note in TODOS instead.

- [ ] **Step 5: Run tests, confirm pass**
- [ ] **Step 6: Commit** — `feat(api): add RequestIdMiddleware with X-Request-Id header echo`

---

## Task 2.5: Test isolation — prevent logs.db pollution

**Why:** Tests in Task 3 deliberately trigger 500 errors. The `_unhandled` handler calls `logger.exception(...)`, which `DbLogHandler` buffers and writes to `backend/data/logs.db` — polluting the real log DB every test run (the "kaboom" test error would show up in the app's LogsOverlay UI).

**Files:**
- Create: `backend/tests/conftest_logs.py`
- Modify: `backend/tests/conftest.py` (import the fixture)

- [ ] **Step 1: Create isolation fixture**

```python
# backend/tests/conftest_logs.py
import logging
import pytest

@pytest.fixture(autouse=True)
def _silence_db_log_handler():
    """Detach DbLogHandler from the root logger during tests so
    error logs emitted by exception handlers don't hit backend/data/logs.db.

    Test assertions that need to verify logging should attach a caplog fixture
    or a MemoryHandler explicitly."""
    root = logging.getLogger()
    removed = [h for h in list(root.handlers) if h.__class__.__name__ == "DbLogHandler"]
    for h in removed:
        root.removeHandler(h)
    yield
    for h in removed:
        root.addHandler(h)
```

- [ ] **Step 2: Import into conftest.py**

Append to `backend/tests/conftest.py`:
```python
from .conftest_logs import _silence_db_log_handler  # noqa: F401  # autouse fixture
```

- [ ] **Step 3: Verify**

```bash
.venv/bin/pytest backend/tests/test_api_errors.py -v
# Then check logs.db hasn't grown:
sqlite3 backend/data/logs.db "SELECT COUNT(*) FROM app_logs WHERE message LIKE '%kaboom%'"
# Expected: 0
```

- [ ] **Step 4: Commit** — `test: isolate test runs from logs.db pollution`

---

## Task 3: Four exception handlers

**Files:**
- Modify: `backend/src/api/main.py` (add handlers near top after imports + near existing route includes)
- Test: `backend/tests/test_api_errors.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_api_errors.py (append)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.errors import (
    NotFoundError as NousNotFound, RateLimitError,
)

# --- Fixtures: mount test-only routes onto a fresh app ---
def _client_with_test_routes():
    from src.api.main import create_app
    app = create_app()
    r = APIRouter()

    class Body(BaseModel):
        name: str

    @r.get("/__test__/notfound")
    async def _nf():
        raise NousNotFound("widget missing", code="widget_not_found")

    @r.get("/__test__/http404")
    async def _http404():
        raise HTTPException(404, "route raised 404")

    @r.post("/__test__/validate")
    async def _val(body: Body):
        return body

    @r.get("/__test__/boom")
    async def _boom():
        raise RuntimeError("kaboom")

    @r.get("/__test__/ratelimit")
    async def _rl():
        raise RateLimitError("too fast", code="rpm_exceeded")

    app.include_router(r)
    return TestClient(app, raise_server_exceptions=False)

def test_nous_error_serialized():
    r = _client_with_test_routes().get("/__test__/notfound")
    assert r.status_code == 404
    assert r.json() == {"error": {
        "message": "widget missing", "type": "not_found_error",
        "code": "widget_not_found",
        "request_id": r.headers["x-request-id"],
    }}

def test_httpexception_converted_to_openai_shape():
    r = _client_with_test_routes().get("/__test__/http404")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["type"] == "not_found_error"
    assert body["error"]["message"] == "route raised 404"
    assert body["error"]["request_id"] == r.headers["x-request-id"]

def test_httpexception_with_list_detail_preserves_param():
    """When a route raises HTTPException(detail=[{'loc': [...], 'msg': ...}]),
    the param should be extracted from loc, not lost in stringification."""
    from fastapi import APIRouter, HTTPException
    from src.api.main import create_app
    app = create_app()
    r = APIRouter()
    @r.get("/__test__/http422-list")
    async def _h():
        raise HTTPException(
            422,
            detail=[{"loc": ["body", "messages", 0, "role"], "msg": "bad role"}],
        )
    app.include_router(r)
    c = TestClient(app, raise_server_exceptions=False)
    resp = c.get("/__test__/http422-list")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["message"] == "bad role"
    assert body["error"]["param"] == "messages.0.role"

def test_validation_error_shape():
    r = _client_with_test_routes().post("/__test__/validate", json={})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["param"] == "name"

def test_500_generic_message_no_traceback_leak():
    r = _client_with_test_routes().get("/__test__/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["message"] == "Internal server error"
    assert body["error"]["type"] == "api_error"
    assert body["error"]["code"] == "internal_error"
    assert "kaboom" not in r.text  # traceback not leaked

def test_rate_limit_status():
    r = _client_with_test_routes().get("/__test__/ratelimit")
    assert r.status_code == 429
```

- [ ] **Step 2: Run tests, confirm they fail**

- [ ] **Step 3: Implement handlers in `backend/src/api/main.py`** (add after `create_app()` starts, before `return app`)

```python
import logging, traceback
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from src.errors import (
    NousError, InvalidRequestError, AuthenticationError,
    PermissionError as NousPermissionError, NotFoundError,
    RateLimitError, APIError,
)

logger = logging.getLogger(__name__)

_HTTP_STATUS_TO_ERROR = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: NousPermissionError,
    404: NotFoundError,
    429: RateLimitError,
}

def _detail_to_message_and_param(detail) -> tuple[str, str | None]:
    """Return (message, param) from an HTTPException.detail.

    Handles 3 forms:
    - str: the message itself, no param
    - list of Pydantic error dicts (from validation): extract msg + loc into param
    - anything else: str() fallback
    """
    if isinstance(detail, str):
        return detail, None
    if isinstance(detail, list) and detail:
        first = detail[0] if isinstance(detail[0], dict) else {}
        msg = first.get("msg") or "; ".join(
            e.get("msg", str(e)) if isinstance(e, dict) else str(e) for e in detail
        )
        loc = first.get("loc") or []
        param = ".".join(str(x) for x in loc if x != "body") or None
        return msg, param
    return str(detail), None

def _with_request_id(err: NousError, request) -> NousError:
    err.request_id = getattr(request.state, "request_id", None)
    return err

def _response(err: NousError) -> JSONResponse:
    headers = {"X-Request-Id": err.request_id} if err.request_id else {}
    return JSONResponse(err.to_dict(), status_code=err.http_status, headers=headers)

def _register_error_handlers(app):
    @app.exception_handler(NousError)
    async def _nous(request, exc: NousError):
        return _response(_with_request_id(exc, request))

    @app.exception_handler(HTTPException)
    async def _http(request, exc: HTTPException):
        status = exc.status_code
        cls = _HTTP_STATUS_TO_ERROR.get(status)
        if cls is None:
            cls = InvalidRequestError if 400 <= status < 500 else APIError
        msg, param = _detail_to_message_and_param(exc.detail)
        err = cls(msg, param=param)
        err.http_status = status  # preserve original 4xx nuance (e.g. 409, 422 via HTTPException)
        return _response(_with_request_id(err, request))

    @app.exception_handler(RequestValidationError)
    async def _validation(request, exc: RequestValidationError):
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        err = InvalidRequestError(
            message=first.get("msg", "Invalid request"),
            code="validation_error",
            param=loc or None,
        )
        return _response(_with_request_id(err, request))

    @app.exception_handler(Exception)
    async def _unhandled(request, exc: Exception):
        # Outer safety net: this handler itself must never raise, or FastAPI
        # falls through to a bare 500 with no X-Request-Id and no OpenAI shape.
        rid = None
        try:
            rid = getattr(request.state, "request_id", None)
            try:
                logger.exception(
                    "unhandled exception | req_id=%s | %s %s",
                    rid, request.method, request.url.path,
                )
            except Exception:
                pass  # logging must never crash the handler
            err = APIError("Internal server error", code="internal_error", request_id=rid)
            return _response(err)
        except Exception:
            # Last-resort fallback: hand-built response that bypasses .to_dict()
            headers = {"X-Request-Id": rid} if rid else {}
            return JSONResponse(
                {"error": {"message": "Internal server error", "type": "api_error",
                           "code": "internal_error"}},
                status_code=500, headers=headers,
            )
```

Then in `create_app()` after middleware setup, call `_register_error_handlers(app)` before `return app`.

- [ ] **Step 4: Run tests, confirm pass**
- [ ] **Step 5: Commit** — `feat(api): register 4 global exception handlers for OpenAI-style errors`

---

## Task 4: SSE error protocol for streaming

**Files:**
- Modify: `backend/src/api/routes/openai_compat.py`
- Test: `backend/tests/test_api_errors.py` (extend)

- [ ] **Step 1: Read current streaming implementation**

```bash
grep -n "StreamingResponse\|async def\|yield" backend/src/api/routes/openai_compat.py
```

Identify the async generator that drives SSE (`stream=true` branch).

- [ ] **Step 2: Write failing test**

```python
# backend/tests/test_api_errors.py (append)
import json

def test_sse_error_protocol():
    """When a streaming request fails mid-flight, emit OpenAI-style error chunk."""
    # Simulate by POSTing with an invalid bearer token (should 401 before stream starts)
    # then via a test route that raises inside an async generator.
    from fastapi import APIRouter
    from fastapi.responses import StreamingResponse
    from src.api.main import create_app
    from src.errors import NotFoundError

    app = create_app()
    r = APIRouter()

    @r.get("/__test__/sse-boom")
    async def _sse():
        async def gen():
            yield 'data: {"first": "chunk"}\n\n'
            raise NotFoundError("mid-stream boom")
        # IMPORTANT: the route must wrap gen() in a helper that catches NousError
        # and yields the error envelope + [DONE]. This helper is added in Task 4 step 3.
        from src.api.routes.openai_compat import sse_with_error_envelope
        return StreamingResponse(sse_with_error_envelope(gen()),
                                 media_type="text/event-stream")

    app.include_router(r)
    c = TestClient(app, raise_server_exceptions=False)
    with c.stream("GET", "/__test__/sse-boom") as resp:
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                chunks.append(line[6:])
    assert chunks[0] == '{"first": "chunk"}'
    err_chunk = json.loads(chunks[1])
    assert err_chunk["error"]["type"] == "not_found_error"
    assert err_chunk["error"]["message"] == "mid-stream boom"
    assert chunks[2] == "[DONE]"
```

- [ ] **Step 3: Implement `sse_with_error_envelope` helper in `openai_compat.py`**

```python
import json, logging
from src.errors import NousError, APIError
logger = logging.getLogger(__name__)

async def sse_with_error_envelope(inner):
    """Wrap an async generator so any NousError/Exception is emitted as an
    OpenAI-style SSE error chunk. Guarantees a single `data: [DONE]` closes the
    stream whether inner finished normally, raised, or inner itself yielded [DONE].

    Invariant: the inner generator MUST NOT yield its own `data: [DONE]` terminator.
    All inner [DONE] emissions should be removed when this wrapper is adopted —
    otherwise clients will see a duplicate [DONE] marker.
    """
    sent_done = False
    try:
        async for chunk in inner:
            # Defensive: strip any stray [DONE] the inner generator tried to emit.
            if chunk.strip() == "data: [DONE]":
                continue
            yield chunk
    except NousError as e:
        yield f"data: {json.dumps(e.to_dict())}\n\n"
    except Exception:
        logger.exception("SSE stream failure")
        err = APIError("Internal server error", code="internal_error")
        yield f"data: {json.dumps(err.to_dict())}\n\n"
    finally:
        if not sent_done:
            yield "data: [DONE]\n\n"
            sent_done = True
```

- [ ] **Step 4: Wrap the existing SSE generator in `openai_compat.py`**

1. Find the `StreamingResponse(...)` call in the `stream=true` branch of `/v1/chat/completions`.
2. **Remove** any existing inline `yield "data: [DONE]\n\n"` inside the inner async generator — the wrapper now owns that terminator.
3. **Remove** any existing error catch that yields a non-OpenAI-shape error payload — let exceptions bubble out of `inner` so `sse_with_error_envelope` can format them.
4. Replace the generator argument: `StreamingResponse(sse_with_error_envelope(existing_generator), ...)`.

**Why strict:** leaving old error/[DONE] code in the inner generator results in two different error formats visible to clients (legacy + wrapper) and potentially two `[DONE]` markers. Either fully delegate to the wrapper or don't use it at all.

- [ ] **Step 5: Run tests, confirm pass**
- [ ] **Step 6: Commit** — `feat(openai): emit OpenAI-style error chunk + [DONE] on SSE failure`

---

## Task 5: Frontend NousApiError + apiFetch rewrite

**Files:**
- Create: `frontend/src/api/errors.ts`
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Create `frontend/src/api/errors.ts`**

```typescript
export class NousApiError extends Error {
  type: string
  code?: string
  param?: string
  requestId?: string
  httpStatus: number

  constructor(payload: any, httpStatus: number, fallbackRequestId?: string) {
    const err = payload?.error ?? {}
    super(err.message ?? `HTTP ${httpStatus}`)
    this.name = 'NousApiError'
    this.type = err.type ?? 'api_error'
    this.code = err.code
    this.param = err.param
    this.requestId = err.request_id ?? fallbackRequestId
    this.httpStatus = httpStatus
  }
}
```

- [ ] **Step 2: Rewrite `frontend/src/api/client.ts`**

```typescript
import { NousApiError } from './errors'

const BASE = ''

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    const reqId = resp.headers.get('x-request-id') ?? undefined
    throw new NousApiError(body, resp.status, reqId)
  }
  if (resp.status === 204) return undefined as T
  return resp.json()
}
```

- [ ] **Step 3: Manual verify in browser**

1. `npm run dev`
2. Open Models overlay, trigger an error (e.g. try to load an engine that doesn't exist)
3. DevTools console: `error.message`, `error.type`, `error.requestId` all populated

- [ ] **Step 4: Commit** — `feat(frontend): typed NousApiError from OpenAI-style responses`

**Note:** Backend and frontend changes must ship together. If the two are committed separately, confirm backend is already deployed before the frontend commit is released (the frontend change alone is tolerant to old-style `{detail: "..."}` since `err.message` falls back to `HTTP <status>`).

---

## Task 6: End-to-end integration verification

**Files:** none created, pure verification.

- [ ] **Step 1: Start stack**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center/backend
.venv/bin/python -m uvicorn src.api.main:create_app --factory --port 8000 --reload &
```

- [ ] **Step 2: Run all 5 curl cases from spec §Verification**

Copy-paste from `docs/superpowers/specs/2026-04-14-step1-error-code-standardization-design.md` §Verification. Each case must match the "预期" comment.

- [ ] **Step 2a: Verify CORS exposes X-Request-Id to browser**

```bash
curl --noproxy '*' -I -H "Origin: http://localhost:5173" \
  http://localhost:8000/api/v1/instances 2>&1 | grep -i 'access-control-expose-headers'
# Expected: the response header includes "X-Request-Id" (case-insensitive)
```

If missing, re-check Task 2 Step 4 CORS modification.

- [ ] **Step 3: Inspect `logs.db` app_logs after a 500**

```bash
# Trigger a 500 via the test route (only works if tests ran recently — otherwise add a temp route)
# Then query:
sqlite3 backend/data/logs.db "SELECT level, module, message, location FROM app_logs WHERE level='ERROR' ORDER BY timestamp DESC LIMIT 1;"
```

Confirm: traceback appears in `message` / `location`; request_id in the message line.

- [ ] **Step 4: Manual frontend smoke**

- Open Dashboard, Models, Workflows pages — nothing should visibly break
- Trigger a known error (e.g. delete a non-existent instance) — toast shows a readable message (not `API error: 404`)
- DevTools Network tab: response header has `X-Request-Id`

- [ ] **Step 5: Commit any fixes, then merge**

---

## Post-merge cleanup (optional)

- Over the next few PRs, start using new `NousError` subclasses directly in routes for new code:
  - `raise NotFoundError("...", code="instance_not_found", param="instance_id")`
  - Existing `HTTPException` raises are untouched (still work via auto-conversion).
- Eventually consider a lint rule / pre-commit to prefer `NousError` subclasses over `HTTPException` in new code.

---

## Notes for Implementers

- **Test file location convention:** this project uses `backend/tests/test_api_*.py`. Follow the same prefix.
- **Running tests:** `.venv/bin/pytest backend/tests/test_api_errors.py -v`
- **Backend is running under `uvicorn --reload`.** After Python file changes, no restart needed. After `.env` / docker-compose changes, restart.
- **Frontend hot reload** via vite — code changes apply immediately.
- **Avoid `detail=[...]` forms** in new `HTTPException` raises; the list-to-string conversion is lossy. Prefer `raise NotFoundError(...)` for new code.
- **logs.db is SQLite** (separate from the main PG). Don't move it without updating `log_db.py`.
- **`RequestIdMiddleware` ordering:** It must be the OUTERMOST middleware so `request.state.request_id` is set before any other middleware (audit, logging) runs. In FastAPI/Starlette, `add_middleware` LIFO — so add it LAST.
