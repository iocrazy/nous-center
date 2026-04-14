"""Integration tests: RequestIdMiddleware + 4 exception handlers.

We construct a fresh TestClient with extra test-only routes mounted. The
conftest autouse fixture detaches DbLogHandler so these tests (which trigger
500s and logger.exception) don't pollute backend/data/logs.db.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Body, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.api.main import create_app
from src.errors import NotFoundError, RateLimitError


class ValidateBody(BaseModel):
    name: str


def _client_no_routes() -> TestClient:
    return TestClient(create_app())


def _client_with_test_routes() -> TestClient:
    app = create_app()
    r = APIRouter()

    @r.get("/__test__/notfound")
    async def _nf():
        raise NotFoundError("widget missing", code="widget_not_found")

    @r.get("/__test__/http404")
    async def _http404():
        raise HTTPException(404, "route raised 404")

    @r.post("/__test__/validate")
    async def _val(payload: ValidateBody):
        return payload

    @r.get("/__test__/boom")
    async def _boom():
        raise RuntimeError("kaboom")

    @r.get("/__test__/ratelimit")
    async def _rl():
        raise RateLimitError("too fast", code="rpm_exceeded")

    @r.get("/__test__/http422-list")
    async def _http422_list():
        raise HTTPException(
            422,
            detail=[{"loc": ["body", "messages", 0, "role"], "msg": "bad role"}],
        )

    app.include_router(r)
    # raise_server_exceptions=False lets the 500 handler run instead of TestClient re-raising
    return TestClient(app, raise_server_exceptions=False)


# --- RequestIdMiddleware ---


def test_request_id_generated_when_absent():
    r = _client_no_routes().get("/health")
    assert "x-request-id" in r.headers
    assert len(r.headers["x-request-id"]) >= 8


def test_request_id_echoed_when_provided():
    r = _client_no_routes().get(
        "/health", headers={"X-Request-Id": "my-trace-abc"}
    )
    assert r.headers["x-request-id"] == "my-trace-abc"


# --- NousError handler ---


def test_nous_error_serialized():
    r = _client_with_test_routes().get("/__test__/notfound")
    assert r.status_code == 404
    assert r.json() == {
        "error": {
            "message": "widget missing",
            "type": "not_found_error",
            "code": "widget_not_found",
            "request_id": r.headers["x-request-id"],
        }
    }


# --- HTTPException handler ---


def test_httpexception_converted_to_openai_shape():
    r = _client_with_test_routes().get("/__test__/http404")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["type"] == "not_found_error"
    assert body["error"]["message"] == "route raised 404"
    assert body["error"]["request_id"] == r.headers["x-request-id"]


def test_httpexception_with_list_detail_preserves_param():
    """Pydantic-style list detail: extract msg + loc into param."""
    r = _client_with_test_routes().get("/__test__/http422-list")
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["message"] == "bad role"
    assert body["error"]["param"] == "messages.0.role"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "validation_error"


# --- RequestValidationError handler ---


def test_validation_error_shape():
    r = _client_with_test_routes().post("/__test__/validate", json={})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["param"] == "name"


# --- Exception (500) handler ---


def test_500_generic_message_no_traceback_leak():
    r = _client_with_test_routes().get("/__test__/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["message"] == "Internal server error"
    assert body["error"]["type"] == "api_error"
    assert body["error"]["code"] == "internal_error"
    # CRITICAL: traceback/exception text must not leak
    assert "kaboom" not in r.text
    assert "RuntimeError" not in r.text


# --- Rate limit shortcut ---


def test_rate_limit_status_and_type():
    r = _client_with_test_routes().get("/__test__/ratelimit")
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["code"] == "rpm_exceeded"
