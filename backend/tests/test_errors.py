"""Unit tests for NousError classes."""

from src.errors import (
    NousError,
    InvalidRequestError,
    AuthenticationError,
    PermissionError as NousPermissionError,
    NotFoundError,
    RateLimitError,
    APIError,
)


def test_default_fields():
    e = APIError("boom")
    assert e.type == "api_error"
    assert e.http_status == 500
    assert e.to_dict() == {"error": {"message": "boom", "type": "api_error"}}


def test_with_all_fields():
    e = InvalidRequestError(
        "bad model", code="model_not_found", param="model", request_id="req-123"
    )
    assert e.http_status == 400
    assert e.to_dict() == {
        "error": {
            "message": "bad model",
            "type": "invalid_request_error",
            "code": "model_not_found",
            "param": "model",
            "request_id": "req-123",
        }
    }


def test_http_status_per_subclass():
    assert AuthenticationError("").http_status == 401
    assert NousPermissionError("").http_status == 403
    assert NotFoundError("").http_status == 404
    assert RateLimitError("").http_status == 429


def test_nouserror_is_exception():
    try:
        raise NotFoundError("x")
    except NousError as e:
        assert str(e) == "x"


def test_to_dict_omits_none_fields():
    assert NotFoundError("x").to_dict() == {
        "error": {"message": "x", "type": "not_found_error"}
    }


from src.errors import ConflictError


def test_conflict_error_409():
    e = ConflictError("dup write", code="session_concurrent_write")
    assert e.http_status == 409
    assert e.type == "invalid_request_error"
    assert e.to_dict() == {"error": {
        "message": "dup write",
        "type": "invalid_request_error",
        "code": "session_concurrent_write",
    }}
