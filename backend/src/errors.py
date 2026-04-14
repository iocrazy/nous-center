"""OpenAI-style error classes for nous-center.

Raise these from route handlers or services; the global exception handlers in
``src.api.main`` serialize them into ``{"error": {"message", "type", "code",
"param", "request_id"}}`` and set the right HTTP status.

New code should prefer these over ``HTTPException`` — they carry ``code`` and
``param`` fields that survive into the response.
"""

from __future__ import annotations


class NousError(Exception):
    """Base class. Subclasses declare ``type`` and ``http_status``."""

    type: str = "api_error"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        request_id: str | None = None,
    ):
        self.message = message
        self.code = code
        self.param = param
        self.request_id = request_id
        super().__init__(message)

    def to_dict(self) -> dict:
        err: dict = {"message": self.message, "type": self.type}
        if self.code:
            err["code"] = self.code
        if self.param:
            err["param"] = self.param
        if self.request_id:
            err["request_id"] = self.request_id
        return {"error": err}


class InvalidRequestError(NousError):
    type = "invalid_request_error"
    http_status = 400


class AuthenticationError(NousError):
    type = "authentication_error"
    http_status = 401


class PermissionError(NousError):  # noqa: A001 — shadow builtin on purpose
    type = "permission_error"
    http_status = 403


class NotFoundError(NousError):
    type = "not_found_error"
    http_status = 404


class RateLimitError(NousError):
    type = "rate_limit_error"
    http_status = 429


class APIError(NousError):
    type = "api_error"
    http_status = 500
