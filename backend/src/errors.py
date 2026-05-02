"""OpenAI-style error classes for nous-center.

Raise these from route handlers or services; the global exception handlers in
``src.api.main`` serialize them into ``{"error": {"message", "type", "code",
"param", "request_id"}}`` and set the right HTTP status.

New code should prefer these over ``HTTPException`` — they carry ``code`` and
``param`` fields that survive into the response.
"""

from __future__ import annotations


class NousError(Exception):
    """Base class. Subclasses declare ``type`` and ``http_status``.

    Stripe-style structured envelope: every error carries ``message``,
    optional ``code`` (machine-readable), optional ``fix`` (one-line
    actionable hint for the caller), optional ``doc_url`` (deep link
    to docs). plan-devex-review D3.1 = B requires every service error
    follow this shape so caller code can ``if e.type == "lora_not_found":``
    program around it.
    """

    type: str = "api_error"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        fix: str | None = None,
        doc_url: str | None = None,
        request_id: str | None = None,
    ):
        self.message = message
        self.code = code
        self.param = param
        self.fix = fix
        self.doc_url = doc_url
        self.request_id = request_id
        super().__init__(message)

    def to_dict(self) -> dict:
        err: dict = {"message": self.message, "type": self.type}
        if self.code:
            err["code"] = self.code
        if self.param:
            err["param"] = self.param
        if self.fix:
            err["fix"] = self.fix
        if self.doc_url:
            err["doc_url"] = self.doc_url
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


class ConflictError(NousError):
    # Matches main.py _HTTP_STATUS_TO_ERROR[409] -> InvalidRequestError type
    type = "invalid_request_error"
    http_status = 409


class ServiceUnavailableError(NousError):
    """503 — model still loading, OOM, transient inference failure, etc."""

    type = "service_unavailable"
    http_status = 503


class ModelNotFoundError(NotFoundError):
    """Adapter requested for a model id that has no spec (yaml or scan miss).

    Surfaces as 404 with code='model_not_found'. Image / LLM / TTS service
    routes raise this when caller's `model_key` doesn't resolve.
    """

    def __init__(self, model_id: str, **kwargs):
        super().__init__(
            f"model {model_id!r} not found",
            code="model_not_found",
            param="model_key",
            fix="Check available models via GET /api/v1/engines",
            **kwargs,
        )


class ModelLoadError(ServiceUnavailableError):
    """Model spec exists but load failed (recorded in model_manager._load_failures).

    Surfaces as 503. Caller can retry after admin intervention.
    """

    def __init__(self, model_id: str, reason: str, **kwargs):
        super().__init__(
            f"model {model_id!r} failed to load: {reason}",
            code="model_load_failed",
            param="model_key",
            fix="Check backend logs; admin must call load_model explicitly to retry",
            **kwargs,
        )
