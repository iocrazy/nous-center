"""Request logging and audit middleware."""
import logging
import re
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("nous.access")

# Paths to skip logging (avoid recursion and noise)
_SKIP_PATHS = {"/health", "/favicon.ico", "/api/v1/tasks", "/api/v1/engines/gpus", "/api/v1/nodes/definitions"}
_SKIP_PREFIXES = ("/api/v1/logs/", "/api/v1/monitor/")

# Audit action derivation rules
_AUDIT_RULES: list[tuple[str, str, str]] = [
    (r"POST", r"/api/v1/engines/[^/]+/load$", "load_engine"),
    (r"POST", r"/api/v1/engines/[^/]+/unload$", "unload_engine"),
    (r"POST", r"/api/v1/engines/reload$", "reload_registry"),
    (r"POST", r"/api/v1/workflows$", "create_workflow"),
    (r"PATCH", r"/api/v1/workflows/\d+$", "update_workflow"),
    (r"DELETE", r"/api/v1/workflows/\d+$", "delete_workflow"),
    (r"POST", r"/api/v1/workflows/\d+/publish-app$", "publish_app"),
    (r"DELETE", r"/api/v1/apps/[^/]+$", "unpublish_app"),
]


def derive_audit_action(method: str, path: str) -> str:
    for rule_method, pattern, action in _AUDIT_RULES:
        if method == rule_method and re.match(pattern, path):
            return action
    # Fallback: method_last_segment (skip numeric-only segments)
    segments = [s for s in path.rstrip("/").split("/") if s and not s.isdigit()]
    last = segments[-1] if segments else "unknown"
    return f"{method.lower()}_{last}"


def _audit_detail(body: bytes, content_type: str) -> str:
    """审计 detail = 请求 body 文本。两条护栏:

    1. multipart / 二进制上传(音频转写、图片编辑等)的 body 是二进制,记进 detail 既
       无意义又含 NUL → 只记占位符「<类型 body, N bytes>」。
    2. 文本 body 仍可能含 NUL(0x00):decode(errors="replace") **不**替换合法的 0x00
       (它解码成 U+0000),但 PostgreSQL text 列存不下 NUL → flush 报
       `invalid byte sequence for encoding "UTF8": 0x00`、该审计行丢失(2026-06-21 真机踩:
       音频转写请求每次刷日志失败)。显式剔除 NUL。
    """
    if not body:
        return ""
    ct = content_type.split(";")[0].strip().lower()
    if ct.startswith("multipart/") or ct in ("application/octet-stream",) or ct.startswith(("audio/", "image/", "video/")):
        return f"<{ct or 'binary'} body, {len(body)} bytes>"
    return body.decode("utf-8", errors="replace").replace("\x00", "")[:2000]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        path = request.url.path
        if path in _SKIP_PATHS or any(path.startswith(p) for p in _SKIP_PREFIXES):
            return response

        logger.info("%s %s %d %dms", request.method, path, response.status_code, elapsed_ms)

        # Enqueue log row (non-blocking, silent-fail; batched into main DB).
        try:
            from src.services.log_store import enqueue
            enqueue("request", {
                "method": request.method,
                "path": path,
                "status": response.status_code,
                "duration_ms": elapsed_ms,
                "ip": request.client.host if request.client else "",
                "user_agent": request.headers.get("user-agent", ""),
            })
        except Exception:
            pass

        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Captures admin operations for audit trail."""

    async def dispatch(self, request: Request, call_next):
        # 审计判定:bearer/x-admin-token header **或** 浏览器 admin session cookie。
        # 旧逻辑只看 header → 生产里 admin 走 nous_admin_session cookie 登录,所有经 Web UI
        # 的变更操作(load/unload engine、create/delete workflow、publish、删 key 等)全不进
        # 审计,只剩 CLI/bearer 调用,与「admin operations audit trail」意图矛盾(round2 #5)。
        from src.api.admin_session import request_is_authed  # noqa: PLC0415
        is_admin_request = (
            "authorization" in request.headers
            or "x-admin-token" in request.headers
            or request_is_authed(request)
        )
        path = request.url.path

        if is_admin_request and not path.startswith("/api/v1/logs/"):
            # Read body for detail (cache it for downstream)
            body = b""
            try:
                body = await request.body()
            except Exception:
                pass

            response = await call_next(request)

            # Only log mutating operations that succeeded
            if request.method in ("POST", "PUT", "PATCH", "DELETE") and response.status_code < 500:
                try:
                    from src.services.log_store import enqueue
                    action = derive_audit_action(request.method, path)
                    detail = _audit_detail(body, request.headers.get("content-type", ""))
                    enqueue("audit", {
                        "action": action,
                        "path": path,
                        "method": request.method,
                        "ip": request.client.host if request.client else "",
                        "detail": detail,
                    })
                except Exception:
                    pass

            return response

        return await call_next(request)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Generate or propagate X-Request-Id header.

    Must run FIRST so ``request.state.request_id`` is populated before any
    downstream middleware or exception handler reads it. Starlette's
    ``add_middleware`` is LIFO — add this LAST to run first.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response


# Path prefixes that require an admin browser session when ADMIN_PASSWORD is set.
# /v1/* keeps its own bearer-token auth (external LLM clients), /sys/admin/*
# must stay reachable so the user can log in, and static SPA paths must load
# the login page itself.
_ADMIN_GATE_PREFIXES = ("/api/",)

# External (API-key authed) endpoints that unavoidably live under /api/ because
# an external protocol fixes their path. Ollama clients hardcode /api/chat etc.,
# so we can't move them under /v1/. They authenticate in-route via
# verify_bearer_token_any (the user's API key), NOT the admin cookie — they must
# bypass the admin gate or every external Ollama client gets 401 before its
# route ever runs. Keep in sync with ollama_compat.py's route decorators.
_ADMIN_GATE_EXEMPT_PATHS = frozenset({
    "/api/chat",
    "/api/generate",
    "/api/tags",
    "/api/show",
})


class AdminSessionGateMiddleware(BaseHTTPMiddleware):
    """Block /api/* requests without a valid admin session cookie.

    Disabled when ADMIN_PASSWORD is empty (dev mode). WebSocket auth is handled
    in the endpoint functions because BaseHTTPMiddleware doesn't see WS upgrades.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith(_ADMIN_GATE_PREFIXES) and path not in _ADMIN_GATE_EXEMPT_PATHS:
            from src.api.admin_session import request_is_authed
            if not request_is_authed(request):
                from starlette.responses import JSONResponse
                return JSONResponse({"detail": "admin login required"}, status_code=401)
        return await call_next(request)
