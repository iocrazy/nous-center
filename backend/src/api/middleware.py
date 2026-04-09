"""Request logging and audit middleware."""
import asyncio
import json
import logging
import re
import time
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


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        path = request.url.path
        if path in _SKIP_PATHS or any(path.startswith(p) for p in _SKIP_PREFIXES):
            return response

        logger.info("%s %s %d %dms", request.method, path, response.status_code, elapsed_ms)

        # Write to log DB (fire-and-forget, non-blocking)
        try:
            from src.services.log_db import insert_request_log
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, lambda: insert_request_log(
                method=request.method,
                path=path,
                status=response.status_code,
                duration_ms=elapsed_ms,
                ip=request.client.host if request.client else "",
                user_agent=request.headers.get("user-agent", ""),
            ))
        except Exception:
            pass

        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Captures admin operations for audit trail."""

    async def dispatch(self, request: Request, call_next):
        # Only audit requests with admin token
        has_admin_token = "authorization" in request.headers or "x-admin-token" in request.headers
        path = request.url.path

        if has_admin_token and not path.startswith("/api/v1/logs/"):
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
                    from src.services.log_db import insert_audit_log
                    action = derive_audit_action(request.method, path)
                    detail = body.decode("utf-8", errors="replace")[:2000] if body else ""
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(None, lambda: insert_audit_log(
                        action=action,
                        path=path,
                        method=request.method,
                        ip=request.client.host if request.client else "",
                        detail=detail,
                    ))
                except Exception:
                    pass

            return response

        return await call_next(request)
