import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("nous.access")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Skip health checks and static files from logging
        path = request.url.path
        if path in ("/health", "/favicon.ico"):
            return response

        logger.info(
            "%s %s %d %dms",
            request.method,
            path,
            response.status_code,
            elapsed_ms,
        )
        return response
