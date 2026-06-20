"""Admin authentication dependency for management API routes."""

from fastapi import HTTPException, Request

from src.api.admin_session import request_is_authed


async def require_admin(request: Request):
    """Require an authenticated admin for management routes.

    Delegates to the SAME predicate the global ``AdminSessionGateMiddleware``
    uses (``request_is_authed`` = valid admin session cookie OR
    ``Authorization: Bearer <ADMIN_TOKEN>``), so there is one source of truth and
    no divergence between the route guard and the middleware.

    Passes in dev mode (no ADMIN_PASSWORD configured → ``is_login_required`` is
    False). Previously this keyed its bypass off ADMIN_TOKEN alone, which made it
    a silent no-op in production (ADMIN_TOKEN empty but ADMIN_PASSWORD set) — the
    guard enforced nothing and only the middleware was actually protecting these
    routes.
    """
    if not request_is_authed(request):
        raise HTTPException(401, "admin login required")
