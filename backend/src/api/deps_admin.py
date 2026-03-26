"""Admin token authentication for management API routes."""

from fastapi import HTTPException, Request

from src.config import get_settings


async def require_admin(request: Request):
    """Require admin token for management routes. Skip if ADMIN_TOKEN is empty (dev mode)."""
    settings = get_settings()
    if not settings.ADMIN_TOKEN:
        return  # No auth in dev mode

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing authorization")
    token = auth[7:]
    if token != settings.ADMIN_TOKEN:
        raise HTTPException(403, "Invalid admin token")
