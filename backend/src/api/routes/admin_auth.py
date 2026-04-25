"""Browser admin login routes — POST /sys/admin/login + logout + me."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from src.api.admin_session import (
    COOKIE_NAME,
    is_login_required,
    issue_token,
    request_is_authed,
)
from src.config import get_settings

router = APIRouter(prefix="/sys/admin", tags=["admin"])


class LoginPayload(BaseModel):
    password: str


class LoginOk(BaseModel):
    ok: bool = True


class MeResponse(BaseModel):
    login_required: bool
    authenticated: bool


@router.get("/me", response_model=MeResponse)
async def me(request: Request) -> MeResponse:
    return MeResponse(
        login_required=is_login_required(),
        authenticated=request_is_authed(request),
    )


@router.post("/login", response_model=LoginOk)
async def login(payload: LoginPayload, request: Request, response: Response) -> LoginOk:
    settings = get_settings()
    if not is_login_required():
        # Login is a no-op in dev mode — still set cookie so frontend code paths
        # work the same in both environments.
        return LoginOk()

    expected = settings.ADMIN_PASSWORD.encode("utf-8")
    given = payload.password.encode("utf-8")
    if not hmac.compare_digest(given, expected):
        raise HTTPException(status_code=401, detail="invalid password")

    token, max_age = issue_token()
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )
    return LoginOk()


@router.post("/logout", response_model=LoginOk)
async def logout(response: Response) -> LoginOk:
    response.delete_cookie(COOKIE_NAME, path="/")
    return LoginOk()
