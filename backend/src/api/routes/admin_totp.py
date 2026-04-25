"""TOTP (RFC 6238) backup auth for the single-admin gate.

POST /sys/admin/totp/setup    (need admin cookie) — issue secret + QR
POST /sys/admin/totp/verify   (need admin cookie) — finish setup with first code
POST /sys/admin/totp/login    (open) → sets admin session cookie
GET  /sys/admin/totp          (need admin cookie) — list verified secrets
DELETE /sys/admin/totp/{id}   (need admin cookie) — drop one

Login uses pyotp.TOTP.verify with valid_window=1 (accepts ±30s drift),
sufficient for clock drift between authenticator app and server.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pyotp
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.admin_session import (
    COOKIE_NAME,
    is_login_required,
    issue_token,
    request_is_authed,
)
from src.config import get_settings
from src.models.admin_credentials import TotpSecret
from src.models.database import get_async_session

router = APIRouter(prefix="/sys/admin/totp", tags=["admin-totp"])


def _require_admin_cookie(request: Request) -> None:
    if not is_login_required():
        raise HTTPException(status_code=400, detail="totp requires ADMIN_PASSWORD to be set")
    if not request_is_authed(request):
        raise HTTPException(status_code=401, detail="admin login required")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TotpStatusOut(BaseModel):
    enabled: bool
    has_verified: bool


@router.get("/status", response_model=TotpStatusOut)
async def totp_status(session: AsyncSession = Depends(get_async_session)) -> TotpStatusOut:
    """Public — Login UI calls this to decide whether to show the TOTP button."""
    enabled = bool(get_settings().ADMIN_PASSWORD)
    if not enabled:
        return TotpStatusOut(enabled=False, has_verified=False)
    row = (
        await session.execute(
            select(TotpSecret.id).where(TotpSecret.verified_at.is_not(None)).limit(1)
        )
    ).first()
    return TotpStatusOut(enabled=True, has_verified=row is not None)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


class SetupBody(BaseModel):
    label: str = Field(..., min_length=1, max_length=100)


class SetupOut(BaseModel):
    id: int
    label: str
    secret: str  # base32, shown ONCE; user should not need to see it again
    otpauth_url: str  # for QR rendering


@router.post("/setup", response_model=SetupOut)
async def totp_setup(
    body: SetupBody,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
) -> SetupOut:
    _require_admin_cookie(request)

    secret = pyotp.random_base32()
    row = TotpSecret(label=body.label[:100], secret=secret)
    session.add(row)
    await session.commit()
    await session.refresh(row)

    otpauth = pyotp.totp.TOTP(secret).provisioning_uri(
        name=body.label, issuer_name="nous-center"
    )
    return SetupOut(id=row.id, label=row.label, secret=secret, otpauth_url=otpauth)


class VerifyBody(BaseModel):
    id: int
    code: str = Field(..., min_length=6, max_length=10)


@router.post("/verify")
async def totp_verify(
    body: VerifyBody,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    _require_admin_cookie(request)

    row = await session.get(TotpSecret, body.id)
    if row is None:
        raise HTTPException(status_code=404, detail="secret not found")
    if not pyotp.TOTP(row.secret).verify(body.code, valid_window=1):
        raise HTTPException(status_code=400, detail="invalid code")

    row.verified_at = datetime.now(timezone.utc)
    row.last_used_at = row.verified_at
    await session.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class LoginBody(BaseModel):
    code: str = Field(..., min_length=6, max_length=10)


@router.post("/login")
async def totp_login(
    body: LoginBody,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_async_session),
):
    if not is_login_required():
        raise HTTPException(status_code=400, detail="admin login is disabled")

    rows = (
        await session.execute(
            select(TotpSecret).where(TotpSecret.verified_at.is_not(None))
        )
    ).scalars().all()
    if not rows:
        raise HTTPException(status_code=409, detail="no totp secret configured")

    matched: TotpSecret | None = None
    for r in rows:
        if pyotp.TOTP(r.secret).verify(body.code, valid_window=1):
            matched = r
            break
    if matched is None:
        raise HTTPException(status_code=401, detail="invalid code")

    matched.last_used_at = datetime.now(timezone.utc)
    await session.commit()

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
    return {"ok": True}


# ---------------------------------------------------------------------------
# Manage
# ---------------------------------------------------------------------------


class TotpOut(BaseModel):
    id: int
    label: str
    created_at: datetime
    verified_at: datetime | None
    last_used_at: datetime | None


@router.get("", response_model=list[TotpOut])
async def list_totp(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    _require_admin_cookie(request)
    rows = (
        await session.execute(select(TotpSecret).order_by(TotpSecret.created_at.desc()))
    ).scalars().all()
    return [
        TotpOut(
            id=r.id,
            label=r.label,
            created_at=r.created_at,
            verified_at=r.verified_at,
            last_used_at=r.last_used_at,
        )
        for r in rows
    ]


@router.delete("/{secret_id}", status_code=204)
async def delete_totp(
    secret_id: int,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    _require_admin_cookie(request)
    row = await session.get(TotpSecret, secret_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    await session.delete(row)
    await session.commit()
