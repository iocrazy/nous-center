"""Passkey (WebAuthn) routes for the single-admin gate.

POST /sys/admin/passkey/register/start  (need admin cookie)
POST /sys/admin/passkey/register/finish (need admin cookie)
POST /sys/admin/passkey/login/start     (open)
POST /sys/admin/passkey/login/finish    (open) → sets admin session cookie
GET  /sys/admin/passkey                 (need admin cookie) — list registered
DELETE /sys/admin/passkey/{cred_id}     (need admin cookie) — drop one

## Why we don't replicate ai-tracker's design as-is

ai-tracker has a CRITICAL bug in /api/auth/passkey/poll: any made-up auth_id
returns a valid JWT because it checks "auth_id is NOT in auth_states map"
without distinguishing "never created" from "successfully consumed". We avoid
the entire poll flow by using a single-hop set-cookie response, which has no
mid-flight cancellation problem in the cookie domain.

Other deliberate departures:
- challenge state has 5-minute TTL (ai-tracker's grows unbounded)
- user_handle is fixed b"nous-admin" so multiple passkeys are recognized as
  the SAME user by platform authenticators (iOS Keychain, etc.)
- session cookie reuses ADMIN_SESSION_SECRET (no separate JWT secret derived
  from the bootstrap password)
"""

from __future__ import annotations

import base64
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers.cose import COSEAlgorithmIdentifier
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from src.api.admin_session import (
    COOKIE_NAME,
    is_login_required,
    issue_token,
    request_is_authed,
)
from src.config import get_settings
from src.models.admin_credentials import WebauthnCredential
from src.models.database import get_async_session

router = APIRouter(prefix="/sys/admin/passkey", tags=["admin-passkey"])

# Single-admin user handle. Stable across all passkey registrations so platform
# authenticators (iCloud Keychain, 1Password) treat them as ONE user with
# multiple keys, not multiple distinct users.
_USER_HANDLE = b"nous-admin"
_USER_NAME = "admin"
_USER_DISPLAY_NAME = "nous-center admin"

# In-process challenge stores. Single-process backend; never grows unbounded
# because of TTL eviction in _store_challenge / _take_challenge.
_CHALLENGE_TTL_SECONDS = 5 * 60
_reg_challenges: dict[str, tuple[bytes, float]] = {}
_login_challenges: dict[str, tuple[bytes, float]] = {}


def _now() -> float:
    return time.monotonic()


def _evict_expired(store: dict[str, tuple[bytes, float]]) -> None:
    cutoff = _now()
    expired = [k for k, (_, exp) in store.items() if exp < cutoff]
    for k in expired:
        store.pop(k, None)


def _store_challenge(store: dict[str, tuple[bytes, float]], key: str, challenge: bytes) -> None:
    _evict_expired(store)
    store[key] = (challenge, _now() + _CHALLENGE_TTL_SECONDS)


def _take_challenge(store: dict[str, tuple[bytes, float]], key: str) -> bytes | None:
    _evict_expired(store)
    item = store.pop(key, None)
    if item is None:
        return None
    challenge, exp = item
    if exp < _now():
        return None
    return challenge


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _from_b64u(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _origins() -> list[str]:
    raw = get_settings().ADMIN_PASSKEY_RP_ORIGINS or ""
    return [o.strip() for o in raw.split(",") if o.strip()]


def _require_admin_cookie(request: Request) -> None:
    """Routes that mutate credentials need an active admin session.

    In dev mode (ADMIN_PASSWORD empty) the gate is off — registering passkeys
    without prior auth would be a footgun. Always require explicit login.
    """
    if not is_login_required():
        raise HTTPException(
            status_code=400,
            detail="passkey registration requires ADMIN_PASSWORD to be set",
        )
    if not request_is_authed(request):
        raise HTTPException(status_code=401, detail="admin login required")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class PasskeyStatusOut(BaseModel):
    enabled: bool
    has_credentials: bool


@router.get("/status", response_model=PasskeyStatusOut)
async def passkey_status(session: AsyncSession = Depends(get_async_session)) -> PasskeyStatusOut:
    """Public — Login UI calls this to decide whether to show the passkey button."""
    settings = get_settings()
    enabled = bool(settings.ADMIN_PASSKEY_RP_ID and settings.ADMIN_PASSWORD)
    if not enabled:
        return PasskeyStatusOut(enabled=False, has_credentials=False)
    count = (
        await session.execute(select(WebauthnCredential.id).limit(1))
    ).first()
    return PasskeyStatusOut(enabled=True, has_credentials=count is not None)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class RegisterFinishBody(BaseModel):
    challenge_id: str
    label: str
    # The full RegistrationResponseJSON the browser produced — we keep it as a
    # generic dict because shape comes from navigator.credentials.create().
    credential: dict


@router.post("/register/start")
async def register_start(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    _require_admin_cookie(request)
    settings = get_settings()

    existing = (await session.execute(select(WebauthnCredential.credential_id))).scalars().all()
    exclude = [
        PublicKeyCredentialDescriptor(id=cid) for cid in existing
    ]

    options = generate_registration_options(
        rp_id=settings.ADMIN_PASSKEY_RP_ID,
        rp_name=settings.ADMIN_PASSKEY_RP_NAME,
        user_id=_USER_HANDLE,
        user_name=_USER_NAME,
        user_display_name=_USER_DISPLAY_NAME,
        exclude_credentials=exclude or None,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
        supported_pub_key_algs=[
            COSEAlgorithmIdentifier.ECDSA_SHA_256,
            COSEAlgorithmIdentifier.RSASSA_PKCS1_v1_5_SHA_256,
        ],
    )

    challenge_id = _b64u(options.challenge[:16])
    _store_challenge(_reg_challenges, challenge_id, options.challenge)

    return {
        "challenge_id": challenge_id,
        "publicKey": options_to_json(options),  # JSON string already encoded
    }


@router.post("/register/finish")
async def register_finish(
    body: RegisterFinishBody,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    _require_admin_cookie(request)
    settings = get_settings()

    challenge = _take_challenge(_reg_challenges, body.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=400, detail="challenge expired or unknown")

    try:
        verification = verify_registration_response(
            credential=body.credential,
            expected_challenge=challenge,
            expected_origin=_origins(),
            expected_rp_id=settings.ADMIN_PASSKEY_RP_ID,
            require_user_verification=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"verification failed: {exc}") from exc

    transports = body.credential.get("response", {}).get("transports") or []
    cred = WebauthnCredential(
        label=body.label[:100],
        credential_id=verification.credential_id,
        public_key=verification.credential_public_key,
        sign_count=verification.sign_count,
        transports=",".join(transports)[:200] if transports else None,
        aaguid=verification.aaguid.bytes if verification.aaguid else None,
        backup_eligible=bool(getattr(verification, "credential_backed_up", False)),
        backup_state=bool(getattr(verification, "credential_backed_up", False)),
    )
    session.add(cred)
    await session.commit()
    await session.refresh(cred)
    return {"id": cred.id, "label": cred.label, "created_at": cred.created_at.isoformat()}


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class LoginFinishBody(BaseModel):
    challenge_id: str
    credential: dict


@router.post("/login/start")
async def login_start(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    if not is_login_required():
        raise HTTPException(status_code=400, detail="admin login is disabled")
    settings = get_settings()

    creds = (await session.execute(select(WebauthnCredential))).scalars().all()
    if not creds:
        raise HTTPException(status_code=409, detail="no passkeys registered")

    options = generate_authentication_options(
        rp_id=settings.ADMIN_PASSKEY_RP_ID,
        allow_credentials=[
            PublicKeyCredentialDescriptor(id=c.credential_id) for c in creds
        ],
        user_verification=UserVerificationRequirement.PREFERRED,
    )
    challenge_id = _b64u(options.challenge[:16])
    _store_challenge(_login_challenges, challenge_id, options.challenge)
    return {
        "challenge_id": challenge_id,
        "publicKey": options_to_json(options),
    }


@router.post("/login/finish")
async def login_finish(
    body: LoginFinishBody,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_async_session),
):
    if not is_login_required():
        raise HTTPException(status_code=400, detail="admin login is disabled")
    settings = get_settings()

    challenge = _take_challenge(_login_challenges, body.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=400, detail="challenge expired or unknown")

    raw_id = body.credential.get("rawId") or body.credential.get("id")
    if not raw_id:
        raise HTTPException(status_code=400, detail="missing credential id")
    credential_id_bytes = _from_b64u(raw_id)

    cred = (
        await session.execute(
            select(WebauthnCredential).where(WebauthnCredential.credential_id == credential_id_bytes)
        )
    ).scalar_one_or_none()
    if cred is None:
        raise HTTPException(status_code=401, detail="unknown credential")

    try:
        verification = verify_authentication_response(
            credential=body.credential,
            expected_challenge=challenge,
            expected_rp_id=settings.ADMIN_PASSKEY_RP_ID,
            expected_origin=_origins(),
            credential_public_key=cred.public_key,
            credential_current_sign_count=cred.sign_count,
            require_user_verification=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"verification failed: {exc}") from exc

    # Replay defense: only update sign_count if it advanced (or is 0 — some
    # platform authenticators always return 0 by spec).
    if verification.new_sign_count > cred.sign_count or verification.new_sign_count == 0:
        cred.sign_count = verification.new_sign_count
    cred.last_used_at = datetime.now(timezone.utc)
    await session.commit()

    # Same cookie as password login → all downstream code (admin gate, WS gate)
    # works identically regardless of how the admin authenticated.
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
    return {"ok": True, "credential_label": cred.label}


# ---------------------------------------------------------------------------
# Manage
# ---------------------------------------------------------------------------


class CredentialOut(BaseModel):
    id: int
    label: str
    transports: str | None
    backup_eligible: bool
    backup_state: bool
    created_at: datetime
    last_used_at: datetime | None


@router.get("", response_model=list[CredentialOut])
async def list_credentials(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    _require_admin_cookie(request)
    rows = (
        await session.execute(
            select(WebauthnCredential).order_by(WebauthnCredential.created_at.desc())
        )
    ).scalars().all()
    return [
        CredentialOut(
            id=c.id,
            label=c.label,
            transports=c.transports,
            backup_eligible=c.backup_eligible,
            backup_state=c.backup_state,
            created_at=c.created_at,
            last_used_at=c.last_used_at,
        )
        for c in rows
    ]


@router.delete("/{cred_id}", status_code=204)
async def delete_credential(
    cred_id: int,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    _require_admin_cookie(request)
    cred = await session.get(WebauthnCredential, cred_id)
    if cred is None:
        raise HTTPException(status_code=404, detail="not found")
    await session.delete(cred)
    await session.commit()
