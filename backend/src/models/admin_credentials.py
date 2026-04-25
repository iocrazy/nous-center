"""Passkey (WebAuthn) + TOTP credentials for the single-admin auth gate.

Single-admin scope: there is no users table, so credentials are global —
all rows belong to "the admin". Any registered passkey can log in;
deleting all of them disables passkey login but ADMIN_PASSWORD always
remains as fallback.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Column, DateTime, LargeBinary, String, func

from src.models.database import Base


class WebauthnCredential(Base):
    __tablename__ = "webauthn_credentials"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    label = Column(String(100), nullable=False)
    # Raw bytes from the authenticator. Unique because two different physical
    # authenticators can never produce the same credential_id.
    credential_id = Column(LargeBinary, nullable=False, unique=True, index=True)
    # Opaque COSE-encoded public key — py_webauthn handles serialization.
    public_key = Column(LargeBinary, nullable=False)
    # Replay defense: monotonic counter the authenticator increments per
    # signature. Must be persisted between logins.
    sign_count = Column(BigInteger, nullable=False, default=0)
    # Comma-separated transport hints from the browser (usb,nfc,internal,...).
    transports = Column(String(200), nullable=True)
    # Authenticator model id (16 bytes), nullable for older keys.
    aaguid = Column(LargeBinary, nullable=True)
    # WebAuthn L3 backup flags — visible in UI to spot device loss vs synced
    # passkey (e.g. iCloud Keychain).
    backup_eligible = Column(Boolean, nullable=False, default=False)
    backup_state = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_used_at: datetime | None = Column(DateTime(timezone=True), nullable=True)


class TotpSecret(Base):
    __tablename__ = "totp_secrets"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    label = Column(String(100), nullable=False)
    secret = Column(String(64), nullable=False)  # base32 encoded
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    # NULL = setup in progress (QR scanned, first code not yet entered).
    # Login only accepts secrets where verified_at is NOT NULL.
    verified_at: datetime | None = Column(DateTime(timezone=True), nullable=True)
    last_used_at: datetime | None = Column(DateTime(timezone=True), nullable=True)
