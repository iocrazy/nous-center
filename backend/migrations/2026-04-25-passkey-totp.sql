-- backend/migrations/2026-04-25-passkey-totp.sql
-- Passkey (WebAuthn) + TOTP for /sys/admin auth — adds two new tables.
--
-- Single-admin scope: there is no users table, so credentials are global —
-- registered passkeys / TOTP secrets all belong to "the admin". Any of the
-- registered passkeys can log in; deleting all of them disables passkey
-- login (password fallback always remains via ADMIN_PASSWORD).
--
-- Single transaction, IF NOT EXISTS idempotent (re-runnable).
--
-- Why these design choices (vs replicating ai-tracker which has critical bugs):
--   * `webauthn_credentials.public_key` stored as bytea (raw COSE key) —
--     opaque to the DB; py_webauthn handles serialization.
--   * `sign_count` must be persisted between logins (replay defense).
--   * `last_used_at` shown in UI so admin can spot a stolen authenticator.
--   * `totp_secrets.verified_at IS NULL` means setup-in-progress (user
--     scanned QR but hasn't entered the first code yet); login only accepts
--     verified secrets.

BEGIN;

CREATE TABLE IF NOT EXISTS webauthn_credentials (
    id              BIGSERIAL PRIMARY KEY,
    label           VARCHAR(100) NOT NULL,           -- "MacBook TouchID", "YubiKey 5"
    credential_id   BYTEA NOT NULL UNIQUE,           -- raw bytes from authenticator
    public_key      BYTEA NOT NULL,                  -- COSE public key (opaque)
    sign_count      BIGINT NOT NULL DEFAULT 0,       -- replay defense counter
    transports      VARCHAR(200),                    -- comma list: "usb,nfc,internal"
    aaguid          BYTEA,                           -- authenticator model id (16 bytes)
    backup_eligible BOOLEAN NOT NULL DEFAULT FALSE,  -- can sync across devices?
    backup_state    BOOLEAN NOT NULL DEFAULT FALSE,  -- is currently synced?
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_webauthn_credentials_credential_id
    ON webauthn_credentials (credential_id);

CREATE TABLE IF NOT EXISTS totp_secrets (
    id           BIGSERIAL PRIMARY KEY,
    label        VARCHAR(100) NOT NULL,              -- "Authy", "1Password TOTP"
    secret       VARCHAR(64) NOT NULL,               -- base32 (32 bytes encoded)
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    verified_at  TIMESTAMPTZ,                        -- NULL = setup in progress
    last_used_at TIMESTAMPTZ
);

COMMIT;
