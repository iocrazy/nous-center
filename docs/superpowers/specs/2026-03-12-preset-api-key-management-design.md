# Preset API Key Management System

## Goal

Turn each Voice Preset into an independently deployable service instance with its own API endpoint and multiple API keys. External frontend applications authenticate via Bearer Token to call preset-specific endpoints. Usage is tracked per-key.

## Architecture

**Core concept:** Preset = Instance = Endpoint. Each preset gets a stable endpoint path. Multiple API keys can be issued per preset, each with a label, independent usage counters, and revocation capability. Keys are stored as bcrypt hashes — the full key is shown only once at creation time.

**Security model:** Single-user system but hardened for network exposure. All external-facing endpoints (`/v1/preset/*`) require Bearer Token auth. Internal management endpoints (`/api/v1/presets/*`) remain unauthenticated (accessed only from the nous-center frontend on localhost).

**Future extensibility:** The preset type field (`tts`, `image`, `inference`) determines which service endpoint is available. Only TTS is implemented in v1; image and inference follow the same pattern later.

## Data Model

### Existing: VoicePreset (extend)

Add two fields to the existing `VoicePreset` model:

```python
# New columns on VoicePreset
status: str = "active"          # "active" | "inactive"
endpoint_path: str              # Auto-generated: "/v1/preset/{id}/synthesize"
```

### New: PresetApiKey

```python
class PresetApiKey(Base):
    __tablename__ = "preset_api_keys"

    id: BigInteger              # Snowflake ID
    preset_id: BigInteger       # FK → voice_presets.id, ON DELETE CASCADE
    label: str                  # Human-readable name, e.g. "有声小说App"
    key_hash: str               # bcrypt hash of the full key
    key_prefix: str             # First 10 chars for display, e.g. "sk-zw-a3f8"
    is_active: bool = True      # Can be revoked without deleting
    usage_calls: int = 0        # Total API calls made with this key
    usage_chars: int = 0        # Total characters synthesized
    last_used_at: DateTime | None
    created_at: DateTime
```

### Key Format

Generated keys follow the pattern: `sk-{preset_name_prefix}-{32_random_hex}`

Example: `sk-zw-a3f8b7c2d9e1f0456789abcdef012345`

- `sk-` prefix identifies it as a service key
- `{preset_name_prefix}` is 2-4 chars derived from preset name (for human recognition)
- `{32_random_hex}` provides 128 bits of entropy

## API Endpoints

### Key Management (internal, no auth)

```
POST   /api/v1/presets/{preset_id}/keys
  Body: { "label": "有声小说App" }
  Response: { "id": "...", "key": "sk-zw-a3f8...(full key)", "label": "...", "prefix": "sk-zw-a3f8" }
  Note: Full key returned ONLY in this response

GET    /api/v1/presets/{preset_id}/keys
  Response: [{ "id": "...", "label": "...", "prefix": "sk-zw-a3f8", "is_active": true,
               "usage_calls": 1234, "usage_chars": 56000, "last_used_at": "...", "created_at": "..." }]

DELETE /api/v1/presets/{preset_id}/keys/{key_id}
  Response: 204 No Content

PATCH  /api/v1/presets/{preset_id}/status
  Body: { "status": "active" | "inactive" }
  Response: { "id": "...", "status": "active" }
```

### Service Endpoints (external, Bearer Token auth)

```
POST   /v1/preset/{preset_id}/synthesize
  Headers: Authorization: Bearer sk-zw-a3f8b7c2d9e1f0456789abcdef012345
  Body: { "text": "你好世界", "emotion": "happy" }
  Response: audio/wav binary (or JSON with base64, matching existing TTS API)

  Auth flow:
    1. Extract Bearer token from header
    2. Query PresetApiKey WHERE preset_id = {preset_id} AND is_active = true
    3. bcrypt.verify(token, key_hash) for each matching key
    4. If no match → 401 Unauthorized
    5. If preset.status != "active" → 403 Forbidden ("Preset is inactive")
    6. Update key: usage_calls++, usage_chars += len(text), last_used_at = now
    7. Resolve preset params (engine, speed, voice, sample_rate, etc.)
    8. Execute TTS via existing engine infrastructure
    9. Return audio

  Future endpoints (same auth pattern):
    POST /v1/preset/{preset_id}/generate    (image)
    POST /v1/preset/{preset_id}/chat        (inference)
```

### Auth Middleware

A reusable FastAPI dependency:

```python
async def verify_preset_key(
    preset_id: int,
    authorization: str = Header(...),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[VoicePreset, PresetApiKey]:
    """Verify Bearer token and return (preset, matched_key)."""
```

This dependency is injected into all `/v1/preset/*` route handlers.

## Frontend Changes

### Preset Detail Overlay (new)

A new full-screen overlay (consistent with existing Models/Settings overlays), opened by clicking a preset in the Presets panel.

**Layout — two columns:**

Left column:
- Preset name + status badge (active/inactive toggle)
- Configuration summary (engine, speed, sample_rate, voice, etc.)
- Endpoint path (copyable)
- cURL example (copyable)

Right column:
- API Keys list with label, prefix, usage stats, last_used_at
- "+ New Key" button → modal with label input → shows full key once (copy button)
- "Revoke" button per key (with confirmation)
- Usage summary (7-day totals)

### Files to Create/Modify

**New files:**
- `backend/src/models/preset_api_key.py` — PresetApiKey ORM model
- `backend/src/api/routes/preset_service.py` — External `/v1/preset/*` endpoints
- `backend/src/api/deps_auth.py` — Auth dependency (verify_preset_key)
- `frontend/src/api/presetKeys.ts` — React Query hooks for key CRUD
- `frontend/src/components/overlays/PresetDetailOverlay.tsx` — Preset detail UI

**Modified files:**
- `backend/src/models/voice_preset.py` — Add status, endpoint_path columns
- `backend/src/models/__init__.py` — Export new model
- `backend/src/api/main.py` — Register preset_service router
- `backend/src/api/routes/voices.py` — Add PATCH status endpoint
- `frontend/src/api/voices.ts` — Extend VoicePreset interface with status field
- `frontend/src/components/panels/PresetsPanel.tsx` — Click handler to open overlay
- `frontend/src/stores/panel.ts` — Add 'preset-detail' overlay with preset_id param

## Testing Strategy

**Backend:**
- Unit test: key generation format and bcrypt hashing
- Unit test: auth dependency with valid/invalid/revoked keys
- Integration test: full flow — create preset → create key → call synthesize endpoint → verify usage increment
- Integration test: revoked key returns 401
- Integration test: inactive preset returns 403

**Frontend:**
- Verify PresetDetailOverlay renders with mock data
- Verify key creation shows full key, subsequent views show only prefix
- Verify revoke confirmation flow

## Migration

Alembic migration to:
1. Add `status` (default "active") and `endpoint_path` columns to `voice_presets`
2. Create `preset_api_keys` table
3. Backfill `endpoint_path` for existing presets

## Scope Boundaries (v1)

**In scope:**
- PresetApiKey model + CRUD
- Bearer Token auth middleware
- `/v1/preset/{id}/synthesize` endpoint
- PresetDetailOverlay UI
- Basic usage tracking (calls, chars)

**Out of scope (future):**
- Usage dashboard with charts
- Rate limiting / quotas per key
- Third-party credential vault
- Image/inference preset types
- APP ID + Secret signature auth upgrade
- Key expiration dates
