# Preset API Key Management — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn each Voice Preset into an independently deployable service instance with its own API endpoint and multiple API keys, with usage tracking.

**Architecture:** Extend VoicePreset model with status/endpoint_path. New PresetApiKey ORM model with bcrypt-hashed keys. Bearer Token auth middleware for external `/v1/preset/*` endpoints. React Query hooks + PresetDetailOverlay for frontend key management.

**Tech Stack:** FastAPI, SQLAlchemy async, bcrypt, React, TanStack Query, Zustand

---

## File Structure

### New Files
- `backend/src/models/preset_api_key.py` — PresetApiKey ORM model
- `backend/src/api/deps_auth.py` — Bearer Token auth dependency
- `backend/src/api/routes/preset_keys.py` — Key management CRUD (internal, no auth)
- `backend/src/api/routes/preset_service.py` — External `/v1/preset/*` endpoints (auth required)
- `frontend/src/api/presetKeys.ts` — React Query hooks for key CRUD + status
- `frontend/src/components/overlays/PresetDetailOverlay.tsx` — Two-column preset detail UI

### Modified Files
- `backend/pyproject.toml` — Add `bcrypt` dependency
- `backend/src/models/voice_preset.py` — Add `status`, `endpoint_path` columns
- `backend/src/models/schemas.py` — Add PresetApiKey schemas, extend VoicePresetOut
- `backend/src/api/main.py` — Register new routers
- `frontend/src/api/voices.ts` — Add `status` to VoicePreset interface
- `frontend/src/stores/panel.ts` — Add `'preset-detail'` overlay with presetId param
- `frontend/src/components/panels/PresetsPanel.tsx` — Click to open overlay
- `frontend/src/components/nodes/NodeEditor.tsx` — Render PresetDetailOverlay
- `frontend/src/components/layout/Topbar.tsx` — Add preset-detail title

---

## Chunk 1: Backend Data Model + Dependencies

### Task 1: Add bcrypt dependency

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1:** Add `bcrypt>=4.0` to dependencies in pyproject.toml

### Task 2: Extend VoicePreset model

**Files:**
- Modify: `backend/src/models/voice_preset.py`
- Modify: `backend/src/models/schemas.py`

- [ ] **Step 1:** Add `status` and `endpoint_path` columns to VoicePreset

```python
# New columns after existing ones:
status = Column(String(20), default="active", nullable=False)
endpoint_path = Column(String(200), nullable=True)
```

- [ ] **Step 2:** Add `status` and `endpoint_path` to VoicePresetOut schema

```python
class VoicePresetOut(BaseModel):
    # ... existing fields ...
    status: str = "active"
    endpoint_path: str | None = None
```

- [ ] **Step 3:** Commit

### Task 3: Create PresetApiKey model

**Files:**
- Create: `backend/src/models/preset_api_key.py`
- Modify: `backend/src/models/schemas.py`

- [ ] **Step 1:** Create PresetApiKey ORM model

```python
from datetime import datetime, timezone
from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, String
from src.models.database import Base
from src.utils.snowflake import snowflake_id

class PresetApiKey(Base):
    __tablename__ = "preset_api_keys"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    preset_id = Column(BigInteger, ForeignKey("voice_presets.id", ondelete="CASCADE"), nullable=False, index=True)
    label = Column(String(100), nullable=False)
    key_hash = Column(String(200), nullable=False)
    key_prefix = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    usage_calls = Column(Integer, default=0, nullable=False)
    usage_chars = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
```

- [ ] **Step 2:** Add Pydantic schemas for PresetApiKey

```python
class PresetApiKeyCreate(BaseModel):
    label: str

class PresetApiKeyOut(BaseModel):
    id: int
    preset_id: int
    label: str
    key_prefix: str
    is_active: bool
    usage_calls: int
    usage_chars: int
    last_used_at: datetime | None
    created_at: datetime
    model_config = {"from_attributes": True}

class PresetApiKeyCreated(PresetApiKeyOut):
    """Returned only on creation — includes the full key."""
    key: str

class PresetStatusUpdate(BaseModel):
    status: Literal["active", "inactive"]
```

- [ ] **Step 3:** Commit

---

## Chunk 2: Backend API — Key Management + Auth

### Task 4: Create auth dependency

**Files:**
- Create: `backend/src/api/deps_auth.py`

- [ ] **Step 1:** Implement `verify_preset_key` dependency

```python
import bcrypt
from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.database import get_async_session
from src.models.preset_api_key import PresetApiKey
from src.models.voice_preset import VoicePreset

async def verify_preset_key(
    preset_id: int,
    authorization: str = Header(...),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[VoicePreset, PresetApiKey]:
    # Extract Bearer token
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="Invalid authorization header")
    token = authorization[7:]

    # Check preset exists
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Preset not found")
    if preset.status != "active":
        raise HTTPException(403, detail="Preset is inactive")

    # Find matching key
    result = await session.execute(
        select(PresetApiKey).where(
            PresetApiKey.preset_id == preset_id,
            PresetApiKey.is_active == True,
        )
    )
    keys = result.scalars().all()
    for key in keys:
        if bcrypt.checkpw(token.encode(), key.key_hash.encode()):
            return preset, key

    raise HTTPException(401, detail="Invalid API key")
```

- [ ] **Step 2:** Commit

### Task 5: Create key management routes

**Files:**
- Create: `backend/src/api/routes/preset_keys.py`

- [ ] **Step 1:** Implement key CRUD endpoints

Key generation helper:
```python
import os, re, bcrypt
from src.utils.snowflake import snowflake_id

def _generate_key(preset_name: str) -> tuple[str, str, str]:
    """Returns (full_key, key_hash, key_prefix)."""
    # Derive 2-4 char prefix from preset name (pinyin initials or first chars)
    clean = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', preset_name)[:4].lower() or 'key'
    random_hex = os.urandom(16).hex()
    full_key = f"sk-{clean}-{random_hex}"
    key_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt()).decode()
    key_prefix = full_key[:10]
    return full_key, key_hash, key_prefix
```

Routes:
- `POST /api/v1/presets/{preset_id}/keys` — Create key
- `GET /api/v1/presets/{preset_id}/keys` — List keys
- `DELETE /api/v1/presets/{preset_id}/keys/{key_id}` — Delete key
- `PATCH /api/v1/presets/{preset_id}/status` — Toggle status

- [ ] **Step 2:** Commit

### Task 6: Create preset service endpoint

**Files:**
- Create: `backend/src/api/routes/preset_service.py`

- [ ] **Step 1:** Implement `/v1/preset/{preset_id}/synthesize`

This endpoint:
1. Uses `verify_preset_key` dependency for auth
2. Resolves preset engine + params
3. Calls existing TTS engine infrastructure
4. Updates usage counters (calls, chars)
5. Returns audio as base64 JSON (matching existing SynthesizeResponse)

```python
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps_auth import verify_preset_key
from src.models.database import get_async_session
from src.models.voice_preset import VoicePreset
from src.models.preset_api_key import PresetApiKey

router = APIRouter(prefix="/v1/preset", tags=["preset-service"])

class PresetSynthesizeRequest(BaseModel):
    text: str
    emotion: str | None = None

@router.post("/{preset_id}/synthesize")
async def preset_synthesize(
    req: PresetSynthesizeRequest,
    auth: tuple[VoicePreset, PresetApiKey] = Depends(verify_preset_key),
    session: AsyncSession = Depends(get_async_session),
):
    preset, api_key = auth
    # Resolve engine + params from preset
    # Synthesize using existing engine infrastructure
    # Update usage: api_key.usage_calls += 1, api_key.usage_chars += len(req.text)
    # Return SynthesizeResponse
```

- [ ] **Step 2:** Commit

### Task 7: Register new routers in main.py

**Files:**
- Modify: `backend/src/api/main.py`

- [ ] **Step 1:** Import and register `preset_keys` and `preset_service` routers

```python
from src.api.routes import preset_keys, preset_service
app.include_router(preset_keys.router)
app.include_router(preset_service.router)
```

- [ ] **Step 2:** Commit

---

## Chunk 3: Frontend — API Hooks + Store

### Task 8: Add preset key API hooks

**Files:**
- Create: `frontend/src/api/presetKeys.ts`

- [ ] **Step 1:** Create React Query hooks

```typescript
// usePresetKeys(presetId) — GET /api/v1/presets/{id}/keys
// useCreatePresetKey(presetId) — POST /api/v1/presets/{id}/keys
// useDeletePresetKey(presetId) — DELETE /api/v1/presets/{id}/keys/{keyId}
// useUpdatePresetStatus(presetId) — PATCH /api/v1/presets/{id}/status
```

- [ ] **Step 2:** Extend VoicePreset interface in `frontend/src/api/voices.ts` with `status` and `endpoint_path`

- [ ] **Step 3:** Commit

### Task 9: Update panel store

**Files:**
- Modify: `frontend/src/stores/panel.ts`

- [ ] **Step 1:** Add `'preset-detail'` to OverlayId union, add `selectedPresetId` state

```typescript
export type OverlayId = 'dashboard' | 'models' | 'settings' | 'preset-detail'

interface PanelState {
    // ... existing ...
    selectedPresetId: string | null
    openPresetDetail: (presetId: string) => void
}
```

- [ ] **Step 2:** Commit

---

## Chunk 4: Frontend — PresetDetailOverlay + Wiring

### Task 10: Create PresetDetailOverlay

**Files:**
- Create: `frontend/src/components/overlays/PresetDetailOverlay.tsx`

- [ ] **Step 1:** Build two-column overlay

Left column:
- Preset name + status badge (active/inactive toggle button)
- Configuration summary (engine, speed, sample_rate, voice from params)
- Endpoint path (copyable)
- cURL example (copyable)

Right column:
- API Keys list (label, prefix, usage_calls, usage_chars, last_used_at)
- "+ New Key" button → inline form with label input → shows full key once with copy
- "Revoke" button per key with confirmation

Follows existing overlay pattern: `className="absolute inset-0 overflow-y-auto z-[16]"` with `background: 'var(--bg)'`

- [ ] **Step 2:** Commit

### Task 11: Wire up PresetsPanel + Topbar + NodeEditor

**Files:**
- Modify: `frontend/src/components/panels/PresetsPanel.tsx`
- Modify: `frontend/src/components/nodes/NodeEditor.tsx`
- Modify: `frontend/src/components/layout/Topbar.tsx`

- [ ] **Step 1:** In PresetsPanel, add click handler to open preset-detail overlay

```tsx
onClick={() => {
    usePanelStore.getState().openPresetDetail(p.id)
}}
```

- [ ] **Step 2:** In NodeEditor, add PresetDetailOverlay rendering

```tsx
{activeOverlay === 'preset-detail' && <PresetDetailOverlay />}
```

- [ ] **Step 3:** In Topbar, add preset-detail title mapping

```tsx
activeOverlay === 'preset-detail' ? 'Preset 详情' : ...
```

- [ ] **Step 4:** Commit
