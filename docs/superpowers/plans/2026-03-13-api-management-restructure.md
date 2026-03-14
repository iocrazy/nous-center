# API Management Restructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure instance management from preset-centric navigation to a centralized API management overlay with master-detail layout, eliminate Collection concept, and support polymorphic instance sources (preset or workflow).

**Architecture:** Replace `preset_id` FK on ServiceInstance with `source_type` + `source_id` polymorphic pattern. Frontend replaces InstanceDetailOverlay + CollectionsPanel + ApiNodesPanel with a single ApiManagementOverlay (three-column: IconRail + instance list + detail). PresetDetailOverlay simplified to show config + deploy button only.

**Tech Stack:** FastAPI + SQLAlchemy + Pydantic (backend), React + TypeScript + Zustand + TanStack Query (frontend)

**Spec:** `docs/superpowers/specs/2026-03-13-api-management-restructure-design.md`

---

## File Structure

### Backend — Modify

| File | Responsibility |
|------|---------------|
| `backend/src/models/service_instance.py` | Replace `preset_id` FK with `source_type` + `source_id` columns |
| `backend/src/models/schemas.py` | Update ServiceInstanceCreate/Out for new fields |
| `backend/src/api/routes/instances.py` | Update create/list endpoints for source_type pattern, add `?type=` filter |
| `backend/src/api/routes/instance_service.py` | Resolve preset via source_type + source_id instead of preset_id |

### Backend — Test

| File | Responsibility |
|------|---------------|
| `backend/tests/test_instance_keys.py` | Update existing tests to use new `source_type`/`source_id` fields |
| `backend/tests/test_schemas.py` | Add schema validation tests for new fields |

### Frontend — Delete

| File | Reason |
|------|--------|
| `frontend/src/components/panels/CollectionsPanel.tsx` | Collection concept removed |
| `frontend/src/components/panels/ApiNodesPanel.tsx` | Replaced by API Management overlay |
| `frontend/src/components/overlays/InstanceDetailOverlay.tsx` | Content migrated into ApiManagementOverlay |

### Frontend — Create

| File | Responsibility |
|------|---------------|
| `frontend/src/components/overlays/ApiManagementOverlay.tsx` | Three-column overlay: instance list sidebar + detail view |
| `frontend/src/components/overlays/CreateInstanceForm.tsx` | New instance creation form with source_type/source_id |

### Frontend — Modify

| File | Responsibility |
|------|---------------|
| `frontend/src/stores/panel.ts` | Remove collections/api PanelId, replace instance-detail with api-management OverlayId |
| `frontend/src/api/instances.ts` | Update interface + hooks for new schema fields |
| `frontend/src/components/layout/IconRail.tsx` | Remove Collections icon, change API icon to open overlay |
| `frontend/src/components/nodes/NodeEditor.tsx` | Replace deleted panels/overlays with ApiManagementOverlay |
| `frontend/src/components/layout/Topbar.tsx` | Update overlay title + back navigation for api-management |
| `frontend/src/components/overlays/PresetDetailOverlay.tsx` | Remove Instance list, add deploy button |

---

## Chunk 1: Backend Data Model & API Changes

> **Migration note:** This project uses SQLite for development. Existing `service_instances` rows (if any) have a `preset_id` column that must become `source_type='preset'` + `source_id=<old preset_id>`. For development, the simplest approach is to drop and recreate the database. For production, write an Alembic migration: add `source_type`/`source_id` columns, copy `preset_id` → `source_id`, set `source_type='preset'`, then drop `preset_id`. The migration script is out of scope for this plan (no production DB yet).

### Task 1: Update ServiceInstance model (source_type + source_id)

**Files:**
- Modify: `backend/src/models/service_instance.py`

- [ ] **Step 1: Write the failing test**

Add test to `backend/tests/test_instance_keys.py` that creates an instance with `source_type` + `source_id` instead of `preset_id`:

```python
async def test_create_instance_with_source_type(db_client):
    """Instance creation uses source_type + source_id instead of preset_id."""
    resp = await db_client.post("/api/v1/voices", json={
        "name": "source-type-test",
        "engine": "cosyvoice2",
        "params": {"voice": "default"},
        "tags": [],
    })
    assert resp.status_code == 201
    preset_id = resp.json()["id"]

    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": preset_id,
        "name": "new-style-instance",
        "type": "tts",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["source_type"] == "preset"
    assert data["source_id"] == preset_id
    assert data["name"] == "new-style-instance"
    assert "preset_id" not in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_instance_keys.py::test_create_instance_with_source_type -v`
Expected: FAIL — endpoint still expects `preset_id`

- [ ] **Step 3: Update ServiceInstance model**

Replace `preset_id` FK with `source_type` + `source_id` in `backend/src/models/service_instance.py`:

```python
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, JSON, String, Index

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class ServiceInstance(Base):
    __tablename__ = "service_instances"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    source_type = Column(String(20), nullable=False, default="preset")  # "preset" or "workflow"
    source_id = Column(BigInteger, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    type = Column(String(20), default="tts", nullable=False)  # tts, image, inference
    status = Column(String(20), default="active", nullable=False)
    endpoint_path = Column(String(200), nullable=True)
    params_override = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_service_instances_source", "source_type", "source_id"),
    )
```

- [ ] **Step 4: Update schemas**

Update `backend/src/models/schemas.py` — replace `preset_id` in `ServiceInstanceCreate` and `ServiceInstanceOut`:

```python
# Replace ServiceInstanceCreate (lines 230-239)
class ServiceInstanceCreate(BaseModel):
    source_type: Literal["preset"] = "preset"
    source_id: int
    name: str
    type: str = "tts"
    params_override: dict = {}

    @field_validator("source_id", mode="before")
    @classmethod
    def coerce_source_id(cls, v: int | str) -> int:
        return int(v)


# Replace ServiceInstanceOut (lines 247-262)
class ServiceInstanceOut(BaseModel):
    id: int
    source_type: str
    source_id: int
    source_name: str | None = None
    name: str
    type: str
    status: str
    endpoint_path: str | None
    params_override: dict
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("id", "source_id")
    def serialize_ids(self, v: int) -> str:
        return str(v)
```

- [ ] **Step 5: Update create/list routes**

Update `backend/src/api/routes/instances.py`:

```python
"""Service instance CRUD routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.service_instance import ServiceInstance
from src.models.voice_preset import VoicePreset
from src.models.schemas import (
    ServiceInstanceCreate,
    ServiceInstanceOut,
    ServiceInstanceUpdate,
    InstanceStatusUpdate,
)

router = APIRouter(prefix="/api/v1/instances", tags=["instances"])


async def _resolve_source_name(session: AsyncSession, source_type: str, source_id: int) -> str | None:
    """Resolve a human-readable name for the instance source."""
    if source_type == "preset":
        preset = await session.get(VoicePreset, source_id)
        return preset.name if preset else None
    return None


def _instance_to_out(instance: ServiceInstance, source_name: str | None = None) -> dict:
    """Convert instance + resolved source_name to dict for ServiceInstanceOut."""
    return {
        "id": instance.id,
        "source_type": instance.source_type,
        "source_id": instance.source_id,
        "source_name": source_name,
        "name": instance.name,
        "type": instance.type,
        "status": instance.status,
        "endpoint_path": instance.endpoint_path,
        "params_override": instance.params_override,
        "created_at": instance.created_at,
        "updated_at": instance.updated_at,
    }


@router.post("", response_model=ServiceInstanceOut, status_code=201)
async def create_instance(
    data: ServiceInstanceCreate,
    session: AsyncSession = Depends(get_async_session),
):
    # Validate source exists
    if data.source_type == "preset":
        preset = await session.get(VoicePreset, data.source_id)
        if not preset:
            raise HTTPException(404, detail="Source preset not found")
        source_name = preset.name
    else:
        raise HTTPException(400, detail=f"Unsupported source_type: {data.source_type}")

    instance = ServiceInstance(
        source_type=data.source_type,
        source_id=data.source_id,
        name=data.name,
        type=data.type,
        params_override=data.params_override,
    )
    session.add(instance)
    await session.commit()
    await session.refresh(instance)

    # Auto-set endpoint_path
    instance.endpoint_path = f"/v1/instances/{instance.id}/synthesize"
    await session.commit()
    await session.refresh(instance)

    return _instance_to_out(instance, source_name)


@router.get("", response_model=list[ServiceInstanceOut])
async def list_instances(
    type: str | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    query = select(ServiceInstance).order_by(ServiceInstance.created_at.desc())
    if type is not None:
        query = query.where(ServiceInstance.type == type)
    result = await session.execute(query)
    instances = result.scalars().all()

    out = []
    for inst in instances:
        source_name = await _resolve_source_name(session, inst.source_type, inst.source_id)
        out.append(_instance_to_out(inst, source_name))
    return out


@router.get("/{instance_id}", response_model=ServiceInstanceOut)
async def get_instance(
    instance_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")
    source_name = await _resolve_source_name(session, instance.source_type, instance.source_id)
    return _instance_to_out(instance, source_name)


@router.patch("/{instance_id}", response_model=ServiceInstanceOut)
async def update_instance(
    instance_id: int,
    data: ServiceInstanceUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")

    if data.name is not None:
        instance.name = data.name
    if data.params_override is not None:
        instance.params_override = data.params_override

    await session.commit()
    await session.refresh(instance)
    source_name = await _resolve_source_name(session, instance.source_type, instance.source_id)
    return _instance_to_out(instance, source_name)


@router.patch("/{instance_id}/status")
async def update_instance_status(
    instance_id: int,
    data: InstanceStatusUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")
    instance.status = data.status
    await session.commit()
    await session.refresh(instance)
    return {"id": instance.id, "status": instance.status}


@router.delete("/{instance_id}", status_code=204)
async def delete_instance(
    instance_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")
    await session.delete(instance)
    await session.commit()
```

- [ ] **Step 6: Update instance_service.py**

Update `backend/src/api/routes/instance_service.py` line 34 — replace `instance.preset_id` with source resolution:

```python
    # Load preset to get engine config (resolve from source_type)
    if instance.source_type != "preset":
        raise HTTPException(501, detail="Only preset-based instances support synthesis currently")
    preset = await session.get(VoicePreset, instance.source_id)
    if not preset:
        raise HTTPException(500, detail="Linked preset not found")
```

- [ ] **Step 7: Update all existing tests**

Update `backend/tests/test_instance_keys.py` — replace `_create_preset_and_instance` helper and all assertions that reference `preset_id`:

```python
async def _create_preset_and_instance(db_client):
    """Helper: create a preset then an instance based on it."""
    resp = await db_client.post("/api/v1/voices", json={
        "name": "testvoice",
        "engine": "cosyvoice2",
        "params": {"voice": "default", "speed": 1.0},
        "tags": [],
    })
    assert resp.status_code == 201
    preset_id = resp.json()["id"]

    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": preset_id,
        "name": "test-instance",
    })
    assert resp.status_code == 201
    instance = resp.json()
    return preset_id, instance
```

Update `test_create_instance`:
```python
async def test_create_instance(db_client):
    preset_id, instance = await _create_preset_and_instance(db_client)
    assert instance["source_type"] == "preset"
    assert instance["source_id"] == preset_id
    assert instance["source_name"] == "testvoice"
    assert instance["name"] == "test-instance"
    assert instance["type"] == "tts"
    assert instance["status"] == "active"
    assert instance["endpoint_path"] == f"/v1/instances/{instance['id']}/synthesize"
    assert instance["params_override"] == {}
```

Update `test_list_instances_by_preset` → `test_list_instances_by_type`:
```python
async def test_list_instances_by_type(db_client):
    preset_id, _ = await _create_preset_and_instance(db_client)

    await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": preset_id,
        "name": "second-instance",
    })

    # Filter by type
    resp = await db_client.get("/api/v1/instances?type=tts")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    # No image instances
    resp = await db_client.get("/api/v1/instances?type=image")
    assert resp.status_code == 200
    assert len(resp.json()) == 0
```

Update `test_instance_params_override`:
```python
async def test_instance_params_override(db_client):
    resp = await db_client.post("/api/v1/voices", json={
        "name": "override-test",
        "engine": "cosyvoice2",
        "params": {"voice": "default", "speed": 1.0},
        "tags": [],
    })
    preset_id = resp.json()["id"]

    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": preset_id,
        "name": "fast-instance",
        "params_override": {"speed": 1.5},
    })
    assert resp.status_code == 201
    assert resp.json()["params_override"] == {"speed": 1.5}

    instance_id = resp.json()["id"]
    resp = await db_client.patch(f"/api/v1/instances/{instance_id}", json={
        "params_override": {"speed": 2.0, "voice": "narrator"},
    })
    assert resp.status_code == 200
    assert resp.json()["params_override"] == {"speed": 2.0, "voice": "narrator"}
```

- [ ] **Step 8: Add new edge-case tests**

Add to `backend/tests/test_instance_keys.py`:

```python
async def test_create_instance_invalid_source_type(db_client):
    """source_type='workflow' should be rejected (not yet supported)."""
    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "workflow",
        "source_id": 12345,
        "name": "workflow-instance",
    })
    assert resp.status_code == 422  # Literal["preset"] validation


async def test_create_instance_nonexistent_source(db_client):
    """Non-existent source_id should 404."""
    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": 9999999999999,
        "name": "ghost-instance",
    })
    assert resp.status_code == 404


async def test_list_all_instances(db_client):
    """List without type filter returns all instances."""
    preset_id, _ = await _create_preset_and_instance(db_client)
    resp = await db_client.get("/api/v1/instances")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
```

- [ ] **Step 9: Run all tests to verify**

Run: `cd backend && python -m pytest tests/test_instance_keys.py -v`
Expected: ALL PASS

- [ ] **Step 10: Also run schema tests**

Run: `cd backend && python -m pytest tests/test_schemas.py -v`
Expected: ALL PASS (schemas.py changes should not break other schema tests)

- [ ] **Step 11: Commit backend changes**

```bash
git add backend/src/models/service_instance.py backend/src/models/schemas.py backend/src/api/routes/instances.py backend/src/api/routes/instance_service.py backend/tests/test_instance_keys.py
git commit -m "refactor: replace preset_id with source_type+source_id on ServiceInstance"
```

---

## Chunk 2: Frontend Store, API Hooks & Navigation Changes

### Task 2: Update panel store types and actions

**Files:**
- Modify: `frontend/src/stores/panel.ts`

- [ ] **Step 1: Update PanelId and OverlayId types**

```typescript
export type PanelId = 'nodes' | 'workflows' | 'presets'
export type OverlayId = 'dashboard' | 'models' | 'settings' | 'preset-detail' | 'api-management'
```

- [ ] **Step 2: Remove selectedInstanceId, openInstanceDetail; add openApiManagement**

Replace the full store with:

```typescript
import { create } from 'zustand'

export type PanelId = 'nodes' | 'workflows' | 'presets'
export type OverlayId = 'dashboard' | 'models' | 'settings' | 'preset-detail' | 'api-management'

export interface ApiManagementOptions {
  source_type?: 'preset' | 'workflow'
  source_id?: string
}

interface PanelState {
  activePanel: PanelId | null
  activeOverlay: OverlayId | null
  selectedPresetId: string | null
  apiManagementOptions: ApiManagementOptions | null
  panelWidth: number
  setPanel: (id: PanelId | null) => void
  togglePanel: (id: PanelId) => void
  setOverlay: (id: OverlayId | null) => void
  toggleOverlay: (id: OverlayId) => void
  openPresetDetail: (presetId: string) => void
  openApiManagement: (options?: ApiManagementOptions) => void
  setPanelWidth: (width: number) => void
}

export const usePanelStore = create<PanelState>((set, get) => ({
  activePanel: 'nodes',
  activeOverlay: null,
  selectedPresetId: null,
  apiManagementOptions: null,
  panelWidth: 260,

  setPanel: (id) => set({ activePanel: id, activeOverlay: null }),

  togglePanel: (id) => {
    const { activePanel } = get()
    set({
      activePanel: activePanel === id ? null : id,
      activeOverlay: null,
    })
  },

  setOverlay: (id) => set({ activeOverlay: id, activePanel: null }),

  toggleOverlay: (id) => {
    const { activeOverlay } = get()
    set({
      activeOverlay: activeOverlay === id ? null : id,
      activePanel: null,
    })
  },

  openPresetDetail: (presetId) =>
    set({ activeOverlay: 'preset-detail', activePanel: null, selectedPresetId: presetId }),

  openApiManagement: (options) =>
    set({ activeOverlay: 'api-management', activePanel: null, apiManagementOptions: options ?? null }),

  setPanelWidth: (width) => set({ panelWidth: Math.max(200, Math.min(400, width)) }),
}))
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/stores/panel.ts
git commit -m "refactor: update panel store — remove collections/api panel, add api-management overlay"
```

### Task 3: Update frontend API hooks

**Files:**
- Modify: `frontend/src/api/instances.ts`

- [ ] **Step 1: Update ServiceInstance interface**

Replace `preset_id` with `source_type`, `source_id`, `source_name`:

```typescript
export interface ServiceInstance {
  id: string
  source_type: string
  source_id: string
  source_name: string | null
  name: string
  type: string
  status: string
  endpoint_path: string | null
  params_override: Record<string, unknown>
  created_at: string
  updated_at: string
}
```

- [ ] **Step 2: Update useInstances hook**

Replace `useInstances(presetId)` with `useInstances(type?)`:

```typescript
export function useInstances(type?: string) {
  return useQuery({
    queryKey: ['instances', type ?? 'all'],
    queryFn: () =>
      apiFetch<ServiceInstance[]>(
        type
          ? `/api/v1/instances?type=${type}`
          : '/api/v1/instances',
      ),
    refetchOnWindowFocus: false,
    retry: false,
  })
}
```

- [ ] **Step 3: Update useCreateInstance hook**

```typescript
export function useCreateInstance() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: {
      source_type: string
      source_id: string
      name: string
      type?: string
      params_override?: Record<string, unknown>
    }) =>
      apiFetch<ServiceInstance>('/api/v1/instances', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['instances'] })
    },
  })
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api/instances.ts
git commit -m "refactor: update instance API hooks for source_type+source_id"
```

### Task 4: Update IconRail navigation

**Files:**
- Modify: `frontend/src/components/layout/IconRail.tsx`

- [ ] **Step 1: Remove collections and api from PANEL_ITEMS, add API to OVERLAY_ITEMS**

```typescript
const PANEL_ITEMS: { id: PanelId; icon: typeof CircuitBoard; label: string }[] = [
  { id: 'nodes', icon: CircuitBoard, label: 'Nodes' },
  { id: 'workflows', icon: GitBranch, label: 'Workflows' },
  { id: 'presets', icon: Settings, label: 'Presets' },
]

const OVERLAY_ITEMS: { id: OverlayId; icon: typeof LayoutDashboard; label: string }[] = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { id: 'models', icon: Layers, label: 'Models' },
  { id: 'api-management', icon: Link, label: 'API' },
]
```

Remove unused imports: `Package` (was for collections).

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/layout/IconRail.tsx
git commit -m "refactor: IconRail — remove Collections, move API to overlay"
```

### Task 5: Update Topbar overlay title and back navigation

**Files:**
- Modify: `frontend/src/components/layout/Topbar.tsx`

- [ ] **Step 1: Update overlay title mapping and back button logic**

Replace the overlayTitle line (line 19):
```typescript
const overlayTitle = activeOverlay === 'dashboard' ? 'Dashboard' : activeOverlay === 'models' ? 'Models' : activeOverlay === 'settings' ? '设置' : activeOverlay === 'preset-detail' ? '预设详情' : activeOverlay === 'api-management' ? 'API 管理' : null
```

Update back button onClick (lines 74-79):
```typescript
onClick={() => setOverlay(null)}
```
(Remove the special `instance-detail → preset-detail` navigation since instance-detail no longer exists)

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/layout/Topbar.tsx
git commit -m "refactor: Topbar — update overlay titles for api-management"
```

---

## Chunk 3: ApiManagementOverlay, CreateInstanceForm & NodeEditor

### Task 7: Create ApiManagementOverlay

**Files:**
- Create: `frontend/src/components/overlays/ApiManagementOverlay.tsx`

- [ ] **Step 1: Create the component**

This component implements the three-column master-detail layout. It contains:
- Left sidebar (240px): Instance list grouped by type (tts/image/inference), with search and "新建实例" button
- Right area (flex): Instance detail (migrated from InstanceDetailOverlay) or CreateInstanceForm
- Local state: `selectedInstanceId`, `showCreateForm`

```typescript
import { useState, useMemo } from 'react'
import { Search, Plus, Key, Activity, Copy, Check, ChevronDown, ChevronUp, Trash2 } from 'lucide-react'
import { usePanelStore, type ApiManagementOptions } from '../../stores/panel'
import { useSettingsStore } from '../../stores/settings'
import {
  useInstances,
  useInstance,
  useInstanceKeys,
  useCreateInstanceKey,
  useDeleteInstanceKey,
  useUpdateInstanceStatus,
  type ServiceInstance,
  type InstanceApiKeyCreated,
} from '../../api/instances'
import CreateInstanceForm from './CreateInstanceForm'

const TYPE_GROUPS = [
  { type: 'tts', label: '语音合成 TTS' },
  { type: 'image', label: '图像生成 Image' },
  { type: 'inference', label: '文本推理 LLM' },
] as const

export default function ApiManagementOverlay() {
  const apiManagementOptions = usePanelStore((s) => s.apiManagementOptions)
  const { data: allInstances, isLoading } = useInstances()

  const [selectedInstanceId, setSelectedInstanceId] = useState<string | null>(null)
  const [showCreateForm, setShowCreateForm] = useState(!!apiManagementOptions?.source_type)
  const [searchQuery, setSearchQuery] = useState('')
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())

  const filteredInstances = useMemo(() => {
    if (!allInstances) return []
    if (!searchQuery.trim()) return allInstances
    const q = searchQuery.toLowerCase()
    return allInstances.filter(
      (i) => i.name.toLowerCase().includes(q) || i.source_name?.toLowerCase().includes(q),
    )
  }, [allInstances, searchQuery])

  const groupedInstances = useMemo(() => {
    const groups: Record<string, ServiceInstance[]> = {}
    for (const g of TYPE_GROUPS) groups[g.type] = []
    for (const inst of filteredInstances) {
      if (groups[inst.type]) groups[inst.type].push(inst)
      else groups[inst.type] = [inst]
    }
    return groups
  }, [filteredInstances])

  const toggleGroup = (type: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(type)) next.delete(type)
      else next.add(type)
      return next
    })
  }

  const handleCreateClick = () => {
    setShowCreateForm(true)
    setSelectedInstanceId(null)
  }

  const handleInstanceCreated = (instanceId: string) => {
    setShowCreateForm(false)
    setSelectedInstanceId(instanceId)
  }

  const handleSelectInstance = (id: string) => {
    setSelectedInstanceId(id)
    setShowCreateForm(false)
  }

  return (
    <div
      className="absolute inset-0 z-[16] flex"
      style={{ background: 'var(--bg)' }}
    >
      {/* Instance List Sidebar */}
      <div
        className="flex flex-col shrink-0"
        style={{
          width: 240,
          borderRight: '1px solid var(--border)',
          background: 'var(--bg-accent)',
          overflow: 'hidden',
        }}
      >
        {/* Search */}
        <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)' }}>
          <div
            className="flex items-center gap-1.5"
            style={{
              background: 'var(--card)',
              border: '1px solid var(--border)',
              borderRadius: 5,
              padding: '5px 8px',
            }}
          >
            <Search size={12} style={{ color: 'var(--muted)', flexShrink: 0 }} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="搜索实例..."
              style={{
                flex: 1,
                background: 'none',
                border: 'none',
                outline: 'none',
                fontSize: 11,
                color: 'var(--text-strong)',
              }}
            />
          </div>
        </div>

        {/* Grouped instance list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
          {isLoading && (
            <div style={{ padding: 12, fontSize: 11, color: 'var(--muted)' }}>加载中...</div>
          )}
          {TYPE_GROUPS.map(({ type, label }) => {
            const instances = groupedInstances[type] ?? []
            const collapsed = collapsedGroups.has(type)
            return (
              <div key={type} style={{ padding: '4px 12px' }}>
                <div
                  onClick={() => toggleGroup(type)}
                  className="flex items-center justify-between"
                  style={{
                    cursor: 'pointer',
                    marginBottom: 4,
                    userSelect: 'none',
                  }}
                >
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 600,
                      color: 'var(--muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                    }}
                  >
                    {collapsed ? '▸' : '▾'} {label}
                  </span>
                  <span
                    style={{
                      fontSize: 9,
                      color: 'var(--muted)',
                      background: 'var(--bg)',
                      padding: '1px 6px',
                      borderRadius: 8,
                    }}
                  >
                    {instances.length}
                  </span>
                </div>
                {!collapsed &&
                  instances.map((inst) => (
                    <InstanceCard
                      key={inst.id}
                      instance={inst}
                      selected={inst.id === selectedInstanceId}
                      onClick={() => handleSelectInstance(inst.id)}
                    />
                  ))}
              </div>
            )
          })}
        </div>

        {/* New instance button */}
        <div style={{ padding: '8px 12px', borderTop: '1px solid var(--border)' }}>
          <button
            onClick={handleCreateClick}
            style={{
              width: '100%',
              border: '1px dashed var(--border)',
              borderRadius: 5,
              padding: 8,
              textAlign: 'center',
              fontSize: 10,
              color: 'var(--muted)',
              background: 'none',
              cursor: 'pointer',
            }}
          >
            + 新建实例
          </button>
        </div>
      </div>

      {/* Detail Area */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {showCreateForm ? (
          <CreateInstanceForm
            defaultOptions={apiManagementOptions ?? undefined}
            onCreated={handleInstanceCreated}
            onCancel={() => setShowCreateForm(false)}
          />
        ) : selectedInstanceId ? (
          <InstanceDetail instanceId={selectedInstanceId} />
        ) : (
          <div
            className="flex items-center justify-center"
            style={{ height: '100%', color: 'var(--muted)', fontSize: 12 }}
          >
            选择一个实例查看详情，或创建新实例
          </div>
        )}
      </div>
    </div>
  )
}

/* --- Instance Card (sidebar) --- */

function InstanceCard({
  instance,
  selected,
  onClick,
}: {
  instance: ServiceInstance
  selected: boolean
  onClick: () => void
}) {
  const isInactive = instance.status !== 'active'

  return (
    <div
      onClick={onClick}
      style={{
        background: selected ? 'var(--accent-subtle)' : 'var(--card)',
        border: `1px solid ${selected ? 'var(--accent)' : 'var(--border)'}`,
        borderRadius: 5,
        padding: '7px 10px',
        marginBottom: 4,
        cursor: 'pointer',
        opacity: isInactive ? 0.45 : 1,
        transition: 'all 0.12s',
      }}
    >
      <div className="flex items-center gap-1.5" style={{ marginBottom: 2 }}>
        <div
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: instance.status === 'active' ? '#4ade80' : '#f87171',
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-strong)' }}>
          {instance.name}
        </span>
      </div>
      <div style={{ fontSize: 9, color: 'var(--muted)', paddingLeft: 12 }}>
        <SourceTag sourceType={instance.source_type} />
        {instance.source_name && (
          <span style={{ marginLeft: 4 }}>{instance.source_name}</span>
        )}
      </div>
    </div>
  )
}

function SourceTag({ sourceType }: { sourceType: string }) {
  const isWorkflow = sourceType === 'workflow'
  return (
    <span
      style={{
        fontSize: 8,
        padding: '1px 5px',
        borderRadius: 3,
        background: isWorkflow ? 'rgba(139,92,246,0.15)' : 'rgba(34,197,94,0.12)',
        color: isWorkflow ? '#a78bfa' : '#4ade80',
      }}
    >
      {sourceType}
    </span>
  )
}

/* --- Instance Detail (right panel, migrated from InstanceDetailOverlay) --- */

function InstanceDetail({ instanceId }: { instanceId: string }) {
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl)
  const { data: instance } = useInstance(instanceId)
  const { data: keys, isLoading: keysLoading } = useInstanceKeys(instanceId)
  const createKey = useCreateInstanceKey(instanceId)
  const deleteKey = useDeleteInstanceKey(instanceId)
  const updateStatus = useUpdateInstanceStatus(instanceId)

  const [newKeyLabel, setNewKeyLabel] = useState('')
  const [showNewKeyForm, setShowNewKeyForm] = useState(false)
  const [createdKey, setCreatedKey] = useState<InstanceApiKeyCreated | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [curlExpanded, setCurlExpanded] = useState(false)

  if (!instance) {
    return (
      <div style={{ padding: 16, fontSize: 11, color: 'var(--muted)' }}>加载中...</div>
    )
  }

  const endpointPath = instance.endpoint_path || `/v1/instances/${instance.id}/synthesize`

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text)
    setCopied(label)
    setTimeout(() => setCopied(null), 2000)
  }

  const handleCreateKey = async () => {
    if (!newKeyLabel.trim()) return
    const result = await createKey.mutateAsync(newKeyLabel.trim())
    setCreatedKey(result)
    setNewKeyLabel('')
    setShowNewKeyForm(false)
  }

  const handleDeleteKey = async (keyId: string) => {
    await deleteKey.mutateAsync(keyId)
    setConfirmDeleteId(null)
  }

  const handleToggleStatus = () => {
    updateStatus.mutate(instance.status === 'active' ? 'inactive' : 'active')
  }

  const curlExample = `curl -X POST ${apiBaseUrl}${endpointPath} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <your-api-key>" \\
  -d '{"text": "你好世界"}'`

  const totalCalls = keys?.reduce((s, k) => s + k.usage_calls, 0) ?? 0
  const totalChars = keys?.reduce((s, k) => s + k.usage_chars, 0) ?? 0

  return (
    <div style={{ padding: '16px 20px', maxWidth: 960 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-strong)', margin: 0 }}>
          {instance.name}
        </h2>
        <button
          onClick={handleToggleStatus}
          disabled={updateStatus.isPending}
          style={{
            fontSize: 10,
            padding: '2px 10px',
            borderRadius: 10,
            border: 'none',
            cursor: 'pointer',
            background: instance.status === 'active' ? 'rgba(34,197,94,0.15)' : 'rgba(248,113,113,0.15)',
            color: instance.status === 'active' ? '#4ade80' : '#f87171',
            fontWeight: 500,
          }}
        >
          {instance.status}
        </button>
      </div>
      <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 16 }}>
        <SourceTag sourceType={instance.source_type} />
        <span style={{ marginLeft: 6 }}>
          {instance.source_name && `${instance.source_name} · `}
          {instance.type} · 服务实例
        </span>
      </div>

      {/* Stats Row */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
        <StatCard icon={<Key size={12} />} label="API Keys" value={String(keys?.length ?? 0)} />
        <StatCard icon={<Activity size={12} />} label="总调用" value={totalCalls.toLocaleString()} />
        <StatCard label="总字符" value={totalChars.toLocaleString()} />
      </div>

      <div style={{ display: 'flex', gap: 20 }}>
        {/* Left Column — Endpoint & Config */}
        <div style={{ width: 360, flexShrink: 0 }}>
          <SectionTitle>Endpoint</SectionTitle>
          <div
            onClick={() => copyToClipboard(`POST ${apiBaseUrl}${endpointPath}`, 'endpoint')}
            className="flex items-center gap-2"
            style={{
              fontSize: 11,
              fontFamily: 'var(--mono)',
              background: 'var(--card)',
              padding: '8px 10px',
              borderRadius: 6,
              marginBottom: 8,
              cursor: 'pointer',
              color: 'var(--text-strong)',
              border: '1px solid var(--border)',
            }}
          >
            <span
              style={{
                fontSize: 9,
                fontWeight: 700,
                padding: '1px 5px',
                borderRadius: 3,
                background: 'rgba(34,197,94,0.15)',
                color: 'var(--ok)',
                flexShrink: 0,
              }}
            >
              POST
            </span>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
              {endpointPath}
            </span>
            {copied === 'endpoint' ? (
              <Check size={12} style={{ color: 'var(--ok)', flexShrink: 0 }} />
            ) : (
              <Copy size={12} style={{ color: 'var(--muted)', flexShrink: 0 }} />
            )}
          </div>

          {/* cURL */}
          <div
            style={{
              background: 'var(--card)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              marginBottom: 16,
              overflow: 'hidden',
            }}
          >
            <div
              onClick={() => setCurlExpanded(!curlExpanded)}
              className="flex items-center justify-between"
              style={{ padding: '6px 10px', cursor: 'pointer', fontSize: 10, color: 'var(--muted)', fontWeight: 500 }}
            >
              <span>cURL 示例</span>
              {curlExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </div>
            {curlExpanded && (
              <pre
                onClick={() => copyToClipboard(curlExample, 'curl')}
                style={{
                  fontSize: 10,
                  fontFamily: 'var(--mono)',
                  padding: '4px 10px 8px',
                  color: 'var(--text-strong)',
                  whiteSpace: 'pre-wrap',
                  lineHeight: 1.6,
                  cursor: 'pointer',
                  margin: 0,
                  borderTop: '1px solid var(--border)',
                }}
              >
                {curlExample}
                <span style={{ fontSize: 9, color: 'var(--muted)', display: 'block', marginTop: 4 }}>
                  {copied === 'curl' ? '✓ 已复制' : '点击复制'}
                </span>
              </pre>
            )}
          </div>

          {/* Params Override */}
          {Object.keys(instance.params_override).length > 0 && (
            <>
              <SectionTitle>参数覆盖</SectionTitle>
              <div
                style={{
                  background: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  padding: '6px 0',
                }}
              >
                {Object.entries(instance.params_override).map(([k, v]) => (
                  <div
                    key={k}
                    style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 10px', fontSize: 11 }}
                  >
                    <span style={{ color: 'var(--muted)' }}>{k}</span>
                    <span style={{ color: 'var(--text-strong)', fontFamily: 'var(--mono)' }}>{String(v)}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* Right Column — API Keys */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <SectionTitle style={{ marginBottom: 0 }}>API Keys ({keys?.length ?? 0})</SectionTitle>
            <button
              onClick={() => { setShowNewKeyForm(true); setCreatedKey(null) }}
              className="flex items-center gap-1"
              style={{
                fontSize: 10,
                padding: '4px 10px',
                background: 'var(--accent)',
                color: '#fff',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
              }}
            >
              <Plus size={10} /> New Key
            </button>
          </div>

          {/* Created Key Banner */}
          {createdKey && (
            <div
              style={{
                background: 'rgba(34,197,94,0.08)',
                border: '1px solid rgba(34,197,94,0.3)',
                borderRadius: 6,
                padding: 10,
                marginBottom: 10,
              }}
            >
              <div style={{ fontSize: 10, color: '#4ade80', fontWeight: 600, marginBottom: 6 }}>
                请立即保存此 Key，关闭后无法再次查看
              </div>
              <div
                onClick={() => copyToClipboard(createdKey.key, 'created-key')}
                className="flex items-center gap-2"
                style={{
                  fontSize: 11,
                  fontFamily: 'var(--mono)',
                  background: 'var(--bg)',
                  padding: '6px 8px',
                  borderRadius: 4,
                  cursor: 'pointer',
                  color: '#4ade80',
                  wordBreak: 'break-all',
                }}
              >
                <span style={{ flex: 1 }}>{createdKey.key}</span>
                {copied === 'created-key' ? (
                  <Check size={12} style={{ flexShrink: 0 }} />
                ) : (
                  <Copy size={12} style={{ color: 'var(--muted)', flexShrink: 0 }} />
                )}
              </div>
            </div>
          )}

          {/* New Key Form */}
          {showNewKeyForm && (
            <div
              style={{
                background: 'var(--card)',
                border: '1px solid var(--accent)',
                borderRadius: 6,
                padding: 10,
                marginBottom: 10,
              }}
            >
              <label style={{ fontSize: 10, color: 'var(--muted)', display: 'block', marginBottom: 4 }}>
                Key 标签
              </label>
              <div style={{ display: 'flex', gap: 6 }}>
                <input
                  type="text"
                  value={newKeyLabel}
                  onChange={(e) => setNewKeyLabel(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCreateKey()}
                  placeholder="如：有声小说App"
                  autoFocus
                  style={{
                    flex: 1,
                    padding: '5px 8px',
                    fontSize: 11,
                    background: 'var(--bg)',
                    border: '1px solid var(--border)',
                    borderRadius: 4,
                    color: 'var(--text-strong)',
                    outline: 'none',
                  }}
                />
                <button
                  onClick={handleCreateKey}
                  disabled={createKey.isPending || !newKeyLabel.trim()}
                  style={{
                    padding: '5px 14px',
                    fontSize: 10,
                    borderRadius: 4,
                    border: 'none',
                    background: 'var(--accent)',
                    color: '#fff',
                    cursor: 'pointer',
                    opacity: createKey.isPending || !newKeyLabel.trim() ? 0.5 : 1,
                  }}
                >
                  {createKey.isPending ? '...' : '创建'}
                </button>
                <button
                  onClick={() => { setShowNewKeyForm(false); setNewKeyLabel('') }}
                  style={{
                    padding: '5px 8px',
                    fontSize: 10,
                    borderRadius: 4,
                    border: '1px solid var(--border)',
                    background: 'none',
                    color: 'var(--muted)',
                    cursor: 'pointer',
                  }}
                >
                  取消
                </button>
              </div>
            </div>
          )}

          {keysLoading && (
            <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>加载中...</div>
          )}

          {/* Key List */}
          {keys?.map((k) => (
            <div
              key={k.id}
              style={{
                background: 'var(--card)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: '8px 10px',
                marginBottom: 6,
                transition: 'border-color 0.12s',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--border-strong)' }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div className="flex items-center gap-2">
                  <Key size={11} style={{ color: 'var(--muted)' }} />
                  <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-strong)' }}>{k.label}</span>
                </div>
                {confirmDeleteId === k.id ? (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleDeleteKey(k.id)}
                      style={{ fontSize: 10, color: '#f87171', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 500 }}
                    >
                      确认
                    </button>
                    <button
                      onClick={() => setConfirmDeleteId(null)}
                      style={{ fontSize: 10, color: 'var(--muted)', background: 'none', border: 'none', cursor: 'pointer' }}
                    >
                      取消
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setConfirmDeleteId(k.id)}
                    className="flex items-center gap-1"
                    style={{ fontSize: 10, color: 'var(--muted)', background: 'none', border: 'none', cursor: 'pointer' }}
                  >
                    <Trash2 size={10} /> 撤销
                  </button>
                )}
              </div>
              <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 3, fontFamily: 'var(--mono)' }}>
                {k.key_prefix}...
                <span style={{ fontFamily: 'var(--font)', marginLeft: 8 }}>
                  {k.usage_calls.toLocaleString()} 次调用 · {k.usage_chars.toLocaleString()} 字符
                </span>
                {k.last_used_at && (
                  <span style={{ marginLeft: 8 }}>· {formatRelativeTime(k.last_used_at)}</span>
                )}
              </div>
            </div>
          ))}

          {!keysLoading && keys?.length === 0 && !showNewKeyForm && (
            <div
              style={{
                padding: '20px 16px',
                textAlign: 'center',
                border: '1px dashed var(--border)',
                borderRadius: 6,
                color: 'var(--muted)',
                fontSize: 11,
              }}
            >
              暂无 API Key — 创建一个用于鉴权访问
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/* --- Shared helpers --- */

function StatCard({ icon, label, value }: { icon?: React.ReactNode; label: string; value: string }) {
  return (
    <div
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '8px 12px',
        flex: 1,
        minWidth: 0,
      }}
    >
      <div className="flex items-center gap-1.5" style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 2 }}>
        {icon}
        {label}
      </div>
      <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-strong)' }}>{value}</div>
    </div>
  )
}

function SectionTitle({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div
      style={{
        fontSize: 10,
        fontWeight: 600,
        color: 'var(--muted)',
        marginBottom: 6,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

function formatRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return '刚刚'
  if (mins < 60) return `${mins}分钟前`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}小时前`
  const days = Math.floor(hours / 24)
  return `${days}天前`
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/overlays/ApiManagementOverlay.tsx
git commit -m "feat: create ApiManagementOverlay with master-detail layout"
```

### Task 8: Create CreateInstanceForm

**Files:**
- Create: `frontend/src/components/overlays/CreateInstanceForm.tsx`

- [ ] **Step 1: Create the component**

```typescript
import { useState, useEffect } from 'react'
import { useVoicePresets } from '../../api/voices'
import { useCreateInstance } from '../../api/instances'
import type { ApiManagementOptions } from '../../stores/panel'

interface Props {
  defaultOptions?: ApiManagementOptions
  onCreated: (instanceId: string) => void
  onCancel: () => void
}

export default function CreateInstanceForm({ defaultOptions, onCreated, onCancel }: Props) {
  const { data: presets } = useVoicePresets()
  const createInstance = useCreateInstance()

  const [sourceType] = useState<'preset'>('preset') // Only preset for now
  const [sourceId, setSourceId] = useState(defaultOptions?.source_id ?? '')
  const [name, setName] = useState('')
  const [instanceType, setInstanceType] = useState('tts')

  // Pre-fill source_id from options
  useEffect(() => {
    if (defaultOptions?.source_id) {
      setSourceId(defaultOptions.source_id)
    }
  }, [defaultOptions?.source_id])

  const handleSubmit = async () => {
    if (!name.trim() || !sourceId) return
    const result = await createInstance.mutateAsync({
      source_type: sourceType,
      source_id: sourceId,
      name: name.trim(),
      type: instanceType,
    })
    onCreated(result.id)
  }

  return (
    <div style={{ padding: '16px 20px', maxWidth: 480 }}>
      <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-strong)', margin: '0 0 16px' }}>
        新建服务实例
      </h2>

      {/* Instance Type */}
      <FormField label="服务类型">
        <select
          value={instanceType}
          onChange={(e) => setInstanceType(e.target.value)}
          style={selectStyle}
        >
          <option value="tts">语音合成 TTS</option>
          <option value="image">图像生成 Image</option>
          <option value="inference">文本推理 LLM</option>
        </select>
      </FormField>

      {/* Source Type (read-only for now) */}
      <FormField label="来源类型">
        <div style={{ ...inputStyle, background: 'var(--bg-accent)', color: 'var(--muted)' }}>
          预设 (Preset)
        </div>
      </FormField>

      {/* Source Selection */}
      <FormField label="选择预设">
        <select
          value={sourceId}
          onChange={(e) => setSourceId(e.target.value)}
          style={selectStyle}
        >
          <option value="">-- 请选择 --</option>
          {presets?.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} ({p.engine})
            </option>
          ))}
        </select>
      </FormField>

      {/* Instance Name */}
      <FormField label="实例名称">
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
          placeholder="如：播客语音包、新闻TTS"
          autoFocus
          style={inputStyle}
        />
      </FormField>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8, marginTop: 20 }}>
        <button
          onClick={handleSubmit}
          disabled={createInstance.isPending || !name.trim() || !sourceId}
          style={{
            padding: '8px 24px',
            fontSize: 12,
            borderRadius: 5,
            border: 'none',
            background: 'var(--accent)',
            color: '#fff',
            cursor: 'pointer',
            fontWeight: 500,
            opacity: createInstance.isPending || !name.trim() || !sourceId ? 0.5 : 1,
          }}
        >
          {createInstance.isPending ? '创建中...' : '创建实例'}
        </button>
        <button
          onClick={onCancel}
          style={{
            padding: '8px 16px',
            fontSize: 12,
            borderRadius: 5,
            border: '1px solid var(--border)',
            background: 'none',
            color: 'var(--muted)',
            cursor: 'pointer',
          }}
        >
          取消
        </button>
      </div>

      {createInstance.isError && (
        <div style={{ marginTop: 10, fontSize: 11, color: '#f87171' }}>
          创建失败：{createInstance.error?.message ?? '未知错误'}
        </div>
      )}
    </div>
  )
}

function FormField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontSize: 10, color: 'var(--muted)', display: 'block', marginBottom: 4, fontWeight: 500 }}>
        {label}
      </label>
      {children}
    </div>
  )
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '7px 10px',
  fontSize: 12,
  background: 'var(--card)',
  border: '1px solid var(--border)',
  borderRadius: 5,
  color: 'var(--text-strong)',
  outline: 'none',
  boxSizing: 'border-box',
}

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  cursor: 'pointer',
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/overlays/CreateInstanceForm.tsx
git commit -m "feat: create CreateInstanceForm component"
```

### Task 9: Update NodeEditor — remove deleted panels/overlays

**Files:**
- Modify: `frontend/src/components/nodes/NodeEditor.tsx`

**Note:** This task must come AFTER Tasks 7-8 (ApiManagementOverlay/CreateInstanceForm exist) so the import resolves.

- [ ] **Step 1: Remove imports and references**

Remove these imports:
```typescript
// DELETE these lines
import CollectionsPanel from '../panels/CollectionsPanel'
import ApiNodesPanel from '../panels/ApiNodesPanel'
import InstanceDetailOverlay from '../overlays/InstanceDetailOverlay'
```

Add import:
```typescript
import ApiManagementOverlay from '../overlays/ApiManagementOverlay'
```

Update PANEL_MAP:
```typescript
const PANEL_MAP: Record<string, React.FC> = {
  nodes: NodeLibraryPanel,
  workflows: WorkflowsPanel,
  presets: PresetsPanel,
}
```

Replace overlay rendering:
```typescript
{/* Replace instance-detail with api-management */}
{activeOverlay === 'api-management' && <ApiManagementOverlay />}
```
(Remove the `{activeOverlay === 'instance-detail' && <InstanceDetailOverlay />}` line)

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/nodes/NodeEditor.tsx
git commit -m "refactor: NodeEditor — remove deleted panels, add ApiManagementOverlay"
```

---

## Chunk 4: Simplify PresetDetailOverlay & Cleanup

### Task 10: Simplify PresetDetailOverlay

**Files:**
- Modify: `frontend/src/components/overlays/PresetDetailOverlay.tsx`

- [ ] **Step 1: Remove Instance list, add deploy button**

Replace the full component. Remove imports for `useInstances`, `useCreateInstance`, `ServiceInstance`. Add `usePanelStore.openApiManagement`. Remove `InstanceCard` sub-component. Right column becomes: deploy button + reference audio/text.

```typescript
import { Rocket } from 'lucide-react'
import { usePanelStore } from '../../stores/panel'
import { useVoicePresets } from '../../api/voices'

export default function PresetDetailOverlay() {
  const presetId = usePanelStore((s) => s.selectedPresetId)
  const openApiManagement = usePanelStore((s) => s.openApiManagement)
  const { data: presets } = useVoicePresets()
  const preset = presets?.find((p) => p.id === presetId)

  if (!preset) {
    return (
      <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
        <div style={{ padding: 16, fontSize: 11, color: 'var(--muted)' }}>Preset not found</div>
      </div>
    )
  }

  const params = preset.params as Record<string, unknown>

  return (
    <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
      <div style={{ padding: '16px 20px', maxWidth: 960 }}>
        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-strong)', margin: 0 }}>
              {preset.name}
            </h2>
            {preset.tags.map((tag) => (
              <span
                key={tag}
                style={{
                  fontSize: 9,
                  padding: '2px 7px',
                  borderRadius: 3,
                  background: 'var(--bg-hover)',
                  color: 'var(--muted)',
                }}
              >
                {tag}
              </span>
            ))}
          </div>
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            {preset.engine} · 声音预设模板
          </div>
        </div>

        <div style={{ display: 'flex', gap: 20 }}>
          {/* Left Column — Configuration */}
          <div style={{ width: 280, flexShrink: 0 }}>
            <SectionTitle>配置参数</SectionTitle>
            <div
              style={{
                background: 'var(--card)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: '8px 0',
                marginBottom: 16,
              }}
            >
              <ConfigRow label="engine" value={preset.engine} />
              {Object.entries(params).map(([k, v]) => (
                <ConfigRow key={k} label={k} value={String(v)} />
              ))}
            </div>

            {preset.reference_audio_path && (
              <>
                <SectionTitle>参考音频</SectionTitle>
                <div
                  style={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    padding: '6px 10px',
                    fontSize: 10,
                    color: 'var(--text-strong)',
                    fontFamily: 'var(--mono)',
                    wordBreak: 'break-all',
                    marginBottom: 16,
                  }}
                >
                  {preset.reference_audio_path}
                </div>
              </>
            )}

            {preset.reference_text && (
              <>
                <SectionTitle>参考文本</SectionTitle>
                <div
                  style={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    padding: '6px 10px',
                    fontSize: 10,
                    color: 'var(--text-strong)',
                    lineHeight: 1.5,
                    marginBottom: 16,
                  }}
                >
                  {preset.reference_text}
                </div>
              </>
            )}
          </div>

          {/* Right Column — Deploy action */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <SectionTitle>快捷操作</SectionTitle>
            <div
              style={{
                background: 'var(--card)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: 20,
                textAlign: 'center',
              }}
            >
              <button
                onClick={() => openApiManagement({ source_type: 'preset', source_id: presetId! })}
                className="flex items-center gap-2 justify-center"
                style={{
                  fontSize: 12,
                  padding: '10px 24px',
                  background: 'var(--accent)',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 5,
                  cursor: 'pointer',
                  fontWeight: 500,
                  display: 'inline-flex',
                }}
              >
                <Rocket size={14} />
                部署为服务实例
              </button>
              <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 10 }}>
                将此预设部署为独立 API 服务，可在 API 管理页面管理
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 10px', fontSize: 11 }}>
      <span style={{ color: 'var(--muted)' }}>{label}</span>
      <span style={{ color: 'var(--text-strong)', fontFamily: 'var(--mono)' }}>{value}</span>
    </div>
  )
}

function SectionTitle({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div
      style={{
        fontSize: 10,
        fontWeight: 600,
        color: 'var(--muted)',
        marginBottom: 6,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        ...style,
      }}
    >
      {children}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/overlays/PresetDetailOverlay.tsx
git commit -m "refactor: simplify PresetDetailOverlay — replace instance list with deploy button"
```

### Task 11: Delete obsolete components

**Files:**
- Delete: `frontend/src/components/panels/CollectionsPanel.tsx`
- Delete: `frontend/src/components/panels/ApiNodesPanel.tsx`
- Delete: `frontend/src/components/overlays/InstanceDetailOverlay.tsx`

- [ ] **Step 1: Delete the three files**

```bash
git rm frontend/src/components/panels/CollectionsPanel.tsx
git rm frontend/src/components/panels/ApiNodesPanel.tsx
git rm frontend/src/components/overlays/InstanceDetailOverlay.tsx
```

- [ ] **Step 2: Verify no remaining imports**

Search for any remaining imports of the deleted components:

```bash
cd frontend && grep -r "CollectionsPanel\|ApiNodesPanel\|InstanceDetailOverlay" src/ --include="*.tsx" --include="*.ts"
```

Expected: No results (all references already updated in Task 9)

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: delete CollectionsPanel, ApiNodesPanel, InstanceDetailOverlay"
```

### Task 12: Build & verify

- [ ] **Step 1: Run frontend type check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: No type errors

- [ ] **Step 2: Run backend tests**

```bash
cd backend && python -m pytest -v
```

Expected: ALL PASS

- [ ] **Step 3: Run frontend dev build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A && git commit -m "fix: address build/test issues from API management restructure"
```
(Only if fixes were needed)
