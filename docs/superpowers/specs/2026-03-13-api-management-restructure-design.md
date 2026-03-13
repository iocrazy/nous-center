# API Management Restructure Design

## Goal

Restructure instance management from a preset-centric navigation to a centralized API management overlay with master-detail layout, remove the Collection concept, and adapt the data model so instances can originate from either presets or workflows.

## Architecture

The current navigation flow (Presets panel → PresetDetail → InstanceDetail) is replaced by a dedicated API Management overlay accessible from the main IconRail. Instances are listed in a sidebar grouped by service type (tts/image/inference), with the selected instance's detail shown inline. The Collection concept is eliminated; multi-voice composition is handled by workflow composite nodes instead.

## Tech Stack

- Frontend: React + TypeScript + Zustand + TanStack Query + lucide-react
- Backend: FastAPI + SQLAlchemy + Pydantic
- Existing component library: FloatingPanel, overlay system, IconRail

---

## 1. Concept Model Changes

### Removed

- **Collection** — eliminated entirely. Multi-voice grouping is handled by workflow composite nodes on the canvas.

### Retained Three-Layer Hierarchy

```
Preset (voice template, atomic unit)
  │
  │  Can deploy directly, or drag into Workflow to compose
  ↓
Instance (service instance, externally-facing service unit)
  │  type: tts | image | inference
  │  source_type: preset | workflow
  │  source_id: references a preset or workflow ID
  │
  ↓
InstanceApiKey (API key, bound to Instance)
```

### Data Model Changes

**ServiceInstance table:**

```
Remove:
  preset_id: BigInt FK → VoicePreset

Add:
  source_type: String(20)   — "preset" or "workflow"
  source_id: BigInt          — ID of the source preset or workflow

Keep:
  type: String(20)           — "tts", "image", "inference" (already exists)
  All other fields unchanged
```

The `source_type` + `source_id` pattern replaces the hard foreign key to VoicePreset, allowing instances to originate from either presets or workflows.

**Schema changes:**

- `ServiceInstanceCreate`: replace `preset_id` with `source_type` + `source_id`
- `ServiceInstanceOut`: replace `preset_id` with `source_type` + `source_id`
- Add `source_name` as a computed field in responses (resolved from source_type + source_id)

**API route changes:**

- `GET /api/v1/instances` — add `?type=tts` filter parameter, remove `?preset_id=` filter
- `POST /api/v1/instances` — accept `source_type` + `source_id` instead of `preset_id`
- Instance synthesize endpoint unchanged

---

## 2. UI Layout: API Management Overlay

Three-column layout as a full-screen overlay:

```
┌──────────┬────────────────┬──────────────────────────────────┐
│ IconRail │ Instance List   │ Instance Detail                  │
│  (44px)  │   (240px)      │    (flex, remaining space)       │
│          │                │                                  │
│  [📊]    │ ▾ TTS (3)      │ Header: name + status badge      │
│  [🧠]    │  ● 播客语音包   │ Source: workflow/preset tag       │
│  ────    │  ● 新闻TTS     │                                  │
│  [🔲]    │  ○ 旁白测试    │ Stats: [Keys] [Calls] [Chars]    │
│  [🔀]    │                │                                  │
│  [🎤]    │ ▾ Image (1)    │ ┌─────────────┬──────────────┐   │
│  [🔗]◀   │  ● 封面图生成   │ │ Endpoint    │ API Keys     │   │
│          │                │ │ POST /v1/.. │ + New Key     │   │
│  ────    │ ▸ LLM (0)     │ │ cURL ▾      │ 🔑 Key1      │   │
│  [⚙️]    │                │ │ Params      │ 🔑 Key2      │   │
│  [🌙]    │ [+ 新建实例]   │ └─────────────┴──────────────┘   │
│  🟢 GPU  │                │                                  │
└──────────┴────────────────┴──────────────────────────────────┘
```

### Instance List (Column 2)

- Grouped by `type`: 语音合成 TTS, 图像生成 Image, 文本推理 LLM
- Each group header is collapsible with count badge
- Instance card shows: status dot (green/red) + name + source tag (workflow/preset) + summary (key count, voice count)
- Selected instance has purple highlight border
- Inactive instances shown at reduced opacity (0.45)
- Bottom: "+ 新建实例" button

### Instance Detail (Column 3)

- Reuses existing InstanceDetailOverlay content (stats cards, endpoint, cURL, API keys)
- Additions: source tag badge (workflow/preset) next to instance name
- When no instance selected: empty state with prompt
- When creating new instance: replaced by CreateInstanceForm

---

## 3. Simplified Preset Detail Overlay

PresetDetailOverlay is simplified — the right-side Instance list is removed:

```
┌──────────────────────────────────────────┐
│ Header: name + tags                      │
│ Subtitle: engine · 声音预设模板            │
├──────────────────────────────────────────┤
│ Left (260px)       │ Right (flex)        │
│                    │                     │
│ 配置参数            │ 快捷操作             │
│ engine: cosyvoice2 │ ┌─────────────────┐│
│ voice: default     │ │ 🚀 部署为服务实例 ││
│ speed: 1.0         │ │ (跳转API管理页)   ││
│ sample_rate: 24000 │ └─────────────────┘│
│                    │                     │
│ 参考音频            │                     │
│ /nas/audio/...     │                     │
│                    │                     │
│ 参考文本            │                     │
│ (if exists)        │                     │
└──────────────────────────────────────────┘
```

The "部署为服务实例" button navigates to the API Management overlay and opens the create form with `source_type=preset` and `source_id` pre-filled.

---

## 4. Navigation & State Management Changes

### IconRail Changes

| Icon | Before | After |
|------|--------|-------|
| Presets 🎤 | Opens FloatingPanel | **No change** |
| Collections 📚 | Opens FloatingPanel (mock) | **Remove icon** |
| API 🔗 | Opens FloatingPanel (mock) | **Opens api-management overlay** |

### Panel Store Changes

**PanelId:**
```
Before: 'nodes' | 'workflows' | 'presets' | 'collections' | 'api'
After:  'nodes' | 'workflows' | 'presets'
```

**OverlayId:**
```
Before: 'dashboard' | 'models' | 'settings' | 'preset-detail' | 'instance-detail'
After:  'dashboard' | 'models' | 'settings' | 'preset-detail' | 'api-management'
```

**Removed from store:**
- `selectedInstanceId` — becomes local state inside ApiManagementOverlay
- `openInstanceDetail()` — removed, no longer needed

**Added to store:**
- `openApiManagement(options?)` — opens api-management overlay, optionally with pre-filled create form params (source_type, source_id)

---

## 5. Create Instance Flow

### Entry 1: From API Management "新建实例" button

1. Click "+ 新建实例" at bottom of instance list
2. Detail area replaces with CreateInstanceForm
3. Form fields:
   - Type: dropdown (tts / image / inference)
   - Source: dropdown (preset / workflow)
   - Source selection: dropdown filtered by source type
   - Instance name: text input
4. Submit → POST /api/v1/instances → new instance appears in list, auto-selected

### Entry 2: From Preset Detail "部署为服务实例" button

1. Click "🚀 部署为服务实例" on PresetDetailOverlay
2. Calls `openApiManagement({ source_type: 'preset', source_id: presetId })`
3. API Management overlay opens with CreateInstanceForm pre-filled
4. User only needs to enter instance name → create

---

## 6. File Change Summary

### Backend

| File | Action | Description |
|------|--------|-------------|
| `backend/src/models/service_instance.py` | Modify | Replace `preset_id` FK with `source_type` + `source_id` columns |
| `backend/src/models/schemas.py` | Modify | Update ServiceInstanceCreate/Out schemas |
| `backend/src/api/routes/instances.py` | Modify | Update create/list endpoints for new fields, add `?type=` filter |
| `backend/src/api/routes/instance_service.py` | Modify | Resolve source params from source_type + source_id |

### Frontend — Delete

| File | Reason |
|------|--------|
| `frontend/src/components/panels/CollectionsPanel.tsx` | Collection concept removed |
| `frontend/src/components/panels/ApiNodesPanel.tsx` | Replaced by API Management overlay |
| `frontend/src/components/overlays/InstanceDetailOverlay.tsx` | Content migrated into ApiManagementOverlay |

### Frontend — Create

| File | Description |
|------|-------------|
| `frontend/src/components/overlays/ApiManagementOverlay.tsx` | Three-column overlay: instance list + detail |
| `frontend/src/components/overlays/CreateInstanceForm.tsx` | New instance creation form |

### Frontend — Modify

| File | Description |
|------|-------------|
| `frontend/src/stores/panel.ts` | Remove collections/api PanelId, add api-management OverlayId, remove selectedInstanceId, add openApiManagement() |
| `frontend/src/components/layout/IconRail.tsx` | Remove Collections icon, change API icon to open overlay |
| `frontend/src/components/overlays/PresetDetailOverlay.tsx` | Remove Instance list, add "部署为实例" button |
| `frontend/src/components/nodes/NodeEditor.tsx` | Render ApiManagementOverlay instead of InstanceDetailOverlay |
| `frontend/src/components/layout/Topbar.tsx` | Handle api-management overlay title |
| `frontend/src/api/instances.ts` | Update hooks for new schema fields, add useInstancesByType() |

---

## 7. Implementation Notes

- **Database migration**: Existing `service_instances` rows have `preset_id`. Migration must copy `preset_id` → `source_id`, set `source_type='preset'`, then drop the `preset_id` column.
- **`source_name` resolution**: `ServiceInstanceOut` should include a computed `source_name` field, resolved by joining against presets or workflows based on `source_type`. For now, only `source_type='preset'` needs resolution; workflow resolution is deferred.
- **`source_type='workflow'` validation**: Since workflows are not yet persisted entities, the create endpoint should only accept `source_type='preset'` initially. Workflow support is added when workflow persistence is implemented.

---

## 8. Out of Scope

- Workflow composite node implementation (multi-voice node) — separate spec
- Workflow deployment as instance — requires workflow persistence, separate spec
- Instance usage analytics/charts — future enhancement
- Instance log viewer — future enhancement
