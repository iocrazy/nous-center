# Image Component Multi-GPU — PR-5b (Frontend: loader nodes + component state UI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** ComfyUI-style component loader UX in the workflow editor: 4 loader nodes (unet/clip/vae/lora_apply) with file dropdowns sourced from `GET /api/v1/components`, a live four-state status header (loaded/cold/loading/failed) fed by the `/ws/models` `component_state_changed` push (PR-5a), and `image_generate` reshaped with unet/clip/vae input ports.

**Architecture:** Reuse the existing **declarative node** system (`DeclarativeNode` + `DECLARATIVE_NODES` widget catalog) rather than handwritten components — the loader nodes are forms (file/device/dtype selects) that write to `node.data`; the backend (PR-4 `image_components.py`) builds the descriptor at run time. Add a `component_select` widget (role-based dropdown) and a `componentRole` flag on the node def that turns on a live status header computed from `data` via a `useComponentState` hook (WS-driven zustand store). `image_generate` gains unet/clip/vae input ports (keeps `model_key` widget for legacy back-compat — backend inline-expands).

**Tech Stack:** React + TypeScript + `@xyflow/react` (ReactFlow) + `@tanstack/react-query` + zustand + lucide-react; CSS-var inline styles (no tailwind classes in components); vitest + React Testing Library. Conventions: no emoji.

**Branch:** `feat/image-component-multigpu-pr5b-frontend` (from master).

**Prereqs (merged):** PR-1..PR-5a (#112–#118). PR-5a backend endpoints live: `GET /api/v1/components?role=`, `GET /api/v1/models/components/state?keys=`, `POST /api/v1/models/components/preload`, WS `/ws/models` `component_state_changed` (payload `{event, component_key, state, error}`).

**Spec:** design doc §6.1 (four states), §6.3 (`useComponentState`), §7.1 (palette subcategory), §7.2 (loader node form), §7.3 (image_generate multi-input).

**Scope note:** ModelsOverlay "GPU 分配表单" (spec §6) is **deferred** — the per-node `device` dropdown on each loader IS the GPU-allocation UX for components; a separate overlay form is redundant for PR-5b. Note as future work.

---

## Key facts (verified from the current frontend)

- `NodeType` union + `PortType` + `NODE_DEFS: Record<NodeType, NodeDef>` live in `frontend/src/models/workflow.ts`. `PortType = 'text'|'audio'|'image'|'message'|'data'|'any'`. `NodeDef = {type,label,inputs:PortDef[],outputs:PortDef[]}`, `PortDef={id,type:PortType,label}`.
- Declarative catalog in `frontend/src/models/nodeRegistry.ts`: `WidgetType`, `WidgetDef`, `DeclarativeNodeDef={type,label,category,badge,badgeColor,widgets}`, `DECLARATIVE_NODES`.
- `frontend/src/components/nodes/nodeTypes.ts` auto-maps every `DECLARATIVE_NODES` key → `DeclarativeNode` (no change needed when adding declarative nodes).
- `frontend/src/components/nodes/DeclarativeNode.tsx`: `WidgetRenderer` switch over `widget.widget`; node body renders `declDef.widgets` via `NodeWidgetRow`. Exports from `BaseNode`: `BaseNode, NodeWidgetRow, NodeInput, NodeSelect, NodeNumberDrag, NodeTextarea`.
- API: `apiFetch<T>(path, init?)` in `frontend/src/api/client.ts`. WS: `useLiveChannel(url, {onMessage, onReconnect})` in `frontend/src/api/useLiveChannel.ts` (existing example `useEnginesLiveSync` builds `${proto}//${host}/ws/models`). `useQuery` from `@tanstack/react-query`.
- Palette: `frontend/src/components/panels/NodeLibraryPanel.tsx` renders its own `BUILTIN_CATEGORIES` (+ merges `PLUGIN_CATEGORIES`). `NodeCategory = {name,label,color,nodes:{type,dotColor}[]}`.
- `GET /api/v1/components?role=unet` → `{components: [{filename, abs_path, size_mb, quant_type, mtime}]}` (PR-3 `component_scanner`).
- Backend `component_state_key(spec)` = `"{file}|{device}|{dtype}|{lora_sig}"`, `lora_sig = "+".join(sorted("name@strength"))`. The frontend must compute the identical string from a loader node's `data` (file/device/dtype, no loras on a base loader → trailing `|`).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `frontend/src/models/workflow.ts` | Modify | `NodeType` += 4 loaders; `PortType` += `unet`/`clip`/`vae`; `NODE_DEFS` += 4 loaders + image_generate input ports |
| `frontend/src/models/nodeRegistry.ts` | Modify | `WidgetType` += `component_select`; `WidgetDef.role`; `DeclarativeNodeDef.componentRole`; `DECLARATIVE_NODES` += 4 loaders |
| `frontend/src/api/components.ts` | Create | `ComponentInfo`, `useComponents(role)`, `componentStateKey(...)`, component-state zustand store, `useComponentState(key)` (WS-driven) |
| `frontend/src/components/nodes/DeclarativeNode.tsx` | Modify | `ComponentSelectWidget` + `component_select` case; live four-state header gated by `componentRole` |
| `frontend/src/components/panels/NodeLibraryPanel.tsx` | Modify | add "组件加载" category with the 4 loader nodes |
| `frontend/src/**/*.test.tsx` | Create | per task |

---

## Task 1: Types + catalog (loaders + image_generate ports)

**Files:** Modify `frontend/src/models/workflow.ts`, `frontend/src/models/nodeRegistry.ts`. Test: tsc (no runtime test).

- [ ] **Step 1: `workflow.ts`** — extend `NodeType` union (find its definition) to include `'image_unet_load' | 'image_clip_load' | 'image_vae_load' | 'image_lora_apply'`. Extend `PortType`:

```typescript
export type PortType = 'text' | 'audio' | 'image' | 'message' | 'data' | 'any' | 'unet' | 'clip' | 'vae'
```

Add to `NODE_DEFS` (the 4 loaders + reshape `image_generate`):

```typescript
  image_unet_load: {
    type: 'image_unet_load', label: 'UNET 加载',
    inputs: [],
    outputs: [{ id: 'unet', type: 'unet', label: 'UNET' }],
  },
  image_clip_load: {
    type: 'image_clip_load', label: 'CLIP 加载',
    inputs: [],
    outputs: [{ id: 'clip', type: 'clip', label: 'CLIP' }],
  },
  image_vae_load: {
    type: 'image_vae_load', label: 'VAE 加载',
    inputs: [],
    outputs: [{ id: 'vae', type: 'vae', label: 'VAE' }],
  },
  image_lora_apply: {
    type: 'image_lora_apply', label: 'LoRA 应用',
    inputs: [{ id: 'unet', type: 'unet', label: 'UNET' }],
    outputs: [{ id: 'unet', type: 'unet', label: 'UNET' }],
  },
```

Reshape `image_generate` inputs (keep the existing `prompt`, add unet/clip/vae + negative_prompt):

```typescript
  image_generate: {
    type: 'image_generate', label: '图像生成',
    inputs: [
      { id: 'unet', type: 'unet', label: 'UNET' },
      { id: 'clip', type: 'clip', label: 'CLIP' },
      { id: 'vae', type: 'vae', label: 'VAE' },
      { id: 'prompt', type: 'text', label: '提示' },
      { id: 'negative_prompt', type: 'text', label: '负面' },
    ],
    outputs: [{ id: 'image', type: 'image', label: '图像' }],
  },
```

- [ ] **Step 2: `nodeRegistry.ts`** — extend `WidgetType`:

```typescript
export type WidgetType = 'input' | 'textarea' | 'select' | 'slider' | 'checkbox' | 'agent_select' | 'model_select' | 'lora_stack' | 'lora_select' | 'component_select'
```

Add to `WidgetDef`: `role?: 'unet' | 'clip' | 'vae' | 'loras'`. Add to `DeclarativeNodeDef`: `componentRole?: 'unet' | 'clip' | 'vae'`.

Add 4 entries to `DECLARATIVE_NODES`:

```typescript
  image_unet_load: {
    type: 'image_unet_load', label: 'UNET 加载', category: 'image_loading',
    badge: 'UNET', badgeColor: 'rgba(244,114,182,0.9)', componentRole: 'unet',
    widgets: [
      { name: 'file', label: '文件', widget: 'component_select', role: 'unet' },
      { name: 'device', label: '设备', widget: 'select', options: [
        { value: 'auto', label: 'auto' }, { value: 'cuda:0', label: 'cuda:0' },
        { value: 'cuda:1', label: 'cuda:1' }, { value: 'cuda:2', label: 'cuda:2' }, { value: 'cpu', label: 'cpu' },
      ], default: 'auto' },
      { name: 'dtype', label: '精度', widget: 'select', options: [
        { value: 'bfloat16', label: 'bfloat16' }, { value: 'float16', label: 'float16' }, { value: 'fp8_e4m3', label: 'fp8_e4m3' },
      ], default: 'bfloat16' },
      { name: 'adapter_arch', label: '架构', widget: 'select', options: [
        { value: 'flux2', label: 'flux2' }, { value: 'flux1', label: 'flux1' },
      ], default: 'flux2' },
    ],
  },
  image_clip_load: {
    type: 'image_clip_load', label: 'CLIP 加载', category: 'image_loading',
    badge: 'CLIP', badgeColor: 'rgba(234,179,8,0.9)', componentRole: 'clip',
    widgets: [
      { name: 'file', label: '文件', widget: 'component_select', role: 'clip' },
      { name: 'device', label: '设备', widget: 'select', options: [
        { value: 'auto', label: 'auto' }, { value: 'cuda:0', label: 'cuda:0' },
        { value: 'cuda:1', label: 'cuda:1' }, { value: 'cuda:2', label: 'cuda:2' }, { value: 'cpu', label: 'cpu' },
      ], default: 'auto' },
      { name: 'dtype', label: '精度', widget: 'select', options: [
        { value: 'bfloat16', label: 'bfloat16' }, { value: 'fp8_e4m3', label: 'fp8_e4m3' },
      ], default: 'bfloat16' },
      { name: 'clip_arch', label: '架构', widget: 'select', options: [
        { value: 'flux2', label: 'flux2' }, { value: 'flux1', label: 'flux1' },
        { value: 'sdxl', label: 'sdxl' }, { value: 'qwen', label: 'qwen' },
      ], default: 'flux2' },
    ],
  },
  image_vae_load: {
    type: 'image_vae_load', label: 'VAE 加载', category: 'image_loading',
    badge: 'VAE', badgeColor: 'rgba(239,68,68,0.85)', componentRole: 'vae',
    widgets: [
      { name: 'file', label: '文件', widget: 'component_select', role: 'vae' },
      { name: 'device', label: '设备', widget: 'select', options: [
        { value: 'auto', label: 'auto' }, { value: 'cuda:0', label: 'cuda:0' },
        { value: 'cuda:1', label: 'cuda:1' }, { value: 'cuda:2', label: 'cuda:2' }, { value: 'cpu', label: 'cpu' },
      ], default: 'auto' },
      { name: 'dtype', label: '精度', widget: 'select', options: [
        { value: 'bfloat16', label: 'bfloat16' }, { value: 'float16', label: 'float16' },
      ], default: 'bfloat16' },
    ],
  },
  image_lora_apply: {
    type: 'image_lora_apply', label: 'LoRA 应用', category: 'image_loading',
    badge: 'LoRA', badgeColor: 'rgba(168,85,247,0.85)',
    widgets: [
      { name: 'lora_path', label: 'LoRA', widget: 'component_select', role: 'loras' },
      { name: 'strength', label: '强度', widget: 'slider', min: -2, max: 2, step: 0.05, precision: 2, default: 1.0 },
      { name: 'bypass', label: '旁路', widget: 'checkbox', default: false },
    ],
  },
```

(Note: `image_lora_apply` has no `componentRole` — it's a transform, not a base loader, so no status header.)

- [ ] **Step 3: verify tsc** — `cd frontend && npx tsc --noEmit` → no errors. (`NodeType`-keyed `Record` forces you to add all 4 to NODE_DEFS — tsc catches omissions.)

- [ ] **Step 4: commit**

```bash
git add frontend/src/models/workflow.ts frontend/src/models/nodeRegistry.ts
git commit -m "feat(image): PR-5b — loader node types + catalog + image_generate ports"
```

---

## Task 1b: Backend glue — `image_lora_apply` accepts `lora_path` (abs path)

**Files:** Modify `backend/src/services/nodes/image_components.py`. Test: `backend/tests/test_image_component_nodes.py`.

> The PR-4 `image_lora_apply` node requires `data["lora_file"]` (a name) + optional `lora_path`. The PR-5b frontend `component_select` emits a single value (the abs path) into `data.lora_path`. Make the node derive the LoRA `name` from the path basename when `lora_file` is absent, and require `lora_path` (or `lora_file`). This is the only backend change in PR-5b — the contract glue for the LoRA node.

- [ ] **Step 1: failing test** — add to `backend/tests/test_image_component_nodes.py`:

```python
@pytest.mark.asyncio
async def test_lora_apply_accepts_lora_path_only():
    base = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []}
    out = await get_node_class("image_lora_apply")().invoke(
        {"lora_path": "/m/loras/style-xl.safetensors", "strength": 0.7}, {"unet": base})
    lo = out["unet"]["loras"][0]
    assert lo["path"] == "/m/loras/style-xl.safetensors"
    assert lo["name"] == "style-xl.safetensors"   # derived from basename
    assert lo["strength"] == 0.7
```

- [ ] **Step 2: run → FAIL** — `cd backend && uv run pytest tests/test_image_component_nodes.py -q` (current node does `data["lora_file"]` → KeyError).

- [ ] **Step 3: implement** — in `image_components.py` `ImageLoraApplyNode.invoke`, replace the `appended` construction (currently `name=data["lora_file"]`) with path-aware derivation:

```python
        import os
        lora_path = data.get("lora_path")
        lora_name = data.get("lora_file") or (os.path.basename(lora_path) if lora_path else None)
        if not lora_name:
            from src.services.workflow_executor import ExecutionError
            raise ExecutionError("image_lora_apply 需要 lora_path 或 lora_file")
        appended = {
            "name": lora_name,
            "path": lora_path,
            "strength": float(data.get("strength", 1.0)),
        }
        return {"unet": {**upstream, "loras": [*upstream.get("loras", []), appended]}}
```

Keep the existing `bypass` passthrough + upstream-validation guards above it unchanged.

- [ ] **Step 4: run → PASS** — `cd backend && uv run pytest tests/test_image_component_nodes.py -q` (existing lora_apply tests that pass `lora_file` still pass — the `or` keeps back-compat).

- [ ] **Step 5: lint + commit**

```bash
git add backend/src/services/nodes/image_components.py backend/tests/test_image_component_nodes.py
git commit -m "feat(image): PR-5b — image_lora_apply derives LoRA name from lora_path"
```

> NOTE: this backend change must keep the full backend suite green — run `cd backend && uv run pytest tests/test_image_component_nodes.py tests/test_runner_build_request.py -q` before moving on.

---

## Task 2: `api/components.ts` — fetch + live state hook

**Files:** Create `frontend/src/api/components.ts`. Test: `frontend/src/api/components.test.tsx`.

- [ ] **Step 1: failing test** (`components.test.tsx`) — mirror `api/runners.test.tsx` patterns (mock `./client`, QueryClient wrapper):

```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('./client', () => ({ apiFetch: vi.fn() }))
vi.mock('./useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
import { apiFetch } from './client'
import { useComponents, componentStateKey, useComponentStateStore } from './components'

function wrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  )
}

describe('components api', () => {
  beforeEach(() => vi.clearAllMocks())

  it('componentStateKey matches backend format (file|device|dtype|)', () => {
    expect(componentStateKey({ file: '/m/u.safe', device: 'cuda:1', dtype: 'bfloat16' }))
      .toBe('/m/u.safe|cuda:1|bfloat16|')
  })

  it('useComponents fetches by role', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ components: [{ filename: 'u.safe', abs_path: '/m/u.safe', size_mb: 1, quant_type: 'bf16', mtime: 0 }] })
    const { result } = renderHook(() => useComponents('unet'), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(apiFetch).toHaveBeenCalledWith('/api/v1/components?role=unet')
    expect(result.current.data?.[0].abs_path).toBe('/m/u.safe')
  })

  it('store update is read back', () => {
    act(() => useComponentStateStore.getState().set('k1', 'loaded', null))
    expect(useComponentStateStore.getState().states['k1']).toEqual({ state: 'loaded', error: null })
  })
})
```

- [ ] **Step 2: run → FAIL** — `cd frontend && npx vitest run src/api/components.test.tsx` → module missing.

- [ ] **Step 3: implement** (`frontend/src/api/components.ts`):

```typescript
import { useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { create } from 'zustand'
import { apiFetch } from './client'
import { useLiveChannel } from './useLiveChannel'

export interface ComponentInfo {
  filename: string
  abs_path: string
  size_mb: number
  quant_type: string
  mtime: number
}

export type ComponentRole = 'unet' | 'clip' | 'vae' | 'loras'
export type ComponentLoadState = 'cold' | 'loading' | 'loaded' | 'failed'

/** Same format as backend component_state_key: file|device|dtype|lora_sig.
 * Base loader nodes carry no loras → trailing '|'. */
export function componentStateKey(d: { file?: string; device?: string; dtype?: string }): string {
  return `${d.file ?? ''}|${d.device ?? ''}|${d.dtype ?? ''}|`
}

export function useComponents(role: ComponentRole) {
  return useQuery({
    queryKey: ['components', role],
    queryFn: async () => (await apiFetch<{ components: ComponentInfo[] }>(`/api/v1/components?role=${role}`)).components,
    staleTime: 30_000,
  })
}

interface ComponentStateStore {
  states: Record<string, { state: ComponentLoadState; error: string | null }>
  set: (key: string, state: ComponentLoadState, error: string | null) => void
}
export const useComponentStateStore = create<ComponentStateStore>((set) => ({
  states: {},
  set: (key, state, error) => set((s) => ({ states: { ...s.states, [key]: { state, error } } })),
}))

/** Subscribe /ws/models for component_state_changed → update the store. Shared
 * connection via useLiveChannel; mount once anywhere a loader node renders. */
export function useComponentStateLiveSync(): void {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${window.location.host}/ws/models`
  const setState = useComponentStateStore((s) => s.set)
  useLiveChannel(url, {
    onMessage: (data) => {
      if (data.event === 'component_state_changed') {
        setState(String(data.component_key), (data.state as ComponentLoadState) ?? 'cold', (data.error as string) ?? null)
      }
    },
  })
}

/** Live four-state for one component key. Seeds from GET /state on mount, then
 * tracks WS pushes. Unknown → 'cold'. */
export function useComponentState(key: string): { state: ComponentLoadState; error: string | null } {
  useComponentStateLiveSync()
  const qc = useQueryClient()
  const setState = useComponentStateStore((s) => s.set)
  const entry = useComponentStateStore((s) => s.states[key])
  useEffect(() => {
    if (!key.replace(/\|/g, '')) return  // empty descriptor (no file picked yet)
    let cancelled = false
    apiFetch<{ components: { key: string; state: ComponentLoadState; error: string | null }[] }>(
      `/api/v1/models/components/state?keys=${encodeURIComponent(key)}`,
    ).then((resp) => {
      if (cancelled) return
      const row = resp.components.find((r) => r.key === key)
      if (row) setState(key, row.state, row.error)
    }).catch(() => {})
    return () => { cancelled = true }
    // qc kept in deps list for lint parity with other hooks
  }, [key, setState, qc])
  return entry ?? { state: 'cold', error: null }
}
```

- [ ] **Step 4: run → PASS** — `cd frontend && npx vitest run src/api/components.test.tsx`.

- [ ] **Step 5: commit**

```bash
git add frontend/src/api/components.ts frontend/src/api/components.test.tsx
git commit -m "feat(image): PR-5b — components API: useComponents + useComponentState (WS-driven)"
```

---

## Task 3: `component_select` widget

**Files:** Modify `frontend/src/components/nodes/DeclarativeNode.tsx`. Test: `frontend/src/components/nodes/ComponentSelectWidget.test.tsx`.

- [ ] **Step 1: failing test**

```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('../../api/client', () => ({ apiFetch: vi.fn() }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
import { apiFetch } from '../../api/client'
import { ComponentSelectWidget } from './DeclarativeNode'

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

describe('ComponentSelectWidget', () => {
  beforeEach(() => vi.clearAllMocks())
  it('lists components for the role by abs_path', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ components: [
      { filename: 'flux-unet.safetensors', abs_path: '/m/flux-unet.safetensors', size_mb: 18000, quant_type: 'bf16', mtime: 0 },
    ] })
    wrap(<ComponentSelectWidget value="" onChange={() => {}} role="unet" />)
    await waitFor(() => expect(screen.getByRole('option', { name: /flux-unet/ })).toBeInTheDocument())
    const opt = screen.getByRole('option', { name: /flux-unet/ }) as HTMLOptionElement
    expect(opt.value).toBe('/m/flux-unet.safetensors')
  })
})
```

- [ ] **Step 2: run → FAIL** — `ComponentSelectWidget` not exported.

- [ ] **Step 3: implement** — in `DeclarativeNode.tsx`, add the component import + the widget + a `component_select` case. Add near the other widget imports:

```typescript
import { useComponents, type ComponentRole } from '../../api/components'
```

Add the widget (export it for the test):

```typescript
export function ComponentSelectWidget({
  value, onChange, role,
}: { value: string; onChange: (v: string) => void; role: ComponentRole }) {
  const { data: components } = useComponents(role)
  return (
    <NodeSelect value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">选择 {role}...</option>
      {(components ?? []).map((c) => (
        <option key={c.abs_path} value={c.abs_path}>
          {c.filename}{c.quant_type && c.quant_type !== 'bf16' ? ` · ${c.quant_type}` : ''}
        </option>
      ))}
    </NodeSelect>
  )
}
```

Add the case to `WidgetRenderer`'s switch (it needs `widget.role`):

```typescript
    case 'component_select':
      return (
        <ComponentSelectWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
          role={(widget.role ?? 'unet') as ComponentRole}
        />
      )
```

- [ ] **Step 4: run → PASS** — `cd frontend && npx vitest run src/components/nodes/ComponentSelectWidget.test.tsx`.

- [ ] **Step 5: commit**

```bash
git add frontend/src/components/nodes/DeclarativeNode.tsx frontend/src/components/nodes/ComponentSelectWidget.test.tsx
git commit -m "feat(image): PR-5b — component_select widget (role dropdown from /api/v1/components)"
```

---

## Task 4: Four-state status header on loader nodes

**Files:** Modify `frontend/src/components/nodes/DeclarativeNode.tsx`. Test: `frontend/src/components/nodes/ComponentStatusHeader.test.tsx`.

- [ ] **Step 1: failing test**

```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ComponentStatusHeader } from './DeclarativeNode'
import { useComponentStateStore } from '../../api/components'

vi.mock('../../api/client', () => ({ apiFetch: vi.fn(() => Promise.resolve({ components: [] })) }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))

describe('ComponentStatusHeader', () => {
  beforeEach(() => useComponentStateStore.setState({ states: {} }))
  it('shows 未加载 (cold) by default', () => {
    render(<ComponentStatusHeader data={{ file: '/m/u.safe', device: 'cuda:1', dtype: 'bfloat16' }} />)
    expect(screen.getByText(/未加载/)).toBeInTheDocument()
  })
  it('shows 已加载 when store says loaded', () => {
    useComponentStateStore.getState().set('/m/u.safe|cuda:1|bfloat16|', 'loaded', null)
    render(<ComponentStatusHeader data={{ file: '/m/u.safe', device: 'cuda:1', dtype: 'bfloat16' }} />)
    expect(screen.getByText(/已加载/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: run → FAIL** — `ComponentStatusHeader` not exported.

- [ ] **Step 3: implement** — in `DeclarativeNode.tsx` add the import + component + render it in the node body when `declDef.componentRole` is set. Import:

```typescript
import { useComponentState, componentStateKey } from '../../api/components'
```

Component (export for test):

```typescript
const _STATE_VIS: Record<string, { label: string; color: string }> = {
  loaded:  { label: '已加载', color: 'var(--ok)' },
  loading: { label: '加载中', color: 'var(--warn)' },
  failed:  { label: '失败',   color: 'var(--accent)' },
  cold:    { label: '未加载', color: 'var(--muted)' },
}

export function ComponentStatusHeader({ data }: { data: Record<string, unknown> }) {
  const key = componentStateKey({
    file: data.file as string | undefined,
    device: (data.device as string) || 'auto',
    dtype: (data.dtype as string) || 'bfloat16',
  })
  const { state } = useComponentState(key)
  const vis = _STATE_VIS[state] ?? _STATE_VIS.cold
  return (
    <div className="flex items-center gap-1.5" style={{ fontSize: 9, color: 'var(--muted)', padding: '2px 10px 4px' }}>
      <span style={{ width: 6, height: 6, borderRadius: 3, background: vis.color, flexShrink: 0 }} />
      <span style={{ color: vis.color }}>{vis.label}</span>
    </div>
  )
}
```

In the `DeclarativeNode` body, render the header right after the opening `<BaseNode ...>` (before the widgets map), gated by `componentRole`:

```typescript
      {declDef.componentRole && <ComponentStatusHeader data={data as Record<string, unknown>} />}
```

- [ ] **Step 4: run → PASS** — `cd frontend && npx vitest run src/components/nodes/ComponentStatusHeader.test.tsx`.

- [ ] **Step 5: commit**

```bash
git add frontend/src/components/nodes/DeclarativeNode.tsx frontend/src/components/nodes/ComponentStatusHeader.test.tsx
git commit -m "feat(image): PR-5b — four-state component status header (WS-driven)"
```

---

## Task 5: Palette "组件加载" category

**Files:** Modify `frontend/src/components/panels/NodeLibraryPanel.tsx`. Test: `frontend/src/components/panels/NodeLibraryPanel.test.tsx` (if one exists; else a small new render test).

- [ ] **Step 1: failing test** (`NodeLibraryPanel.test.tsx` — if a test file exists, add a case; else create minimal):

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import NodeLibraryPanel from './NodeLibraryPanel'

describe('NodeLibraryPanel', () => {
  it('shows the 组件加载 category with loader nodes', () => {
    render(<NodeLibraryPanel />)
    expect(screen.getByText('组件加载')).toBeInTheDocument()
    expect(screen.getByText('UNET 加载')).toBeInTheDocument()
  })
})
```

> If `NodeLibraryPanel` requires props/context (store, drag handlers), wrap as the existing panel test does, or render within the minimal providers it needs. Inspect the component's props first; adapt the test to construct it correctly while keeping the two assertions.

- [ ] **Step 2: run → FAIL** — category absent.

- [ ] **Step 3: implement** — add a `NodeCategory` to `BUILTIN_CATEGORIES` in `NodeLibraryPanel.tsx`:

```typescript
  {
    name: 'image_loading',
    label: '组件加载',
    color: 'rgba(244,114,182,0.9)',
    nodes: [
      { type: 'image_unet_load', dotColor: 'rgba(244,114,182,0.9)' },
      { type: 'image_clip_load', dotColor: 'rgba(234,179,8,0.9)' },
      { type: 'image_vae_load', dotColor: 'rgba(239,68,68,0.85)' },
      { type: 'image_lora_apply', dotColor: 'rgba(168,85,247,0.85)' },
    ],
  },
```

(Place it right after the existing `image` category. The label text the palette renders comes from `NODE_DEFS[type].label`, set in Task 1.)

- [ ] **Step 4: run → PASS** — `cd frontend && npx vitest run src/components/panels/NodeLibraryPanel.test.tsx`.

- [ ] **Step 5: commit**

```bash
git add frontend/src/components/panels/NodeLibraryPanel.tsx frontend/src/components/panels/NodeLibraryPanel.test.tsx
git commit -m "feat(image): PR-5b — palette 组件加载 category (4 loader nodes)"
```

---

## Task 6: Frontend gates (tsc + vitest + build)

**Files:** none (verification).

- [ ] **Step 1: typecheck** — `cd frontend && npx tsc --noEmit` → 0 errors.
- [ ] **Step 2: unit tests** — `cd frontend && npx vitest run` → all pass (new + existing).
- [ ] **Step 3: production build** ([[feedback-preflight-lint]]) — `cd frontend && npm run build` → succeeds.
- [ ] **Step 4: commit** (only if any fix needed; else skip).

---

## Task 7: Real-app verification (vite + browser) — CHECKPOINT

> Frontend correctness is visual; vitest + build are the CI gates, but the live UX (palette → drag loader → dropdown populated → wire to image_generate → four-state header → preload flips loading→loaded over WS) must be eyeballed. Needs a backend with PR-5a on `:8000`.

**Backend for verification (per user's port note):** the production backend (PID 3529010, pre-PR-4) lacks PR-5a. Start a local backend on **master** (which has PR-5a) on `:8000` — **kill or reuse 3529010 first** to avoid the port clash. vite dev is `:9999` (local-only, proxies to `:8000`).

- [ ] **Step 1: backend up** — `git checkout master` in a clean checkout (or this branch — backend code is identical to master for PR-5a), ensure `:8000` is free (`kill 3529010` or confirm it's the one to replace), start `backend serve` (or `uvicorn`) on `:8000` with real `.env`.
- [ ] **Step 2: vite up** — `cd frontend && npm run dev` (`:9999`).
- [ ] **Step 3: drive the editor** (chrome-devtools MCP or a browse skill): open `/workflows` editor → confirm "组件加载" palette section with 4 nodes → drag `image_unet_load` → its file dropdown lists `GET /api/v1/components?role=unet` entries → status header shows "未加载" → wire unet/clip/vae loaders into `image_generate`'s new ports.
- [ ] **Step 4: live state** — `POST /api/v1/models/components/preload` (curl with the chosen unet/clip/vae abs paths) → watch the loader node headers flip 未加载→加载中→已加载 via the WS push (real GPU load; or assert the WS event arrives + store updates if no GPU is free).
- [ ] **Step 5: screenshot** the editor with the loader nodes + four-state header as evidence; note any visual issues for a follow-up design polish.

---

## Full verification + finish

- [ ] `cd frontend && npx tsc --noEmit && npx vitest run && npm run build` all green.
- [ ] superpowers:finishing-a-development-branch → push `feat/image-component-multigpu-pr5b-frontend` → PR → CI (frontend job builds; backend job unaffected) → merge on green ([[feedback-auto-merge]]).
- [ ] **Production note:** frontend changes need `cd frontend && npm run build` after merge — backend serves `frontend/dist/`, not source (CLAUDE.md).

---

## Self-Review (vs spec §6/§7)

- **§7.1 palette subcategory**: Task 5 "组件加载" category. ✓ (flat category, not nested — matches the palette's actual structure.)
- **§7.2 loader node form** (file dropdown + device + dtype + arch + status header): Tasks 1+3+4. ✓
- **§7.3 image_generate multi-input** (unet/clip/vae ports): Task 1. ✓ (model_key widget kept for legacy back-compat — backend inline-expands.)
- **§6.1 four states**: Task 4 header (loaded/loading/failed/cold with color + label). ✓
- **§6.3 useComponentState** (batch GET state + WS subscribe): Task 2. ✓
- **§6.2 preload**: triggered via the PR-5a endpoint; a UI "prewarm" button is **deferred** (per-run load + the status header already cover the core UX; add a prewarm affordance in a follow-up if desired).
- **Type consistency**: `componentStateKey` (Task 2) used by Task 4; `ComponentRole` (Task 2) used by Task 3; `component_select`/`role`/`componentRole` (Task 1 catalog) consumed by Tasks 3/4. ✓
- **Deferred (future)**: ModelsOverlay GPU 分配表单 (per-node device dropdown covers it); preload "prewarm" button; LoRA `path` on the lora_apply descriptor (frontend currently sends `lora_file`=abs_path; backend reads `lora_file`/`lora_path` — confirm PR-4 `image_lora_apply` reads the abs path the dropdown emits; if it expects a separate `lora_path`, set both in the node data).
