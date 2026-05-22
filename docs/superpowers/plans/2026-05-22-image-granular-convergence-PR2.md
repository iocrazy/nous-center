# Image 细粒度图收敛 — PR-2(前端:flux2 loader 接 component_select + 四态 + device)Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development / executing-plans。checkbox 跟踪。

**Goal:** 让 `flux2-components` 的三个 loader(Load Diffusion / CLIP / VAE)在编辑器里真正显示 PR-1 已加的 `file`(component_select 组件下拉)+ `weight_dtype` + `device`(仅 Diffusion)控件 + **四态加载头**(loaded/cold/loading/failed),复用 PR-5b 已建的声明式渲染。

**根因(已查实):** `/api/v1/nodes/definitions` 直接返回 node.yaml 原始 dict,**已含** `componentRole`(节点级)+ widget `role`。前端 `loadPluginDefinitions()`(nodeRegistry.ts)注册插件节点时:widget `role` 因 `as WidgetDef[]` cast **已保留**;但 **`componentRole` 没拷进 `DECLARATIVE_NODES`** → `DeclarativeNode.tsx:554` 的 `{declDef.componentRole && <ComponentStatusHeader/>}` 不触发 → 四态头不显示。component_select widget(`DeclarativeNode.tsx:368`,用 `widget.role`)本就能渲染。

**范围:** 纯前端一处透传 + 类型补全 + 真机验证。**不含**:device 特殊处理(普通 select 已通用渲染)、PortType 收敛(Family B 还在,留 PR-4 删 Family B 时一起)。

**Tech:** React + TS;CI = `tsc -b && vite build` + `vitest run`。

**Branch:** `feat/image-granular-convergence-pr2`。

**Spec:** `2026-05-21-image-granular-convergence-design.md` §5.1/§5.3。

---

## Task 1: loadPluginDefinitions 透传 componentRole

**Files:**
- Modify: `frontend/src/models/nodeRegistry.ts`(`PluginNodeDef` 接口 + `loadPluginDefinitions` 注册体)
- Test: `frontend/src/models/nodeRegistry.test.ts`(新建,或并入现有)

- [ ] **Step 1: 失败测试** —mock `/api/v1/nodes/definitions` 返回一个带 `componentRole: 'unet'` + widget `{name:file, widget:component_select, role:unet}` 的插件节点;调 `loadPluginDefinitions()`;断言 `DECLARATIVE_NODES[type].componentRole === 'unet'` 且该 widget 的 `role === 'unet'`。

```ts
// 关键断言
global.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({
  flux2_load_diffusion_model: {
    label: 'Load Diffusion Model', category: 'image', badge: 'Loader', badgeColor: 'x',
    componentRole: 'unet',
    widgets: [{ name: 'file', label: '文件', widget: 'component_select', role: 'unet' }],
    outputs: [{ id: 'model', type: 'MODEL', label: 'MODEL' }],
  },
}) })
await loadPluginDefinitions()
expect(DECLARATIVE_NODES['flux2_load_diffusion_model'].componentRole).toBe('unet')
expect(DECLARATIVE_NODES['flux2_load_diffusion_model'].widgets[0].role).toBe('unet')
```

- [ ] **Step 2: 跑确认失败**(componentRole undefined)。`cd frontend && npx vitest run src/models/nodeRegistry.test.ts`
- [ ] **Step 3: 实现** —
  - `PluginNodeDef` 接口加 `componentRole?: 'unet' | 'clip' | 'vae'`。
  - 注册体加 `componentRole: def.componentRole`:
    ```ts
    DECLARATIVE_NODES[nodeType] = {
      type: nodeType, label: def.label, category: def.category,
      badge: def.badge, badgeColor: def.badgeColor,
      widgets: (def.widgets ?? []) as WidgetDef[],
      componentRole: def.componentRole,   // ← 新增透传(四态头靠它)
    }
    ```
- [ ] **Step 4: 跑确认通过** + `npx tsc -b`。
- [ ] **Step 5: Commit** `feat(image): PR-2 — loadPluginDefinitions 透传 componentRole(flux2 loader 四态头)`

---

## Task 2: 预检 + 真机验证(vite)

- [ ] **Step 1: 全量前端检查** `cd frontend && npx tsc -b && npx vitest run && npm run build`(遵守 feedback_preflight_lint)。
- [ ] **Step 2: 真机验证(feedback_verify_real_model)** —本地起 backend(:8000,gate-off)+ vite(:9999),在 workflow 编辑器:
  - 拖 Load Diffusion Model → 见 **文件下拉(component_select,真组件列表)+ 精度 + 显卡 + 架构** + **四态头**(默认「未加载/cold」)。
  - 拖 Load CLIP / Load VAE → 各见 文件 + 精度 + 四态头。
  - 零 console error。截图留证。
- [ ] **Step 3:** 把验证结论(截图/状态)写进 PR 描述。

---

## 收尾
- [ ] 开 PR `feat/image-granular-convergence-pr2` → CI(tsc+vite+vitest 前端、后端不变)绿 → auto-merge。
- [ ] 过渡期:Family B 仍在(PR-4 删),两套图像节点并存正常。
