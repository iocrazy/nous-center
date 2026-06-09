# 复刻 Infinite-Canvas 节点画布(样式 + 四项交互)设计

**Status**: Draft
**Author**: heygo(设计 by Claude)
**Date**: 2026-06-09

## 0. 目标

把 Infinite-Canvas(`github-repos/Infinite-Canvas` 的 `static/js/smart-canvas.js`,媒体板范式)的
**节点卡视觉 + 四项交互**复刻到 nous-center 的节点画布:

1. 节点卡样式(截图 1):圆角卡 / dotted 网格背景 / 端口圆点 / 选中环。
2. 文字输入(截图 2):PROMPT 卡,字数计数「N / 20,000」+ 提示库。
3. 打组 + 快捷键(截图 3):Ctrl+G 成组 / Ctrl+Shift+G 解组,容器框 + 计数副标题。
4. 拖拽选择(截图 4):框选多选,洋红选框。
5. 分组自适应(截图 5):框随成员内容自动适配尺寸。

用户拍板两条:**(1) 增强现有 React Flow 画布**(非新页);**(2) 全部节点类型统一换 Infinite-Canvas 卡片风**。

## 1. 关键事实(已读双方真代码)

### 1.1 落地画布 = `NodeEditor`,不是 `Studio.tsx`

- `frontend/src/components/nodes/NodeEditor.tsx` 是真正的节点画布(React Flow v12 `@xyflow/react ^12.10.1`)。
- `frontend/src/pages/Studio.tsx` 是表单式「创作控制台」(左导航 + 面板),**非画布**,本次不改。

### 1.2 nous-center 已具备的能力(不重写,只扩展)

| 能力 | 现状位置 |
|---|---|
| 端口拖空白 → 建相连节点菜单 | `NodeEditor.onConnectEnd` + `NodeCreateMenu` + `spawnConnectedNode` |
| Ctrl+G 分组 | `NodeEditor.groupSelected`(右键菜单也有「分组」) |
| 复制/粘贴/原地复制/旁路 快捷键 | `NodeEditor` keydown handler |
| 分组框拖头跟随移动 | `GroupLayer.tsx`(按 position 落框内隐式判定成员) |
| 端口类型色 | `components/nodes/portColors.ts` `PORT_TYPE_COLORS` |
| 文本 IME portal 编辑 | `components/nodes/TextareaPortalEditor`(Linux fcitx 坑) |

**缺的正是四张截图本身**:Infinite-Canvas 视觉、PROMPT 卡、框选、分组自适应/解组/计数副标题。

### 1.3 Infinite-Canvas 对应实现(已读 `smart-canvas.js`)

- 端口 `.node-port` 14px 圆点,默认 `opacity:0`,hover/选中显示,`scale(1.22)` 高亮。
- 卡片 16px 圆角,默认阴影 `0 12px 34px rgba(15,23,42,.08)`,选中 `0 0 0 1px var(--strong)` 环。
- 网格背景:`radial-gradient(var(--grid) 1px, transparent 1px)` `background-size:24px 24px`。
- 框选:`Ctrl/Cmd+drag`(或 `R+drag`)拉框,AABB 相交即选;选框 `.selection-box` 描边。
- 分组:`groupSelectedNodes()` Ctrl+G / `ungroupNode()` Ctrl+Shift+G;自适应用 `groupImageGridLayout()`
  按显式尺寸算缩略图网格(Infinite-Canvas 是「组拥有图片数组」范式)。

## 2. 设计决策

### 2.1 端口策略——折中,不照搬「完全隐藏」

Infinite-Canvas 端口默认隐藏。但 nous-center 是多端口强类型工作流,完全隐藏伤可用性。
**采用折中**:端口常驻但默认低调(小圆点 + 类型色,标签弱化),hover/选中放大高亮 + 显标签。
卡片其余视觉(圆角 / 头部 / dotted 网格 / 选中环)完全对齐 Infinite-Canvas。

### 2.2 分组成员模型改为显式 `nodeIds`

`WorkflowGroup` 增加可选 `nodeIds?: string[]`(向后兼容)。Ctrl+G 写入选中节点 id;
自适应与拖动按 `nodeIds` 的**实测包围盒**(`node.measured`)计算,取代现有「按 position 落框内」隐式判定
(节点拖出后判定失效)。

### 2.3 框选用 React Flow 原生,不手写

对齐截图提示「拖拽画布移动,Ctrl 框选多选」:`selectionOnDrag` + `selectionKeyCode=['Control','Meta']` +
`SelectionMode.Partial`(相交即选)+ `panOnDrag`(左键平移)。比 Infinite-Canvas 手写画布事件更稳健。
多选拖动 React Flow 原生支持。

## 3. 实施(4 个独立 PR,每 PR 一分支,过 CI 绿后自动合)

### PR-1 — 节点卡视觉统一换风(截图 1)
- `BaseNode.tsx`:卡片圆角 8→16、阴影/选中环对齐;头部标题大写字重;端口圆点 14px、默认低调、
  hover/选中放大 + 显标签(保留 `PORT_TYPE_COLORS`)。
- `NodeEditor.tsx`:`<Background variant="dots" gap={24}>`;画布背景对齐网格。
- 全局 CSS:`.react-flow__node` hover/选中、端口显隐。暗/亮双主题都验。

### PR-2 — PROMPT 文字输入节点(截图 2)
- 文本输入节点(`TextInputNode.tsx` / `text_input`)重做成 Infinite-Canvas PROMPT 卡:
  头部 `PROMPT`、占位「输入提示词...」、**字数计数「N / 20,000」**、**提示库 pill**(lucide `library`,无 emoji)。
- 保留 `TextareaPortalEditor`(IME),只换壳。
- 计数/提示库筛选抽纯函数 + 单测。提示库内容参考 `Infinite-Canvas/static/system-prompts/infinite-canvas-prompt-templates.md`,落 nous 本地常量。

### PR-3 — 框选(截图 4)
- `NodeEditor.tsx`:加 `selectionOnDrag` / `selectionKeyCode` / `selectionMode=Partial` / `panOnDrag`。
- 选框视觉:`.react-flow__selection` / `.react-flow__nodesselection` 改洋红/强调描边。
- 底部提示条:「拖拽画布移动,Ctrl 框选多选,拖动选中节点可一起移动」。

### PR-4 — 打组增强:解组 + 计数副标题 + 自适应(截图 3 + 5)
- 模型:`models/workflow.ts` `WorkflowGroup` 加 `nodeIds?`;`stores/workspace.ts` add/update 透传。
- `NodeEditor.tsx`:`groupSelected` 写 `nodeIds`,包围盒用 `getNodes().measured`;新增 **Ctrl+Shift+G 解组**。
- `GroupLayer.tsx`:容器视觉(半透明圆角 16 + `GROUP` 标签 + **副标题「N张图片 · M个提示词 已成组」**);
  **自适应**(订阅节点位移/缩放,RAF/debounce 重算 group 矩形包住成员);拖组头按 `nodeIds` 移动成员。
- 分类计数(图片节点 vs 提示词节点,依据 `NODE_DEFS` 端口类型 / category)抽纯函数 + 单测。

## 4. 验证(每 PR)

- **Preflight**:`cd frontend && npm run lint && npm run build && npm test`。
- **真机 chrome-devtools**(`reference_reactflow_devtools_verify`):PR-1/2/4 肉眼 + 合成事件可点处验;
  **框选/连接拖拽合成事件验不了 → 靠单测**验相交/包围盒纯函数。
- **CI**:合前显式确认 Frontend job 绿(仓库无 required-check)。

## 5. 风险

- 端口 hover-reveal 折中若用户更想要「完全隐藏」,PR-1 可调(决策 2.1)。
- dotted 网格 + 浅卡在暗主题需调(Infinite-Canvas 原图是亮主题),双主题都过目。
- 自适应重算 throttle(RAF/debounce),避免拖动每帧 `setWorkflow` 抖动(参考 GroupLayer 现有 `zoomRef` 模式)。
