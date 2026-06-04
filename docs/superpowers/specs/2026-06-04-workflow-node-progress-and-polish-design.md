# 工作流节点逐组件进度/时长 + 连接 & 节点展示精修

**Status**: Draft
**Author**: heygo (设计 by Claude)
**Date**: 2026-06-04

## 0. 诉求与现状

### 用户诉求
1. **逐组件(节点)单独进度 + 时长**:Load CLIP / KSampler / VAE Decode 各自显示自己阶段的进度条 + 耗时(text encode / denoise / vae decode 分别多久)。
2. **连接(edge)美化**:当前是 React Flow 默认细灰虚线,弱。要实线 + 按数据类型着色 + hover 高亮 +(运行时)流动感。
3. **节点(node)展示精修**:间距/对齐/四态/进度区/端口标签等优化。**nous 自有风格**(不照搬 ComfyUI,保持简洁深色)。

### 现状(确认自源码)
- **后端进度数据已齐**:`protocol.py:143 NodeProgress` 带 `node_id` / `node_type` / `stage`(候选 `text_encode` / `dit_denoise` / `vae_decode` / `tts_synth` / `llm_gen`)/ `step` / `total` / `step_latency_ms` / `eta_ms` / `preview_url`。`image_modular.py` infer 经 `ProgressTracker` 已发这三 stage(`pt.stage("text_encode")` line 700;`pt.step(..., stage="dit_denoise")` line 751;pipe 后 `vae_decode`)。
- **但 stage 全挂在末端节点**:flux2 整条链(load→encode→denoise→decode)在 **VAE Decode 节点 dispatch 的一次 infer()** 里执行(runner `_build_request` 摊平),所以 `NodeProgress.node_id` = vae_decode 节点。前端 `stores/execution.ts` 的 `currentNodeId` / `currentNodeStage` / `currentNodeProgress` / `currentNodeStepLatencyMs` / `currentNodeEtaMs` 都只反映「当前在 vae_decode 节点 + stage=X」,**没分发到 Load CLIP / KSampler 各自节点**。
- **已有 `NodeDenoiseProgress` 组件**(`NodeDenoiseProgress.test.tsx`):KSampler 节点已能显示 denoise 逐步进度 + preview。→ 本 spec **扩展**它到 text_encode→CLIP / vae_decode→VAE Decode,不新造轮子。
- **edge 无自定义组件**:用 React Flow 默认 edge(细灰虚线)。
- **node 渲染**:`DeclarativeNode.tsx` 内联 style(深色卡片 + badge + widget 行 + 四态 header)。
- **Load 节点惰性**:`exec_load_clip/vae` 只产描述符(~0ms,不计算)。「Load CLIP 节点时长」语义 = infer 内 text_encode stage 的耗时(逻辑映射,非节点自执行)。

### 关键洞察
进度数据后端全有,**这是纯前端工作**:把 `NodeProgress` 按 `stage` 映射到工作流图里对应类型的节点(前端有图拓扑),更新该节点的进度/时长,而非只挂末端节点。

## 1. 目标 / 非目标 / 成功标准

### 目标
- **stage→节点映射**:`text_encode`→Load CLIP / `dit_denoise`→KSampler / `vae_decode`→VAE Decode 节点。每节点显示自己 stage 的进度条 + 耗时(denoise 含逐步 + ETA + preview;encode/decode 是阶段态 + 耗时)。
- **连接精修**:自定义 edge —— 实线 + 按 PortType 着色 + hover 高亮 + 运行时沿活跃边流动动画。
- **节点精修**:标题/间距/对齐统一;四态 header + 进度区视觉协调;端口标签清晰。nous 风格。
- **零功能回归**:不碰 GPU 计算 / runner 协议;纯前端渲染 + store 分发。

### 非目标
- 不改后端进度协议(NodeProgress 已够)。
- 不改 runner 摊平执行(load→decode 仍一次 infer)。
- 不做多图并发 / 模型并行(沿用单 runner 串行)。
- 不照搬 ComfyUI 视觉(nous 自有风格;ComfyUI 源仅作交互参考)。

### 成功标准
- 真机 chrome-devtools 核(`project_realmachine_ui_audit` 方法论 —— 进度跨进程可见性是复发坑):跑一次 flux2 出图,**Load CLIP→KSampler→VAE Decode 依次亮起各自进度**,KSampler 显示逐步 + ETA,各节点收尾显示耗时(ms)。
- 连接实线着色 + hover 高亮 + 运行时活跃边动画,视觉核过。
- `tsc` + `vite build` + vitest 绿;每 PR 独立绿。

## 2. 设计

### 2.1 进度 stage→节点映射(PR-1)
- `stores/execution.ts`:从「单 currentNode + stage」改为 **`nodeStages: Record<nodeId, {stage, progress, step, total, stepLatencyMs, etaMs, previewUrl, elapsedMs, state}>`**。收到 `NodeProgress(stage=S)` 时:
  - 查当前工作流图(nodes),找 `node_type` 匹配 S 的节点(`STAGE_TO_NODE_TYPE = {text_encode: 'flux2_load_clip'|'flux2_encode_prompt', dit_denoise: 'flux2_ksampler', vae_decode: 'flux2_vae_decode'}`)。
  - 更新该节点条目;阶段切换时给上一 stage 节点打 `done` + 记 `elapsedMs`(由 stage 首末时间戳算,或 step_latency 累加)。
  - 映射歧义(同类型多节点)→ 取该类型**唯一**节点;多于一个则 fallback 到 dispatch 节点(打 log,不静默)。
- `text_encode` / `vae_decode` 是单点阶段(非逐步)→ 节点显示「进行中…」spinner + 完成后耗时;`dit_denoise` 逐步 → 复用 `NodeDenoiseProgress`(进度条 + step x/N + ETA + preview)。
- **耗时来源**:阶段进入/退出时间差(前端按 stage 变化打点),或后端 `step_latency_ms` 累加(denoise)。优先后端 latency(准),前端打点兜底 encode/decode。

### 2.2 连接美化(PR-2)
- 自定义 React Flow edge 组件 `PortTypedEdge`:
  - **实线** bezier(去默认虚线)。
  - **按 PortType 着色**:复用端口颜色(MODEL/LATENT/CONDITIONING/CLIP/VAE/IMAGE/文本/音频各色)。edge stroke = source handle 的类型色。
  - **hover 高亮**:加粗 + 提亮 + 可选 tooltip(类型名)。
  - **运行时流动**:当前活跃节点的**入边**用 `stroke-dasharray` + `animate`(沿边流动),跑完转静态实线。活跃边由 execution store 的 `nodeStages` 推。
- 选中/相关边高亮(选中节点时其连边提亮)。

### 2.3 节点展示精修(PR-3)
- 统一标题栏(badge 色条 + 节点名 + 四态 header 对齐),widget 行间距/标签宽度统一。
- 四态 header(PR-C 已修口径)+ 本 spec 进度区协调:未运行显四态,运行中显 stage 进度。
- 端口标签(MODEL/LATENT…)字号/位置/对比度精修。
- 节点 hover/选中态(边框/阴影)统一。
- nous 风格:深色 + 现有 `--accent`/`--ok`/`--warn` 变量,不引入 ComfyUI 配色。

## 3. PR 拆分(每个独立分支 + CI + 真机核)
- **PR-1 进度 stage→节点映射**(核心):execution store `nodeStages` + STAGE_TO_NODE_TYPE 分发 + 节点进度/时长 UI(扩展 NodeDenoiseProgress)。真机核三节点依次亮 + 耗时。
- **PR-2 连接精修**:`PortTypedEdge`(实线+类型色+hover+运行时流动)。真机核视觉 + 活跃边动画。
- **PR-3 节点精修**:DeclarativeNode style 统一(标题/间距/端口/四态+进度协调)。真机核视觉。
> 落序 PR-1 → PR-3 → PR-2(进度先通,节点容纳进度区,边动画依赖活跃节点态)。各 PR 纯前端,tsc+vite+vitest 绿 + chrome-devtools 真机核。

## 4. 风险
- **进度可见性是复发坑**([[project_workflow_ui_bugs]]:节点高亮错位/RUNNING 无进度/跨进程子进程 vs 主进程边界)。stage→节点映射必须真机核(单测 mock 看不出跨进程口径)。
- **stage→node_type 映射歧义**:工作流可能多个同类型节点(多 CLIP?);本 spec 取唯一节点,多个 fallback dispatch 节点 + log。
- **edge 运行时动画性能**:大图多边动画;只对活跃边动画(非全图),避免卡顿。
- **text_encode/vae_decode 耗时精度**:单点阶段无逐步 latency,靠前端打点;denoise 用后端 step_latency 累加(准)。
