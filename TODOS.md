# TODOS

Project-level deferred work items captured during reviews. Each item: what / why / context / depends-on.

---

## IA 重构 v3 · 延后项（2026-04-23）

### [ ] 前端测试：非核心组件 unit test 扩展

- **What:** 给 IconRail / ServiceCard / 其他辅助组件补 Vitest unit test（SchemaDrivenForm + SchemaDrivenOutput 的 unit 已在 PR-B 里做）
- **Why:** PR-B 上了 Playwright 3 E2E + 两个核心组件 unit，其他小组件缺 unit 覆盖；未来重构时无保护
- **Context:** 本轮 plan-eng-review 的 T12 决策已把 SchemaDrivenForm/Output 从 TODO 拉进 PR-B，本条剩下的是非核心组件的覆盖
- **Depends on:** v3 PR-B 合并后

### [ ] ComfyUI `*_api.json` 导入功能

- **What:** 拖动一个 ComfyUI 社区的 `*_api.json` → 解析 → 自动建 Workflow + 发布为服务
- **Why:** 复用 ComfyUI 生态的 workflow 资产；降低搭 workflow 的门槛
- **Context:** v3 设计文档已经选择 ComfyUI 双 JSON 格式作为底层存储，兼容性天然就在。需要：(1) ComfyUI 的整数 node_id (`"391"`) → 我们的 snowflake/UUID (`"nd_xxx"`) 映射表 (2) `class_type` 白名单：能对应到我们的 node package 里的实现类
- **Depends on:** v3 PR-A 合并后（有了 workflow.auto_generated + service.workflow_snapshot 两个字段才能挂）

### [ ] 发布向导「高级模式」：任意节点任意字段暴露

- **What:** 发布向导 Step 1/3 从"只扫 Primitive/Load/Save/Preview 节点"扩展到"可手动勾选任意节点的任意 input 字段"
- **Why:** LTX 短剧这类 workflow 有中间节点（CFGGuider / KSamplerSelect）的 cfg / steps 参数也值得暴露给调用方
- **Context:** 首期 v3 为了控制复杂度只扫 I/O 节点；v2 放开后，UI 上需要一个"高级模式"折叠栏展示所有节点 + 所有 input name 的勾选树
- **Depends on:** v3 PR-A + PR-B 合并后

---

## V1.5 workflow queue + GPU scheduler · 延后项（2026-05-14）

### [ ] auto GPU topo 探测（解析 `nvidia-smi topo -m`）

- **What:** 实现解析 `nvidia-smi topo -m` 的 auto GPU 拓扑探测，与 `hardware.yaml` 互补（yaml override）。启动时探测出 NVLink group 划分，与 yaml diff 不一致告警。
- **Why:** 加卡/换卡不用手改 `hardware.yaml`；且 Pro 6000 到货后有 2 套硬件配置（双 3090 / 三卡）能真正验证跨驱动版本的解析器。
- **Context:** V1.5 plan-eng-review 决策 D15 把这个推迟——`nvidia-smi topo -m` 输出格式跨驱动版本变化（NV4/NV2/SYS/PHB/PIX），单一硬件无法验证脆弱解析器，所以 V1.5 只发 manual 模式（`hardware.yaml` 是唯一真相源）。
- **Depends on:** Pro 6000 到货 + V1.5 完成（spec: `docs/superpowers/specs/2026-05-13-workflow-queue-and-gpu-scheduler-design.md`）

### [ ] 点击历史任务恢复 workflow 到 canvas（ComfyUI 式）

- **What:** 点 TaskPanel 里一个完成的任务 → 把那次的 workflow 定义 + 输入参数加载回 canvas 编辑器，方便调一个参数重跑。
- **Why:** 快速迭代——不用从头重搭 workflow。ComfyUI 新版 queue 侧栏有这个交互。
- **Context:** V1.5 plan-design-review 决策 D10 把这个推迟。需要每个 task 存 workflow 快照；`execution_tasks` 表有 `workflow_id` 但不一定有完整快照，V1 的「重试」读原 workflow 定义。「恢复到 canvas」是前端动作 + 需快照可用。比缩略图历史重（需快照存储 schema + canvas 加载路径），可独立交付。
- **Depends on:** V1.5 TaskPanel 重构（direction A）+ workflow 快照存储设计
