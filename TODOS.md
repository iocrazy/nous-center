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
