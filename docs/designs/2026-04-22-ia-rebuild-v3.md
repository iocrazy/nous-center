---
status: SHIPPED
supersedes: (architectural decisions after api-gateway lanes A–G merged on master)
shipped_via:
  - PR #8 (PR-A backend + migration + frontend stubs) — merged 2026-04-23
  - PR #9 (PR-B frontend pages + dialogs + Vitest unit + e2e + manual-gate hotfixes) — merged 2026-04-23
  - PR #10 (docs SHIPPED 标记) — merged 2026-04-23
  - PR #11 (manual-gate 3 个 bug 的回归测试：migration PG / navigate / mutation deps) — merged 2026-04-23
  - PR #12 (m08 完整 workflow 列表页) — merged 2026-04-23
  - PR #13 (m13 用量统计页 + 3 个 backend 聚合端点) — merged 2026-04-23
  - PR #14 (UsagePage vitest 与其他页面对齐) — merged 2026-04-24
  - PR #15 (m02/m08/m13 与 mockup 对齐：split-button、卡片 footer、category 区分色、+新建 Workflow 卡片、导出 CSV) — merged 2026-04-24
---

# Design: IA 重构 v3 — 单人管理员控制台

Generated during mockup-review iteration on 2026-04-22
Branch: docs/ia-rebuild-v3-design
Repo: nous-center
Related gstack artifact: `~/.gstack/projects/iocrazy-nous-center/heygo-master-design-20260422-111743.md` (v3 addendum at top)
Related mockups: `~/.gstack/projects/iocrazy-nous-center/designs/ia-rebuild-20260422/` (13 张 HTML + index)

## TL;DR

前一版（v2）把 IA 按"公开 SaaS 多用户"设计出来：能力中心 / Apps / 演示 Agent / 独立 Playground 并列做一级入口。画完 16 张 mockup 审核时发现定位错了 — nous-center 实际是**单人管理员控制台**，对外只暴露 API。本次修订（v3）把所有面向公开用户群的概念砍掉、术语统一，收敛成 4 层简洁架构。

| 指标 | v2 | v3 |
|---|---|---|
| Rail 一级入口 | 14 | 8 |
| 核心概念数 | 引擎 / 实例 / App / Agent / Workflow | **引擎 / Workflow / 服务 / API Key**（4 个） |
| Mockup 张数 | 16 + index | 13 + index |
| IA 分组 | 能力中心 / 体验中心 / 编辑器 / 管理 | 首屏 / 服务 / 编辑器 / 管理 |
| Playground | 独立一级入口 | 内嵌服务详情 tab |
| Apps | 独立一级入口 | 合并进"服务"（术语统一） |

## 定位修正

**v2 假设：** Web UI 要服务"普通用户 / 开发者 / 管理员"三种角色，有"访客打开体验页试玩"场景。

**v3 实际：**
- Web UI 的唯一使用者 = admin（heygo 自己）
- "外部用户" = 调 API 的程序（自己的 mediahub / 朋友的脚本 / 其他 AI 应用）
- 没有"浏览器访客"这个场景

这个修正直接删掉一层面向公开用户群的设计：
- 演示 Agent（不需要给访客看 demo）
- Apps 独立一级（就是"服务"的同义词）
- 能力中心独立一级（只是服务的 category 筛选视图）
- 独立 Playground（合并进服务详情）

## 核心架构（4 层）

```
[引擎]          模型文件 + GPU 占用（内部，不对外）
    ↓ 被使用
[Workflow]      DAG 蓝图。ComfyUI 风格双格式：
                  - 编辑态：full JSON (nodes + links + 坐标 / widget 值)
                  - 运行态：api JSON ({node_id: {inputs, class_type, _meta}})
    ↓ 发布（冻结快照 + I/O schema）
[服务]          对外可调用的单元
                术语统一：服务 = 实例 = App，同一概念同一张表
                字段：name / endpoint / schema / 配额 / 版本 / 授权
    ↑ 通过 Grant 授权
[API Key]       唯一外部鉴权凭据
                Key → N 条 Grant → N 个服务（M:N）
                每条 Grant 独立：配额、过期、启用
```

**要点：**
- "服务" 是唯一对外可见的 unit。所有外部调用必须先挑一个服务。
- 每个服务背后**必有**一个 workflow（可能是 1-3 节点的 trivial workflow，也可能是几十节点的 DAG）。
- workflow 和服务解耦：改 workflow 不影响已发布服务（快照冻结），需要显式"发布新版本"产生 v2。

## 服务的两条创建路径

| 路径 | 入口 | 流程 | 适用 |
|---|---|---|---|
| **快速开通** | 服务列表页 → "+新建服务 → 快速开通" | 选引擎 + 填参数 → 系统后台自动生成 trivial workflow → 得服务 | 单步调用：LLM chat / 单纯 TTS / 单纯 VL |
| **从 Workflow 发布** | Workflow 列表卡片 → "发布为服务" | 选输入节点 + 选输出节点 + 填 schema → 发布 | 多步流程：LTX 短剧配音 / 图像 pipeline / 播客生成 |

**两条路径产出同一种对象** — 都是 `services` 表的一行。简单服务对应的 trivial workflow 默认在 Workflow 列表里不显示（`auto_generated: true` 字段过滤），避免列表被系统生成的噪音淹没。

## Workflow 节点寻址规范（采自 ComfyUI）

**决策：** 采用 ComfyUI 的 `*_api.json` 结构作为服务发布快照的存储格式。

### 为什么选 ComfyUI 格式

参考文件（user 提供）：
- `LTX-2.3配音术升级版：输入台词就能生成短剧！FishAudio S2 + LTX-2.3实战.json` (105KB, 编辑态)
- `LTX-2.3..._api.json` (15KB, 运行态)

两种格式的职责：
- **编辑态 JSON** — `nodes[]` + `links[]` + 坐标 / 尺寸 / widget 定义，打开画布能继续编辑
- **API JSON** — 扁平 dict 按节点 ID keyed：`{ "269": { inputs: {...}, class_type, _meta } }`，发布服务时冻结这份

### 节点 ID 规则

- 节点 ID = workflow 内稳定的整数（`last_node_id` 单调递增，**删除节点不回收编号**）
- 节点内字段用 name 寻址
- 任意"点位"可用三元组唯一定位：`(workflow_id / snapshot_id, node_id, input_name)`

### 连线表达

```json
"341": {
  "inputs": {
    "video": ["377", 0]   // 来自节点 377 的 output slot 0
  },
  "class_type": "SaveVideo"
}
```

### 服务发布时暴露的 Schema 结构

```typescript
Service {
  id, name, endpoint, version, status
  source: {
    workflow_id: "wf_xxx",
    snapshot: { ... },              // 冻结的 api JSON
    snapshot_hash: "sha256:...",    // 防篡改
  }
  inputs: [
    {
      key: "prompt",                // 外部字段名
      label: "台词脚本",
      node_id: 391,                 // 映射到快照里哪个节点
      input_name: "value",          // 哪个 input 字段
      type: "string_multiline",
      required: true,
      default: "...",
      constraints: { min, max, enum }
    },
    ...
  ]
  outputs: [
    { key, label, node_id, output_slot, mime }
  ]
}
```

### 兼容性收益（非首期）

ComfyUI 社区的 `*_api.json` 可以直接导入 nous-center 发布成服务。生态复用的杠杆。

## 对外调用契约

| 服务类型 | endpoint 形式 | body | 触发 |
|---|---|---|---|
| LLM chat | `POST /v1/chat/completions` | OpenAI 格式 + `model` = 服务 name | 快速开通的 LLM 服务默认走这条 |
| 其他（TTS / VL / 视频 / 自定义 workflow） | `POST /v1/apps/{service_name}/{action}` | JSON 或 `multipart/form-data` | 任何从 workflow 发布的服务 |

两种形式都走 `ApiKeyGrant` 鉴权 → 查 Grant → 执行对应 workflow 快照。

## IconRail 终版（8 主 nav + 主题 3 + 设置）

```
N (logo)
├─ Dashboard                概览
├─ ─sep─
├─ 服务                     核心入口（列表 / 详情 / 内嵌 Playground）
├─ Workflow                 DAG 编辑器
├─ ─sep─
├─ 引擎库                   模型文件管理（右键菜单加载/自动加载/GPU 分配/删除）
├─ API Key                  Key + Grant 管理
├─ 用量                     统计
├─ 日志                     timeline
├─ (spacer)
├─ 主题 × 3                 浅色 / 深色 / 跟随系统
├─ ─sep─
├─ 设置                     账号 / 外观 / 通知 / 引擎默认 / Workflow 节点包 / 限流 / 数据 / 开发者
└─ ● GPU                    状态指示
```

## 删除 / 合并 / 降级清单（相对 v2）

| v2 概念 | v3 处置 | 理由 |
|---|---|---|
| 能力中心（LLM / TTS / VL / Apps 四个一级） | **删除 IA 层**，合进服务页做 category tab | 只是 `services.category` 字段的筛选视图，不值一级入口 |
| Apps 作为一级入口 | **合并进服务**，术语统一 | App = 服务 = 实例，同一概念 |
| 独立 Playground 页 | **合并进服务详情 Playground tab** | 每个服务详情含 5 tab：总览 / Playground / API 文档 / Key 授权 / 用量 |
| 演示 Agent（Settings 子页） | **整个删除** | 单人管理员不需要 demo 给访客看 |
| 服务开通一级入口 | **降级为服务页的弹窗向导** | 从服务页 `+新建服务 → 快速开通` 打开 |
| 节点包一级入口 | **降级为 设置 → Workflow 节点包** 子页 | 低频操作 |

## Mockup 清单（13 张 + index）

所有文件路径：`~/.gstack/projects/iocrazy-nous-center/designs/ia-rebuild-20260422/`

| 文件 | 内容 | 状态 |
|---|---|---|
| `index.html` | 总览 + 架构图 + 变更说明 | v3 新 |
| `m04-dashboard.html` | 概览 | 保留 |
| `m02-services.html` | **服务列表（核心）** | v3 新 |
| `m03-service-detail.html` | **服务详情 · 5 tabs · 内嵌 Playground** | v3 新 |
| `mockup-1-admin-activation.html` | 快速开通向导（服务页弹窗） | 改名 |
| `m08-workflow-list.html` | Workflow 列表（卡片底部关联服务 + 再次发布） | 改 |
| `m09-workflow-canvas.html` | Workflow 画布 | 保留 |
| `m11-engines.html` | 引擎库（卡片 grid + 右键菜单） | v2 已重做 |
| `m10-api-keys.html` | API Key · Grant 目标统一为服务 | 改 |
| `m13-usage.html` | 用量统计 | 保留 |
| `m14-logs.html` | 日志 | 保留 |
| `m15-task-panel.html` | 任务面板抽屉 | 保留 |
| `m16-settings.html` | 设置（节点包作子页） | 保留 |
| `m12-packages.html` | 节点包（设置子页） | 保留 |

**已删除的 v2 mockup：**
- `mockup-2-playground.html`（消化进 m03 Playground tab）
- `mockup-3-capabilities-llm.html`（消化进 m02 服务列表）
- `m05-tts.html` / `m06-vl.html`（同上）
- `m07-apps.html`（App = 服务，不独立）

## 给 /plan-eng-review 的开放问题

实现前要锁定这些决策。

### Q1. `services` 表要复用现有 `service_instances` 吗？

- v2 的 `service_instances.source_type ∈ (model | workflow | app | preset)` 是漏抽象
- v3 想统一成"服务 = 带 workflow 快照的对外端点"
- 要不要新建 `services` 表 + migration 把 v2 数据回填？还是直接在 `service_instances` 上加 `workflow_snapshot_id / input_schema / output_schema` 列？
- Migration 策略：老 service_instances 怎么生成它背后的 "auto-generated workflow"？

### Q2. Workflow 节点 ID 稳定性保证

- 创建时如何分配（自增 last_node_id？还是 UUID 转整数？）
- 删除节点后 ID **不复用**是硬要求（否则已发布服务的 schema 中 `node_id=391` 会指到错的节点）
- 如果 workflow 改了但不重新发布，服务继续跑冻结快照**不升级**（OK）
- 要不要在服务详情页提示 "源 workflow 比 v1 快照多出 3 个节点，要不要发布 v2"？

### Q3. 服务 endpoint 命名规则

- LLM 走 OpenAI 兼容 `/v1/chat/completions?model=xxx` — `model` 字段 = 服务 name
- 其他走 `/v1/apps/{service_name}/{action}` — name 的合法字符集（小写 + 数字 + `-`）？
- 冲突检测：同名服务能存在吗（比如 v1 暂停、v2 运行）？建议 name 唯一，版本内化在 `service.version`
- 重命名策略：已发布后允许改 name 吗？（改会破坏调用方，建议不允许，只允许 deprecate + new name）

### Q4. Grant 粒度 + 配额 / 计费

- 当前每条 Grant 独立挂一个 ResourcePack
- v3 下要不要允许"跨服务共享一个 ResourcePack"（一个客户的总 token 预算）？
- 过期时间：Key 级 vs Grant 级优先级？(Key 先到 = Key 禁，哪条先到哪个先生效)
- 配额单位：LLM 用 token，TTS 用字符数/时长，视频用调用次数/GPU·h — 每类服务声明自己的 unit，ResourcePack 带 unit 字段

### Q5. 快速开通向导的 trivial workflow 存储

- 选 LLM 引擎 + 填 system prompt + 温度 → 系统建 3 节点 workflow：PrimitiveString → LLMChat → Output
- 这个 auto-generated workflow 默认**不在 m08 workflow 列表里显示**（列表带 `auto_generated` filter，默认 false）
- 在服务详情"源 Workflow"链接可以点进去看/改（改完发布产生服务 v2）
- DB 字段：`workflow.auto_generated: bool` + `workflow.generated_for_service_id: FK`

### Q6. 发布向导的节点自动识别范围

- 首期扫出 `Primitive* / Load* / Save* / Preview*` 节点作为 I/O 候选
- 其他节点的字段（如 CFGGuider 的 cfg 值）要不要也可以手动勾选暴露？
- 建议：**首期只扫 Primitive/Load/Save**，v2 再加"高级模式：任意节点任意字段暴露"

## 实施前的文件级影响评估（供 /plan-eng-review 扩展）

### Backend

```
backend/src/models/
  service.py                 (新/改: services 表 or service_instances 扩展)
  workflow.py                (加 snapshot_json, auto_generated, generated_for_service_id)
  workflow_snapshot.py       (新: 冻结的 api_json 存储)

backend/src/api/routes/
  services.py                (改: CRUD 两路径入口统一)
  workflow_publish.py        (新: 发布 workflow 为 service 的事务)
  openai_compat.py           (改: dispatch 从 service 拿 workflow snapshot 执行)
  generic_apps.py            (新: /v1/apps/{name}/{action} 路由)

backend/migrations/
  2026-04-??-services-v3.sql (数据迁移 + 表结构)
```

### Frontend

```
frontend/src/pages/
  ServicesList.tsx           (新: m02 服务列表)
  ServiceDetail.tsx          (新: m03 详情，含 5 tabs)

frontend/src/components/
  services/CreateServiceDialog.tsx    (新: 快速开通向导)
  workflow/PublishDialog.tsx          (新: 发布为服务向导)
  playground/SchemaDrivenForm.tsx     (新: 从 service schema 自动渲染表单)
  playground/SchemaDrivenOutput.tsx   (新: 按 output schema 渲染 video/audio/text 预览)
  IconRail.tsx                        (改: 精简到 8 主 nav)

frontend/src/api/
  services.ts                (新: React Query hooks for services CRUD + publish)
  workflow-publish.ts        (新: 发布相关 hooks)
```

## 后续计划

1. **本 PR 合入**：v3 设计文档落 git
2. **另开 session 跑 `/plan-eng-review`**：基于本文档 + 6 个开放问题，锁定实现 plan
3. **再另开 session 开始实现**：按 plan 走，建议先做骨架（数据模型 + API 契约 + 1 张关键页 m02）跑通最简闭环

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 10 issues, 2 critical gaps (migration 幂等性 + deferred() perf), 0 unresolved |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |
| Outside Voice | Claude subagent | Cross-model challenge | 1 | issues_found | 11 findings (9 采纳 → PR-A/PR-B · 2 拒绝 · 用户 sovereignty) |

**CODEX:** Codex CLI 本地配置有格式错（`~/.codex/config.toml` `tui.alternate_screen`），回落 Claude 子 agent 做独立评审。输出已合入本文档 "Review Findings" 追加。

**CROSS-MODEL:**
- 最根本的分歧：Outside voice 认为"不该合表，两表保留 + 路由层 normalize" — 用户坚持 v3 合表（方案 C）。User sovereignty 记录在案。
- Outside voice 认为 "SchemaDrivenForm 仅 Playwright E2E 不够" — 用户采纳，PR-B 升级含 SchemaDrivenForm/Output Vitest unit test。

**UNRESOLVED:** 0 项（所有决策都有明确答案）

**VERDICT:** ENG CLEARED — ready to implement. 建议顺序：PR-A（backend + migration + frontend placeholder stub）→ PR-B（frontend）。

## 实施状态（2026-04-24 收尾）

**SHIPPED + 视觉对齐 + 回归测试.** v3 主线（PR-A/PR-B）+ 次主线（m08/m13）
+ mockup 对齐 + 回归测试均已 merge。manual gate 在本地 dev 环境通过（通过浏览器
walk-through + 直接对照 mockup 截图）。

| 阶段 | 状态 | PR | 关键证据 |
|---|---|---|---|
| PR-A backend skeleton | ✅ merged | [#8](https://github.com/iocrazy/nous-center/pull/8) | migration + 10 backend 文件改 + 7 个新 test，全套 pytest green |
| PR-B frontend | ✅ merged | [#9](https://github.com/iocrazy/nous-center/pull/9) | m02 + m03 + 4 dialogs + Schema-driven Playground + IconRail 8 nav；14 个 vitest unit、3 个 Playwright spec |
| Migration manual gate | ✅ done | — | dev DB pg_dump → 跑 migration → 重跑 no-op；6 service_instances name normalize、2 collision 加 id 后缀；9 个 instance_api_keys 回填到 api_key_grants |
| Browser walk-through | ✅ done | — | 快速开通 → 详情页 5 tabs；publish wizard 3 步走通 |
| 回归测试（manual-gate 3 bug） | ✅ merged | [#11](https://github.com/iocrazy/nous-center/pull/11) | PG migration 真跑 + 幂等 + 名字冲突；ServicesList navigate；CreateServiceDialog/PublishDialog mutation-deps loop guard |
| m08 Workflow 列表页 | ✅ merged | [#12](https://github.com/iocrazy/nous-center/pull/12) | `/workflows` 切到 m08 列表，`/workflows/:id` 仍是 canvas；卡片关联服务 + 再次发布按钮 + 5 vitest |
| m13 用量统计页 | ✅ merged | [#13](https://github.com/iocrazy/nous-center/pull/13) | 后端 3 端点 (summary / timeseries / top-keys) + 6 pytest；前端 recharts stacked bar + Top Key 表 |
| UsagePage vitest 对齐 | ✅ merged | [#14](https://github.com/iocrazy/nous-center/pull/14) | 7 个 case，stat 渲染 / 错误率 null / 环比 / 范围切换 / 空态 / 错误显示 |
| Mockup 视觉对齐 | ✅ merged | [#15](https://github.com/iocrazy/nous-center/pull/15) | m02 split-button + footer 三按钮 + category 区分色 + 提示框；m08 + 新建 Workflow 卡；m13 导出 CSV |

### Manual gate 期间发现并修的 3 个 bug

落在 PR-B 的 `fix(services-v3): manual-gate hotfixes` 提交里，回归测试在 PR #11：

1. **Migration 幂等性破窗**：`api_key_grants` 回填 INSERT 在第二次跑时失败（步骤 7 已把 `instance_id` rename 成 `service_id`）。修：把 INSERT 包进 `DO $$ ... IF EXISTS (instance_id 列) THEN EXECUTE ...`，重跑变 no-op。原 `test_services_v3_migration.py` 只查文件结构没真跑 SQL，所以静态测试漏了这个。
2. **`window.history.pushState` 不触发 react-router**：`ServicesList` 的卡片点击导航到 `/services/:id` 但 `RouteSync` 的 `useLocation` 不会订阅原生 history 改动。修：改用 `useNavigate()`。
3. **`useEffect` 依赖了 `useMutation` 整个对象 → 无限渲染循环**：`CreateServiceDialog` 与 `PublishDialog` 的 reset effect 依赖了 mutation result 的引用，每次 render 引用变化触发 reset → React Query 状态变 → 再 render → 死循环。修：依赖 stable 的 `.reset` 字段。

### 后续可做（不阻塞，需要后端补数据）

- m02 卡片配额条 + 24h 调用量 / P95 / 错误率（要 LLMUsage 加 status 字段或新建错误统计表）
- m13 错误率 + P95 真实数据（同上 + PG `percentile_cont` / SQLite fallback）
- m03 service detail 加"新版本"按钮（语义=新建一个新 name 的服务，源 workflow 不变）
- vite chunk splitting：拆 react / recharts / @xyflow 等大依赖到独立 vendor chunk（消除 build 警告）

## 本轮 Plan-Eng-Review 锁定的实施细节

以下是 /plan-eng-review 过程中（Step 0 + 4 sections + outside voice）拍板的**实施规范**。PR-A/PR-B 执行时以此为准。

### PR-A Backend（~10 files）

**Migration `2026-04-22-services-v3.sql`** — 单事务内完成：
1. `service_instances` 加列：`workflow_id BIGINT NULL`, `workflow_snapshot JSONB DEFAULT '{}'::jsonb`, `exposed_inputs JSONB DEFAULT '[]'::jsonb`, `exposed_outputs JSONB DEFAULT '[]'::jsonb`, `snapshot_hash TEXT NULL`, `snapshot_schema_version INT DEFAULT 1`, `version INT DEFAULT 1`。
2. `ALTER TABLE service_instances ADD CONSTRAINT name_unique UNIQUE (name)` + `CHECK (name ~ '^[a-z][a-z0-9-]{1,62}$')`。
3. `workflows` 加列：`auto_generated BOOLEAN DEFAULT FALSE`, `generated_for_service_id BIGINT NULL REFERENCES service_instances(id) ON DELETE SET NULL`。
4. `ApiKeyGrant.instance_id` → `service_id`（rename column + rename unique constraint + rename index）。
5. 数据回填：每个 `workflow_apps` 行 → `service_instances` 行（拷 name/snapshot/exposed_i/o），每个 `InstanceApiKey.instance_id NOT NULL` → `api_key_grants` 新行。
6. `UPDATE instance_api_keys SET instance_id = NULL WHERE instance_id IS NOT NULL`。
7. `DROP TABLE workflow_apps`（按现 ORM 关系清理）。
8. `CREATE INDEX idx_service_snapshot_hash ON service_instances (snapshot_hash)` — 非 unique，为 dedup 留门。
9. 幂等保护：每步用 `IF NOT EXISTS` / `IF EXISTS`；重跑应 no-op。

**Models：**
- `service_instance.py`: 加字段 + `deferred(workflow_snapshot, exposed_inputs, exposed_outputs)`
- `workflow.py`: 加 `auto_generated`, `generated_for_service_id`
- `api_gateway.py`: 改 column 名
- `workflow_app.py`: **删除文件** + 更新 `__init__.py` 导入

**Services 层：**
- `model_resolver.py`: 删 Legacy 分支，重命名 `resolve_target_instance` → `resolve_target_service`
- `quota_gate.py`: 参数 `instance_id` → `service_id`

**Routes：**
- `apps.py`: `execute_app()` 改走 `resolve_target_service` + `consume_for_request`，units=1 dim="calls"
- `openai_compat.py`: dispatch 改 `resolve_target_service` + `.options(undefer(ServiceInstance.workflow_snapshot))`
- `services.py`（新）：CRUD + 快速开通路径。name regex 校验通过 Pydantic validator
- `workflow_publish.py`（新）：POST `/api/v1/workflows/{id}/publish`。发布事务内断言每个 `exposed.node_id ∈ snapshot.nodes`，否则 422

**Frontend placeholder stubs**（防 master 破坏）：
- `frontend/src/pages/ServicesListStub.tsx` — "v3 重构中，功能在 PR-B"
- `frontend/src/pages/WorkflowsListStub.tsx` — 同
- 清除老 `ServicesOverlay` / `AppsOverlay` 路由指向占位

**Tests (PR-A)**：
- `test_services_v3_migration.py`: Legacy→Grant 回填、WorkflowApp→Service 回填、rerun idempotent
- `test_services_crud.py`: name regex、冲突、deprecated 仍服务
- `test_workflow_publish.py`: version bump、snapshot_hash 计算、exposed_node_id 校验
- `test_services_dispatch.py`: 统一 dispatch + SQL query counter 断言（防 deferred 漏 undefer）
- `test_apps_grant_auth.py`: /v1/apps/{name} 无 grant → 403，配额用尽 → 402
- `test_model_resolver.py`: 删 Legacy 分支 asserts
- `test_api_gateway_e2e.py`: 字段重命名回归

**手动门禁**：PR-A merge 前 `pg_dump` 当前 dev DB → fresh PG → run migration → 验证。

### PR-B Frontend（~9 files + e2e + unit）

**Pages：**
- `ServicesList.tsx` (m02)
- `ServiceDetail.tsx` (m03，含 5 tabs：总览 / Playground / API 文档 / Key 授权 / 用量)

**Components：**
- `services/CreateServiceDialog.tsx` — 快速开通向导
- `workflow/PublishDialog.tsx` — 发布向导，3 step
- `playground/SchemaDrivenForm.tsx` — 从 exposed_inputs 自动渲染（核心）
- `playground/SchemaDrivenOutput.tsx` — 按 output mime 渲染
- `IconRail.tsx` — 精简到 8 主 nav

**API hooks：**
- `frontend/src/api/services.ts` — CRUD + publish hooks

**Tests (PR-B)**：
- `e2e/quick-provision-llm.spec.ts` — 快速开通 → Playground run
- `e2e/publish-workflow.spec.ts` — 画布发布 → 生成服务
- `e2e/playground-run.spec.ts` — 服务详情 Playground 全流程
- `components/playground/SchemaDrivenForm.test.tsx` — Vitest unit (T12 采纳)
- `components/playground/SchemaDrivenOutput.test.tsx` — Vitest unit (T12 采纳)

### 设计细节补充（Plan-Eng-Review 锁定）

- **服务命名** = `^[a-z][a-z0-9-]{1,62}$`，禁止重命名，只允许 deprecate + new name
- **节点 ID** = snowflake / UUID 字符串（如 `"nd_5g2k9m"`），删除不复用。schema 里 `node_id` 字段 type = string
- **service.status 生命周期**：`active` → `paused` → `deprecated` → `retired`。deprecated 仍响应调用但 log warn；retired 返 410
- **trivial workflow**: `workflow.auto_generated=True` + `generated_for_service_id` FK (ondelete=SET NULL)。m08 列表默认 filter `auto_generated=false`
- **snapshot_hash**: index=True, unique=False（允多服务共享同 snapshot，为 dedup 留门）
- **snapshot_schema_version**: 加在 services 行上，默认 1。ComfyUI 导入未来可增版本
- **Grant.service_id**: `ApiKeyGrant.instance_id` 改名
- **Grant-level ResourcePack** 唯一：不做跨服务共享池
- **deferred() + 强制 undefer**: dispatch 路径显式 `.options(undefer(...))`，有 SQL counter 测试

## 附：术语表

| 术语 | 定义 |
|---|---|
| **引擎（Engine）** | 底层模型文件 + GPU 占用。内部概念，不对外 |
| **Workflow** | DAG 蓝图。ComfyUI 风格双格式（编辑态 full / 运行态 api） |
| **服务（Service）** | = 实例 = App. 对外可调用的单元。有 endpoint + schema + 配额 + 授权 |
| **节点 ID（node_id）** | Workflow 内稳定整数，删除不回收 |
| **发布（Publish）** | 把 workflow 冻结为快照 + 定义 I/O schema → 产生一个 service 版本 |
| **快速开通（Quick Provision）** | 选引擎填参数 → 系统后台生成 trivial workflow → 得服务 |
| **Grant** | API Key 授权到某个服务的记录。携带独立配额 / 过期 / 启用状态 |
| **ResourcePack** | 挂在 Grant 上的配额单位：token / 字符数 / 时长 / 调用次数 |
