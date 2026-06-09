# 运行历史回看 + 产物画廊 + 参数回填重跑(借鉴 Infinite-Canvas)

- 日期: 2026-06-09
- 状态: 设计
- 参考: Infinite-Canvas(`history.json` 全局生成历史 + `canvas.logs` 画布运行日志 +
  `applySameStyle` 参数回填)
- 相关: [task-panel-reset](./2026-05-27-task-panel-reset-design.md)、
  [workflow-to-webui-app-editor](./2026-06-09-workflow-to-webui-app-editor-design.md)

## 1. 背景与定位

对照 Infinite-Canvas 的「历史操作记录 + logs」,nous 现状盘点:

| 能力 | Infinite-Canvas | nous 现状 | 缺口 |
|---|---|---|---|
| 日志查看 UI | 仅 uvicorn 访问日志 + 画布 run 弹窗 | ✅ `LogsOverlay`(Request/App/Frontend/Audit 四类 + 搜索/时间范围/Live/导出)+ log_db(7天/10万行/每小时清) | **无 —— nous 更全,不复刻** |
| 每次运行记录 | history.json(5000)+ canvas.logs(500) | ✅ `ExecutionTask`(含 `input_json`/`result`/`duration_ms`/`node_timings`)+ `/api/v1/tasks` + `TaskPanel` | 无 |
| 产物存储 | 本地文件 | ✅ 签名 URL+TTL(`image_output_storage.py`) | 无 |
| **参数回填重跑** | ✅ `applySameStyle` 回填表单 | `input_json` 已存,但前端「重跑」按钮**静态占位**(task-panel spec PR-6 未做) | **PR-A** |
| 历史出图画廊 | ✅ 各页历史库(瀑布流+无限滚动+批量删) | `TaskPanel` 有缩略图但 `output_thumbnails` 后端未序列化;无独立画廊;Studio gallery 内存态刷新即丢 | **PR-B** |
| 服务级历史/用量 | — | `ServiceDetail` 用量 tab 纯占位;`/api/v1/tasks` 无 workflow/service 过滤 | **PR-C** |
| 手动启动日志去向 | — | 生产 journald;手动起 stdout→/tmp(无规整) | **PR-D** |

结论:**logs 已完备(优于参考),本 arc 聚焦「运行历史回看 + 产物画廊 + 参数回填重跑」**。

## 2. 关键现状约束

- `ExecutionTask`(`models/execution_task.py`):有 `workflow_id`(nullable)、`input_json`
  (PR-2 加,历史参数)、`result`(完整输出)、`status/duration_ms/created_at/node_timings`。
  **无 `api_key_id` / `service_id` 列** → 服务归属只能经 `workflow_id` / `workflow_name`
  (`run_published_workflow` 建 task 时 `workflow_name=svc.name`)。按 api_key 归属是更大改动
  (需加列 + 回填),本 arc **不做**,标 follow-up。
- `/api/v1/tasks`(`routes/execution_tasks.py`):仅 `status` 过滤 + created_at desc + limit/offset。
- `retry`(`POST /tasks/{id}/retry`):只重跑**相同 workflow**,不回填/不改参数。
- `SchemaDrivenForm`:目前从 `exposed_inputs` 的 default 算初值,**不接受外部 initialValues**。
- 产物:`result.outputs[node_id].image_url` 是签名 URL(1h TTL)。

## 3. PR 切分

### PR-A 参数回填重跑(最高价值,对齐 applySameStyle)
- `SchemaDrivenForm` 加 `initialValues?: Record<string, unknown>` prop(覆盖 default 初值)。
- `ExecutionTask` 已存 `input_json`;`/api/v1/tasks/{id}` 详情已返回它。
- 前端:任务面板/`TaskDetailModal` 的「重跑(相同参数)」按钮接真 action ——
  - 服务来源任务(能由 `workflow_id` 找到对应 service):跳到该服务详情页 Playground tab,
    用 `input_json` 预填表单(经 `initialValues`),用户可改参数再点运行。
  - 用 React Router state 或 query 传 `input_json`(避免塞进 URL 过长;用 router state)。
- 「复制参数」按钮:把 `input_json` 复制为 JSON。
- 测试:initialValues 覆盖 default;重跑导航带上 input_json。

### PR-B 历史出图画廊
- 后端 `output_thumbnails` 序列化:`/api/v1/tasks` 列表项从 `result` 抽出 image url 列表
  (`result.outputs[*].image_url`,最多 N 张),前端任务卡缩略图直接用(填上现有半接线字段)。
- 后端 `GET /api/v1/tasks` 加 `type` 过滤(image/tts/llm/vl,从 task 已检测的 type)。
- 前端独立 `/history` 路由:瀑布流画廊,拉 `/api/v1/tasks?type=image`,点图开 lightbox,
  卡片带「重跑」(复用 PR-A)+「删除」(已有 DELETE API)。无限滚动(offset 翻页)。
- Studio gallery:不强改持久化(产物已在 ExecutionTask),改为「历史」入口跳 /history。
- 测试:画廊渲染 + type 过滤 + 缩略图抽取纯函数单测。

### PR-C 服务详情「用量/历史」tab 填真数据
- 后端 `/api/v1/tasks` 加 `workflow_id` 过滤(配合 PR-B 的 type)。
- `ServiceDetail` 用量 tab:拉 `/api/v1/tasks?workflow_id={svc.workflow_id}` →
  历史调用记录表(时间/状态/耗时/入参摘要/出参缩略),点行看详情(复用 TaskDetailModal)。
- 简单聚合:总调用数 / 成功率 / 均耗时(从这批 task 前端算,够用;重统计走 follow-up 用量子系统)。
- 按 api_key 归属:**本 PR 不做**(无列),UI 标注「按 key 细分待用量子系统」。
- 测试:用量 tab 拉对 workflow_id 的 tasks + 表格渲染。

### PR-D 启动/日志规整(运维,小)
- 加 dev 启动脚本 `backend/scripts/dev-serve.sh`(或 infra 下):`set -a; . .env; set +a` +
  uvicorn,stdout/stderr → `backend/logs/backend-dev.log`(gitignore),配合 `logrotate` 片段或
  脚本内简单 size-cap。文档化:生产走 journald(`journalctl -u nous-backend`),dev 用此脚本。
- 不动结构化 log_db(已完善)。

> 依赖:PR-A 是 PR-B/PR-C 的「重跑」复用基础;PR-D 独立。

## 4. 真机验证

起后端(PR-D 脚本)+ Playwright/chrome-devtools:跑一次出图 → /history 画廊看到该图 →
点「重跑」回填参数到 Playground 改了再跑 → 服务用量 tab 看到这两条调用记录。
(本 session 已验过 img-flux2 真出图链路,产物落 `/files/images/...`。)

## 5. Follow-up(不在本 arc)
- `ExecutionTask.api_key_id` 列 + 回填 → 按 key 的用量细分 / 配额可视化。
- 画廊批量删除(对齐 Infinite-Canvas history-bulk-manager)。
- 产物保留策略 UI(reap_orphans 现有,暴露成可配)。
