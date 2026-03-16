# Nous Center — Workflow + Agent 平台设计

## 概述

将 nous-center 从 TTS 工具升级为**本地 AI 工作流平台**：

- **Workflow 画布**编排 AI 流程（LLM、TTS、图像、视频）
- **Agent 系统**管理智能体（Skills + Prompts + 工具）
- **声明式组件模型**驱动节点 UI 自动生成
- **一键发布**将 workflow 部署为常驻 API 服务

三阶段交付：A（持久化+部署）→ B（Agent 系统）→ C（LLM 节点+声明式组件）

---

## 核心概念

```
预设 = 已保存的 Workflow 模板（is_template=true）
Workflow = 节点 + 连线 + 配置
发布 = Workflow → 常驻服务（模型加载到显存 + API endpoint）
Agent = 配置了 Skills/Prompts 的智能体（可作为 Workflow 节点）
Skill = SKILL.md（YAML frontmatter + Markdown 指令）
```

**实体关系：**

```
Workflow ←1:1→ ServiceInstance（发布后）
Workflow ←contains→ Node[]
Agent ←has→ Skill[]（MD 文件）
Agent ←has→ Prompt[]（MD 文件：SOUL.md, IDENTITY.md 等）
Agent Node ←references→ Agent 配置
```

---

## A 阶段：Workflow 持久化 + 部署发布

### 数据模型

```sql
-- workflows 表
id            BIGINT PRIMARY KEY (snowflake)
name          VARCHAR(100)
description   TEXT
nodes         JSON          -- WorkflowNode[]
edges         JSON          -- WorkflowEdge[]
is_template   BOOLEAN       -- true = 预设模板
status        VARCHAR(20)   -- draft / published
created_at    TIMESTAMP
updated_at    TIMESTAMP

-- service_instances 表（已有，扩展）
-- source_type = 'workflow', source_id = workflow.id
```

### API 端点

```
CRUD:
  POST   /api/v1/workflows          — 创建
  GET    /api/v1/workflows          — 列表（?is_template=true 过滤预设）
  GET    /api/v1/workflows/{id}     — 详情
  PATCH  /api/v1/workflows/{id}     — 更新（节点/连线/名称）
  DELETE /api/v1/workflows/{id}     — 删除

发布:
  POST   /api/v1/workflows/{id}/publish    — 发布为服务
    → 创建 ServiceInstance (source_type=workflow)
    → 加载依赖模型到显存
    → 返回 instance_id + endpoint

  POST   /api/v1/workflows/{id}/unpublish  — 下线
    → 停止服务、释放资源

运行:
  POST   /api/v1/instances/{instance_id}/run  — 调用已发布的 workflow
    → 请求体根据 workflow 的输入节点自动推导
    → 返回输出节点的结果
```

### 前端变化

- 现有 voice_presets 面板改为 workflows 面板（列表 is_template=true 的 workflow）
- 点击预设 → 打开 workflow tab（从 DB 加载）
- Tab 自动保存（debounce 写回 DB）
- Topbar 加"发布"按钮 → 调 publish API
- 发布状态在 tab 上显示（draft/published 徽章）

### Voice Preset 迁移

现有 `voice_presets` 表保留不删除，渐进式迁移：

1. 为每个现有 preset 自动生成一个 workflow（text_input → tts_engine[preset params] → output）
2. `is_template = true`，名称沿用 preset 名
3. `ServiceInstance` 的 `source_type="preset"` 继续支持，新发布的用 `source_type="workflow"`
4. 前端预设面板切换为 workflow 列表，旧 preset 作为只读兼容项

### 后端 DAG 执行器

```
POST /api/v1/instances/{id}/run
  → 加载 workflow 定义（发布时快照，不随编辑变化）
  → 拓扑排序节点
  → 按依赖序逐节点执行
  → 返回输出节点结果
```

**执行模型：**
- 异步执行（FastAPI async），长任务通过 WebSocket 推送进度
- 每个节点执行器注册在 `node_executors: dict[str, Callable]`
- 节点输出缓存在内存 dict 中，按 edge 传递给下游节点

**错误处理：**
- 节点执行失败 → 标记该节点 error → 停止整个 workflow → 返回错误详情
- 超时：每个节点可配 timeout（默认 60s，LLM/视频节点更长）

**发布快照：**
- publish 时将当前 workflow 的 nodes/edges 快照存入 `ServiceInstance.params_override`
- 编辑 workflow 不影响已发布的服务
- 重新 publish = 更新快照（原子替换，无停机）

**并发：**
- 单用户本地使用，暂不考虑高并发
- GPU 资源由现有的 vram_tracker 管理

### 执行层

- 已发布的 workflow 执行在**后端**（Python DAG 执行器）
- 未发布的 workflow 仍在前端执行（调试/预览用）
- 执行进度通过 WebSocket `/ws/workflow/{instance_id}` 推送

---

## B 阶段：Agent 系统 + Skills/Prompts 管理

### 文件体系（借鉴 OpenClaw）

```
~/.nous-center/
├── agents/
│   ├── podcast-host/              # Agent: 播客主播
│   │   ├── AGENT.md               # 操作指令（角色、行为规则）
│   │   ├── SOUL.md                # 人设、语气、边界
│   │   ├── IDENTITY.md            # 名称、风格
│   │   └── sessions/              # 对话历史 JSONL
│   └── translator/
│       ├── AGENT.md
│       ├── SOUL.md
│       └── IDENTITY.md
├── skills/
│   ├── web-search/
│   │   └── SKILL.md
│   ├── tts-synthesis/
│   │   └── SKILL.md
│   └── document-qa/
│       └── SKILL.md
└── config.json
```

### SKILL.md 格式

```markdown
---
name: tts-synthesis
description: 将文本合成为语音
requires:
  models: ["cosyvoice2"]
---

## 使用说明
当用户要求语音合成时，调用 tts_synthesize 工具。

## 参数
- text: 要合成的文本
- voice: 音色名称（默认 "default"）
- engine: TTS 引擎（默认 "cosyvoice2"）
```

### 数据模型

**文件系统是唯一数据源**，数据库不存 agent/skill 配置。启动时扫描 `~/.nous-center/` 目录加载。

Agent 配置存在各目录下的 `config.json` 中：

```json
// ~/.nous-center/agents/podcast-host/config.json
{
  "display_name": "播客主播",
  "model": { "engine_key": "qwen3_tts_base", "fallback_api": null },
  "skills": ["tts-synthesis", "web-search"],
  "tools_policy": {},
  "status": "active"
}
```

Skills 的元数据在 `SKILL.md` 的 YAML frontmatter 中，不需要额外数据库表。

`requires.models` 在 Agent 激活时校验：检查 models.yaml 中对应模型是否存在且本地已下载。

### API 端点

```
Agents:
  POST   /api/v1/agents
  GET    /api/v1/agents
  GET    /api/v1/agents/{id}
  PATCH  /api/v1/agents/{id}
  PUT    /api/v1/agents/{id}/prompts/{filename}  — 更新 MD 文件
  DELETE /api/v1/agents/{id}

Skills:
  GET    /api/v1/skills
  GET    /api/v1/skills/{name}
  POST   /api/v1/skills
  PUT    /api/v1/skills/{name}
  DELETE /api/v1/skills/{name}
```

### Agent 在 Workflow 中的使用

Agent 是**预配置好的实体**。在 workflow 画布中放一个 Agent 节点，选择已配置的 agent 即可：

```
文本输入 ──→ [Agent: 播客主播] ──→ TTS ──→ 输出
              选择 agent ▾
```

Agent 的 Skills/Prompts 在独立的 Agent 管理页面编辑，不在画布上操作。

### 前端

- IconRail 新增 Agent 图标 → Agent 管理 Overlay
- Agent 列表 + 详情（内嵌 MD 编辑器）
- Skills 标签页（SKILL.md 编辑器）
- Agent 配置面板（model 选择、skills 勾选、tools 策略）

---

## C 阶段：LLM 节点 + 声明式组件模型

### 声明式节点定义（借鉴 LangFlow）

从手写 React 组件改为数据驱动，UI 自动生成：

```typescript
{
  "llm": {
    "label": "LLM",
    "category": "ai",
    "icon": "brain",
    "inputs": [
      { "name": "prompt", "type": "text", "widget": "textarea", "required": true },
      { "name": "system", "type": "text", "widget": "textarea" },
      { "name": "model", "type": "string", "widget": "model_select",
        "filter": { "type": "llm" } },
      { "name": "temperature", "type": "number", "widget": "slider",
        "min": 0, "max": 2, "default": 0.7 }
    ],
    "outputs": [
      { "name": "text", "type": "text" }
    ]
  }
}
```

**渲染方式：**
- `BaseNode` 读取定义 → 自动生成 inputs/outputs 的 UI
- 现有 8 个节点迁移为声明式定义
- 自定义渲染仍可覆盖（复杂节点如 OutputNode）

### LLM 模型管理

与 TTS 引擎管理模式相同：

1. Models 页面下载 LLM 模型到 `models/nous/llm/`
2. 点"启动" → nous-center 内部启动 vLLM 进程
3. LLM 节点选模型名 → 自动连接本地 vLLM 实例
4. 云端 API 作为可选 fallback（配 api_key 即可）

### 节点规划（按梯队）

**第一梯队 — 核心必备（C 阶段）：**

| 节点 | 类别 | 来源 | 说明 |
|------|------|------|------|
| LLM | ai | LangFlow 借鉴 | 本地 vLLM / 云端 API |
| Agent | ai | 自建 | 调用预配置 Agent |
| Prompt Template | ai | LangFlow 借鉴 | 变量替换模板 |
| Memory | ai | LangFlow 借鉴 | 对话历史管理 |
| If-Else | control | LangFlow 借鉴 | 条件分支 |
| Loop | control | LangFlow 借鉴 | 循环遍历 |
| API Request | data | LangFlow 借鉴 | HTTP 外部调用 |
| Structured Output | ai | LangFlow 借鉴 | LLM 输出结构化 |
| Chat Input/Output | io | LangFlow 借鉴 | 对话式 IO |

**第二梯队 — 数据处理：**

| 节点 | 类别 | 来源 | 说明 |
|------|------|------|------|
| JSON Parser | processing | LangFlow 借鉴 | JSON 解析提取 |
| Regex | processing | LangFlow 借鉴 | 正则匹配 |
| Combine Text | processing | LangFlow 借鉴 | 合并文本 |
| Split Text | processing | LangFlow 借鉴 | 文本分块 |
| Filter Data | processing | LangFlow 借鉴 | 数据过滤 |
| Python REPL | utility | LangFlow 借鉴 | 执行 Python 代码 |
| CSV/JSON to Data | data | LangFlow 借鉴 | 文件解析 |

**第三梯队 — RAG / 知识库：**

| 节点 | 类别 | 来源 | 说明 |
|------|------|------|------|
| File Loader | knowledge | LangFlow 借鉴 | 加载 PDF/DOCX/TXT |
| Embedding | knowledge | LangFlow 借鉴 | 文本向量化 |
| Vector Store | knowledge | LangFlow 借鉴 | 向量存储检索 |
| Retrieval | knowledge | LangFlow 借鉴 | RAG 检索 |
| Web Search | data | LangFlow 借鉴 | 网络搜索 |
| URL Loader | data | LangFlow 借鉴 | 抓取网页 |

**第四梯队 — 多媒体（nous-center 特有）：**

| 节点 | 类别 | 来源 | 说明 |
|------|------|------|------|
| TTS Engine | media | 已有 | 语音合成 |
| Ref Audio | media | 已有 | 参考音频 |
| Resample/Mixer/Concat/BGM | media | 已有 | 音频处理 |
| Image Generate | media | 自建 | SDXL 图像生成 |
| Video Generate | media | 自建 | Wan2.1 视频生成 |
| Output | io | 已有 | 播放预览 |

**第五梯队 — 高级流程控制：**

| 节点 | 类别 | 来源 | 说明 |
|------|------|------|------|
| Run Flow | control | LangFlow 借鉴 | 子流程调用 |
| Flow Tool | control | LangFlow 借鉴 | workflow 包装为 Agent 工具 |
| Notify/Listen | control | LangFlow 借鉴 | 事件驱动 |
| Batch Run | control | LangFlow 借鉴 | 批量执行 |
| Guardrails | ai | LangFlow 借鉴 | 安全过滤 |
| LLM Router | ai | LangFlow 借鉴 | 动态路由 |

### 端口类型扩展

```
现有：text | audio
新增：message | data | image | video | any
```

---

## 技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| Workflow 引擎 | 渐进增强现有系统 | 已有画布+执行器，重建成本高 |
| LangFlow 关系 | 借鉴组件模型，不 fork | fork 维护负担太重 |
| LLM 推理 | nous-center 托管 vLLM | 用户无外部推理服务，本地显卡运行 |
| Agent 架构 | 预配置实体 + workflow 节点引用 | 职责分离：画布管编排，Agent 页面管配置 |
| 节点 UI | 声明式定义自动生成 | 减少重复代码，新增节点只需写定义 |
| Prompt/Skill 格式 | MD 文件（借鉴 OpenClaw） | 人类可读、版本可控、易编辑 |
| 预设 | 等同于 is_template=true 的 workflow | 统一概念，消除歧义 |
| 部署粒度 | workflow 级别发布 | 单节点部署无实际场景 |
| 已发布 workflow 执行 | 后端 Python DAG 执行器 | 常驻服务需要后端运行 |
| 云端 API | 可选 fallback | 本地显存不够时使用 |
| Agent/Skill 存储 | 文件系统为唯一数据源 | 避免双源同步问题，MD 文件可外部编辑 |
| 发布快照 | publish 时冻结 workflow 定义 | 编辑不影响已发布服务 |
| 执行进度 | WebSocket 推送 | LLM+TTS+视频可能耗时数分钟 |
| 并发 | 单用户，暂不考虑 | 本地工具，非云服务 |

---

## 非目标

- 多用户/多租户
- 云端部署
- Workflow 版本历史（Git 管理即可）
- 实时协作编辑
- 模型训练/微调
