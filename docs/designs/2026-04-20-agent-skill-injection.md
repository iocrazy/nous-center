---
status: SUPERSEDED
date: 2026-04-20
branch: feature/nous-center-v2
superseded_by: 战略定位调整 — agent/skill 在 mediahub 层实现，nous-center 保持干净推理 infra
---

> **2026-04-20 战略决策**：经 CEO review 后确认，agent/skill 语义**不在 nous-center 实现**。
>
> **nous-center 的定位**：干净的推理基础设施（本地托管、OpenAI 协议兼容 + 协议级扩展如 Files / Responses session / Context cache / Memory ABC）。
>
> **mediahub 的定位**：agent / skill / persona 的实现层，通过标准 OpenAI 协议调用 nous-center。
>
> **理由**：
> 1. 对齐 OpenClaw / Claude Code 范式（agent 在应用层，不在 infra 层）
> 2. 避免双层 agent 抽象（mediahub 和 nous-center 都有 agent 概念会冲突）
> 3. 让 mediahub 可切换到别的 infra 零成本，nous-center 的差异化聚焦到"本地跑 + 协议级扩展"
>
> **本文档的价值**：Skill = Lazy-readable File（OpenClaw 路线）+ Claude Code 风格 `Skill` tool 的对比结论，迁移到 mediahub 的 agent/skill 设计时可直接复用。
>
> **下一步**：进 Wave 1（`docs/superpowers/specs/2026-04-16-wave1-platform-contracts-design.md`），专注 MemoryProvider ABC / ContextEngine ABC / 事件扩展等**协议级契约**。本 spec 作为设计演进记录保留，不再实施。

---

# Agent / Skill 注入设计（SUPERSEDED）

## 背景

nous-center 已有 agent 与 skill 的 CRUD 基础设施（`agent_manager.py` / `skill_manager.py` / `skill_tools.py`），但推理入口（`/v1/responses`、`/v1/chat/completions`、`workflow_executor`）都没有消费这两个概念 —— 没有 `agent` 入参，也没有把 `AGENT.md / SOUL.md / IDENTITY.md` 拼进 system 消息的路径。本文档锁定这条缺失链路的设计。

本次对比并选型的对象是 **"一个配好的 agent + skills 如何出现在对 LLM 的请求里"**，不是 agent/skill 管理界面、workflow 节点或评估框架。

## 当前状态

| 模块 | 位置 | 现状 |
|------|------|------|
| Agent 存储 | `backend/src/services/agent_manager.py` | 目录结构：`~/.nous-center/agents/<name>/{config.json, AGENT.md, SOUL.md, IDENTITY.md}`；CRUD 完整 |
| Skill 存储 | `backend/src/services/skill_manager.py` | 目录：`~/.nous-center/skills/<name>/SKILL.md`（frontmatter + body）；CRUD 完整 |
| Skill → OpenAI tools | `backend/src/services/skill_tools.py` | `skills_to_tools()` 把每个 skill 变成一个 function tool；带 `execute_python` 沙箱 |
| 推理入口 | `backend/src/api/routes/responses.py:*`, `openai_compat.py:*` | 不读 agent / 不读 skill；`instructions` 字段仅透传 OpenAI 原生语义 |
| Session 模型 | `response_sessions` 表 | 有 session 概念（`previous_response_id` 串联），**没有** `agent_id` 列 |

## 设计目标

1. 调用方通过一个明确字段指定 agent，对 `/v1/responses` 与 `/v1/chat/completions` 统一生效
2. Session 首次创建时与 agent 绑定；同 session 不能中途换 agent（一致性校验）
3. Agent 的 `AGENT.md / SOUL.md / IDENTITY.md` 装配进 system message 的稳定段
4. Skills 以 **lazy-readable** 形式注入：system prompt 只放 `<available_skills>` 清单（name + description），模型按需通过专用工具读取 SKILL.md body
5. Prompt cache 前缀按 `(agent_id, skills_fingerprint)` 键，同 agent 多 session 复用
6. 向下兼容：不传 `agent` 时完全退回现有行为，无 persona、无 skills

Out of scope（见文末）：workflow LLM 节点接入、请求级 skills override、agent 版本管理、多租户 skill 可见性。

## 方案对比

三种 skill → 模型注入路线：

| 维度 | 方案 A：Skill-as-Function-Tool | 方案 B：Lazy-Readable（本设计） | 方案 C：混合 |
|------|---------|-----|-----|
| 注入位置 | `tools[]` 数组（每 skill 一个 function） | system prompt 的 `<available_skills>` XML + 一个专用 `Skill` tool | frontmatter 声明 `mode`，两种路径并存 |
| 静态 token 成本 | O(skills × 平均 schema 大小) | O(skills × 50 tokens description) | 两路叠加 |
| SKILL.md body 可见性 | 丢失（仅 description + params） | 完整（tool_result 里返回 body） | 分裂：tool 模式丢失，lazy 模式保留 |
| 对本地 vLLM 模型友好度 | 高（function-calling 是 OpenAI 协议原生） | 中（需模型理解"先 read 再 follow"的两阶段） | 低（混合路径容易训练混乱） |
| 实现复杂度 | 低（`skills_to_tools()` 已有雏形） | 中 | 高（双装配器 + 双测试） |
| 与 OpenClaw / Claude Code 对齐 | ✗ | ✓（两家都选此路线） | ✗ |

**选型结论：方案 B。**

**理由**：
- OpenClaw 和 Claude Code 这两个成熟 agent 产品独立地都选了 lazy-readable 路线，说明该路线在真实 agent 场景下 context 效率、指令保真度、可维护性综合最优
- SKILL.md body 往往是几百到上千字的操作手册（调用规范、中英文提示词示例、注意事项），塞进 tool schema 的 description 既违反 schema 原意也装不下
- nous-center 的目标是"Python 世界 AI infra 标杆"（见 `2026-04-16-nous-center-v3-platform.md`），和 OpenClaw / Claude Code 的生态对齐比"跟 OpenAI function-calling 范式绑死"更有长期价值
- 本地 Qwen/Gemma 对 lazy 模式的两阶段行为不如 Claude 稳，但通过 system prompt 的明确指令段（"scan → select → Skill() → follow"）可以训练/引导，且工具数量从 N 个降为 1 个也反过来降低了模型的决策面

方案 A 不完全抛弃：`skill_tools.py` 的 `execute_python` 沙箱作为通用执行器继续存在，SKILL.md body 读回后模型可以在下一轮显式调用。

## 架构

```
Client ──POST /v1/responses──▶ api/routes/responses.py
       (agent="tutor")             │
                                   ▼
                      services/prompt_composer.py   ← 新模块
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   agent_manager              skill_manager             context_cache
   .get_agent("tutor")        .list_skills(...)         (已有，复用)
     → config.json              → frontmatter only
     → AGENT.md                 → [{name, description}, ...]
     → SOUL.md
     → IDENTITY.md
                                   │
                                   ▼
                       拼装 system message（单条 str）
                        + tools = [Skill, execute_python]
                                   │
                                   ▼
                   services/responses_service.py
                    / services/openai_compat.py
                                   │
                                   ▼
                          vLLM /chat/completions
```

新增单元：

- `backend/src/services/prompt_composer.py` —— 单一职责：`(agent_id, skills_override?) → (system_message: str, tools: list[dict])`
- `backend/src/services/skill_tools.py` 扩展：加入 `Skill` tool 执行分支

修改：

- `backend/src/api/routes/responses.py`、`backend/src/api/routes/openai_compat.py` —— 请求 schema 加 `agent` 字段；调 composer
- `backend/src/models/response_session.py` —— 加 `agent_id` 列
- Migration：alembic 迁移加列

## API 契约

两个 endpoint 同形扩展：

```python
class ResponsesCreateRequest(BaseModel):
    model: str
    agent: str | None = None              # 新增：首请求时指定
    input: str | list[dict]
    instructions: str | None = None       # OpenAI 原生字段；保留
    previous_response_id: str | None = None
    tools: list[dict] | None = None       # 调用方自定义 tools；和系统注入的 Skill 工具合并
    # ... 其他字段维持不变
```

`POST /v1/chat/completions` 对应：

```python
class ChatCompletionRequest(BaseModel):
    model: str
    agent: str | None = None              # 新增
    messages: list[dict]
    # ... 其他字段维持不变
```

**首请求 / 续请求判别**：`previous_response_id is None` 即首请求。

Binding 规则：

| 情况 | 行为 |
|------|------|
| 首请求，`agent` 传了 | 使用该 agent；写入 `response_sessions.agent_id` |
| 首请求，`agent` 空 | 无 persona，裸模型 —— 和现有行为一致；`response_sessions.agent_id = NULL` |
| 续请求，`agent` 空 | 从 `response_sessions.agent_id` 恢复（可能为 NULL，即裸模型续用） |
| 续请求，`agent` 值 ≠ session 绑定值 | **400 `agent_session_mismatch`** |
| 续请求，`agent` 值 == session 绑定值 | 允许（幂等） |
| 续请求，`agent` 非空但 session.agent_id 为 NULL | 400 `agent_session_mismatch` |

续请求不允许换 agent 的设计原则对齐 OpenClaw：`agent-scope.ts:48-65` 的 `resolveSessionAgentIds()` 对显式 agent 与 session 解析出的 agent 做等值校验，不匹配即报错。切 agent 用户需开新 session（不传 `previous_response_id`）。

本设计**不包含** API key 级的默认 agent 绑定；调用方必须在首请求显式传 `agent` 字段。见 NOT in scope。

## System message 装配

**文件顺序**（对齐 OpenClaw 的 `CONTEXT_FILE_ORDER`，本地化到 nous-center 三件套）：

```
┌─ [CACHE STABLE PREFIX] ──────────────────────
│
│  # Identity
│  <IDENTITY.md 内容>
│
│  # Soul
│  <SOUL.md 内容>
│  Embody the persona and tone described above. Avoid generic
│  or stiff replies unless higher-priority instructions override it.
│
│  # Agent Instructions
│  <AGENT.md 内容>
│
│  ## Available Skills
│  Before replying: scan <available_skills> <description> entries.
│  - If one clearly applies: call Skill(skill="<name>") first,
│    then follow the returned instructions.
│  - If none apply: do not call Skill.
│  Never call Skill more than once per turn unless the task
│  clearly requires chaining.
│
│  <available_skills>
│    <skill>
│      <name>search</name>
│      <description>网页搜索，返回可引用链接</description>
│    </skill>
│    ...
│  </available_skills>
│
├─ <!-- CACHE_BOUNDARY --> ────────────────────
│
│  # Request Instructions
│  <request.instructions 内容，若有>
│
│  # Runtime
│  Model: qwen3.5 | Time: 2026-04-20 15:30 UTC | Session: resp_xyz
│
└──────────────────────────────────────────────
```

**装配细节**：

- 三个 md 文件读入后 `.strip()`，任一为空则跳过整个对应 section（包括 `# Identity` 标题）
- SOUL.md 存在时才追加 persona 指令，不存在不加
- `<available_skills>` 段仅在 agent.config.json 的 `skills` 非空时生成
- 位置字段（`<location>`）省略，与 Claude Code 对齐；模型只需 name，skill 查找由后端 `skill_manager.get_skill(name)` 完成
- `<!-- CACHE_BOUNDARY -->` 是纯文本注释（对模型无意义），只作为本地后续 cache 计算的分隔符
- `instructions`（OpenAI 原生字段）定位为"本次请求临时指令"，放在 cache boundary 之后 —— **不替换、不修改** agent 段，分层共存
- 运行时行（Runtime）放最后，含模型名、当前时间、session id，便于模型自我定位

## Skill tool

**Tools 数组注入**（替代 `skills_to_tools()` 生成多个 function 的做法）：

```json
{
  "type": "function",
  "function": {
    "name": "Skill",
    "description": "Load a local skill definition and its instructions.",
    "parameters": {
      "type": "object",
      "properties": {
        "skill": {
          "type": "string",
          "description": "Skill name from <available_skills>."
        },
        "args": {
          "type": "string",
          "description": "Optional arguments to pass to the skill."
        }
      },
      "required": ["skill"]
    }
  }
}
```

配合现有 `execute_python` 保留为独立 sandbox 工具 —— tools 数组形如 `[Skill, execute_python, ...caller_tools]`，三者正交。

**后端执行**（在 `skill_tools.execute_tool` 里加分支）：

```python
def _execute_skill_tool(args: dict) -> dict:
    name = args.get("skill", "").strip()
    if not name:
        return {"error": "skill name required"}
    try:
        sk = skill_manager.get_skill(name)
    except FileNotFoundError:
        return {"error": f"unknown skill: {name}"}
    return {
        "skill": name,
        "description": sk.get("description", ""),
        "prompt": sk.get("body", ""),   # SKILL.md body 原文
        "args": args.get("args"),
    }
```

返回对象序列化为 JSON，作为 tool_result message 回给模型。下一轮模型看到 `prompt` 字段（完整 SKILL.md body）→ 按其指令行动。

**边界情况**：
- `skill` 为空 / 不存在 → 返回 `{"error": "..."}` 作为正常 tool_result，**不抛异常、不 500**（让模型纠错，保持会话）
- 模型调用某个 **不在** agent.skills 列表里的 skill → 放行（agent.skills 只控制 `<available_skills>` 清单的可见度，不控制工具执行的白名单），与 Claude Code 的多 root 查找哲学一致；若未来需要收紧可加 `strict: true` 配置

## Session 绑定与一致性

**Schema 改动**：

```sql
ALTER TABLE response_sessions ADD COLUMN agent_id VARCHAR(128) NULL;
CREATE INDEX idx_response_sessions_agent_id ON response_sessions(agent_id);
```

**写入时机**：首请求（`previous_response_id` 为空）且 `agent` 参数解析成非空值时，创建 `response_sessions` 行时一并写入 `agent_id`。

**读取路径**：续请求解析 `previous_response_id` → `fetch_session_for_turn()` 已有路径（`responses_service.py:240`）→ 从 session 对象读 `agent_id`。

**一致性校验**（新增函数，建议位置 `responses_service.py`）：

```python
def assert_agent_matches_session(
    sess: ResponseSession, request_agent: str | None
) -> str | None:
    """Return the agent_id to use, or raise 400 if mismatch."""
    if request_agent is None:
        return sess.agent_id  # 允许省略，从 session 恢复
    if sess.agent_id is None:
        # session 创建时无 agent，续请求也不应提供
        raise HTTPException(
            400,
            {"error": "agent_session_mismatch",
             "message": f"session has no agent binding; got {request_agent!r}"},
        )
    if request_agent != sess.agent_id:
        raise HTTPException(
            400,
            {"error": "agent_session_mismatch",
             "message": f"session bound to {sess.agent_id!r}, got {request_agent!r}"},
        )
    return sess.agent_id
```

## Cache 策略

复用现有 `context_cache_service`（已实现 KV 前缀复用）。

**Cache key**：`sha1(agent_id + agent_mtime_sum + skills_fingerprint)`，其中：
- `agent_mtime_sum` = `config.json`、`AGENT.md`、`SOUL.md`、`IDENTITY.md` 四个文件的 `os.path.getmtime()` 之和（浮点）
- `skills_fingerprint` = `sha1(sorted([(name, os.path.getmtime(SKILL.md)) for name in agent.skills]))`

**失效触发**：
- Agent 任一 md 或 config.json 文件 mtime 改变 → cache miss 一次重建
- Agent.skills 列表增删（反映在 config.json mtime 上）→ fingerprint 变
- 任一 skill 的 SKILL.md 任何变更（frontmatter 或 body）→ mtime 变 → fingerprint 变

注意：由于 cache key 里含 SKILL.md 的 mtime，body 更改也会失效 cache 稳定段；但 body 本身并未进入稳定段文本（仅 description 进），所以失效只是保守起见，实际影响是一次重建。这个保守性换来实现简单（不需要区分 frontmatter 变 vs body 变）。

**稳定段** 与 **动态段** 的分界即系统消息中的 `<!-- CACHE_BOUNDARY -->` 标记。动态段（`instructions` + runtime line）每请求重算，不参与 cache。

同一 agent 多 session 共享稳定段 cache，这是 lazy-readable 路线相对 function-tool 路线的额外收益（每 skill 独立 function 的 schema 本身也会随数量膨胀 cache 占用）。

## 错误与边界

| 场景 | HTTP / 行为 |
|------|------------|
| `agent` 指向不存在的 agent | 400 `agent_not_found` |
| `agent.skills` 中某 skill 目录不存在 | 跳过该 skill，warning 日志，不中断组装 |
| 模型 tool call `Skill(skill="foo")` 但 foo 不存在 | tool_result 返回 `{"error": "unknown skill: foo"}`，200 OK |
| `Skill(skill="")` / 缺 `skill` 字段 | tool_result 返回 `{"error": "skill name required"}`，200 OK |
| 续请求 `agent` 与 session 不符 | 400 `agent_session_mismatch`（含双方值） |
| 三个 md 全为空文件 + agent.skills 为空 | 退化为裸模型，debug 日志 `agent=xxx resolved to empty persona` |
| 首请求传了 `agent` 但 `agent.skills` 引用的 skill 全部无效 | system prompt 生成，但不包含 `<available_skills>` 段；warning 日志 |
| SOUL.md 存在但全是空白 | 跳过整个 Soul section（包括 persona 指令） |

## 测试策略

新增三个测试模块：

**`backend/tests/test_prompt_composer.py`**（单元）：
- 三 md 全存在 → 顺序 IDENTITY → SOUL → AGENT
- 只有 IDENTITY.md → 无 persona 指令，无 Agent section
- SOUL.md 存在 → persona 追加指令出现
- agent.skills 非空 → `<available_skills>` XML 段存在；每条 entry 有 `<name>` `<description>`
- `<!-- CACHE_BOUNDARY -->` 在正确位置（三件套 + skills 之后，instructions 之前）
- `instructions` 参数 → 出现在 cache boundary 之后
- agent 不存在 → 抛 `AgentNotFound`

**`backend/tests/test_responses_agent_binding.py`**（集成，SQLite in-memory）：
- 首请求传 `agent` → `response_sessions.agent_id` 被写入
- 续请求不传 `agent` → 能从 session 恢复，system prompt 正确
- 续请求传不同 `agent` → 400 `agent_session_mismatch`
- 续请求传相同 `agent` → 200（幂等）
- 首请求不传 `agent` → session.agent_id 为 NULL，裸模型行为

**`backend/tests/test_skill_tool_exec.py`**（单元）：
- 合法 skill name → 返回 `{skill, description, prompt, args}` 且 prompt 是 SKILL.md body
- 非法 skill name → 返回 `{error: "unknown skill: ..."}`，不抛异常
- 空 skill 字段 → 返回 `{error: "skill name required"}`
- `args` 透传

E2E（可选，第二阶段）：mock vLLM 端，实际发起 `/v1/responses` 调用，断言发给 vLLM 的 messages 数组的 system message 包含预期内容。

## 实现清单

变更影响的文件（按改动顺序）：

1. **新建** `backend/src/services/prompt_composer.py`
2. **修改** `backend/src/services/skill_tools.py` —— 加入 `Skill` tool 执行分支
3. **修改** `backend/src/models/response_session.py` —— 加 `agent_id` 列
4. **新建** `backend/src/db/migrations/versions/<new>_add_agent_id.py` —— Alembic 迁移
5. **修改** `backend/src/api/routes/responses.py` —— 请求 schema + composer 接入 + 一致性校验
6. **修改** `backend/src/api/routes/openai_compat.py` —— 同上
7. **修改** `backend/src/services/responses_service.py` —— 加 `assert_agent_matches_session()` helper
8. **新建** `backend/tests/test_prompt_composer.py`
9. **新建** `backend/tests/test_responses_agent_binding.py`
10. **新建** `backend/tests/test_skill_tool_exec.py`

预估工作量：2 天（含测试）。

## 参考实现

| 项目 | 路径 | 关键点 |
|------|------|--------|
| OpenClaw | `github-repos/openclaw/src/agents/system-prompt.ts` | `buildAgentSystemPrompt()` 定义了文件顺序、skills 段、cache boundary；本设计直接参考 |
| OpenClaw | `github-repos/openclaw/src/agents/skills/skill-contract.ts` | `formatSkillsForPrompt()` 的 XML 布局，本设计减去 `<location>` 字段 |
| OpenClaw | `github-repos/openclaw/src/agents/agent-scope.ts:48-65` | `resolveSessionAgentIds()` 的一致性校验语义 |
| Claude Code (claw-code) | `github-repos/claw-code/rust/crates/tools/src/lib.rs:557-570` | `Skill` tool 的 schema；本设计 1:1 对齐 |
| Claude Code (claw-code) | `github-repos/claw-code/rust/crates/tools/src/lib.rs:3176-3188` | `execute_skill()` 的后端执行语义 |

## NOT in scope

本设计明确不包含以下内容，留待后续 spec：

- **Workflow LLM 节点接入** —— workflow_executor 的 LLM 节点目前自构 messages，不走 `/v1/responses`。接入 composer 需要扩展节点配置（选 agent 的 UI 下拉）和 workflow 层的 composer 调用，单独一份 spec。
- **API key 级默认 agent 绑定** —— 允许临时 key 创建时预绑定 agent，调用方不传 `agent` 字段时兜底。第一版要求调用方显式传 `agent`，验证清晰的调用契约后再加此层，避免"到底用了哪个 agent"的排查成本。
- **请求级 skills override** —— 允许 `skills: [...]` 字段临时覆盖 agent 的默认 skills。OpenClaw 都没做，YAGNI；实际有需求时加字段 + cache key 扩展。
- **Agent 版本管理 / 回滚** —— 目前 agent.config.json 是可变文件；若未来需要 "agent v1.2" 这样的语义，要引入 agent 版本表。
- **多租户 skill 可见性** —— 当前 skill 对所有 agent 全局可见；若要做 per-org 或 per-instance 的 skill 仓库，单独设计。
- **Agent 的 model 字段驱动** —— 当前设计里 agent.config.json 的 `model` 字段不自动覆盖请求的 `model`（调用方显式传）。若希望 "agent 自带 model"，后续加参数优先级规则。
- **Skill args 的 structured schema** —— 目前 `args: string`，完全自由格式。若某些 skill 需要强类型参数，加 frontmatter `args_schema` 字段再扩展 `Skill` tool 定义。

## 下一步

1. 本文档提交，`docs/designs/2026-04-20-agent-skill-injection.md`
2. 用户 review → 修订或确认
3. 确认后，invoke writing-plans skill 产出实施 plan（`docs/superpowers/specs/2026-04-20-agent-skill-injection-impl.md`）
