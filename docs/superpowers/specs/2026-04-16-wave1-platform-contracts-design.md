# Wave 1 · Platform Contracts 实现设计

> 父 plan：`docs/designs/2026-04-16-nous-center-v3-platform.md`
> 来源：coze-studio + hermes-agent 综合研判
> 工作量：**~7.5 天**（原估 4.5 天 + Eng Review 加强 3 天）

## Eng Review 修订（2026-04-16）

下方决策 + 架构细节都已经过 `/plan-eng-review`，并按 review 结果修订。修订点：
1. **async system_prompt_block**（Issue 1A）—— 见决策 #8
2. **ClientError vs ProviderError 分级**（Issue 1B）—— 见决策 #3 修订版
3. **ContextEngine 无状态**（Issue 1C）—— 见决策 #9
4. **全 15 个 `_exec_*` 改 class**（Issue 2A）—— Task 4 展开
5. **nous-center 只存，mediahub 抽取**（Issue 2B）—— `sync_turn` → `add_entries`
6. **空 entries 幂等**（Issue 3A）—— 决策 #10
7. **Eval harness 完整版**（Issue 3B）—— 决策 #12 新增 Task 6
8. **批量写入 100 条硬上限**（Issue 4B）—— 决策 #11

## Context

nous-center 已完成 Ark 对齐 8 步 + VL + 节点包管理。v3 的 Wave 1 把 platform contracts 一次性立好，上游 app（mediahub 等）接入不用等后续波。

本波交付 5 件事：事件扩展 + MemoryProvider ABC + ContextEngine ABC + 节点接口分层 + PGMemoryProvider reference 实现。

## 决策记录

1. **存储**：PostgreSQL（跟现有主业务栈一致）。`logs.db` SQLite 不动，那是独立模块。
2. **ABC 共享基类**：`PluginBase` 提供 `initialize / shutdown / system_prompt_block`；MemoryProvider 和 ContextEngine 都继承。避免两份相似 lifecycle 代码。
3. **hook 语义分三级（Eng Review 修订）：**
   - **fail-fast**：崩就崩。适用：`initialize`、所有 `ClientError` 子类（参数错、鉴权错）
   - **best-effort**：失败记 log 继续。适用：`add_entries`/`prefetch` 遇到 `ProviderError`（DB hiccup）
   - **critical**：失败阻塞。适用：`compress` 时 context 塞不下 → raise `ContextOverflowError`
   - 每个方法 docstring 明标语义。**ABC 暴露两类 Exception：`ClientError`（must raise）+ `ProviderError`（should swallow）**。
4. **prefetch 必须异步**：`asyncio.create_task(provider.prefetch(query))`，不阻塞当前响应。
5. **capabilities 字段**：从 `config.json` 的 `architectures` + `vision_config` + 手动 `configs/models.yaml` 覆盖综合推导（本波不实现 endpoint，留给 Wave 2 E3）。
6. **第三方插件安全边界**：spec 明写"nous-center 不对第三方 MemoryProvider/ContextEngine 的行为背书，用户需自评估"。提供内置 `PGMemoryProvider` 和 `GzipCompactContextEngine`，无需第三方也能跑。
7. **节点接口分层（Eng Review 修订 2A）**：**全 15 个 `_exec_*` function 改造成 class**，每个 class 显式实现 `InvokableNode` / `StreamableNode` / 二者兼具的 Protocol。`runtime_checkable` 验证。
8. **`system_prompt_block` 改 async（Eng Review 修订 1A）**：签名 `async def system_prompt_block(...) -> str`，Python 自然兼容同步实现（直接 `return ""`）。
9. **ContextEngine 无状态（Eng Review 修订 1C）**：engine 实例不持有会话状态。`update_from_response` 如果要累计 token，改写到 `ResponseSession.total_input_tokens/total_output_tokens`（现有字段），engine 不持久化。
10. **add_entries 空列表幂等（Eng Review 修订 3A）**：`add_entries(entries=[])` → 返回 `[]`，不写 DB，不 raise。
11. **批量 entries 硬上限（Eng Review 修订 4B）**：`len(entries) <= 100` per call，超过返回 400 `too_many_entries`。
12. **Wave 1 含 eval harness（Eng Review 修订 3B）**：Task 6 新增，hermes 风格 JSONL fixture + 批量跑 GzipCompactContextEngine + 自动评分。
13. **职责边界（Eng Review 修订 2B）**：nous-center 只存不抽取。`sync_turn(user, asst)` 签名改为 `add_entries(entries: list[MemoryEntry])`。上游 app 负责"这段对话里什么值得记"这件事。
14. **Checkpoint 写入异步**：next wave 的 Checkpoint 表要用 background task，不让 workflow 执行变慢。

## Schema（PostgreSQL）

```sql
-- 2026-04-16 Wave 1
CREATE TABLE memory_entries (
    id            BIGSERIAL PRIMARY KEY,
    instance_id   BIGINT NOT NULL REFERENCES service_instances(id) ON DELETE CASCADE,
    api_key_id    BIGINT,                                   -- 可空，允许 instance 级条目
    category      VARCHAR(32) NOT NULL,                     -- 'preference' | 'fact' | 'instruction' | 'custom'
    content       TEXT NOT NULL,
    context_key   VARCHAR(128),                             -- 业务层自定义分组键（如 project_id）
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_mem_inst_created ON memory_entries (instance_id, created_at DESC);
CREATE INDEX idx_mem_inst_key_cat ON memory_entries (instance_id, context_key, category);
CREATE INDEX idx_mem_content_fts  ON memory_entries USING GIN (to_tsvector('simple', content));

CREATE TABLE memory_embeddings (
    entry_id      BIGINT PRIMARY KEY REFERENCES memory_entries(id) ON DELETE CASCADE,
    model         VARCHAR(64) NOT NULL,                     -- embedding 模型名
    dim           INT NOT NULL,                             -- 向量维度
    vector        BYTEA                                    -- 目前存 raw bytes，未来上 pgvector 换 vector 类型
);
```

## 任务拆解

### Task 1 · 事件扩展（0.5d）

**文件：** `backend/src/services/workflow_executor.py` + `backend/src/api/routes/workflows.py`

现有事件：`node_start` / `node_stream` / `node_complete` / `node_error` / `complete`

新增事件（参考 coze `execute/event.go`）：

| 事件 | 触发时机 | 用途 |
|------|----------|------|
| `node_end_streaming` | 当最后一个 chunk 发出后 | token stats 精准计算（vs `node_complete` 是逻辑完成时刻） |
| `workflow_interrupt` | 执行到 QA 节点时 | 预留 human-in-the-loop hook |
| `workflow_resume` | 从 interrupt 恢复 | 同上 |
| `function_call` | LLM 发起 tool call | 预留 tool-use 事件口 |
| `tool_response` | tool 返回结果 | 同上 |
| `tool_streaming_response` | tool 流式返回 | 同上 |

**实现要点：**
- 只加事件类型定义 + 发事件的位置（不做 QA 节点实现，那是 Wave 外）
- `node_end_streaming` 在 `_exec_llm` 流式路径末尾发（所有 chunk 发完 + usage 塞好后）
- 前端 `openProgressChannel` ws handler 识别新事件并 dispatch 为 CustomEvent

**测试：**
- `tests/test_event_types.py`：每个新事件至少一条 fixture，验证 payload schema
- 现有 workflow 兼容性测试：老 event 不变

### Task 2 · MemoryProvider ABC（1d）

**文件：** `backend/src/services/memory/base.py`（新建）

**Eng Review 修订版**：签名已按决策 3/8/13 调整。

```python
from abc import ABC, abstractmethod
from typing import Any, TypedDict


class MemoryProviderError(Exception):
    """Base class for all MemoryProvider errors."""


class MemoryProviderClientError(MemoryProviderError):
    """Caller did something wrong (bad input, auth, oversized). MUST be raised."""


class MemoryProviderInternalError(MemoryProviderError):
    """Transient infra failure (DB hiccup, etc). Callers' choice to retry;
    best-effort methods swallow this and log."""


class MemoryEntry(TypedDict):
    category: str         # 'preference' | 'fact' | 'instruction' | 'custom'
    content: str          # max 10KB per entry (enforced)
    context_key: str | None


class StoredMemoryEntry(MemoryEntry):
    id: int
    instance_id: int
    created_at: str  # ISO


class PluginBase(ABC):
    """Shared lifecycle for MemoryProvider & ContextEngine."""

    @abstractmethod
    async def initialize(self) -> None:
        """fail-fast: raise if cannot start. Called at app startup."""

    async def shutdown(self) -> None:
        """best-effort: cleanup resources. Log errors, don't raise."""

    async def system_prompt_block(self, *, instance_id: int) -> str:
        """Static text or dynamically fetched. Default empty string.
        Must return quickly (<50ms); do not block on LLM calls here."""
        return ""


class MemoryProvider(PluginBase):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def add_entries(
        self,
        *,
        instance_id: int,
        api_key_id: int | None,
        entries: list[MemoryEntry],
        context_key: str | None = None,
    ) -> list[int]:
        """Store entries; return new entry ids.

        - Empty entries list: return []; no DB write; no raise.
        - entries > 100 OR any content > 10KB: raise MemoryProviderClientError
        - instance_id not authorized: raise MemoryProviderClientError
        - DB transient failure: raise MemoryProviderInternalError (caller may swallow)
        """

    @abstractmethod
    async def prefetch(
        self,
        *,
        instance_id: int,
        query: str,
        limit: int = 10,
        context_key: str | None = None,
    ) -> list[StoredMemoryEntry]:
        """best-effort: recall relevant memories for a query.
        Returns empty list on MemoryProviderInternalError (logs warning).
        Raises MemoryProviderClientError on bad input (unauthorized instance)."""

    async def on_session_end(
        self, *, instance_id: int, turns: list[dict]
    ) -> None:
        """Optional: post-session hook. Default no-op.
        Implementations may extract facts here (nous-center does not)."""

    async def on_pre_compress(
        self, *, instance_id: int, messages: list[dict]
    ) -> str | None:
        """Optional: extract before ContextEngine compression. Return summary or None."""
```

**实现要点：**
- 所有 async 方法有 `try/except` + log + best-effort 返回，符合决策 #3
- `sync_turn` 永不 raise（除非 initialize 失败）
- docstring 明标 fail-fast / best-effort 语义

**测试：**
- `tests/test_memory_provider_abc.py`：`AbstractMemoryProviderTests` mixin，社区实现继承就跑所有 contract
  - `test_sync_writes_entries`
  - `test_prefetch_returns_relevant`
  - `test_sync_swallows_errors`（注入 DB fail → 返回 0 不 raise）
  - `test_cross_instance_isolation`
  - `test_context_key_scoping`

### Task 3 · ContextEngine ABC（1d）

**文件：** `backend/src/services/context/base.py`（新建）

```python
class ContextEngine(PluginBase):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def should_compress(
        self, *, messages: list[dict], max_tokens: int, current_tokens: int | None = None,
    ) -> bool: ...

    @abstractmethod
    async def compress(
        self, *, messages: list[dict], max_tokens: int,
    ) -> tuple[list[dict], bool]:
        """critical: must succeed or raise. If context too large to fit even
        after compression, raise ContextOverflowError."""

    def update_from_response(self, usage: dict) -> None:
        """Optional: track running usage. Default no-op."""
```

**重构：** `backend/src/services/responses_service.py:compact_messages()` 的硬编码实现迁到 `backend/src/services/context/gzip_compact.py` 作为 `GzipCompactContextEngine`（内置默认实现）。

**测试：**
- `tests/test_context_engine_abc.py`：`AbstractContextEngineTests` mixin
  - `test_should_compress_threshold`
  - `test_compress_reduces_tokens`
  - `test_last_turn_preserved`
  - `test_system_message_preserved`
  - `test_overflow_raises`
- 现有 `test_responses_service.py::test_compact_*` 测试改成调 `GzipCompactContextEngine`，确保行为不变。

### Task 4 · 节点接口分层（2d，Eng Review 升级）

**Eng Review 2A 决定：全 15 个 `_exec_*` function 改 class**。工作量从 1d 升到 2d。

**文件：** `backend/src/services/workflow_executor.py` + 新增 `backend/src/services/nodes/` 目录

```python
# backend/src/services/nodes/base.py
from typing import Protocol, runtime_checkable, AsyncIterator

@runtime_checkable
class InvokableNode(Protocol):
    async def invoke(self, data: dict, inputs: dict) -> dict: ...

@runtime_checkable
class StreamableNode(Protocol):
    """Note: accepts explicit on_token callback (no more _on_progress_ref)."""
    async def stream(
        self,
        data: dict,
        inputs: dict,
        on_token,   # async callback(token: str) -> None
    ) -> dict:
        """Returns final result dict after streaming completes."""

@runtime_checkable
class CollectableNode(Protocol):
    async def collect(
        self, data: dict, inputs_stream: AsyncIterator[dict],
    ) -> dict: ...

@runtime_checkable
class TransformableNode(Protocol):
    async def transform(
        self, data: dict, inputs_stream: AsyncIterator[dict],
    ) -> AsyncIterator[dict]: ...
```

**迁移：** 15 个 `_exec_*` 函数一一变 class：

```python
# 例：_exec_text_input
class TextInputNode:  # implements InvokableNode (duck typing)
    async def invoke(self, data, inputs):
        return {"text": data.get("text", "")}

# 例：_exec_llm（多维度）
class LLMNode:  # implements InvokableNode + StreamableNode
    async def invoke(self, data, inputs):
        # 非流式路径
        ...
    async def stream(self, data, inputs, on_token):
        # 流式路径；**on_token 显式传参**，彻底取消 _on_progress_ref
        ...
```

**注册表：** `_NODE_CLASSES: dict[str, type] = {"text_input": TextInputNode, ...}`。
`WorkflowExecutor._execute_node` 改成 instantiate + dispatch based on protocol 断言。

**完整列表（15 个）：** text_input / multimodal_input / ref_audio / tts_engine / resample / mixer / concat / bgm_mix / output / text_output / llm / prompt_template / selector / skill / agent （加上 plugin 包里动态发现的，总共 15+）。

**关键回归测试：** `test_llm_token_stats_regression.py` — 现有 token stats 功能在改造后必须继续 work。

**测试：**
- `tests/test_node_protocols.py`：Invokable / Streamable / Collectable / Transformable 各一个 minimal fixture
- `tests/test_node_class_migration.py`：每个 class 跟老 function 行为等价（参数→输出）
- **CRITICAL REGRESSION：** `test_workflow_llm_token_stats.py` — 确保前端 token stats 不回归

### Task 5 · PGMemoryProvider reference 实现（1d，Eng Review 修订）

**Eng Review 2B 决定：nous-center 只存不抽取**。所以**没有 `_extract_facts`**。签名 `add_entries` 直接接受结构化输入。

**文件：** `backend/src/services/memory/pg_provider.py`（新建） + `backend/src/models/memory.py`（新建）

```python
from src.services.memory.base import (
    MemoryProvider, MemoryEntry, StoredMemoryEntry,
    MemoryProviderClientError, MemoryProviderInternalError,
)

MAX_ENTRY_BYTES = 10 * 1024  # 10KB per entry
MAX_BATCH_SIZE = 100


class PGMemoryProvider(MemoryProvider):
    name = "pg"

    def __init__(self, session_factory):
        self._sf = session_factory

    async def initialize(self):
        """fail-fast: raise ProgrammingError if migration not applied."""
        async with self._sf() as s:
            await s.execute(text("SELECT 1 FROM memory_entries LIMIT 1"))

    async def add_entries(self, *, instance_id, api_key_id, entries, context_key=None):
        # Empty batch: idempotent no-op
        if not entries:
            return []

        # ClientError: batch too large
        if len(entries) > MAX_BATCH_SIZE:
            raise MemoryProviderClientError(
                f"entries exceeds max batch size {MAX_BATCH_SIZE}"
            )

        # ClientError: per-entry content too large
        for i, e in enumerate(entries):
            if len(e.get("content", "").encode()) > MAX_ENTRY_BYTES:
                raise MemoryProviderClientError(
                    f"entries[{i}].content exceeds {MAX_ENTRY_BYTES} bytes"
                )

        # ClientError: instance not authorized (caller's responsibility,
        # but defensive check)
        # (skipped — HTTP layer verifies instance_id via bearer token)

        try:
            async with self._sf() as s:
                rows = [
                    MemoryEntryModel(
                        instance_id=instance_id,
                        api_key_id=api_key_id,
                        category=e["category"],
                        content=e["content"],
                        context_key=context_key,
                    )
                    for e in entries
                ]
                s.add_all(rows)
                await s.flush()
                new_ids = [r.id for r in rows]
                await s.commit()
                return new_ids
        except (DBAPIError, asyncio.TimeoutError) as exc:
            raise MemoryProviderInternalError(str(exc)) from exc

    async def prefetch(self, *, instance_id, query, limit=10, context_key=None):
        try:
            async with self._sf() as s:
                stmt = select(MemoryEntryModel).where(
                    MemoryEntryModel.instance_id == instance_id,
                )
                if context_key:
                    stmt = stmt.where(MemoryEntryModel.context_key == context_key)
                if query:
                    stmt = stmt.where(
                        text("to_tsvector('simple', content) @@ plainto_tsquery(:q)")
                    ).bindparams(q=query)
                stmt = stmt.order_by(desc(MemoryEntryModel.created_at)).limit(limit)
                rows = (await s.execute(stmt)).scalars().all()
                return [_to_stored_entry(r) for r in rows]
        except (DBAPIError, asyncio.TimeoutError):
            logger.warning("PGMemoryProvider.prefetch failed; returning empty")
            return []

    async def system_prompt_block(self, *, instance_id):
        # async version (decision #8). Returns static hint for now.
        # Future: query recent entries and return digest.
        return "You have access to long-term memory for this user (managed by the platform)."
```

**HTTP endpoint：** `POST /api/v1/memory/sync`（接 mediahub 调用）
- body: `{entries: [{category, content, context_key?}], context_key?}`
- 鉴权：`verify_bearer_token`
- 调用 `PGMemoryProvider.add_entries(instance_id, api_key_id, entries, context_key)` 返回 `{entry_ids: [...]}`
- ClientError → 400，ProviderError → 500

**Related endpoint：** `GET /api/v1/memory/prefetch?q=...&limit=10&context_key=...`
- 调用 `PGMemoryProvider.prefetch(...)` 返回 `{entries: [...]}`

**测试：**
- `tests/test_pg_memory_provider.py` 继承 `AbstractMemoryProviderTests`
- + PG-specific：
  - FTS GIN 索引命中
  - context_key 过滤
  - 跨 instance 隔离（instance A add，instance B prefetch 查不到）
  - batch 101 条 → ClientError 400
  - content 11KB → ClientError 400
  - 空 entries → 返回 `[]`（幂等）
  - DB 挂 → InternalError raise（endpoint 转 500）

### Task 6 · Eval Harness（Eng Review 新增，2d）

**Eng Review 3B 决定：完整 eval 框架**。

**目的：** 保证 `GzipCompactContextEngine.compress()` 在真实长对话上质量不回归。

**文件：**
- `backend/tests/evals/compact/fixtures.jsonl` — 人工挑的 10 段真实长对话（30+ 轮）
- `backend/tests/evals/compact/runner.py` — 批量跑 eval 的脚本
- `backend/tests/evals/compact/scorer.py` — 自动评分（基于 LLM-as-judge）
- `backend/tests/evals/compact/baselines/` — 历史 baseline 快照

**Fixture 格式：**
```jsonl
{"id": "multi-turn-summary", "conversation": [...], "test_prompt": "总结前面的对话", "must_contain": ["火锅", "麻辣烫"]}
{"id": "cross-turn-reference", "conversation": [...], "test_prompt": "把第 3 条建议再展开", "must_contain": ["建议 3"]}
...
```

**Runner：**
```python
# pseudocode
for fixture in fixtures:
    compressed = await engine.compress(fixture.conversation, max_tokens=4000)
    response = await llm.chat(compressed + [{"role":"user","content":fixture.test_prompt}])
    score = scorer.check(response, fixture.must_contain)
    append_result(fixture.id, compressed_tok_count, score, response)
```

**Scorer：** LLM-as-judge 用 `qwen3_5_35b_a3b_gptq_int4`（或 auxiliary model）打分：`0-10` 分，含 `must_contain` 所有关键信息得 10，完全缺失得 0。

**Baseline 比对：** 新 compress 实现 → 跑 eval → 跟 `baselines/gzip_compact_latest.json` 比，分差 > 2 视为回归。

**命令：**
```bash
python -m backend.tests.evals.compact.runner
# → 输出平均分、每 fixture 对比、回归警告
```

**不跑 CI（决策：Wave 1 有 harness，CI 联动留 Wave 2+）。**手动 ship 前跑一次。

## Error Map（必填，解决 Section 2 GAP）

| 方法 | 可能出错 | 异常类 | 处理策略 | 用户看到 |
|------|---------|--------|---------|---------|
| `PGMemoryProvider.sync_turn` | PG pool exhausted | `asyncio.TimeoutError` / `DBAPIError` | log + return 0（best-effort） | 无感知 |
| `PGMemoryProvider.sync_turn` | FK violation（instance 被删） | `IntegrityError` | log + return 0 | 无感知 |
| `PGMemoryProvider.prefetch` | query timeout | `asyncio.TimeoutError` | log + return [] | 空记忆，不报错 |
| `PGMemoryProvider.initialize` | 表不存在 | `ProgrammingError` | **raise** | 启动失败，需运行 migration |
| `GzipCompactContextEngine.compress` | 极端情况：最后一轮仍超 | `ContextOverflowError` | **raise** | 400 `input_too_long_for_model` |
| `/api/v1/memory/sync` | instance_id 缺失 | `AuthenticationError` | 401 (auto) | 鉴权错误 |
| `/api/v1/memory/sync` | 非法 JSON body | `InvalidRequestError` | 400 | 请求格式错 |
| Workflow event broadcast | ws channel 无订阅者 | N/A | silent skip | 无影响 |

## 安全边界声明

- 本波**不接受任何 external MemoryProvider 插件**，只内置 `PGMemoryProvider`。
- ContextEngine 同理，只内置 `GzipCompactContextEngine`。
- 未来支持第三方实现时（Wave N+），会在 `docs/plugins/` 写明：
  - 插件运行在 nous-center 进程内，有完整的 DB / 文件系统访问权限
  - 用户需自评估插件来源的可信度
  - 插件不能调用未经用户同意的外部服务

## 不做的事

- **不做** pgvector 集成（留 schema 字段，等真有需求）
- **不做** LLM-based fact extraction（启发式够用于 reference）
- **不做** Redis-based MemoryProvider（未来 wave）
- **不做** workflow 端的 MemoryProvider 自动调用（上游 app 主动调 `/api/v1/memory/sync`）
- **不做** 第三方插件加载机制（Wave N+）

## 验证

```bash
# 1. migration
psql $DATABASE_URL -f backend/migrations/wave1_memory.sql

# 2. 启动 backend，确认 PGMemoryProvider.initialize 不 raise
.venv/bin/python -m uvicorn src.api.main:create_app --factory --host 0.0.0.0 --port 8000

# 3. ABC contract test
.venv/bin/python -m pytest tests/test_memory_provider_abc.py tests/test_context_engine_abc.py tests/test_pg_memory_provider.py -q

# 4. 事件扩展
# 在浏览器跑 workflow，DevTools Network 应能看到 node_end_streaming 事件

# 5. mediahub 侧验收（人工）
curl -X POST /api/v1/memory/sync -H "Authorization: Bearer sk-..." \
  -d '{"user_content":"我喜欢简洁回复","assistant_content":"好的，我会简洁","context_key":"project-1"}'
# → {"entries_written": 1}

curl -X GET "/api/v1/memory/prefetch?q=简洁&limit=5" -H "Authorization: Bearer sk-..."
# → [{"id":1,"category":"preference","content":"用户喜欢简洁回复",...}]
```

## 回滚

```sql
DROP TABLE memory_embeddings;
DROP TABLE memory_entries;
```

ABC 代码留着无害（没人继承就没人调用）。

## Next

Wave 1 完成后：
- 发内部公告："MemoryProvider ABC 可用，mediahub 可对接"
- 进入 Wave 2 spec：`2026-04-17-wave2-reliability-design.md`

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | clean | SCOPE_EXPANSION, 4 expansions accepted |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | clean | 7 issues raised + resolved inline |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | N/A | no UI scope |
| Outside Voice | `/codex plan review` | Independent 2nd opinion | 0 | skipped | user-driven pace |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

**UNRESOLVED:** 0
**VERDICT:** CEO + ENG CLEARED — ready to implement Wave 1
