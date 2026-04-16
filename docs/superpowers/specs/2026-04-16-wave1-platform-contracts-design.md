# Wave 1 · Platform Contracts 实现设计

> 父 plan：`docs/designs/2026-04-16-nous-center-v3-platform.md`
> 来源：coze-studio + hermes-agent 综合研判
> 工作量：~4.5 天

## Context

nous-center 已完成 Ark 对齐 8 步 + VL + 节点包管理。v3 的 Wave 1 把 platform contracts 一次性立好，上游 app（mediahub 等）接入不用等后续波。

本波交付 5 件事：事件扩展 + MemoryProvider ABC + ContextEngine ABC + 节点接口分层 + PGMemoryProvider reference 实现。

## 决策记录

1. **存储**：PostgreSQL（跟现有主业务栈一致）。`logs.db` SQLite 不动，那是独立模块。
2. **ABC 共享基类**：`PluginBase` 提供 `initialize / shutdown / system_prompt_block`；MemoryProvider 和 ContextEngine 都继承。避免两份相似 lifecycle 代码。
3. **hook 语义分级**：`fail-fast`（崩就崩，正确性优先，用于 initialize）vs `best-effort`（失败记 log 继续，用于 sync_turn / prefetch）vs `critical`（失败阻塞，用于 compress 时 context 塞不下）。每个 hook 方法签名里用 docstring 标明。
4. **prefetch 必须异步**：`asyncio.create_task(provider.prefetch(query))`，不阻塞当前响应。
5. **capabilities 字段**：从 `config.json` 的 `architectures` + `vision_config` + 手动 `configs/models.yaml` 覆盖综合推导（本波不实现 endpoint，留给 Wave 2 E3）。
6. **第三方插件安全边界**：spec 明写"nous-center 不对第三方 MemoryProvider/ContextEngine 的行为背书，用户需自评估"。提供内置 `PGMemoryProvider` 和 `GzipCompactContextEngine`，无需第三方也能跑。
7. **节点接口分层**：Python Protocol + runtime duck typing，不强制现有节点实现所有方法。
8. **Checkpoint 写入异步**：next wave 的 Checkpoint 表要用 background task，不让 workflow 执行变慢。

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

```python
from abc import ABC, abstractmethod
from typing import Any, TypedDict


class MemoryEntry(TypedDict):
    id: int
    category: str
    content: str
    context_key: str | None
    created_at: str  # ISO


class PluginBase(ABC):
    """Shared lifecycle for MemoryProvider & ContextEngine."""

    @abstractmethod
    async def initialize(self) -> None:
        """fail-fast: raise if cannot start. Called at app startup."""

    async def shutdown(self) -> None:
        """best-effort: cleanup resources. Log errors, don't raise."""

    def system_prompt_block(self, *, instance_id: int) -> str:
        """Static text injected into system prompt. Default empty."""
        return ""


class MemoryProvider(PluginBase):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def sync_turn(
        self,
        *,
        instance_id: int,
        api_key_id: int | None,
        user_content: str,
        assistant_content: str,
        context_key: str | None = None,
    ) -> int:
        """best-effort: write memory from a completed turn. Returns #entries written.
        Failures are logged; caller continues. Never raise to caller."""

    @abstractmethod
    async def prefetch(
        self,
        *,
        instance_id: int,
        query: str,
        limit: int = 10,
        context_key: str | None = None,
    ) -> list[MemoryEntry]:
        """best-effort: recall relevant memories for a query.
        Failures return empty list and log warning."""

    async def on_session_end(
        self, *, instance_id: int, turns: list[dict]
    ) -> None:
        """Optional: extract end-of-session facts. Default no-op."""

    async def on_pre_compress(
        self, *, instance_id: int, messages: list[dict]
    ) -> str | None:
        """Optional: extract before compression. Return summary text or None."""
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

### Task 4 · 节点接口分层（1d）

**文件：** `backend/src/services/workflow_executor.py`

定义 Protocol（Python 3.12 `typing.Protocol`）：

```python
from typing import Protocol, runtime_checkable, AsyncIterator

@runtime_checkable
class InvokableNode(Protocol):
    async def invoke(self, data: dict, inputs: dict) -> dict: ...

@runtime_checkable
class StreamableNode(Protocol):
    async def stream(
        self, data: dict, inputs: dict, on_token,
    ) -> AsyncIterator[dict]: ...

@runtime_checkable
class CollectableNode(Protocol):
    async def collect(
        self, data: dict, inputs_stream: AsyncIterator[dict],
    ) -> dict: ...
```

**改造：** 现有 `_exec_*` 函数签名改造成实现这些 Protocol。`_exec_llm` 同时实现 Invokable 和 Streamable（两种路径都支持）。

**重点：** **取消全局 `_on_progress_ref`**，改成显式传参（解决 token stats 坑的根因）。

**测试：**
- `tests/test_node_protocols.py`：Invokable / Streamable 各一个 minimal fixture

### Task 5 · PGMemoryProvider reference 实现（1d）

**文件：** `backend/src/services/memory/pg_provider.py`（新建） + `backend/src/models/memory.py`（新建）

```python
class PGMemoryProvider(MemoryProvider):
    name = "pg"

    def __init__(self, session_factory):
        self._sf = session_factory

    async def initialize(self):
        # schema check; raise if tables missing
        async with self._sf() as s:
            await s.execute(text("SELECT 1 FROM memory_entries LIMIT 1"))

    async def sync_turn(self, *, instance_id, api_key_id, user_content, assistant_content, context_key=None):
        try:
            extracted = _extract_facts(user_content, assistant_content)
            async with self._sf() as s:
                for entry in extracted:
                    s.add(MemoryEntryModel(
                        instance_id=instance_id,
                        api_key_id=api_key_id,
                        category=entry["category"],
                        content=entry["content"],
                        context_key=context_key,
                    ))
                await s.commit()
            return len(extracted)
        except Exception as e:
            logger.exception("PGMemoryProvider.sync_turn failed for instance=%s", instance_id)
            return 0

    async def prefetch(self, *, instance_id, query, limit=10, context_key=None):
        try:
            async with self._sf() as s:
                stmt = select(MemoryEntryModel).where(
                    MemoryEntryModel.instance_id == instance_id,
                )
                if context_key:
                    stmt = stmt.where(MemoryEntryModel.context_key == context_key)
                # FTS 或简单 LIKE（本波用 FTS，GIN 索引已建）
                stmt = stmt.where(
                    text("to_tsvector('simple', content) @@ plainto_tsquery(:q)")
                ).bindparams(q=query)
                stmt = stmt.order_by(desc(MemoryEntryModel.created_at)).limit(limit)
                rows = (await s.execute(stmt)).scalars().all()
                return [_to_entry(r) for r in rows]
        except Exception:
            logger.warning("PGMemoryProvider.prefetch failed; returning empty")
            return []

    def system_prompt_block(self, *, instance_id):
        # 同步 API，不能 await；返回静态声明让 LLM 知道有记忆能力
        return "You have access to long-term memory for this user (managed by the platform)."
```

**`_extract_facts`：** Wave 1 用简单启发式（用户显式 "记住"、"我喜欢"、"我是" 等触发词）。Wave 2+ 可以换 LLM 抽取。

**HTTP endpoint：** `POST /api/v1/memory/sync`（接 mediahub 调用）
- body: `{user_content, assistant_content, context_key?}`
- 鉴权：`verify_bearer_token`
- 调用 `PGMemoryProvider.sync_turn(instance_id, ...)` 并返回 `{entries_written}`

**测试：**
- `tests/test_pg_memory_provider.py` 继承 `AbstractMemoryProviderTests`
- + PG-specific：FTS 查询 / context_key 过滤 / 跨 instance 隔离

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

## Reviewer Concerns

（spec review loop 产出，留白。运行 /plan-eng-review 后填。）
