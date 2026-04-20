# Wave 1 · Platform Contracts 实施 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：** 一次性立好 nous-center 的 5 个协议级契约：事件扩展 + MemoryProvider ABC + ContextEngine ABC + 节点接口分层 + PGMemoryProvider reference 实现 + Eval harness。让 mediahub 等上层 app 通过 ABC/契约集成。

**Architecture：** 引入 `PluginBase` 共享生命周期 → `MemoryProvider` 与 `ContextEngine` 继承；workflow 节点从 function 改 class，实现 `InvokableNode` / `StreamableNode` Protocol；新增 `memory_entries` + `memory_embeddings` 两张 PG 表；保留内置实现（PG / GzipCompact），不对第三方插件背书。

**Tech Stack：** Python 3.12, FastAPI, SQLAlchemy async (PG + SQLite), pytest, vLLM, LLM-as-judge (eval)。

**Spec：** `docs/superpowers/specs/2026-04-16-wave1-platform-contracts-design.md`（已过 Eng review，7 issues resolved）

**并行分支策略（选项 C）：** 本 plan 在独立 worktree 下执行，分支 `feature/wave1-platform-contracts`（从 `feature/nous-center-v2` fork）。与 `feature/agent-skill-injection` 分支并行；合并到 `feature/nous-center-v2` 时各自独立 PR。

---

## File Structure

```
backend/
  src/
    services/
      workflow_executor.py       # 改：全 12 个 _exec_* 迁 class + 新事件
      responses_service.py       # 改：compact_messages 迁到 GzipCompactContextEngine
      memory/                    # 新增
        __init__.py
        base.py                  # PluginBase + MemoryProvider ABC + 异常类
        pg_provider.py           # PGMemoryProvider 实现
      context/                   # 新增
        __init__.py
        base.py                  # ContextEngine ABC + ContextOverflowError
        gzip_compact.py          # 内置默认实现（迁移自 compact_messages）
      nodes/                     # 新增
        __init__.py
        base.py                  # InvokableNode / StreamableNode / CollectableNode / TransformableNode Protocol
        text_input.py            # 3 个简单节点放一起
        llm.py                   # LLMNode（Invokable + Streamable）
        others.py                # 剩余节点
        registry.py              # _NODE_CLASSES 注册表
    models/
      memory.py                  # 新：MemoryEntryModel + MemoryEmbeddingModel
    api/routes/
      memory.py                  # 新：POST /api/v1/memory/sync + GET /api/v1/memory/prefetch
      workflows.py               # 改：新事件类型识别
  migrations/
    wave1_memory.sql             # 新：CREATE TABLE memory_entries + memory_embeddings
  tests/
    test_event_types.py          # 新事件 fixtures
    test_memory_provider_abc.py  # AbstractMemoryProviderTests mixin
    test_context_engine_abc.py   # AbstractContextEngineTests mixin
    test_pg_memory_provider.py   # PG 特化测试
    test_gzip_compact_engine.py  # compact 迁移后行为等价
    test_node_protocols.py       # Protocol 断言 + 12 节点迁移等价
    test_workflow_llm_token_stats.py  # CRITICAL REGRESSION
    test_api_memory.py           # /api/v1/memory/sync + /prefetch 集成
    evals/compact/
      fixtures.jsonl
      runner.py
      scorer.py
      baselines/gzip_compact_v1.json
```

---

## Task 1 · 事件扩展（0.5 天）

**Files:**
- Modify: `backend/src/services/workflow_executor.py`
- Modify: `backend/src/api/routes/workflows.py`（ws handler）
- Test: `backend/tests/test_event_types.py`

### Subtask 1.1 — 定义新事件类型常量

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_event_types.py
from src.services.workflow_executor import EVENT_TYPES


def test_all_new_event_types_defined():
    expected_new = {
        "node_end_streaming",
        "workflow_interrupt",
        "workflow_resume",
        "function_call",
        "tool_response",
        "tool_streaming_response",
    }
    assert expected_new.issubset(set(EVENT_TYPES))


def test_existing_event_types_preserved():
    """Ensure no regression on existing events."""
    expected_existing = {"node_start", "node_stream", "node_complete", "node_error", "complete"}
    assert expected_existing.issubset(set(EVENT_TYPES))
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_event_types.py -v`
Expected: FAIL — `ImportError: cannot import 'EVENT_TYPES'`

- [ ] **Step 3: 在 workflow_executor.py 顶部加常量**

```python
# backend/src/services/workflow_executor.py (near top, after imports)

EVENT_TYPES: tuple[str, ...] = (
    # Existing events
    "node_start",
    "node_stream",
    "node_complete",
    "node_error",
    "complete",
    # Wave 1 new events (coze-style)
    "node_end_streaming",        # 流式最后一个 chunk 发出后触发（vs node_complete 是逻辑完成点）
    "workflow_interrupt",        # QA 节点等需要 human-in-the-loop 时触发（本波只占位，不实现节点）
    "workflow_resume",           # 从 interrupt 恢复时触发
    "function_call",             # LLM 发起 tool call 时触发（预留 tool-use 事件）
    "tool_response",             # tool 返回结果
    "tool_streaming_response",   # tool 流式返回
)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_event_types.py -v`
Expected: PASS（2 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/workflow_executor.py backend/tests/test_event_types.py
git commit -m "feat(events): declare Wave 1 event types (node_end_streaming + 5 more)"
```

### Subtask 1.2 — 在 LLM 流式末尾发 `node_end_streaming`

- [ ] **Step 1: 写失败测试**

追加到 `backend/tests/test_event_types.py`：

```python
import pytest


@pytest.mark.asyncio
async def test_llm_stream_emits_node_end_streaming(mock_llm_stream, on_progress_capture):
    """After _exec_llm streams all chunks and resolves usage, node_end_streaming fires."""
    # mock_llm_stream / on_progress_capture 用下面的 fixture
    from src.services.workflow_executor import _exec_llm

    data = {"_node_id": "llm-1", "model": "qwen3.5", "messages": [{"role": "user", "content": "hi"}]}
    await _exec_llm(data, {})
    event_types = [e["type"] for e in on_progress_capture.events]
    assert "node_end_streaming" in event_types
    # 顺序：node_stream* → node_end_streaming
    idx_stream = next(i for i, t in enumerate(event_types) if t == "node_stream")
    idx_end = event_types.index("node_end_streaming")
    assert idx_stream < idx_end
```

Conftest 加 fixture（若不存在）：

```python
# backend/tests/conftest.py 追加
@pytest.fixture
def on_progress_capture(monkeypatch):
    """Capture all events emitted via _on_progress_ref."""
    class _Cap:
        def __init__(self):
            self.events: list[dict] = []
        async def __call__(self, ev):
            self.events.append(ev)
    cap = _Cap()
    from src.services import workflow_executor
    monkeypatch.setattr(workflow_executor, "_on_progress_ref", cap)
    return cap


@pytest.fixture
def mock_llm_stream(monkeypatch):
    """Replace _call_llm streaming with a deterministic 3-chunk stream."""
    # 实际实现依赖 _exec_llm 里调的具体 httpx / vllm 调用点；
    # 此 fixture 的实现细节在 workflow_executor.py 的现状上微调。
    # 若 _exec_llm 用 httpx.AsyncClient post → 用 respx 拦；
    # 若用自定义 adapter → monkeypatch adapter.chat_completion()。
    # TODO during implementation: 打开 _exec_llm 代码挑一种方式。
    import httpx
    from unittest.mock import AsyncMock

    async def _fake_stream(*args, **kwargs):
        chunks = [
            {"choices": [{"delta": {"content": "hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}},
        ]
        for c in chunks:
            yield c

    # 注意：此处需根据 _exec_llm 实际的 streaming 接口 monkeypatch。
    # 占位：打 marker 让执行者在 Subtask 1.2 实现时补全。
    monkeypatch.setattr(
        "src.services.workflow_executor._stream_llm", _fake_stream, raising=False
    )
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_event_types.py::test_llm_stream_emits_node_end_streaming -v`
Expected: FAIL — `node_end_streaming` 未被发

- [ ] **Step 3: 在 `_exec_llm` 流式路径末尾加发事件**

打开 `backend/src/services/workflow_executor.py`，找到 `_exec_llm()` 函数流式分支。在所有 `node_stream` chunk 发完 + usage 已塞好之后，加：

```python
    # 在 _exec_llm 流式分支，usage 塞好之后、return 之前
    if _on_progress_ref is not None:
        await _on_progress_ref({
            "type": "node_end_streaming",
            "node_id": data.get("_node_id"),
            "usage": usage,  # 最终 usage（含 total_tokens）
        })
    return result  # 保持原有返回
```

（具体插入行号取决于 `_exec_llm` 现有代码结构。实施时先 `grep -n "node_stream" workflow_executor.py` 定位。）

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_event_types.py -v`
Expected: PASS（3 tests）

- [ ] **Step 5: 现有 workflow 测试无回归**

Run: `cd backend && pytest tests/test_api_workflows.py -v`
Expected: 所有通过

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/workflow_executor.py backend/tests/test_event_types.py backend/tests/conftest.py
git commit -m "feat(events): emit node_end_streaming after LLM stream completes"
```

### Subtask 1.3 — 前端 ws handler 识别新事件

- [ ] **Step 1: 读现有 ws handler**

Run: `cd backend && grep -n "openProgressChannel\|node_start\|dispatchEvent" src/api/routes/workflows.py`（若前端代码在 frontend/ 目录，参考 frontend）

- [ ] **Step 2: 加测试覆盖新事件派发**

若 frontend 有 TS 测试（`frontend/src/...` 下），查 `openProgressChannel.test.ts`；若不存在，在 backend 的 ws 路由测试里加：

```python
# backend/tests/test_api_workflows.py 追加
@pytest.mark.asyncio
async def test_ws_handler_passes_through_new_event_types(api_client, bearer_headers):
    """Confirm ws endpoint forwards all EVENT_TYPES (not filtered by old whitelist)."""
    from src.services.workflow_executor import EVENT_TYPES
    # 用已有 ws 订阅 fixture 验证每种事件都能到达
    # 具体验证方式依 ws 路由实现而定
    ...  # 填 fixture
```

（若现有 ws 是"透明转发 payload"而非白名单过滤，此 subtask 只需补文档注释，没代码改动。实施时先确认。）

- [ ] **Step 3: 运行 + 确认 / 补改代码**

Run: `cd backend && pytest tests/test_api_workflows.py -v -k event`
若 pass → 说明已是透传模式，ok；若 fail → 改 ws handler 去掉白名单。

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_api_workflows.py backend/src/api/routes/workflows.py
git commit -m "feat(ws): handler forwards Wave 1 event types"
```

---

## Task 2 · MemoryProvider ABC（1 天）

**Files:**
- Create: `backend/src/services/memory/__init__.py`
- Create: `backend/src/services/memory/base.py`
- Test: `backend/tests/test_memory_provider_abc.py`

### Subtask 2.1 — PluginBase + 异常类

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_memory_provider_abc.py
import inspect

import pytest

from src.services.memory.base import (
    MemoryEntry,
    MemoryProvider,
    MemoryProviderClientError,
    MemoryProviderError,
    MemoryProviderInternalError,
    PluginBase,
    StoredMemoryEntry,
)


def test_plugin_base_is_abstract():
    from abc import ABC
    assert issubclass(PluginBase, ABC)


def test_memory_provider_subclass_of_plugin_base():
    assert issubclass(MemoryProvider, PluginBase)


def test_system_prompt_block_is_async():
    sig = inspect.iscoroutinefunction(PluginBase.system_prompt_block)
    assert sig is True


def test_plugin_base_default_system_prompt_block_returns_empty():
    class _Impl(PluginBase):
        async def initialize(self):
            pass
    import asyncio
    result = asyncio.run(_Impl().system_prompt_block(instance_id=1))
    assert result == ""


def test_error_hierarchy():
    assert issubclass(MemoryProviderClientError, MemoryProviderError)
    assert issubclass(MemoryProviderInternalError, MemoryProviderError)
    assert not issubclass(MemoryProviderClientError, MemoryProviderInternalError)


def test_memory_entry_typed_dict_keys():
    entry: MemoryEntry = {
        "category": "preference",
        "content": "user likes brief replies",
        "context_key": None,
    }
    assert entry["category"] == "preference"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && pytest tests/test_memory_provider_abc.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现 base.py**

```python
# backend/src/services/memory/__init__.py
"""Memory subsystem. See base.py for ABC."""
```

```python
# backend/src/services/memory/base.py
"""MemoryProvider ABC + shared PluginBase.

Contract (see spec 2026-04-16-wave1-platform-contracts-design.md 决策 3):
- fail-fast:  initialize(), ClientError subclasses → MUST raise
- best-effort: add_entries/prefetch → should swallow InternalError, log
- critical:    ContextEngine.compress (in context/base.py) → raise ContextOverflowError
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


# ---------- Exceptions ---------- #

class MemoryProviderError(Exception):
    """Base class for all MemoryProvider errors."""


class MemoryProviderClientError(MemoryProviderError):
    """Caller did something wrong (bad input, auth, oversized). MUST be raised."""


class MemoryProviderInternalError(MemoryProviderError):
    """Transient infra failure (DB hiccup). Callers' choice to retry;
    best-effort methods swallow and log."""


# ---------- Types ---------- #

class MemoryEntry(TypedDict):
    category: str         # 'preference' | 'fact' | 'instruction' | 'custom'
    content: str          # max 10KB per entry (enforced by implementations)
    context_key: str | None


class StoredMemoryEntry(MemoryEntry):
    id: int
    instance_id: int
    created_at: str  # ISO8601


# ---------- PluginBase (shared by MemoryProvider + ContextEngine) ---------- #

class PluginBase(ABC):
    """Shared lifecycle for MemoryProvider & ContextEngine.

    Implementations:
    - Override initialize() (required; fail-fast)
    - Override shutdown() (optional; best-effort)
    - Override system_prompt_block() (optional; defaults to "")
    """

    @abstractmethod
    async def initialize(self) -> None:
        """fail-fast: raise if cannot start. Called at app startup."""

    async def shutdown(self) -> None:
        """best-effort: cleanup resources. Log errors, don't raise."""

    async def system_prompt_block(self, *, instance_id: int) -> str:
        """Static text or dynamically fetched. Default empty string.

        MUST return quickly (<50ms); do not block on LLM calls here.
        """
        return ""


# ---------- MemoryProvider ABC ---------- #

class MemoryProvider(PluginBase):
    """ABC for long-term memory storage/retrieval.

    nous-center 只存不抽取（决策 13）：mediahub 等上层 app 负责
    "这段对话里什么值得记"的逻辑，然后调 add_entries 传结构化条目。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'pg', 'redis'). Used in logs/config."""

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

        Contract:
        - entries == []: return []; no DB write; no raise (idempotent).
        - len(entries) > 100: raise MemoryProviderClientError.
        - any content > 10KB: raise MemoryProviderClientError.
        - unauthorized instance_id: raise MemoryProviderClientError.
        - DB transient failure: raise MemoryProviderInternalError
          (caller may swallow; HTTP layer maps to 500).
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
        """best-effort: recall relevant memories.

        Returns [] on InternalError (logs warning).
        Raises ClientError on bad input (unauthorized instance).
        """

    async def on_session_end(
        self, *, instance_id: int, turns: list[dict]
    ) -> None:
        """Optional: post-session hook. Default no-op.

        Implementations MAY extract facts here. nous-center's built-in
        PGMemoryProvider does NOT (see 决策 2B — 只存不抽取).
        """

    async def on_pre_compress(
        self, *, instance_id: int, messages: list[dict]
    ) -> str | None:
        """Optional: extract summary before ContextEngine compression.

        Return summary string or None. Default no-op.
        """
        return None
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_memory_provider_abc.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/memory backend/tests/test_memory_provider_abc.py
git commit -m "feat(memory): PluginBase + MemoryProvider ABC + exception hierarchy"
```

### Subtask 2.2 — AbstractMemoryProviderTests contract mixin

- [ ] **Step 1: 写 mixin 本身 + 最简 FakeMemoryProvider 让 mixin 跑起来**

追加到 `backend/tests/test_memory_provider_abc.py`：

```python
from datetime import datetime, timezone


class _FakeMemoryProvider(MemoryProvider):
    """In-memory MemoryProvider for contract test runner. Not production-ready."""
    name = "fake"

    def __init__(self):
        self._store: list[dict] = []
        self._next_id = 1
        self._simulate_fail = False

    async def initialize(self):
        pass

    async def add_entries(self, *, instance_id, api_key_id, entries, context_key=None):
        if not entries:
            return []
        if len(entries) > 100:
            raise MemoryProviderClientError("batch > 100")
        for i, e in enumerate(entries):
            if len(e["content"].encode()) > 10_240:
                raise MemoryProviderClientError(f"entries[{i}].content > 10KB")
        if self._simulate_fail:
            raise MemoryProviderInternalError("simulated db fail")
        ids: list[int] = []
        for e in entries:
            # Per-entry context_key takes precedence; fall back to outer kwarg
            entry_ck = e.get("context_key")
            effective_ck = entry_ck if entry_ck is not None else context_key
            row = {
                "id": self._next_id,
                "instance_id": instance_id,
                "api_key_id": api_key_id,
                "category": e["category"],
                "content": e["content"],
                "context_key": effective_ck,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._store.append(row)
            ids.append(self._next_id)
            self._next_id += 1
        return ids

    async def prefetch(self, *, instance_id, query, limit=10, context_key=None):
        if self._simulate_fail:
            return []  # best-effort swallow
        results = [r for r in self._store if r["instance_id"] == instance_id]
        if context_key:
            results = [r for r in results if r["context_key"] == context_key]
        if query:
            results = [r for r in results if query in r["content"]]
        return [StoredMemoryEntry(**r) for r in results[:limit]]


class AbstractMemoryProviderTests:
    """Contract tests any MemoryProvider must pass.

    Subclass and override `provider_factory()` to return an instance.
    """

    def provider_factory(self) -> MemoryProvider:
        raise NotImplementedError

    @pytest.fixture
    async def provider(self):
        p = self.provider_factory()
        await p.initialize()
        yield p
        await p.shutdown()

    @pytest.mark.asyncio
    async def test_add_entries_empty_list_idempotent(self, provider):
        assert await provider.add_entries(
            instance_id=1, api_key_id=None, entries=[]
        ) == []

    @pytest.mark.asyncio
    async def test_add_entries_returns_ids(self, provider):
        ids = await provider.add_entries(
            instance_id=1,
            api_key_id=None,
            entries=[{"category": "preference", "content": "short replies", "context_key": None}],
        )
        assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_add_entries_batch_over_100_raises(self, provider):
        big = [{"category": "fact", "content": "x", "context_key": None}] * 101
        with pytest.raises(MemoryProviderClientError):
            await provider.add_entries(instance_id=1, api_key_id=None, entries=big)

    @pytest.mark.asyncio
    async def test_add_entries_content_over_10kb_raises(self, provider):
        big_content = "x" * 11_000
        with pytest.raises(MemoryProviderClientError):
            await provider.add_entries(
                instance_id=1, api_key_id=None,
                entries=[{"category": "fact", "content": big_content, "context_key": None}],
            )

    @pytest.mark.asyncio
    async def test_prefetch_returns_matching(self, provider):
        await provider.add_entries(
            instance_id=1, api_key_id=None,
            entries=[
                {"category": "preference", "content": "simple replies", "context_key": None},
                {"category": "fact", "content": "lives in Tokyo", "context_key": None},
            ],
        )
        results = await provider.prefetch(instance_id=1, query="Tokyo", limit=10)
        assert len(results) == 1
        assert "Tokyo" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_cross_instance_isolation(self, provider):
        await provider.add_entries(
            instance_id=1, api_key_id=None,
            entries=[{"category": "preference", "content": "foo", "context_key": None}],
        )
        results = await provider.prefetch(instance_id=2, query="foo")
        assert results == []

    @pytest.mark.asyncio
    async def test_context_key_filters_prefetch(self, provider):
        await provider.add_entries(
            instance_id=1, api_key_id=None,
            entries=[
                {"category": "fact", "content": "project alpha note", "context_key": "alpha"},
                {"category": "fact", "content": "project beta note", "context_key": "beta"},
            ],
        )
        r = await provider.prefetch(instance_id=1, query="project", context_key="alpha")
        assert len(r) == 1
        assert r[0]["context_key"] == "alpha"

    @pytest.mark.asyncio
    async def test_prefetch_swallows_internal_error(self, provider):
        # Test that best-effort swallow works when implementations inject fail
        if hasattr(provider, "_simulate_fail"):
            provider._simulate_fail = True
            results = await provider.prefetch(instance_id=1, query="anything")
            assert results == []
            provider._simulate_fail = False


class TestFakeMemoryProviderContract(AbstractMemoryProviderTests):
    """Verify the abstract contract test itself works with the Fake impl."""

    def provider_factory(self):
        return _FakeMemoryProvider()
```

- [ ] **Step 2: 运行测试验证通过**

Run: `cd backend && pytest tests/test_memory_provider_abc.py -v`
Expected: PASS（6 + 7 contract tests = 13 total）

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_memory_provider_abc.py
git commit -m "test(memory): AbstractMemoryProviderTests contract mixin + fake impl"
```

---

## Task 3 · ContextEngine ABC（1 天）

**Files:**
- Create: `backend/src/services/context/__init__.py`
- Create: `backend/src/services/context/base.py`
- Create: `backend/src/services/context/gzip_compact.py`
- Modify: `backend/src/services/responses_service.py`
- Test: `backend/tests/test_context_engine_abc.py`
- Test: `backend/tests/test_gzip_compact_engine.py`

### Subtask 3.1 — ContextEngine ABC + ContextOverflowError

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_context_engine_abc.py
import inspect
import pytest

from src.services.context.base import (
    ContextEngine,
    ContextOverflowError,
)
from src.services.memory.base import PluginBase


def test_context_engine_subclass_of_plugin_base():
    assert issubclass(ContextEngine, PluginBase)


def test_compress_is_async():
    assert inspect.iscoroutinefunction(ContextEngine.compress)


def test_should_compress_is_sync():
    assert not inspect.iscoroutinefunction(ContextEngine.should_compress)


def test_context_overflow_error_exists():
    assert issubclass(ContextOverflowError, Exception)
    with pytest.raises(ContextOverflowError):
        raise ContextOverflowError("too big")
```

- [ ] **Step 2: 运行验证失败**

Run: `cd backend && pytest tests/test_context_engine_abc.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现 base.py**

```python
# backend/src/services/context/__init__.py
"""Context management subsystem. See base.py for ABC."""
```

```python
# backend/src/services/context/base.py
"""ContextEngine ABC (stateless, critical-path component).

Contract (spec 决策 9): engine is stateless. Per-session token accounting
lives in ResponseSession.total_input_tokens / total_output_tokens.
"""

from __future__ import annotations

from abc import abstractmethod

from src.services.memory.base import PluginBase


class ContextOverflowError(Exception):
    """Compression failed: even after pruning, context exceeds model's max_tokens."""


class ContextEngine(PluginBase):
    """ABC for context compression strategies.

    Contract:
    - Stateless: do not hold per-session data in instance attributes.
    - compress() is critical: must succeed or raise ContextOverflowError.
    - should_compress() is synchronous + cheap (O(messages)).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name (e.g. 'gzip-compact'). Used in logs/config."""

    @abstractmethod
    def should_compress(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        current_tokens: int | None = None,
    ) -> bool:
        """Return True if messages would exceed max_tokens and need compression."""

    @abstractmethod
    async def compress(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
    ) -> tuple[list[dict], bool]:
        """Compress messages to fit max_tokens.

        Returns (compacted_messages, was_truncated).

        Raises:
            ContextOverflowError: if even the minimal retained set (e.g. last turn)
            exceeds max_tokens. Callers should return HTTP 400 input_too_long.
        """

    def update_from_response(self, usage: dict) -> None:
        """Optional: track running usage. Default no-op.

        Prefer writing to ResponseSession.total_input_tokens etc. for persistence;
        this method is for in-memory engines that need to react to usage.
        """
```

- [ ] **Step 4: 运行验证通过**

Run: `cd backend && pytest tests/test_context_engine_abc.py -v`
Expected: PASS（4 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/context backend/tests/test_context_engine_abc.py
git commit -m "feat(context): ContextEngine ABC + ContextOverflowError"
```

### Subtask 3.2 — GzipCompactContextEngine（迁移 compact_messages）

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_gzip_compact_engine.py
import pytest

from src.services.context.gzip_compact import GzipCompactContextEngine
from src.services.context.base import ContextOverflowError


def make_msgs(user_contents: list[str], with_system: bool = False):
    msgs = []
    if with_system:
        msgs.append({"role": "system", "content": "be brief"})
    for i, c in enumerate(user_contents):
        msgs.append({"role": "user", "content": c})
        msgs.append({"role": "assistant", "content": f"ack{i}"})
    return msgs


@pytest.mark.asyncio
async def test_compress_noop_when_under_budget():
    engine = GzipCompactContextEngine()
    msgs = make_msgs(["short"])
    result, truncated = await engine.compress(messages=msgs, max_tokens=10_000)
    assert result == msgs
    assert truncated is False


@pytest.mark.asyncio
async def test_compress_drops_oldest_nonsystem_turns():
    engine = GzipCompactContextEngine()
    long = "x" * 2000
    msgs = make_msgs([long, long, long, "recent"], with_system=True)
    result, truncated = await engine.compress(messages=msgs, max_tokens=500)
    assert truncated is True
    # system preserved
    assert result[0]["role"] == "system"
    # 最后一轮 user "recent" 应在结果里
    contents = [m["content"] for m in result]
    assert "recent" in contents


@pytest.mark.asyncio
async def test_compress_preserves_last_user_turn():
    engine = GzipCompactContextEngine()
    msgs = make_msgs(["a" * 5000, "b" * 5000, "last"])
    result, _ = await engine.compress(messages=msgs, max_tokens=100)
    assert result[-1]["content"] in ("last", "ack2")
    assert any(m.get("content") == "last" for m in result)


@pytest.mark.asyncio
async def test_compress_overflow_raises():
    engine = GzipCompactContextEngine()
    huge_last = "x" * 100_000
    msgs = make_msgs([huge_last])
    with pytest.raises(ContextOverflowError):
        await engine.compress(messages=msgs, max_tokens=100)


def test_should_compress_threshold():
    engine = GzipCompactContextEngine()
    small = make_msgs(["hi"])
    big = make_msgs(["x" * 10_000])
    assert engine.should_compress(messages=small, max_tokens=10_000) is False
    assert engine.should_compress(messages=big, max_tokens=100) is True


def test_engine_name():
    assert GzipCompactContextEngine().name == "gzip-compact"
```

- [ ] **Step 2: 运行验证失败**

Run: `cd backend && pytest tests/test_gzip_compact_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 迁移 compact_messages 到新类**

```python
# backend/src/services/context/gzip_compact.py
"""Built-in ContextEngine implementation (migrated from responses_service.compact_messages).

Strategy: drop oldest non-system turns until estimated token count fits max_tokens.
Conservative over-estimation (len/2 + 4 per message) to avoid vLLM
`context_length_exceeded` mid-stream.
"""

from __future__ import annotations

import logging

from src.services.context.base import ContextEngine, ContextOverflowError

logger = logging.getLogger(__name__)


def _approx_tokens(messages: list[dict]) -> int:
    """Conservative OVER-estimate. Migrated from responses_service.approx_tokens."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c) // 2 + 4
        elif isinstance(c, list):
            for item in c:
                t = item.get("text", "")
                if isinstance(t, str):
                    total += len(t) // 2 + 4
                else:
                    total += 200  # image / other placeholder
    return total


class GzipCompactContextEngine(ContextEngine):
    name = "gzip-compact"

    async def initialize(self) -> None:
        pass  # stateless, nothing to do

    def should_compress(self, *, messages, max_tokens, current_tokens=None):
        est = current_tokens if current_tokens is not None else _approx_tokens(messages)
        return est > max_tokens

    async def compress(self, *, messages, max_tokens):
        """Drop oldest non-system turns until fit.

        Returns (compacted, was_truncated).
        Raises ContextOverflowError if final set (last turn + system) still exceeds.
        """
        if _approx_tokens(messages) <= max_tokens:
            return messages, False

        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        # Never pop the last user message — it's the current turn's input.
        # If dropping the oldest would leave us with nothing that includes
        # a user turn, break out and let the overflow check raise below.
        while rest and _approx_tokens(system_msgs + rest) > max_tokens:
            # Stop if remaining rest is just [last_user] (or shorter)
            if len(rest) <= 1:
                break
            rest.pop(0)

        compacted = system_msgs + rest

        if _approx_tokens(compacted) > max_tokens:
            raise ContextOverflowError(
                f"context still exceeds max_tokens={max_tokens} "
                f"after compression (est={_approx_tokens(compacted)})"
            )

        if not rest:
            # 所有非 system 全被砍光了 — 说明最后一轮本身就超，raise
            raise ContextOverflowError(
                "last turn alone exceeds max_tokens"
            )

        return compacted, True
```

- [ ] **Step 4: 运行验证通过**

Run: `cd backend && pytest tests/test_gzip_compact_engine.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/context/gzip_compact.py backend/tests/test_gzip_compact_engine.py
git commit -m "feat(context): GzipCompactContextEngine (migrated from compact_messages)"
```

### Subtask 3.3 — 切 responses_service.py 用 engine

- [ ] **Step 1: 写失败测试（regression）**

```python
# backend/tests/test_responses_service_engine_swap.py
"""After migrating compact_messages → GzipCompactContextEngine, behavior must match."""

import pytest
from src.services.responses_service import compact_messages as legacy_compact


@pytest.mark.asyncio
async def test_engine_matches_legacy_for_short_input():
    from src.services.context.gzip_compact import GzipCompactContextEngine
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    legacy_out, legacy_trunc = legacy_compact(msgs, max_history_tokens=10_000)
    engine_out, engine_trunc = await GzipCompactContextEngine().compress(
        messages=msgs, max_tokens=10_000
    )
    assert legacy_out == engine_out
    assert legacy_trunc == engine_trunc
```

- [ ] **Step 2: 运行验证基线一致**

Run: `cd backend && pytest tests/test_responses_service_engine_swap.py -v`
Expected: PASS（行为等价）

- [ ] **Step 3: 用 engine 替换 responses.py 里的 compact_messages 调用**

查 responses.py 里使用 `compact_messages` 的所有位置：

```bash
cd backend && grep -n "compact_messages\|approx_tokens" src/api/routes/responses.py
```

将其替换为 `GzipCompactContextEngine()`：

```python
# backend/src/api/routes/responses.py 顶部 imports 追加：
from src.services.context.base import ContextOverflowError
from src.services.context.gzip_compact import GzipCompactContextEngine

# 原来的 compact_messages(...) 调用改为：
_engine = GzipCompactContextEngine()  # 或放到 app.state 启动时 initialize
try:
    compacted, history_truncated = await _engine.compress(
        messages=messages, max_tokens=max_history_tokens
    )
except ContextOverflowError as e:
    raise InvalidRequestError(str(e), code="input_too_long_for_model")
```

**注意：保留 `responses_service.compact_messages` 函数**，其内部改为调 engine 以保持向下兼容（有其他调用点或测试直接依赖此函数名）：

```python
# backend/src/services/responses_service.py 里 compact_messages 改为：
def compact_messages(messages, *, max_history_tokens, keep_system=True):
    """Legacy shim. New code should use GzipCompactContextEngine directly."""
    import asyncio
    from src.services.context.gzip_compact import GzipCompactContextEngine
    from src.services.context.base import ContextOverflowError
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            GzipCompactContextEngine().compress(
                messages=messages, max_tokens=max_history_tokens
            )
        )
        loop.close()
        return result
    except ContextOverflowError:
        # 原 compact_messages 不 raise, 保持行为：返回原 messages + truncated=True
        return messages, True
```

（实际上旧 compact_messages 不 raise，engine 现在 raise——shim 保持老行为。新代码直接用 engine 拿到 raise 能力。）

- [ ] **Step 4: 运行现有 responses 测试无回归**

Run: `cd backend && pytest tests/ -v -k responses`
Expected: 所有通过

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/routes/responses.py backend/src/services/responses_service.py \
        backend/tests/test_responses_service_engine_swap.py
git commit -m "refactor(responses): use GzipCompactContextEngine (legacy shim retained)"
```

---

## Task 4 · 节点接口分层（2 天）

12 个 `_exec_*` 函数迁移到 class 实现 Protocol。关键约束：**取消全局 `_on_progress_ref`**，改为显式 `on_token` callback。

**Files:**
- Create: `backend/src/services/nodes/__init__.py`
- Create: `backend/src/services/nodes/base.py`
- Create: `backend/src/services/nodes/text_io.py`（text_input / text_output / multimodal_input / ref_audio / output / passthrough）
- Create: `backend/src/services/nodes/llm.py`（LLMNode — Invokable + Streamable）
- Create: `backend/src/services/nodes/audio.py`（tts_engine / resample / mixer / concat / bgm_mix）
- Create: `backend/src/services/nodes/logic.py`（prompt_template / agent / if_else / python_code）
- Create: `backend/src/services/nodes/registry.py`
- Modify: `backend/src/services/workflow_executor.py`
- Test: `backend/tests/test_node_protocols.py`
- Test: `backend/tests/test_workflow_llm_token_stats.py`

### Subtask 4.1 — 定义 Protocols + registry 骨架

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_node_protocols.py
from src.services.nodes.base import (
    CollectableNode,
    InvokableNode,
    StreamableNode,
    TransformableNode,
)


def test_protocols_are_runtime_checkable():
    from typing import runtime_checkable
    # 验证 Protocol 声明了 @runtime_checkable
    assert hasattr(InvokableNode, "_is_runtime_protocol")


def test_invokable_protocol_shape():
    class _Good:
        async def invoke(self, data, inputs):
            return {}
    assert isinstance(_Good(), InvokableNode)


def test_invokable_protocol_rejects_missing_method():
    class _Bad:
        pass
    assert not isinstance(_Bad(), InvokableNode)


def test_streamable_protocol_shape():
    class _Stream:
        async def stream(self, data, inputs, on_token):
            return {}
    assert isinstance(_Stream(), StreamableNode)
```

- [ ] **Step 2: 运行验证失败**

Run: `cd backend && pytest tests/test_node_protocols.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现 Protocol**

```python
# backend/src/services/nodes/__init__.py
"""Node subsystem with Protocol-based interfaces. See base.py."""
```

```python
# backend/src/services/nodes/base.py
"""Node Protocols (Wave 1 Task 4).

Each node class implements one or more of these Protocols, checked via
`isinstance(obj, Proto)` at dispatch time.

KEY CHANGE from Wave 0: StreamableNode accepts explicit on_token callback.
The global _on_progress_ref is eliminated (决策 7).
"""

from __future__ import annotations

from typing import AsyncIterator, Awaitable, Callable, Protocol, runtime_checkable


# Callback type for streaming nodes to push tokens back to caller.
OnTokenFn = Callable[[str], Awaitable[None]]


@runtime_checkable
class InvokableNode(Protocol):
    """Node that executes once, returns final dict."""

    async def invoke(self, data: dict, inputs: dict) -> dict: ...


@runtime_checkable
class StreamableNode(Protocol):
    """Node that streams tokens via on_token callback, returns final dict.

    on_token MUST be awaited for each chunk. Node is responsible for final
    usage aggregation in the returned dict.
    """

    async def stream(
        self,
        data: dict,
        inputs: dict,
        on_token: OnTokenFn,
    ) -> dict: ...


@runtime_checkable
class CollectableNode(Protocol):
    """Node that consumes an async stream of inputs and produces one final dict."""

    async def collect(
        self,
        data: dict,
        inputs_stream: AsyncIterator[dict],
    ) -> dict: ...


@runtime_checkable
class TransformableNode(Protocol):
    """Node that transforms a stream of inputs into a stream of outputs."""

    async def transform(
        self,
        data: dict,
        inputs_stream: AsyncIterator[dict],
    ) -> AsyncIterator[dict]: ...
```

```python
# backend/src/services/nodes/registry.py
"""Registry mapping node type string → class.

Populated by each node module's import (side effect). Lookup is done by
WorkflowExecutor._execute_node.
"""

from __future__ import annotations

_NODE_CLASSES: dict[str, type] = {}


def register(node_type: str):
    """Decorator to register a node class."""
    def _inner(cls: type) -> type:
        _NODE_CLASSES[node_type] = cls
        return cls
    return _inner


def get_node_class(node_type: str) -> type | None:
    return _NODE_CLASSES.get(node_type)


def all_registered() -> dict[str, type]:
    return dict(_NODE_CLASSES)
```

- [ ] **Step 4: 运行验证通过**

Run: `cd backend && pytest tests/test_node_protocols.py -v`
Expected: PASS（4 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/nodes backend/tests/test_node_protocols.py
git commit -m "feat(nodes): InvokableNode/StreamableNode/Collectable/Transformable Protocols + registry"
```

### Subtask 4.2 — 迁移简单 Invokable 节点（3 个：text_input / text_output / passthrough）

- [ ] **Step 1: 写测试（行为等价）**

```python
# backend/tests/test_node_class_migration.py
import pytest

from src.services.nodes.base import InvokableNode
from src.services.nodes.text_io import TextInputNode, TextOutputNode, PassthroughNode


@pytest.mark.asyncio
async def test_text_input_node_returns_data_text():
    node = TextInputNode()
    assert isinstance(node, InvokableNode)
    result = await node.invoke({"text": "hello"}, {})
    assert result == {"text": "hello"}


@pytest.mark.asyncio
async def test_text_input_node_defaults_empty():
    node = TextInputNode()
    result = await node.invoke({}, {})
    assert result == {"text": ""}


@pytest.mark.asyncio
async def test_text_output_node_returns_inputs_text():
    node = TextOutputNode()
    result = await node.invoke({}, {"text": "out"})
    assert result == {"text": "out"}


@pytest.mark.asyncio
async def test_passthrough_node_returns_inputs():
    node = PassthroughNode()
    result = await node.invoke({}, {"key": "value"})
    assert result == {"key": "value"}
```

- [ ] **Step 2: 运行验证失败**

Run: `cd backend && pytest tests/test_node_class_migration.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现 3 个节点**

```python
# backend/src/services/nodes/text_io.py
"""Simple invokable nodes: text in/out, passthrough."""

from src.services.nodes.registry import register


@register("text_input")
class TextInputNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"text": data.get("text", "")}


@register("text_output")
class TextOutputNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"text": inputs.get("text", "")}


@register("passthrough")
class PassthroughNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return dict(inputs)
```

- [ ] **Step 4: 运行验证通过**

Run: `cd backend && pytest tests/test_node_class_migration.py -v`
Expected: PASS（4 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/nodes/text_io.py backend/tests/test_node_class_migration.py
git commit -m "feat(nodes): migrate text_input/text_output/passthrough to classes"
```

### Subtask 4.3 — 迁移 LLMNode（Invokable + Streamable，取消 _on_progress_ref）

- [ ] **Step 1: 写测试（token stats 不回归 + on_token 显式调）**

```python
# backend/tests/test_workflow_llm_token_stats.py
"""CRITICAL REGRESSION: existing frontend token stats must keep working after node migration."""

import pytest

from src.services.nodes.base import InvokableNode, StreamableNode
from src.services.nodes.llm import LLMNode


def test_llm_node_implements_both_protocols():
    node = LLMNode()
    assert isinstance(node, InvokableNode)
    assert isinstance(node, StreamableNode)


@pytest.mark.asyncio
async def test_llm_node_stream_invokes_on_token_per_chunk(mock_llm_stream_v2):
    """on_token should be awaited once per chunk, not via global ref."""
    captured: list[str] = []

    async def on_token(t: str):
        captured.append(t)

    node = LLMNode()
    result = await node.stream(
        data={"_node_id": "llm-1", "model": "qwen3.5"},
        inputs={"messages": [{"role": "user", "content": "hi"}]},
        on_token=on_token,
    )
    assert captured == ["hel", "lo"]  # from fake_stream fixture
    assert result["usage"]["total_tokens"] == 4
    assert "text" in result  # 最终拼好的 assistant text


@pytest.mark.asyncio
async def test_llm_node_invoke_non_stream(mock_llm_nonstream):
    """Non-streaming invoke path."""
    node = LLMNode()
    result = await node.invoke(
        data={"_node_id": "llm-2", "model": "qwen3.5", "stream": False},
        inputs={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert result["text"] == "non-stream response"
    assert result["usage"]["total_tokens"] > 0
```

Conftest 加 fixture（若 `mock_llm_stream_v2` 未定义）：

```python
# backend/tests/conftest.py 追加
@pytest.fixture
def mock_llm_stream_v2(monkeypatch):
    """Mock LLM streaming for the new node-class based architecture."""
    # 具体实现取决于 LLMNode.stream 内部调什么。
    # 占位：执行者在实现 LLMNode 时对齐。
    pass


@pytest.fixture
def mock_llm_nonstream(monkeypatch):
    pass
```

- [ ] **Step 2: 运行验证失败**

Run: `cd backend && pytest tests/test_workflow_llm_token_stats.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现 LLMNode**

读现有 `backend/src/services/workflow_executor.py:297` 的 `_exec_llm` 函数体，整体搬到 LLMNode。关键改造：
1. 去掉对全局 `_on_progress_ref` 的引用
2. 流式路径改为 `stream(...)`，接受 `on_token` 参数
3. 非流式路径变 `invoke(...)`

```python
# backend/src/services/nodes/llm.py
"""LLM node: implements InvokableNode (non-stream) + StreamableNode (stream).

Migrated from workflow_executor._exec_llm. Key change: on_token is passed
explicitly; no more global _on_progress_ref.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from src.services.nodes.base import OnTokenFn
from src.services.nodes.registry import register

logger = logging.getLogger(__name__)


@register("llm")
class LLMNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        """Non-streaming LLM call. Returns {text, usage}."""
        # 搬 _exec_llm 的非流式分支代码。具体依赖 adapter / model_manager
        # 调用方式与原函数一致。
        # 伪代码：
        #   adapter = _resolve_adapter(data["model"])
        #   resp = await adapter.chat_completion(messages=inputs["messages"], stream=False)
        #   return {"text": resp.choices[0].message.content, "usage": resp.usage.dict()}
        raise NotImplementedError("port _exec_llm non-stream branch here")

    async def stream(
        self,
        data: dict,
        inputs: dict,
        on_token: OnTokenFn,
    ) -> dict:
        """Streaming LLM call. Invokes on_token per chunk. Returns {text, usage}."""
        # 搬 _exec_llm 流式分支。对于每个 chunk 的 delta.content：
        #   await on_token(delta_content_str)
        # 最后 return {"text": full_text, "usage": final_usage}
        raise NotImplementedError("port _exec_llm streaming branch here")
```

**注意**：实现此类时必须完整搬 `_exec_llm` 的代码，然后替换所有 `_on_progress_ref({"type": "node_stream", "content": ...})` 为 `await on_token(content_str)`。

- [ ] **Step 4: 运行验证通过**

Run: `cd backend && pytest tests/test_workflow_llm_token_stats.py -v`
Expected: PASS（3 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/nodes/llm.py backend/tests/test_workflow_llm_token_stats.py \
        backend/tests/conftest.py
git commit -m "feat(nodes): LLMNode (Invokable+Streamable) replaces _exec_llm, kills _on_progress_ref"
```

### Subtask 4.4 — 迁移剩余 8 个节点

**迁移目标表**（同 pattern 为 Subtask 4.2 的 3 个简单节点）：

| 老函数 (workflow_executor.py) | 新类 | 文件 | Protocol |
|-------|------|------|---------|
| `_exec_multimodal_input` | `MultimodalInputNode` | `nodes/text_io.py` | Invokable |
| `_exec_ref_audio` | `RefAudioNode` | `nodes/audio.py` | Invokable |
| `_exec_tts_engine` | `TTSEngineNode` | `nodes/audio.py` | Invokable |
| `_exec_output` | `OutputNode` | `nodes/text_io.py` | Invokable |
| `_exec_prompt_template` | `PromptTemplateNode` | `nodes/logic.py` | Invokable |
| `_exec_agent` | `AgentNode` | `nodes/logic.py` | Invokable |
| `_exec_python_code` | `PythonCodeNode` | `nodes/logic.py` | Invokable |
| `_exec_if_else` | `IfElseNode` | `nodes/logic.py` | Invokable |

- [ ] **Step 1: 为每个节点写等价性测试**

追加到 `backend/tests/test_node_class_migration.py`：

```python
# 每个节点一条测试，pattern 相同于 Subtask 4.2 的 text_input:
#
# 1) 从老函数（还在 workflow_executor.py）捕获一个典型输入→输出
# 2) 用同样的 data + inputs 跑新类
# 3) assert 两者输出字典深比相等

# 例：MultimodalInputNode
@pytest.mark.asyncio
async def test_multimodal_input_node_equivalence():
    from src.services.workflow_executor import _exec_multimodal_input
    from src.services.nodes.text_io import MultimodalInputNode
    data = {"text": "hi", "images": []}
    inputs = {}
    old = await _exec_multimodal_input(data, inputs)
    new = await MultimodalInputNode().invoke(data, inputs)
    assert old == new

# 其他 7 个节点同理，每个一条 test_XXX_equivalence
```

- [ ] **Step 2: 运行验证失败（所有新类未定义）**

Run: `cd backend && pytest tests/test_node_class_migration.py -v -k equivalence`
Expected: FAIL — new node classes not found

- [ ] **Step 3: 迁移所有 8 个节点**

对每个节点，按 **Subtask 4.2 的 pattern**：
1. 打开 `workflow_executor.py` 找到对应 `_exec_xxx` 函数
2. 复制函数体到新类的 `async def invoke(self, data, inputs)` 方法内
3. 在新类加 `@register("xxx")` 装饰
4. 放到对应的文件（text_io.py / audio.py / logic.py）

示例（把 `_exec_multimodal_input` 迁到 `MultimodalInputNode`）：

```python
# backend/src/services/nodes/text_io.py 追加
@register("multimodal_input")
class MultimodalInputNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        # ← 把 workflow_executor.py:227 的 _exec_multimodal_input 函数体搬这里
        return {"text": data.get("text", ""), "images": data.get("images", [])}
```

把所有 8 个节点完成，每 4 个节点 commit 一次（避免 commit 过大）。

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && pytest tests/test_node_class_migration.py -v`
Expected: PASS（12 个节点等价性 + 前 4 个简单节点）

- [ ] **Step 5: Commit（分 3 个 commit，每 commit 搞 3-4 个节点）**

```bash
# commit 1: audio/io
git add backend/src/services/nodes/audio.py backend/src/services/nodes/text_io.py backend/tests/test_node_class_migration.py
git commit -m "feat(nodes): migrate multimodal_input/ref_audio/tts_engine/output to classes"

# commit 2: logic
git add backend/src/services/nodes/logic.py
git commit -m "feat(nodes): migrate prompt_template/agent/python_code/if_else to classes"
```

### Subtask 4.5 — WorkflowExecutor._execute_node 换 class dispatch

- [ ] **Step 1: 写 dispatch 测试**

```python
# backend/tests/test_workflow_executor_dispatch.py
import pytest

from src.services.workflow_executor import WorkflowExecutor


@pytest.mark.asyncio
async def test_executor_dispatches_via_registry():
    """_execute_node should look up class from registry, instantiate, call invoke/stream."""
    workflow = {
        "nodes": [{"id": "n1", "type": "text_input", "data": {"text": "hi"}}],
        "edges": [],
    }
    captured: list[dict] = []

    async def on_progress(ev):
        captured.append(ev)

    exe = WorkflowExecutor(workflow, on_progress=on_progress)
    result = await exe.execute()
    assert result["outputs"]["n1"] == {"text": "hi"}
```

- [ ] **Step 2: 运行（现有实现用老 function dict）**

Run: `cd backend && pytest tests/test_workflow_executor_dispatch.py -v`
Expected: 可能 PASS（因为老实现仍 work）或 FAIL（如果已改一半）

- [ ] **Step 3: 改写 _execute_node**

修改 `backend/src/services/workflow_executor.py` 的 `_execute_node`：

```python
async def _execute_node(self, node: dict, inputs: dict) -> dict[str, Any]:
    """Execute a single node via registered class + protocol dispatch."""
    from src.services.nodes.base import InvokableNode, StreamableNode
    from src.services.nodes.registry import get_node_class

    node_type = node["type"]
    data = dict(node.get("data", {}))
    data["_node_id"] = node["id"]

    node_cls = get_node_class(node_type)
    if node_cls is None:
        # Check plugin executors from node packages（保留向下兼容）
        from nodes import get_all_executors
        plugin_executors = get_all_executors()
        legacy_fn = plugin_executors.get(node_type)
        if legacy_fn is None:
            raise ExecutionError(f"未知节点类型: {node_type}")
        # 插件还是老 function 形式，设置 _on_progress_ref 给它们兼容期用
        global _on_progress_ref
        _on_progress_ref = self._on_progress
        return await legacy_fn(data, inputs)

    instance = node_cls()

    if isinstance(instance, StreamableNode) and data.get("stream") is not False:
        # stream 路径
        async def _on_token(token: str):
            if self._on_progress:
                await self._on_progress({
                    "type": "node_stream",
                    "node_id": node["id"],
                    "content": token,
                })
        result = await instance.stream(data, inputs, _on_token)
        # 流式结束触发 node_end_streaming（Task 1 定义的新事件）
        if self._on_progress:
            await self._on_progress({
                "type": "node_end_streaming",
                "node_id": node["id"],
                "usage": result.get("usage"),
            })
        return result

    if isinstance(instance, InvokableNode):
        return await instance.invoke(data, inputs)

    raise ExecutionError(
        f"Node class for {node_type!r} implements neither InvokableNode nor StreamableNode"
    )
```

- [ ] **Step 4: 运行所有 workflow 测试**

Run: `cd backend && pytest tests/ -v -k workflow`
Expected: 全过（含 LLM token stats regression）

- [ ] **Step 5: 删除老的 `_NODE_EXECUTORS` dict 和 `_exec_*` function**

保留插件机制（`nodes/get_all_executors()`），只删 12 个内置 `_exec_*`。

Run: `grep -n "_NODE_EXECUTORS\|def _exec_" backend/src/services/workflow_executor.py`

把这些行全删掉。运行全量测试确认没其他地方直接调 `_exec_*`：

Run: `cd backend && pytest tests/ -v`
Expected: 全过

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/workflow_executor.py backend/tests/test_workflow_executor_dispatch.py
git commit -m "refactor(workflow): dispatch via node class + protocol; remove _NODE_EXECUTORS + _exec_*"
```

### Subtask 4.6 — 确认 `_on_progress_ref` 已删

- [ ] **Step 1: grep 确认**

```bash
cd backend && grep -n "_on_progress_ref" src/
```
Expected: 只剩**向下兼容插件** function 的那 2 行（见 Subtask 4.5 Step 3 的 "legacy_fn" 分支）；所有内置节点不再用它。

- [ ] **Step 2: 写确认测试**

追加到 `backend/tests/test_workflow_executor_dispatch.py`：

```python
def test_on_progress_ref_not_used_by_builtin_nodes():
    """Builtin 12 nodes must NOT reference the global _on_progress_ref."""
    import inspect
    from src.services.nodes import text_io, audio, logic, llm as llm_module
    for mod in (text_io, audio, logic, llm_module):
        src = inspect.getsource(mod)
        assert "_on_progress_ref" not in src, \
            f"{mod.__name__} should not reference global _on_progress_ref"
```

- [ ] **Step 3: Run + Commit**

```bash
cd backend && pytest tests/test_workflow_executor_dispatch.py -v
git add backend/tests/test_workflow_executor_dispatch.py
git commit -m "test(nodes): assert builtin nodes don't touch _on_progress_ref"
```

---

## Task 5 · PGMemoryProvider reference 实现（1 天）

**Files:**
- Create: `backend/migrations/wave1_memory.sql`
- Create: `backend/src/models/memory.py`
- Create: `backend/src/services/memory/pg_provider.py`
- Create: `backend/src/api/routes/memory.py`
- Modify: `backend/src/api/main.py`（注册新 router + 启动时 initialize provider）
- Test: `backend/tests/test_pg_memory_provider.py`
- Test: `backend/tests/test_api_memory.py`

### Subtask 5.1 — SQL migration + ORM model

- [ ] **Step 1: 写 migration SQL**

```sql
-- backend/migrations/wave1_memory.sql
-- Wave 1 · memory tables (2026-04-20)

CREATE TABLE IF NOT EXISTS memory_entries (
    id            BIGSERIAL PRIMARY KEY,
    instance_id   BIGINT NOT NULL REFERENCES service_instances(id) ON DELETE CASCADE,
    api_key_id    BIGINT,
    category      VARCHAR(32) NOT NULL,
    content       TEXT NOT NULL,
    context_key   VARCHAR(128),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mem_inst_created ON memory_entries (instance_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mem_inst_key_cat ON memory_entries (instance_id, context_key, category);
CREATE INDEX IF NOT EXISTS idx_mem_content_fts  ON memory_entries USING GIN (to_tsvector('simple', content));

CREATE TABLE IF NOT EXISTS memory_embeddings (
    entry_id      BIGINT PRIMARY KEY REFERENCES memory_entries(id) ON DELETE CASCADE,
    model         VARCHAR(64) NOT NULL,
    dim           INT NOT NULL,
    vector        BYTEA
);
```

- [ ] **Step 2: 写 ORM model**

```python
# backend/src/models/memory.py
"""Wave 1 memory tables (MemoryEntry + MemoryEmbedding).

Dialect-agnostic declarations; FTS index is PG-only and added in raw SQL migration.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Column, DateTime, ForeignKey, Index, Integer,
    LargeBinary, String, Text,
)

from src.models.database import Base


class MemoryEntryModel(Base):
    __tablename__ = "memory_entries"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    instance_id = Column(
        BigInteger,
        ForeignKey("service_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    api_key_id = Column(BigInteger, nullable=True)
    category = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    context_key = Column(String(128), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("idx_mem_inst_created", "instance_id", "created_at"),
        Index("idx_mem_inst_key_cat", "instance_id", "context_key", "category"),
    )


class MemoryEmbeddingModel(Base):
    __tablename__ = "memory_embeddings"

    entry_id = Column(
        BigInteger,
        ForeignKey("memory_entries.id", ondelete="CASCADE"),
        primary_key=True,
    )
    model = Column(String(64), nullable=False)
    dim = Column(Integer, nullable=False)
    vector = Column(LargeBinary, nullable=True)
```

- [ ] **Step 3: Commit**

```bash
git add backend/migrations/wave1_memory.sql backend/src/models/memory.py
git commit -m "feat(db): memory_entries + memory_embeddings schema + ORM"
```

### Subtask 5.2 — PGMemoryProvider 实现

- [ ] **Step 1: 写测试（继承 AbstractMemoryProviderTests）**

```python
# backend/tests/test_pg_memory_provider.py
import pytest
import pytest_asyncio

from src.services.memory.pg_provider import PGMemoryProvider

from tests.test_memory_provider_abc import AbstractMemoryProviderTests


class TestPGMemoryProviderContract(AbstractMemoryProviderTests):
    """Run the full AbstractMemoryProviderTests suite against PG (SQLite in tests)."""

    @pytest_asyncio.fixture
    async def provider(self, async_session_factory):
        p = PGMemoryProvider(session_factory=async_session_factory)
        await p.initialize()
        yield p
        await p.shutdown()


# PG-specific additional tests
@pytest.mark.asyncio
async def test_fts_index_hit(async_session_factory):
    """FTS GIN index should allow text search (PG only; SQLite fallback uses LIKE)."""
    p = PGMemoryProvider(session_factory=async_session_factory)
    await p.initialize()
    await p.add_entries(
        instance_id=1, api_key_id=None,
        entries=[
            {"category": "fact", "content": "user prefers concise replies", "context_key": None},
            {"category": "fact", "content": "user lives in Tokyo", "context_key": None},
        ],
    )
    results = await p.prefetch(instance_id=1, query="concise")
    assert len(results) == 1
    assert "concise" in results[0]["content"]
```

- [ ] **Step 2: 实现 pg_provider.py**

```python
# backend/src/services/memory/pg_provider.py
"""PGMemoryProvider — reference implementation.

Uses PG FTS (GIN) for content search. Works on SQLite via LIKE fallback
(detected by dialect) — for dev/test convenience.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import desc, select, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.memory import MemoryEntryModel
from src.services.memory.base import (
    MemoryEntry,
    MemoryProvider,
    MemoryProviderClientError,
    MemoryProviderInternalError,
    StoredMemoryEntry,
)

logger = logging.getLogger(__name__)

MAX_ENTRY_BYTES = 10 * 1024
MAX_BATCH_SIZE = 100


def _to_stored_entry(row: MemoryEntryModel) -> StoredMemoryEntry:
    return StoredMemoryEntry(
        id=row.id,
        instance_id=row.instance_id,
        category=row.category,
        content=row.content,
        context_key=row.context_key,
        created_at=row.created_at.isoformat(),
    )


class PGMemoryProvider(MemoryProvider):
    name = "pg"

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._sf = session_factory

    async def initialize(self) -> None:
        """fail-fast: ensure table exists (migration applied)."""
        async with self._sf() as s:
            try:
                await s.execute(text("SELECT 1 FROM memory_entries LIMIT 1"))
            except ProgrammingError as e:
                raise RuntimeError(
                    "memory_entries table not found — run wave1_memory.sql migration"
                ) from e

    async def shutdown(self) -> None:
        pass  # session factory managed externally

    async def add_entries(
        self,
        *,
        instance_id: int,
        api_key_id: int | None,
        entries: list[MemoryEntry],
        context_key: str | None = None,
    ) -> list[int]:
        if not entries:
            return []

        if len(entries) > MAX_BATCH_SIZE:
            raise MemoryProviderClientError(
                f"entries exceeds max batch size {MAX_BATCH_SIZE}"
            )

        for i, e in enumerate(entries):
            if len(e.get("content", "").encode()) > MAX_ENTRY_BYTES:
                raise MemoryProviderClientError(
                    f"entries[{i}].content exceeds {MAX_ENTRY_BYTES} bytes"
                )

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

    async def prefetch(
        self,
        *,
        instance_id: int,
        query: str,
        limit: int = 10,
        context_key: str | None = None,
    ) -> list[StoredMemoryEntry]:
        try:
            async with self._sf() as s:
                stmt = select(MemoryEntryModel).where(
                    MemoryEntryModel.instance_id == instance_id
                )
                if context_key:
                    stmt = stmt.where(MemoryEntryModel.context_key == context_key)
                if query:
                    dialect = s.bind.dialect.name if s.bind else "sqlite"
                    if dialect == "postgresql":
                        stmt = stmt.where(
                            text("to_tsvector('simple', content) @@ plainto_tsquery(:q)")
                        ).params(q=query)
                    else:
                        stmt = stmt.where(MemoryEntryModel.content.contains(query))
                stmt = stmt.order_by(desc(MemoryEntryModel.created_at)).limit(limit)
                rows = (await s.execute(stmt)).scalars().all()
                return [_to_stored_entry(r) for r in rows]
        except (DBAPIError, asyncio.TimeoutError) as exc:
            logger.warning("PGMemoryProvider.prefetch failed: %s; returning empty", exc)
            return []

    async def system_prompt_block(self, *, instance_id: int) -> str:
        return (
            "You have access to long-term memory for this user "
            "(managed by the platform)."
        )
```

- [ ] **Step 3: 运行测试**

Run: `cd backend && pytest tests/test_pg_memory_provider.py -v`
Expected: PASS（contract 7 + PG-specific 1 = 8 tests）

- [ ] **Step 4: Commit**

```bash
git add backend/src/services/memory/pg_provider.py backend/tests/test_pg_memory_provider.py
git commit -m "feat(memory): PGMemoryProvider reference implementation"
```

### Subtask 5.3 — /api/v1/memory HTTP routes

- [ ] **Step 1: 写失败 E2E**

```python
# backend/tests/test_api_memory.py
import pytest


@pytest.mark.asyncio
async def test_sync_endpoint_writes_entries(api_client, bearer_headers):
    resp = await api_client.post(
        "/api/v1/memory/sync",
        json={
            "entries": [
                {"category": "preference", "content": "用户喜欢简洁回复", "context_key": "proj-1"},
            ],
            "context_key": "proj-1",
        },
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    assert "entry_ids" in resp.json()
    assert len(resp.json()["entry_ids"]) == 1


@pytest.mark.asyncio
async def test_prefetch_endpoint_returns_entries(api_client, bearer_headers):
    await api_client.post(
        "/api/v1/memory/sync",
        json={"entries": [{"category": "fact", "content": "Tokyo", "context_key": None}]},
        headers=bearer_headers,
    )
    resp = await api_client.get(
        "/api/v1/memory/prefetch?q=Tokyo&limit=5",
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    assert len(resp.json()["entries"]) == 1


@pytest.mark.asyncio
async def test_sync_over_100_entries_returns_400(api_client, bearer_headers):
    entries = [{"category": "fact", "content": "x", "context_key": None}] * 101
    resp = await api_client.post(
        "/api/v1/memory/sync",
        json={"entries": entries},
        headers=bearer_headers,
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_sync_empty_list_ok(api_client, bearer_headers):
    resp = await api_client.post(
        "/api/v1/memory/sync",
        json={"entries": []},
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["entry_ids"] == []
```

- [ ] **Step 2: 运行验证失败**

Run: `cd backend && pytest tests/test_api_memory.py -v`
Expected: FAIL — 404 (route missing)

- [ ] **Step 3: 实现 route**

```python
# backend/src/api/routes/memory.py
"""Memory API (Wave 1): POST /sync + GET /prefetch."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.deps_auth import verify_bearer_token
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.memory.base import (
    MemoryProviderClientError,
    MemoryProviderInternalError,
)

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])
logger = logging.getLogger(__name__)


class MemoryEntryIn(BaseModel):
    category: str = Field(..., pattern="^(preference|fact|instruction|custom)$")
    content: str
    context_key: str | None = None


class SyncRequest(BaseModel):
    entries: list[MemoryEntryIn]
    context_key: str | None = None


@router.post("/sync")
async def memory_sync(
    body: SyncRequest,
    request: Request,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
):
    instance, api_key = auth
    provider = request.app.state.memory_provider
    try:
        ids = await provider.add_entries(
            instance_id=instance.id,
            api_key_id=api_key.id if api_key else None,
            entries=[e.model_dump() for e in body.entries],
            context_key=body.context_key,
        )
        return {"entry_ids": ids}
    except MemoryProviderClientError as e:
        raise HTTPException(400, {"error": "invalid_entries", "message": str(e)})
    except MemoryProviderInternalError as e:
        logger.error("memory sync internal error: %s", e)
        raise HTTPException(500, {"error": "memory_backend_unavailable"})


@router.get("/prefetch")
async def memory_prefetch(
    q: str = "",
    limit: int = 10,
    context_key: str | None = None,
    request: Request = None,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
):
    instance, _ = auth
    provider = request.app.state.memory_provider
    results = await provider.prefetch(
        instance_id=instance.id,
        query=q,
        limit=limit,
        context_key=context_key,
    )
    return {"entries": results}
```

- [ ] **Step 4: 注册 router + 启动时 initialize provider**

修改 `backend/src/api/main.py` 的 `create_app` / `lifespan`：

```python
# 在 app 启动时（lifespan startup）
from src.services.memory.pg_provider import PGMemoryProvider
from src.models.database import create_session_factory

app.state.memory_provider = PGMemoryProvider(
    session_factory=create_session_factory()
)
await app.state.memory_provider.initialize()

# register router
from src.api.routes import memory as memory_routes
app.include_router(memory_routes.router)
```

- [ ] **Step 5: 运行测试**

Run: `cd backend && pytest tests/test_api_memory.py -v`
Expected: PASS（4 tests）

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/routes/memory.py backend/src/api/main.py backend/tests/test_api_memory.py
git commit -m "feat(api): /api/v1/memory/sync + /prefetch endpoints"
```

---

## Task 6 · Eval Harness（2 天）

**Files:**
- Create: `backend/tests/evals/__init__.py`
- Create: `backend/tests/evals/compact/__init__.py`
- Create: `backend/tests/evals/compact/fixtures.jsonl`
- Create: `backend/tests/evals/compact/runner.py`
- Create: `backend/tests/evals/compact/scorer.py`
- Create: `backend/tests/evals/compact/baselines/.gitkeep`

### Subtask 6.1 — 10 段 fixture 对话

- [ ] **Step 1: 构造 10 段 JSONL fixture**

创建 `backend/tests/evals/compact/fixtures.jsonl`，每行一个 JSON 对象，covering：
- 多轮对话后要求"总结前面的对话"
- 跨轮引用"把第 3 条建议再展开"
- 角色切换连贯性
- 代码生成后的 debug 对话
- 多语言混合
- 长文档摘要请求
- 事实回忆（人名、地点、日期）
- 多模态提及（占位图片）
- 参数偏好复用（用户说过"简洁回复"→ 后续是否仍简洁）
- 指令嵌入在第 1 轮的记忆压缩（"以下对话中你是 XXX"）

示例格式：

```jsonl
{"id":"summary-after-10-turns","conversation":[{"role":"user","content":"我想学 Python"},{"role":"assistant","content":"好的，从基础语法开始"},{"role":"user","content":"先教变量"},{"role":"assistant","content":"变量是..."},{"role":"user","content":"然后呢？"},{"role":"assistant","content":"控制流..."},{"role":"user","content":"循环"},{"role":"assistant","content":"for/while..."},{"role":"user","content":"函数"},{"role":"assistant","content":"def..."}],"test_prompt":"帮我总结前面学了啥","must_contain":["变量","控制流","循环","函数"]}
```

写 10 条。真实内容应基于你手上有的真实对话日志（logs.db 可挖样本）。

- [ ] **Step 2: Commit fixtures**

```bash
git add backend/tests/evals/compact/fixtures.jsonl
git commit -m "test(evals): 10 fixture conversations for compact eval"
```

### Subtask 6.2 — Scorer (LLM-as-judge)

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/evals/compact/test_scorer.py
import pytest

from backend.tests.evals.compact.scorer import score_response


def test_score_perfect_match_gets_10():
    result = score_response(
        response="涉及变量、控制流、循环、函数",
        must_contain=["变量", "控制流", "循环", "函数"],
    )
    assert result["score"] == 10


def test_score_missing_half_gets_5():
    result = score_response(
        response="变量和循环",
        must_contain=["变量", "控制流", "循环", "函数"],
    )
    assert 4 <= result["score"] <= 6


def test_score_missing_all_gets_0():
    result = score_response(
        response="完全无关的内容",
        must_contain=["变量", "控制流"],
    )
    assert result["score"] == 0
```

- [ ] **Step 2: 运行验证失败**

Run: `cd backend && pytest tests/evals/compact/test_scorer.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现 scorer**

```python
# backend/tests/evals/compact/scorer.py
"""Simple keyword-based scorer for compact eval.

LLM-as-judge version is future work; keyword matching is enough for initial baseline.
"""

from __future__ import annotations


def score_response(*, response: str, must_contain: list[str]) -> dict:
    """Score 0-10 based on fraction of must_contain terms present in response."""
    if not must_contain:
        return {"score": 10, "matched": [], "missing": []}
    matched = [term for term in must_contain if term in response]
    missing = [term for term in must_contain if term not in response]
    ratio = len(matched) / len(must_contain)
    return {
        "score": round(ratio * 10),
        "matched": matched,
        "missing": missing,
    }
```

- [ ] **Step 4: 运行验证通过**

Run: `cd backend && pytest tests/evals/compact/test_scorer.py -v`
Expected: PASS（3 tests）

- [ ] **Step 5: Commit**

```bash
git add backend/tests/evals/compact/scorer.py backend/tests/evals/compact/test_scorer.py
git commit -m "test(evals): keyword-based scorer for compact eval"
```

### Subtask 6.3 — Runner

- [ ] **Step 1: 写 runner**

```python
# backend/tests/evals/compact/runner.py
"""Run compact eval against all fixtures, generate report.

Usage:
    cd backend && python -m tests.evals.compact.runner

Output: backend/tests/evals/compact/latest_report.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.services.context.gzip_compact import GzipCompactContextEngine
from src.services.context.base import ContextOverflowError

from tests.evals.compact.scorer import score_response

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures.jsonl"
BASELINES = HERE / "baselines"
REPORT = HERE / "latest_report.json"

MAX_TOKENS_BUDGET = 1000  # simulate tight context


async def _simulate_response(compacted_messages: list[dict], test_prompt: str) -> str:
    """In production, call real LLM here. For initial baseline, synthesize a response
    by concatenating the content of compacted messages — this tests whether compress
    preserved the right info."""
    all_text = " ".join(m.get("content", "") for m in compacted_messages if isinstance(m.get("content"), str))
    return f"{all_text} [prompt: {test_prompt}]"


async def main():
    engine = GzipCompactContextEngine()
    await engine.initialize()

    results = []
    with FIXTURES.open() as f:
        for line in f:
            fix = json.loads(line)
            try:
                compacted, truncated = await engine.compress(
                    messages=fix["conversation"],
                    max_tokens=MAX_TOKENS_BUDGET,
                )
                response = await _simulate_response(compacted, fix["test_prompt"])
                score = score_response(
                    response=response,
                    must_contain=fix["must_contain"],
                )
            except ContextOverflowError as e:
                score = {"score": 0, "matched": [], "missing": fix["must_contain"], "error": str(e)}
                truncated = True
            results.append({
                "id": fix["id"],
                "score": score["score"],
                "truncated": truncated,
                "missing": score.get("missing", []),
            })

    avg = sum(r["score"] for r in results) / len(results)
    report = {
        "engine": engine.name,
        "fixtures_count": len(results),
        "avg_score": round(avg, 2),
        "results": results,
    }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Avg score: {avg:.2f}/10 (fixtures: {len(results)})")
    print(f"Report: {REPORT}")

    # Baseline comparison
    latest_baseline = BASELINES / "gzip_compact_v1.json"
    if latest_baseline.exists():
        baseline = json.loads(latest_baseline.read_text())
        delta = avg - baseline["avg_score"]
        if delta < -2:
            print(f"⚠️  REGRESSION: avg_score dropped {delta:+.2f} from baseline")
            return 1
        print(f"Baseline delta: {delta:+.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: 跑 runner 生成初始 baseline**

Run:
```bash
cd backend && python -m tests.evals.compact.runner
cp tests/evals/compact/latest_report.json tests/evals/compact/baselines/gzip_compact_v1.json
```

确认输出合理（avg_score 不是 0，也不应是 10，典型在 6-9 之间）。

- [ ] **Step 3: Commit**

```bash
git add backend/tests/evals/compact/runner.py backend/tests/evals/compact/baselines/gzip_compact_v1.json
git commit -m "test(evals): compact eval runner + initial baseline"
```

### Subtask 6.4 — 文档化（Wave 1 验收清单）

- [ ] **Step 1: 加 README**

```markdown
<!-- backend/tests/evals/compact/README.md -->
# Compact Eval Harness

Regression guard for `GzipCompactContextEngine` (or any `ContextEngine` impl).

## Run

    cd backend && python -m tests.evals.compact.runner

Outputs `latest_report.json`. Compares against `baselines/gzip_compact_v1.json`:
- Δavg_score > -2 → OK
- Δavg_score ≤ -2 → ⚠️ REGRESSION (fail the ship decision)

## Update baseline

When compress strategy changes intentionally:

    cp tests/evals/compact/latest_report.json tests/evals/compact/baselines/gzip_compact_v2.json
    # 更新 runner.py 里 latest_baseline 指向
```

- [ ] **Step 2: Commit**

```bash
git add backend/tests/evals/compact/README.md
git commit -m "docs(evals): compact eval harness usage"
```

---

## 整体验收 + PR

- [ ] **Step 1: 跑全量测试**

```bash
cd backend && pytest tests/ -v --tb=short
```
Expected: 所有通过。若有回归，修到通过。

- [ ] **Step 2: 跑 compact eval**

```bash
cd backend && python -m tests.evals.compact.runner
```
Expected: avg_score 不报 REGRESSION。

- [ ] **Step 3: 真跑一次 migration on dev PG**

```bash
psql $DATABASE_URL -f backend/migrations/wave1_memory.sql
```
Expected: 无错，表已存在时 `CREATE TABLE IF NOT EXISTS` 静默。

- [ ] **Step 4: 启服务 + curl 验证**

```bash
.venv/bin/python -m uvicorn src.api.main:create_app --factory --host 127.0.0.1 --port 8001

# 另一个 terminal
curl -sS -X POST http://127.0.0.1:8001/api/v1/memory/sync \
  -H "Authorization: Bearer $NOUS_KEY" \
  -H "Content-Type: application/json" \
  -d '{"entries":[{"category":"preference","content":"用户喜欢简洁回复","context_key":"proj-1"}]}' | jq

curl -sS "http://127.0.0.1:8001/api/v1/memory/prefetch?q=简洁&limit=5" \
  -H "Authorization: Bearer $NOUS_KEY" | jq
```

- [ ] **Step 5: 前端验证 node_end_streaming**

在浏览器跑一个 workflow，DevTools Network 过滤 ws，确认 `node_end_streaming` 出现。

- [ ] **Step 6: 开 PR**

```bash
git push -u origin feature/wave1-platform-contracts
gh pr create --base feature/nous-center-v2 --title "feat(wave1): platform contracts (memory ABC + context engine + node classes)" --body "$(cat <<'EOF'
## Summary
- 事件扩展：6 个新 event types（node_end_streaming 等）
- MemoryProvider ABC + PGMemoryProvider reference + /api/v1/memory/{sync,prefetch}
- ContextEngine ABC + GzipCompactContextEngine（迁移自 compact_messages）
- 12 个 _exec_* function 全迁移为 class，实现 Protocol；取消全局 _on_progress_ref
- Eval harness（10 段 fixture + keyword scorer + baseline 比对）

Spec: docs/superpowers/specs/2026-04-16-wave1-platform-contracts-design.md

## Test plan
- [x] Unit: MemoryProvider ABC 6 tests + contract mixin 7 tests
- [x] Unit: ContextEngine ABC 4 tests + gzip engine 6 tests + regression match
- [x] Unit: Node protocols 4 + migration equivalence 12
- [x] CRITICAL REGRESSION: workflow LLM token stats 3 tests
- [x] Integration: PG memory provider contract + 4 PG-specific
- [x] Integration: /api/v1/memory/sync + /prefetch 4 tests
- [x] Eval: compact eval avg_score baseline

## Rollout
1. Deploy migration first: `psql $DATABASE_URL -f backend/migrations/wave1_memory.sql`
2. Deploy code
3. 手动验证 `/api/v1/memory/sync` 与 `node_end_streaming` ws 事件
4. 发内部公告：MemoryProvider ABC 可用，mediahub 可对接

## Next
Wave 2 spec: 2026-04-17-wave2-reliability-design.md
EOF
)"
```

---

## Self-Review Checklist

### 1. Spec coverage

| Spec section | Task |
|--------------|------|
| Task 1 事件扩展（6 个新事件）| Task 1 |
| MemoryProvider ABC + 异常分级 | Task 2 |
| ContextEngine ABC 无状态 | Task 3 |
| 节点 Protocol 分层 + 取消 _on_progress_ref | Task 4 |
| PGMemoryProvider reference | Task 5 |
| Eval harness | Task 6 |
| AbstractMemoryProviderTests mixin | Task 2.2 |
| add_entries 空列表幂等（决策 10）| Task 2.2 test |
| 批量 100 条上限（决策 11）| Task 5.2 + 5.3 |
| nous-center 只存不抽取（决策 13）| MemoryProvider ABC docstring + PGMemoryProvider |
| async system_prompt_block（决策 8）| Task 2.1 test + PluginBase |
| Token stats 不回归 | Task 4.3 critical regression |

### 2. Placeholder scan

- Task 1.2 里 `mock_llm_stream` 和 Task 4.3 的 `mock_llm_stream_v2` fixture 标了"占位，实现时对齐"——这是因为 `_exec_llm` 的真实 streaming 接口依赖具体 adapter，实施者看到时要现场补。**不是"TODO 以后再做"**，而是"现场调整"。
- Task 6.1 fixture 内容是示例，实施者要**真的写 10 条**。提示里明说"真实对话日志可挖 logs.db 样本"。

### 3. Type consistency

- `MemoryEntry` / `StoredMemoryEntry` TypedDict 在 Task 2.1 定义，Task 2.2/5.2/5.3 引用
- `MemoryProvider` / `ContextEngine` / `PluginBase` 签名在各自 Task 定义，后续一致
- `InvokableNode` / `StreamableNode` Protocol 在 Task 4.1 定义，Task 4.2-4.6 一致
- `OnTokenFn` callback 类型在 Task 4.1 定义，Task 4.3 LLMNode.stream 一致使用
- `add_entries(instance_id, api_key_id, entries, context_key)` 签名一致（Task 2 ABC / Task 5 impl / Task 5.3 route）

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-04-20-wave1-platform-contracts.md`。总 6 个 Task、24 个 Subtask、~80 个 steps、~7.5 天工作量。

与 agent/skill plan 的对比：

| 项 | agent/skill | Wave 1 |
|----|------------|--------|
| 任务数 | 14 | 6 (24 subtasks) |
| 文件变更 | ~18 | ~25 |
| 工作量 | 3 天 | 7.5 天 |
| 分支 | `feature/agent-skill-injection` | `feature/wave1-platform-contracts` |
| 核心风险 | 本地 Qwen 是否真调 Skill tool | LLM node 迁移保持 token stats 不回归 |

按选项 C 并行分支执行：
- 两个分支都从 `feature/nous-center-v2` fork
- 各自独立 PR 回 `feature/nous-center-v2`
- 若有人工介入交替（单人串行执行），顺序建议 agent/skill 先（3 天，验证 lazy-load 假设）→ Wave 1（7.5 天）

**Subagent-Driven 模式已选定**。下一步：为两个 plan 各建 worktree 后开始派发 subagent 跑 Task 1。