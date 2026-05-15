# V1.5 Lane S: workflow_executor 重写 + /run 契约变更 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `workflow_executor.py` 的 `execute()`（现 `:102` 扁平顺序循环）重写成「节点分流」执行——`llm` 节点 inline 走 HTTP 调 vLLM，`image` / `tts` 节点 dispatch 到 runner 串行队列（D10）；把 `/v1/instances/{id}/run` 与 `/api/v1/workflows/execute` 两个同步阻塞端点改成纯异步「入队 + 立即返回 202 + task_id」（D17）；给上游（mediahub 等）写迁移指引。

**Architecture:** 三块改动，按依赖顺序交付：

1. **节点分流（dispatch vs inline-HTTP）** —— spec §2.1 step 9 / §4.5「Inline 执行点改道清单」。当前 `execute()` 对所有节点走同一条 `_execute_node` 路径。V1.5 引入 `node_exec_class(node_type) -> "inline" | "dispatch"` 判定：`llm` / `text_*` / `logic` / `agent` 等纯 CPU 或 HTTP 节点走 `inline`（在主进程 event loop 内 await，`llm` 节点的 HTTP-调-vLLM 由现有 `LLMNode` adapter 完成，本来就是 HTTP，无需改）；`image_generate` / `tts_engine` 等 GPU 节点走 `dispatch`（投到 `RunnerClient.run_node`）。同 group 的 dispatch 节点串行 await，无 GPU 依赖的 inline 节点按拓扑序执行。**注意（与 spec 的偏差）见本文件「与 spec 的偏差」节。**

2. **RunnerClient 注入点** —— Lane S 依赖 Lane C 的 `RunnerClient`（spec §3.3 协议、§3.5 `GroupScheduler.runner_client`）。**Lane C 的 plan 尚未编写**，本 Lane 按 spec §3.3 定义的 `run_node(spec, inputs) -> NodeResult` 接口编程，并在 `WorkflowExecutor.__init__` 注入一个 `runner_client` 参数（默认 `None`）。测试用 `tests/fixtures/fake_runner_client.py`（本 Lane 新建的纯 Python stub，不是 Lane C 的 subprocess fake）。Lane C 落地后，main.py 把真 `RunnerClient` 注入即可，`WorkflowExecutor` 无需再改。

3. **`/run` 纯异步契约（D17）** —— 当前 `execute_workflow_direct`（`workflows.py:142`）和 `instance_run`（`instance_service.py:93`）都在 request handler 内同步 `await executor.execute()`，阻塞到 workflow 跑完才返回 result。V1.5 改成：建 `ExecutionTask(status="queued")` → commit → 用 `asyncio.create_task` 在后台跑 executor → 立即返回 `202 {"task_id": ...}`。客户端轮询 `/api/v1/tasks/{id}` 或订阅 WS `/ws/workflow/{task_id}` 拿结果。后台 task 跑完写 `task.status` + `result` + `duration_ms`。

**Tech Stack:** Python 3.12 / FastAPI / asyncio / pytest（`asyncio_mode = "auto"`，conftest 强制 `ADMIN_PASSWORD=""` + `NOUS_DISABLE_FRONTEND_MOUNT=1`）。无新第三方依赖。

> **与 spec 的偏差（已核实，须知会）：**
>
> 1. **spec 写的端点路径 `/v1/workflows/{id}/run` 在本仓库不存在。** spec §2.1 / §「/run 契约变更」反复说 `/v1/workflows/{id}/run`，但 grep 全仓只有两个 workflow 执行入口：`POST /v1/instances/{instance_id}/run`（`instance_service.py:93`，Bearer-token 鉴权，mediahub 等上游用的就是这个）和 `POST /api/v1/workflows/execute`（`workflows.py:142`，前端 Run 按钮用，cookie 鉴权）。spec 把它俩笼统称作「workflow `/run`」。本 Lane 对**两个**端点都做 D17 异步化改造，迁移指引针对 `instance_run`（上游真正调的那个）。已在 Self-Review 标注。
>
> 2. **spec 说 `llm` 节点要「executor inline HTTP 调 vLLM」，暗示当前是「主进程内调 adapter」。** 实际上现有 `LLMNode`（`src/services/nodes/llm.py:127`）已经通过 `InferenceAdapter`（owns httpx + base_url）走 HTTP 调 vLLM——它**本来就是 inline HTTP**。所以 `llm` 节点在 Lane S 不需要改执行方式，只需要在分流判定里被归类为 `inline`。spec §4.5 改道清单里「workflow_executor 的 llm 节点 → executor inline HTTP 调 vLLM」对本仓库而言是「保持现状 + 归类为 inline」。真正变的是 `image` / `tts` 节点从「主进程内调 adapter」改成「dispatch 到 runner」。已在 Self-Review 标注。
>
> 3. **本仓库无 `outputs/{task_id}/` 落盘机制（spec §2.1 step 12a）。** 那是 Lane D（image runner 落盘）+ Lane I（缩略图历史）的事。Lane S 的 `RunnerClient.run_node` 按 spec §3.3 `NodeResult.outputs` 契约接收 runner 返回的 dict，原样塞进 `self._outputs`——Lane S 不关心 outputs 里是路径还是内联数据，只做透传。

---

## File Structure

| 文件 | Lane S 动作 | 责任 |
|---|---|---|
| `backend/src/services/node_routing.py` | **新建** | `node_exec_class(node_type) -> Literal["inline","dispatch"]` —— 节点分流判定的唯一真相源 |
| `backend/src/services/workflow_executor.py` | **修改** | `execute()` 重写为节点分流；`WorkflowExecutor.__init__` 加 `runner_client` 参数；`_execute_node` 拆出 `_execute_inline_node` / `_dispatch_node` |
| `backend/src/services/workflow_runner.py` | **新建** | `run_workflow_task(task_id, workflow_data, ...)` —— 后台 task 入口：建 executor、跑、写 ExecutionTask 终态、推 WS。两个 `/run` 端点共用 |
| `backend/src/api/routes/instance_service.py` | **修改** | `instance_run` 改纯异步：建 task → `create_task(run_workflow_task(...))` → 返回 `202 {task_id}` |
| `backend/src/api/routes/workflows.py` | **修改** | `execute_workflow_direct` 同样改纯异步 202 |
| `backend/tests/fixtures/fake_runner_client.py` | **新建** | 纯 Python `RunnerClient` stub：可配置 run_node 返回值 / 抛错 / 记录调用 |
| `backend/tests/test_node_routing.py` | **新建** | 分流判定：llm→inline、image_generate→dispatch、未知类型默认 inline |
| `backend/tests/test_workflow_executor_split.py` | **新建** | executor 节点分流：inline 节点不碰 runner_client、dispatch 节点走 runner_client、混合 workflow、runner_client=None 时 dispatch 节点报错 |
| `backend/tests/test_run_async_contract.py` | **新建** | **[回归]** 两个 `/run` 端点：202 + task_id；后台跑完 task 落 completed；poll `/tasks/{id}` 拿到 result（enqueue → poll → result 端到端） |

---

## Task 1: 节点分流判定 `node_routing.py`

spec §2.1 step 9 / §4.5 要求 executor 按节点类型分流。本仓库节点类型从 `@register(...)` 注册（`grep` 得：`llm` / `image_generate` / `image_output` / `tts_engine` / `ref_audio` / `text_input` / `text_output` / `prompt_template` / `agent` / `python_exec` / `if_else` / `multimodal_input` / `output` / `passthrough` / `resample` / `mixer` / `concat` / `bgm_mix`）。GPU 节点（要 dispatch 到 runner）目前只有 `image_generate` 和 `tts_engine`——其余都是 CPU / HTTP 节点，inline 执行。

**Files:**
- New: `backend/src/services/node_routing.py`
- Test: `backend/tests/test_node_routing.py`（新建）

- [ ] **Step 1: 写失败测试 —— 分流判定**

新建 `backend/tests/test_node_routing.py`：
```python
"""Lane S: 节点分流判定（dispatch vs inline-HTTP）。"""
import pytest

from src.services.node_routing import node_exec_class, DISPATCH_NODE_TYPES


def test_llm_node_is_inline():
    """llm 节点走 inline —— 现有 LLMNode 已经 HTTP 调 vLLM。"""
    assert node_exec_class("llm") == "inline"


def test_image_generate_is_dispatch():
    """image_generate 是 GPU 节点 —— dispatch 到 runner 串行队列。"""
    assert node_exec_class("image_generate") == "dispatch"


def test_tts_engine_is_dispatch():
    assert node_exec_class("tts_engine") == "dispatch"


@pytest.mark.parametrize("node_type", [
    "text_input", "text_output", "prompt_template", "agent",
    "if_else", "python_exec", "passthrough", "output",
])
def test_cpu_nodes_are_inline(node_type):
    """纯 CPU / 逻辑节点走 inline。"""
    assert node_exec_class(node_type) == "inline"


def test_unknown_node_defaults_inline():
    """未知节点类型（含插件节点）默认 inline —— 保守：不假设它需要 GPU runner。"""
    assert node_exec_class("some_plugin_node") == "inline"


def test_dispatch_set_is_explicit():
    """DISPATCH_NODE_TYPES 是显式白名单，新增 GPU 节点必须在此登记。"""
    assert DISPATCH_NODE_TYPES == {"image_generate", "tts_engine"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_node_routing.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'src.services.node_routing'`。

- [ ] **Step 3: 实现 `node_routing.py`**

新建 `backend/src/services/node_routing.py`：
```python
"""节点分流判定 —— dispatch 节点（GPU runner 串行队列）vs inline 节点（主进程 event loop）。

spec §2.1 step 9 / §4.5「Inline 执行点改道清单」。

- dispatch 节点：在 GPU runner 子进程内执行（image / tts），主进程经 RunnerClient.run_node 投递
- inline 节点：在主进程 event loop 内直接 await（CPU 逻辑节点；llm 节点本身已是 HTTP-调-vLLM）

DISPATCH_NODE_TYPES 是显式白名单 —— 新增任何需要 GPU runner 的节点类型，必须在此登记，
否则会被当作 inline 在主进程内执行（撞 GPU race，正是 V1.5 要消灭的问题）。
"""
from __future__ import annotations

from typing import Literal

ExecClass = Literal["inline", "dispatch"]

# GPU 节点白名单 —— 这些节点 dispatch 到对应 runner 的串行队列执行。
DISPATCH_NODE_TYPES: frozenset[str] = frozenset({"image_generate", "tts_engine"})


def node_exec_class(node_type: str) -> ExecClass:
    """判定一个节点类型走 dispatch 还是 inline。

    未登记的类型（含第三方插件节点）默认 inline —— 保守策略：不假设未知节点
    需要 GPU runner。若某插件节点其实吃 GPU，需显式加进 DISPATCH_NODE_TYPES。
    """
    return "dispatch" if node_type in DISPATCH_NODE_TYPES else "inline"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_node_routing.py -q`
Expected: 全 PASS（7 个用例 / 含 parametrize 展开共 14 个）。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/services/node_routing.py tests/test_node_routing.py
git commit -m "feat(workflow): node routing — dispatch (GPU runner) vs inline classification

spec §2.1 step 9 / §4.5. DISPATCH_NODE_TYPES is an explicit allowlist
of GPU node types (image_generate, tts_engine); everything else runs
inline in the main event loop. V1.5 Lane S."
```

---

## Task 2: fake RunnerClient stub

Lane S 依赖 Lane C 的 `RunnerClient`（spec §3.3）。Lane C plan 尚未编写，本 Lane 按 spec §3.3 的 `run_node` 节点级 RPC 契约编程，并提供一个纯 Python stub 给 executor 测试用。stub 不起 subprocess、不走 pipe —— 它只实现 executor 看到的那个 `async run_node(...)` 接口。

**Files:**
- New: `backend/tests/fixtures/fake_runner_client.py`
- Test: 本 Task 不写独立测试（stub 在 Task 3 被 executor 测试消费）；但加一个 self-check 用例确保 stub 行为可预期。

- [ ] **Step 1: 实现 fake RunnerClient**

新建 `backend/tests/fixtures/fake_runner_client.py`：
```python
"""Lane S: 纯 Python RunnerClient stub —— 给 WorkflowExecutor 节点分流测试用。

不是 Lane C 的 subprocess fake_runner —— 这个 stub 只实现 executor 直接看到的
`async run_node(node_spec, inputs) -> dict` 接口（spec §3.3 RunNode/NodeResult
节点级 RPC）。Lane C 落地真 RunnerClient 后，executor 代码不变，只换注入对象。
"""
from __future__ import annotations

from typing import Any


class FakeRunnerClient:
    """可配置的 RunnerClient stub。

    - results: node_id -> 该节点 run_node 返回的 outputs dict
    - fail_nodes: 这些 node_id 调 run_node 时抛 RuntimeError
    - calls: 按调用顺序记录 (node_id, node_type, inputs)，给断言用
    """

    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        fail_nodes: set[str] | None = None,
    ):
        self._results = results or {}
        self._fail_nodes = fail_nodes or set()
        self.calls: list[tuple[str, str, dict]] = []

    async def run_node(
        self, node: dict, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """对齐 spec §3.3：主进程投 RunNode，runner 回 NodeResult.outputs。"""
        node_id = node["id"]
        node_type = node["type"]
        self.calls.append((node_id, node_type, dict(inputs)))
        if node_id in self._fail_nodes:
            raise RuntimeError(f"fake runner: node {node_id} failed")
        return self._results.get(node_id, {"result": f"dispatched:{node_id}"})
```

- [ ] **Step 2: 加 stub self-check 用例**

在 `backend/tests/test_workflow_executor_split.py` 顶部（Task 3 会建此文件，这里先放骨架 + self-check）。**注意：本 Step 只为让 stub 有一次独立验证；完整 executor 测试在 Task 3。** 新建 `backend/tests/test_workflow_executor_split.py`：
```python
"""Lane S: WorkflowExecutor 节点分流执行测试。"""
import pytest

from tests.fixtures.fake_runner_client import FakeRunnerClient


@pytest.mark.asyncio
async def test_fake_runner_client_records_calls():
    """stub self-check：run_node 记录调用、按 node_id 返回配置结果。"""
    rc = FakeRunnerClient(results={"n1": {"image_url": "x.png"}})
    out = await rc.run_node({"id": "n1", "type": "image_generate"}, {"prompt": "cat"})
    assert out == {"image_url": "x.png"}
    assert rc.calls == [("n1", "image_generate", {"prompt": "cat"})]


@pytest.mark.asyncio
async def test_fake_runner_client_fail_nodes():
    rc = FakeRunnerClient(fail_nodes={"bad"})
    with pytest.raises(RuntimeError, match="node bad failed"):
        await rc.run_node({"id": "bad", "type": "image_generate"}, {})
```

- [ ] **Step 3: 跑 self-check 确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_workflow_executor_split.py -q`
Expected: 2 个 PASS。

- [ ] **Step 4: Commit**

```bash
cd backend && git add tests/fixtures/fake_runner_client.py tests/test_workflow_executor_split.py
git commit -m "test(workflow): FakeRunnerClient stub for executor dispatch tests

Pure-Python stub implementing the run_node RPC contract (spec §3.3),
so Lane S executor tests don't need Lane C's runner subprocess. V1.5 Lane S."
```

---

## Task 3: `WorkflowExecutor.execute()` 重写为节点分流

这是 Lane S 的核心。当前 `execute()`（`workflow_executor.py:102`）对所有节点走同一条 `_execute_node`。重写后：拓扑序遍历，每个节点按 `node_exec_class` 分流——inline 节点走现有 `_execute_node` 逻辑（重命名为 `_execute_inline_node`），dispatch 节点走 `RunnerClient.run_node`。progress 事件、错误包装、`self._outputs` 行为全部保持不变。

**Files:**
- Modify: `backend/src/services/workflow_executor.py`
- Test: `backend/tests/test_workflow_executor_split.py`（Task 2 已建，本 Task 追加用例）

- [ ] **Step 1: 跑现有 executor suite 建基线**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "executor or workflow"`
Expected: PASS（记下通过数。重写 `execute()` 后这些 inline-only workflow 测试必须仍全绿——这是行为不变的回归保证）。

- [ ] **Step 2: 写失败测试 —— 节点分流执行**

在 `backend/tests/test_workflow_executor_split.py` 追加：
```python
from src.services.workflow_executor import WorkflowExecutor, ExecutionError


def _wf(nodes, edges=None):
    return {"nodes": nodes, "edges": edges or []}


@pytest.mark.asyncio
async def test_inline_node_does_not_touch_runner_client():
    """纯 inline workflow：runner_client 一次都不该被调。"""
    rc = FakeRunnerClient()
    wf = _wf([{"id": "t1", "type": "text_input", "data": {"text": "hello"}}])
    ex = WorkflowExecutor(wf, runner_client=rc)
    result = await ex.execute()
    assert rc.calls == []
    assert "t1" in result["outputs"]


@pytest.mark.asyncio
async def test_dispatch_node_routes_to_runner_client():
    """image_generate 节点 → RunnerClient.run_node，结果进 outputs。"""
    rc = FakeRunnerClient(results={"img": {"image_url": "out.png"}})
    wf = _wf([{"id": "img", "type": "image_generate", "data": {"prompt": "cat"}}])
    ex = WorkflowExecutor(wf, runner_client=rc)
    result = await ex.execute()
    assert rc.calls[0][0] == "img"
    assert result["outputs"]["img"] == {"image_url": "out.png"}


@pytest.mark.asyncio
async def test_mixed_workflow_inline_then_dispatch():
    """text_input(inline) → image_generate(dispatch)：上游 inline 输出进下游 dispatch 的 inputs。"""
    rc = FakeRunnerClient(results={"img": {"image_url": "out.png"}})
    wf = _wf(
        nodes=[
            {"id": "t", "type": "text_input", "data": {"text": "a cat"}},
            {"id": "img", "type": "image_generate", "data": {}},
        ],
        edges=[{"source": "t", "target": "img",
                "sourceHandle": "text", "targetHandle": "prompt"}],
    )
    ex = WorkflowExecutor(wf, runner_client=rc)
    result = await ex.execute()
    # dispatch 节点拿到了 inline 上游的输出
    assert "text" in rc.calls[0][2] or "prompt" in rc.calls[0][2]
    assert result["outputs"]["img"] == {"image_url": "out.png"}


@pytest.mark.asyncio
async def test_dispatch_node_without_runner_client_raises():
    """runner_client=None 但 workflow 含 dispatch 节点 → ExecutionError（不静默 inline 跑 GPU 节点）。"""
    wf = _wf([{"id": "img", "type": "image_generate", "data": {}}])
    ex = WorkflowExecutor(wf, runner_client=None)
    with pytest.raises(ExecutionError, match="runner"):
        await ex.execute()


@pytest.mark.asyncio
async def test_dispatch_node_failure_wrapped():
    """runner 抛错 → ExecutionError，node_error progress 事件发出。"""
    events = []
    rc = FakeRunnerClient(fail_nodes={"img"})
    wf = _wf([{"id": "img", "type": "image_generate", "data": {}}])
    ex = WorkflowExecutor(wf, runner_client=rc, on_progress=lambda e: events.append(e))
    with pytest.raises(ExecutionError):
        await ex.execute()
    assert any(e["type"] == "node_error" and e["node_id"] == "img" for e in events)


@pytest.mark.asyncio
async def test_progress_events_unchanged_for_dispatch():
    """dispatch 节点同样发 node_start / node_complete progress 事件。"""
    events = []
    rc = FakeRunnerClient(results={"img": {"image_url": "x"}})
    wf = _wf([{"id": "img", "type": "image_generate", "data": {}}])
    ex = WorkflowExecutor(wf, runner_client=rc, on_progress=lambda e: events.append(e))
    await ex.execute()
    types = [e["type"] for e in events]
    assert "node_start" in types and "node_complete" in types
```
（注：`on_progress` 这里传同步 lambda 仅为收集——executor 内是 `await self._on_progress(...)`，同步 lambda 返回 None 不可 await。改用下面的 async 收集器。)

把上面 `on_progress=lambda e: events.append(e)` 两处替换为一个 async 收集器：
```python
def _collector():
    events: list[dict] = []
    async def _on(e: dict) -> None:
        events.append(e)
    return events, _on
# 用法：events, on_progress = _collector(); WorkflowExecutor(..., on_progress=on_progress)
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_workflow_executor_split.py -q`
Expected: FAIL —— `WorkflowExecutor.__init__` 还不接受 `runner_client` 参数（`TypeError: unexpected keyword argument 'runner_client'`）。

- [ ] **Step 4: 重写 `workflow_executor.py`**

`backend/src/services/workflow_executor.py`：

(a) `__init__` 加 `runner_client` 参数：
```python
    def __init__(self, workflow: dict, on_progress=None, runner_client=None):
        self.nodes: list[dict] = workflow.get("nodes", [])
        self.edges: list[dict] = workflow.get("edges", [])
        self._node_map: dict[str, dict] = {n["id"]: n for n in self.nodes}
        self._outputs: dict[str, dict[str, Any]] = {}
        self._on_progress = on_progress  # async callback(data: dict)
        self._runner_client = runner_client  # Lane C RunnerClient；inline-only workflow 可为 None
```

(b) `execute()` 里的 `output = await self._execute_node(node, inputs)` 一行改成分流调用：
```python
            try:
                output = await self._run_node_routed(node, inputs)
                self._outputs[node_id] = output
            except Exception as e:
```
（`try` / `except` / progress 事件 / `_topological_sort` / `_get_inputs` 全部不动。）

(c) 新增分流方法 + 把原 `_execute_node` 重命名为 `_execute_inline_node`：
```python
    async def _run_node_routed(self, node: dict, inputs: dict) -> dict[str, Any]:
        """按节点类型分流：inline 节点主进程内 await，dispatch 节点投 RunnerClient。

        spec §2.1 step 9 / §4.5。
        """
        from src.services.node_routing import node_exec_class

        if node_exec_class(node["type"]) == "dispatch":
            return await self._dispatch_node(node, inputs)
        return await self._execute_inline_node(node, inputs)

    async def _dispatch_node(self, node: dict, inputs: dict) -> dict[str, Any]:
        """GPU 节点 → RunnerClient.run_node（spec §3.3 RunNode/NodeResult RPC）。

        runner_client 缺失时显式报错 —— 绝不静默在主进程内 inline 跑 GPU 节点
        （那正是 V1.5 要消灭的 GPU race）。
        """
        if self._runner_client is None:
            raise ExecutionError(
                f"节点 {node['id']} ({node['type']}) 需要 GPU runner，"
                f"但 executor 未注入 runner_client"
            )
        return await self._runner_client.run_node(node, inputs)

    async def _execute_inline_node(self, node: dict, inputs: dict) -> dict[str, Any]:
        """主进程 event loop 内直接执行（原 _execute_node 逻辑，零改动）。"""
        # ... 原 _execute_node 函数体整段搬过来，一字不改 ...
```
（`_execute_node` 函数体——`get_node_class` / `StreamableNode` / `InvokableNode` 那整段——原样搬进 `_execute_inline_node`，逻辑不动。`execute()` 里不再有任何地方调 `_execute_node` 这个旧名字。）

- [ ] **Step 5: 跑 Lane S 测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_workflow_executor_split.py tests/test_node_routing.py -q`
Expected: 全 PASS。

- [ ] **Step 6: 跑现有 executor suite 确认零回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "executor or workflow"`
Expected: PASS，通过数 = Step 1 基线 + 新增用例数。inline-only workflow 行为完全不变（这些测试不传 `runner_client`，默认 `None`，且不含 dispatch 节点，走 `_execute_inline_node` 与重写前等价）。

- [ ] **Step 7: Commit**

```bash
cd backend && git add src/services/workflow_executor.py tests/test_workflow_executor_split.py
git commit -m "feat(workflow): executor splits dispatch nodes from inline nodes

execute() now routes each node via node_exec_class: GPU nodes go to
RunnerClient.run_node (Lane C), inline nodes run in the main event loop
as before. Dispatch node with no runner_client raises explicitly rather
than silently running GPU work in-process. spec §2.1 step 9. V1.5 Lane S."
```

---

## Task 4: 后台执行入口 `workflow_runner.py`

D17 异步化后，两个 `/run` 端点不再在 request handler 内 `await executor.execute()`。把「建 executor → 跑 → 写 ExecutionTask 终态 → 推 WS」这段逻辑抽到一个共用函数 `run_workflow_task`，由 `asyncio.create_task` 在后台调度。

**Files:**
- New: `backend/src/services/workflow_runner.py`
- Test: 本 Task 的逻辑在 Task 5 的端到端回归测试里被覆盖；本 Task 加一个直接单测。

- [ ] **Step 1: 写失败测试 —— run_workflow_task 写终态**

新建 `backend/tests/test_workflow_runner.py`：
```python
"""Lane S: 后台 workflow 执行入口 run_workflow_task。"""
import pytest
from sqlalchemy import select

from src.models.execution_task import ExecutionTask
from src.services.workflow_runner import run_workflow_task


@pytest.mark.asyncio
async def test_run_workflow_task_marks_completed(db_session):
    """跑完一个 inline-only workflow → task.status=completed + result + duration_ms。"""
    task = ExecutionTask(workflow_name="t", status="queued", nodes_total=1)
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    wf = {"nodes": [{"id": "t1", "type": "text_input", "data": {"text": "hi"}}],
          "edges": []}
    await run_workflow_task(task.id, wf, runner_client=None, channel_id=None)

    refreshed = await db_session.get(ExecutionTask, task.id)
    await db_session.refresh(refreshed)
    assert refreshed.status == "completed"
    assert refreshed.result is not None
    assert refreshed.duration_ms is not None
    assert refreshed.nodes_done == 1


@pytest.mark.asyncio
async def test_run_workflow_task_marks_failed_on_error(db_session):
    """workflow 抛 ExecutionError → task.status=failed + error 落表，不抛出。"""
    task = ExecutionTask(workflow_name="t", status="queued", nodes_total=1)
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # 空 workflow → _topological_sort 抛 ExecutionError("工作流为空")
    await run_workflow_task(task.id, {"nodes": [], "edges": []},
                            runner_client=None, channel_id=None)

    refreshed = await db_session.get(ExecutionTask, task.id)
    await db_session.refresh(refreshed)
    assert refreshed.status == "failed"
    assert refreshed.error
```
（`db_session` fixture 名对齐 `tests/conftest.py` 现有命名——若是 `db` / `async_session` 之类，照改。）

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_workflow_runner.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'src.services.workflow_runner'`。

- [ ] **Step 3: 实现 `workflow_runner.py`**

新建 `backend/src/services/workflow_runner.py`：
```python
"""后台 workflow 执行入口 —— D17 纯异步契约的执行侧。

两个 /run 端点（instance_run / execute_workflow_direct）建完 ExecutionTask 后，
用 asyncio.create_task(run_workflow_task(...)) 把执行丢到后台，立即返回 202。
本函数负责：建 WorkflowExecutor、跑、把终态写回 ExecutionTask、推 WS complete。

注意：本函数自己开一个独立 DB session —— 它跑在 request 之外的后台 task 里，
不能复用 request-scoped 的 session（那个在 handler 返回时就关了）。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from src.models.database import async_session_factory
from src.models.execution_task import ExecutionTask
from src.services.workflow_executor import ExecutionError, WorkflowExecutor

logger = logging.getLogger(__name__)


async def _broadcast(channel_id: str | None, event: dict) -> None:
    """推 WS 进度事件 —— 复用现有 /ws/workflow/{channel_id} 连接桶。"""
    if not channel_id:
        return
    from src.api.main import _ws_connections

    for ws in list(_ws_connections.get(channel_id, [])):
        try:
            await ws.send_json(event)
        except Exception as e:  # noqa: BLE001 — WS 推送失败静默吞（spec §4.1）
            logger.warning("workflow_runner broadcast failed: %s", e)


async def run_workflow_task(
    task_id: int,
    workflow_data: dict,
    runner_client: Any = None,
    channel_id: str | None = None,
) -> None:
    """后台执行一个 workflow，把终态写回 ExecutionTask。

    本函数不抛出 —— 所有异常都落到 task.status=failed + task.error。调用方
    （create_task）拿不到也不该拿返回值。
    """
    start = time.monotonic()
    nodes = workflow_data.get("nodes", [])

    async def on_progress(event: dict) -> None:
        await _broadcast(channel_id, event)

    executor = WorkflowExecutor(
        workflow_data,
        on_progress=on_progress if channel_id else None,
        runner_client=runner_client,
    )

    async with async_session_factory() as session:
        task = await session.get(ExecutionTask, task_id)
        if task is None:
            logger.error("run_workflow_task: task %s not found", task_id)
            return
        task.status = "running"
        await session.commit()

        try:
            result = await executor.execute()
            elapsed = int((time.monotonic() - start) * 1000)
            task.status = "completed"
            task.result = result
            task.duration_ms = elapsed
            task.nodes_done = len(nodes)
            task.current_node = None
            await session.commit()
            await _broadcast(channel_id, {"type": "complete", "progress": 100})
        except ExecutionError as e:
            elapsed = int((time.monotonic() - start) * 1000)
            task.status = "failed"
            task.error = str(e)
            task.duration_ms = elapsed
            await session.commit()
            logger.error("workflow %s failed: %s", task_id, e)
        except Exception as e:  # noqa: BLE001 — 后台 task 永不冒泡
            elapsed = int((time.monotonic() - start) * 1000)
            task.status = "failed"
            task.error = str(e)
            task.duration_ms = elapsed
            await session.commit()
            logger.error("workflow %s errored: %s", task_id, e, exc_info=True)
```
（**实现前先核对**：`src/models/database.py` 里 session factory 的实际名字。grep `async_sessionmaker\|sessionmaker\|async_session` 确认——若叫 `AsyncSessionLocal` / `async_session` 等，照改 import 与 `async with` 那行。这是唯一需要现场核对的接缝。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_workflow_runner.py -q`
Expected: 2 个 PASS。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/services/workflow_runner.py tests/test_workflow_runner.py
git commit -m "feat(workflow): run_workflow_task background entrypoint for async /run

Shared background runner for the D17 pure-async contract: opens its own
DB session, runs the executor, writes the terminal ExecutionTask state,
broadcasts WS complete. Never raises — failures land in task.error.
V1.5 Lane S."
```

---

## Task 5: 两个 `/run` 端点改纯异步 202（D17）+ 端到端回归

把 `instance_run`（`instance_service.py:93`）和 `execute_workflow_direct`（`workflows.py:142`）从「同步阻塞到 workflow 跑完」改成「建 task → 后台 create_task → 202 + task_id」。这是 Lane S 的回归风险点 —— 必须有 enqueue → poll → result 的端到端测试守住。

**Files:**
- Modify: `backend/src/api/routes/instance_service.py`
- Modify: `backend/src/api/routes/workflows.py`
- Test: `backend/tests/test_run_async_contract.py`（新建）

- [ ] **Step 1: 跑现有 `/run` 相关 suite 建基线**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "instance or workflow_direct or run"`
Expected: PASS。记下哪些用例断言「`/run` 响应 body 直接是 result」—— 那些断言在 D17 下必须改成「响应是 `{task_id}` → poll」。若改不动（外部契约测试），在本 Task Step 5 单独处理。

- [ ] **Step 2: 写失败测试 —— 异步契约 + 端到端**

新建 `backend/tests/test_run_async_contract.py`：
```python
"""Lane S: /run 纯异步契约（D17）回归。

回归风险：/run 从同步阻塞改成 202 + task_id。本测试守住
enqueue → poll /tasks/{id} → result 的端到端链路不断。
"""
import asyncio

import pytest


async def _poll_until_done(client, task_id, timeout=5.0):
    """轮询 /api/v1/tasks/{id} 直到 status 进入终态。"""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/api/v1/tasks/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} 未在 {timeout}s 内完成")


@pytest.mark.asyncio
async def test_execute_workflow_direct_returns_202_task_id(client):
    """POST /api/v1/workflows/execute → 202 + task_id（不再同步阻塞返回 result）。"""
    resp = await client.post("/api/v1/workflows/execute", json={
        "name": "test-async",
        "nodes": [{"id": "t1", "type": "text_input", "data": {"text": "hi"}}],
        "edges": [],
    })
    assert resp.status_code == 202
    body = resp.json()
    assert "task_id" in body


@pytest.mark.asyncio
async def test_execute_workflow_direct_enqueue_poll_result(client):
    """端到端：enqueue → poll → 拿到 completed + result。"""
    resp = await client.post("/api/v1/workflows/execute", json={
        "name": "e2e",
        "nodes": [{"id": "t1", "type": "text_input", "data": {"text": "hello"}}],
        "edges": [],
    })
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    final = await _poll_until_done(client, task_id)
    assert final["status"] == "completed"
    assert final["result"] is not None
    assert final["nodes_done"] == 1


@pytest.mark.asyncio
async def test_instance_run_returns_202(client, published_workflow_instance):
    """POST /v1/instances/{id}/run → 202 + task_id。

    published_workflow_instance fixture：一个 source_type=workflow 的
    ServiceInstance + 一把 InstanceApiKey。若现有 suite 没有此 fixture，
    在 conftest 或本文件内组装（参考 test_instance_service.py 的建实例方式）。
    """
    instance, api_key = published_workflow_instance
    resp = await client.post(
        f"/v1/instances/{instance.id}/run",
        json={"inputs": {}},
        headers={"Authorization": f"Bearer {api_key.raw_key}"},
    )
    assert resp.status_code == 202
    assert "task_id" in resp.json()


@pytest.mark.asyncio
async def test_instance_run_enqueue_poll_result(client, published_workflow_instance):
    """端到端：instance /run enqueue → poll → completed。"""
    instance, api_key = published_workflow_instance
    resp = await client.post(
        f"/v1/instances/{instance.id}/run",
        json={"inputs": {}},
        headers={"Authorization": f"Bearer {api_key.raw_key}"},
    )
    task_id = resp.json()["task_id"]
    final = await _poll_until_done(client, task_id)
    assert final["status"] in ("completed", "failed")
    # 该 instance 的 workflow 是 inline-only（fixture 保证），应 completed
    assert final["status"] == "completed"
```
（`client` / `published_workflow_instance` fixture：`client` 对齐现有 conftest 的 async test client 名字。`published_workflow_instance` 若 suite 里没有，参照 `tests/test_instance_service.py` 现有的「建 ServiceInstance + InstanceApiKey」代码组一个 fixture，instance 的 `params_override` 放一个 inline-only workflow（`text_input` 节点），`raw_key` 是建 key 时拿到的明文。)

- [ ] **Step 3: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_run_async_contract.py -q`
Expected: FAIL —— 当前 `/run` 返回 200 + result body，不是 202 + task_id；且同步阻塞执行。

- [ ] **Step 4: 改 `execute_workflow_direct`（`workflows.py`）**

`backend/src/api/routes/workflows.py` 的 `execute_workflow_direct`（:142-225）整个函数体替换为：
```python
@router.post("/execute", status_code=202)
async def execute_workflow_direct(
    body: dict,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """入队一个 workflow 直接执行（D17 纯异步）：建 task → 后台跑 → 立即返回 202 + task_id。

    客户端拿结果：轮询 GET /api/v1/tasks/{task_id} 或订阅 WS /ws/workflow/{task_id}。
    """
    import asyncio

    from src.models.execution_task import ExecutionTask
    from src.services.workflow_runner import run_workflow_task

    nodes = body.get("nodes", [])
    edges = body.get("edges", [])
    if not nodes:
        raise HTTPException(400, "Workflow is empty")

    task = ExecutionTask(
        workflow_name=body.get("name", "直接执行"),
        status="queued",
        nodes_total=len(nodes),
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    # channel_id：D17 之后客户端订阅 WS 用 task_id 作为 channel；兼容旧的
    # 显式 channel_id（前端在 POST 前先开 WS 的老流程）。
    channel_id = body.get("channel_id") or str(task.id)
    runner_client = getattr(request.app.state, "runner_client", None)

    asyncio.create_task(run_workflow_task(
        task.id, {"nodes": nodes, "edges": edges},
        runner_client=runner_client, channel_id=channel_id,
    ))
    return {"task_id": str(task.id), "status": "queued", "channel_id": channel_id}
```
（删掉原函数里 `import time` / `start` / `broadcast_progress` / 同步 `await executor.execute()` 那一整套——这些逻辑已搬进 `workflow_runner.run_workflow_task`。`request: Request` 参数要加进签名。原来那行 `logger.warning("DIAG ...")` 诊断日志一并删除。)

- [ ] **Step 5: 改 `instance_run`（`instance_service.py`）**

`backend/src/api/routes/instance_service.py` 的 `instance_run`（:93-161）函数体替换为：
```python
@router.post("/{instance_id}/run", status_code=202)
async def instance_run(
    req: InstanceRunRequest,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_instance_key),
    session: AsyncSession = Depends(get_async_session),
):
    """入队一个已发布 workflow 实例的执行（D17 纯异步）。

    返回 202 + task_id。客户端轮询 GET /api/v1/tasks/{task_id} 或订阅
    WS /ws/workflow/{task_id} 拿结果。迁移指引见 docs 的 /run 契约变更节。
    """
    import asyncio

    from src.models.execution_task import ExecutionTask
    from src.services.workflow_runner import run_workflow_task

    instance, api_key = auth

    if instance.source_type != "workflow":
        raise HTTPException(400, detail="Only workflow-based instances support /run")

    workflow_data = instance.params_override or {}
    nodes = workflow_data.get("nodes", [])
    if not nodes:
        raise HTTPException(400, detail="Workflow has no nodes")

    task = ExecutionTask(
        workflow_id=instance.source_id,
        workflow_name=instance.name or "API 执行",
        status="queued",
        nodes_total=len(nodes),
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    # 用量计数：入队即计一次调用（异步契约下无法等执行完）
    api_key.usage_calls += 1
    api_key.last_used_at = datetime.now(timezone.utc)
    await session.commit()

    # WS channel 仍用 instance.id（上游订阅 /ws/workflow/{instance_id} 的老约定不变），
    # 但同时把 task_id 回给客户端用于轮询。
    channel_id = str(instance.id)
    from src.api.main import app_state_runner_client  # 见下方说明
    runner_client = app_state_runner_client()

    asyncio.create_task(run_workflow_task(
        task.id, workflow_data,
        runner_client=runner_client, channel_id=channel_id,
    ))
    return {"task_id": str(task.id), "status": "queued"}
```
**注意**：`instance_run` 没有 `request: Request` 参数（它的签名是 `verify_instance_key` 注入）。拿 `runner_client` 有两条路，二选一：
   - (推荐) 给 `instance_run` 签名加 `request: Request`，然后 `getattr(request.app.state, "runner_client", None)`——与 `execute_workflow_direct` 一致。把上面 `from src.api.main import app_state_runner_client` 那两行删掉，改成 `request.app.state`。
   - 若不想动签名：在 Lane C 落地前 `runner_client` 恒为 `None`（inline-only instance 不需要它），直接传 `runner_client=None`，留 `# TODO(Lane C): wire RunnerClient` 注释。

   选第一条（加 `request: Request`）—— 长远正确，Lane C 落地后零改动。

- [ ] **Step 6: 跑 Lane S 端到端测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_run_async_contract.py -q`
Expected: 全 PASS。enqueue → poll → result 链路通。

- [ ] **Step 7: 修现有 `/run` 测试的同步断言**

Step 1 基线里那些断言「`/run` 响应 body == result」的用例，现在响应是 `202 {task_id}`。逐个改成「assert 202 → 取 task_id → `_poll_until_done` → 断言 result」。
Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "instance or workflow_direct or run"`
Expected: 全 PASS（改完断言后）。若某用例是 mediahub 风格的外部契约快照测试且无法在测试内 poll，标记 `@pytest.mark.xfail(reason="D17 /run async contract — upstream must migrate")` 并在 commit message 点名。

- [ ] **Step 8: 全 suite 回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS（或仅有 Step 7 标的 xfail）。无 collection error、无 `ModuleNotFoundError`。

- [ ] **Step 9: lint 预检**

Run: `cd backend && ruff check src/services/node_routing.py src/services/workflow_runner.py src/services/workflow_executor.py src/api/routes/instance_service.py src/api/routes/workflows.py tests/test_node_routing.py tests/test_workflow_executor_split.py tests/test_workflow_runner.py tests/test_run_async_contract.py tests/fixtures/fake_runner_client.py`
Expected: 无 lint 错误。

- [ ] **Step 10: Commit**

```bash
cd backend && git add src/api/routes/instance_service.py src/api/routes/workflows.py tests/test_run_async_contract.py
git commit -m "feat(workflow): /run endpoints become pure-async — 202 + task_id (D17)

instance_run + execute_workflow_direct no longer block on execute().
They enqueue an ExecutionTask, fire run_workflow_task in the background,
and return 202 immediately. Clients poll /api/v1/tasks/{id} or subscribe
to WS /ws/workflow/{id}. Regression test covers enqueue -> poll -> result
end-to-end. spec §2.1 step 5 / D17. V1.5 Lane S."
```

---

## Task 6: `/run` 契约变更文档 + 上游迁移指引

spec 的「`/run` 契约变更 + 上游迁移指引（D17）」节要求给上游（mediahub）写迁移指引。在 spec 文档里那一节已有骨架表格，本 Task 把它落成可执行的迁移步骤，写进一个独立 doc。

**Files:**
- New: `backend/docs/run-async-migration.md`（若 `backend/docs/` 不存在，建在 `docs/superpowers/` 同级或仓库 `docs/`——先 `ls` 确认）

- [ ] **Step 1: 确认 docs 落点**

Run: `ls backend/docs 2>/dev/null; ls docs`
Expected: 据输出决定——若 `backend/docs/` 存在放那；否则放 `docs/run-async-migration.md`。

- [ ] **Step 2: 写迁移指引**

写入 `docs/run-async-migration.md`（或上一步确认的路径）：
```markdown
# /run 异步契约迁移指引（V1.5 D17）

## 变更摘要

V1.5 起，workflow 执行端点从「同步阻塞到完成」改为「纯异步入队」：

| 端点 | V1（旧） | V1.5（新） |
|---|---|---|
| `POST /v1/instances/{id}/run` | 同步执行，阻塞到 workflow 跑完，body 直接是 result | 入队 → 立即返回 `202 {"task_id": "..."}` |
| `POST /api/v1/workflows/execute` | 同上 | 同上 |

**单次 LLM 调用不受影响** —— OpenAI/Anthropic/Ollama/Responses compat 路由本来就直连
vLLM HTTP、本来就同步，行为不变。D17 只改多节点 workflow 端点。

## 上游（mediahub 等）迁移步骤

旧代码（同步等结果）：
```python
resp = httpx.post(f"{base}/v1/instances/{iid}/run",
                  json={"inputs": ...}, headers=auth)
result = resp.json()          # V1：body 直接是 result
```

新代码（拿 task_id → 轮询）：
```python
resp = httpx.post(f"{base}/v1/instances/{iid}/run",
                  json={"inputs": ...}, headers=auth)
assert resp.status_code == 202
task_id = resp.json()["task_id"]

# 方式 A：轮询
while True:
    t = httpx.get(f"{base}/api/v1/tasks/{task_id}").json()
    if t["status"] in ("completed", "failed", "cancelled"):
        break
    time.sleep(0.5)
result = t["result"]          # status=completed 时

# 方式 B：订阅 WS（实时进度）
#   ws connect {base}/ws/workflow/{instance_id}
#   收 node_start / node_complete / complete 事件
```

## 迁移期兼容（可选）

如上游一时改不动，可在服务端加一个 `?wait=true` 兼容 flag：服务端代为轮询
task 直到终态再返回 result。该 flag 标记 **deprecated**，仅为过渡——长期
所有上游都应走 task_id + 轮询/WS。

（注：`?wait=true` 兼容 flag 本 Lane 不实现；若 mediahub 迁移确实需要缓冲期，
另开 PR 加，并从落地起就标 deprecated。）
```

- [ ] **Step 3: Commit**

```bash
git add docs/run-async-migration.md
git commit -m "docs: /run async contract migration guide for upstream consumers

D17 changes workflow /run from sync-blocking to 202 + task_id. Guide
covers the poll and WS-subscribe patterns mediahub needs to adopt.
V1.5 Lane S."
```

---

## Task 7: 整合验证 + 开 PR

**Files:** 无（验证）

- [ ] **Step 1: 全 suite green**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS（或仅 Task 5 Step 7 标记的 xfail）。无 collection error。

- [ ] **Step 2: 确认分流真相源唯一**

Run: `cd backend && grep -rn "_execute_node\b" src/ --include="*.py"`
Expected: 零输出 —— 旧的 `_execute_node` 名字已全部改为 `_execute_inline_node`，`execute()` 只经 `_run_node_routed` 分流。

- [ ] **Step 3: 确认两个 `/run` 端点都返回 202**

Run: `cd backend && grep -rn "status_code=202\|def execute_workflow_direct\|def instance_run" src/api/routes/`
Expected: `execute_workflow_direct` 与 `instance_run` 都带 `status_code=202`。

- [ ] **Step 4: 后端冒烟 —— enqueue + poll**

启动后端（依项目方式），然后：
```bash
curl -s -X POST localhost:8000/api/v1/workflows/execute \
  -H 'Content-Type: application/json' \
  -d '{"name":"smoke","nodes":[{"id":"t1","type":"text_input","data":{"text":"hi"}}],"edges":[]}'
```
Expected: HTTP 202，body `{"task_id":"...","status":"queued","channel_id":"..."}`。
然后 `curl -s localhost:8000/api/v1/tasks/<task_id>` → 几百 ms 内 `status` 变 `completed`，`result` 非空。无 500、无 traceback。

- [ ] **Step 5: 开 PR**

```bash
git push -u origin <lane-s-branch>
gh pr create --title "feat: V1.5 Lane S — workflow_executor rewrite + /run async contract" --body "$(cat <<'EOF'
## Summary
- `workflow_executor.execute()` 重写为节点分流：GPU 节点（image/tts）dispatch 到 RunnerClient，inline 节点（llm/逻辑节点）主进程内执行
- 新增 `node_routing.py` —— dispatch vs inline 判定的唯一真相源（显式白名单）
- 新增 `workflow_runner.py` —— 后台执行入口，两个 /run 端点共用
- `/v1/instances/{id}/run` + `/api/v1/workflows/execute` 改纯异步：202 + task_id（D17）
- 上游迁移指引文档（mediahub 等）

## 与 spec 的偏差
- spec 写的 `/v1/workflows/{id}/run` 路径本仓库不存在；实际是 `instance_run` + `execute_workflow_direct` 两个端点，本 PR 对两者都做 D17 改造
- `llm` 节点本来就经 adapter HTTP 调 vLLM，无需改执行方式，只归类为 inline
- RunnerClient 来自 Lane C（plan 未编写）；本 PR 按 spec §3.3 接口编程 + 注入点预留，测试用纯 Python stub

## Test plan
- [ ] 全 suite green（pytest tests/）
- [ ] node_routing 分流判定单测
- [ ] executor 节点分流单测（inline / dispatch / 混合 / runner 缺失报错）
- [ ] [回归] /run 异步契约端到端：enqueue → poll → result
- [ ] 冒烟：POST /workflows/execute → 202 → poll /tasks/{id} → completed
EOF
)"
```
（分支名按项目惯例，如 `feat/v15-laneS-workflow-executor-rewrite`。）

---

## Self-Review

**Spec 覆盖检查：** Lane S 在 spec「实施分 Lane」表里的职责是「`execute()` 拆 dispatch vs inline-HTTP 节点（D10）；`execute_workflow_direct` 的 inline 执行收编；`/run` 改纯异步 202 + task_id（D17）；上游迁移指引」。

- `execute()` 拆 dispatch vs inline → Task 1（判定）+ Task 3（执行）
- `execute_workflow_direct` 收编 → Task 5（改成入队 + 后台 run_workflow_task）
- `/run` 纯异步 202 → Task 5（两个端点都改）+ Task 4（后台入口）
- 上游迁移指引 → Task 6

**与 spec 的偏差（3 处，已在文件头「与 spec 的偏差」节详述，须知会 reviewer）：**
1. **端点路径**：spec 反复写 `/v1/workflows/{id}/run`，本仓库无此路径。实际是 `POST /v1/instances/{instance_id}/run`（上游 mediahub 用）+ `POST /api/v1/workflows/execute`（前端 Run 用）。本 Lane 对两者都做 D17 改造。
2. **llm 节点「改道」**：spec §4.5 说 llm 节点要「executor inline HTTP 调 vLLM」，暗示当前是「主进程内调 adapter」。实际现有 `LLMNode` 已通过 `InferenceAdapter` 走 HTTP 调 vLLM——本来就是 inline HTTP。Lane S 对 llm 节点只做「归类为 inline」，不改执行方式。真正改的是 image/tts 节点（主进程内调 adapter → dispatch 到 runner）。
3. **outputs 落盘**：spec §2.1 step 12a 说 image 结果写 `outputs/{task_id}/`，本仓库无此机制（Lane D + Lane I 的事）。Lane S 的 `_dispatch_node` 对 `RunnerClient.run_node` 返回的 `outputs` dict 原样透传，不关心里面是路径还是内联数据。

**依赖说明：** Lane S 依赖 Lane C（RunnerClient）。Lane C plan 尚未编写。本 Lane 按 spec §3.3 定义的 `run_node(node, inputs) -> outputs` 接口编程，`WorkflowExecutor.__init__` 注入 `runner_client`（默认 `None`），测试用纯 Python `FakeRunnerClient` stub。Lane C 落地后 main.py 注入真 `RunnerClient` 即可，executor 代码零改动。**风险**：Lane C 实际接口若与 spec §3.3 偏离（例如 `run_node` 签名带 task_id / cancel_token），需回来调 `_dispatch_node` 一行 + stub。已把接口面压到最小（单方法、透传 dict）以降低此风险。

**回归风险（已守住）：** `/run` 从同步阻塞改纯异步是外部契约变更。Task 5 的 `test_run_async_contract.py` 用 enqueue → poll `/tasks/{id}` → 断言 result 的端到端测试守住链路；Task 5 Step 7 显式要求修现有同步断言用例，无法在测试内 poll 的外部契约快照测试标 `xfail` 并在 commit 点名。Task 3 Step 6 要求重写 `execute()` 后现有 inline-only workflow 测试全绿（行为不变保证）。

**未决接缝（实现时现场核对，已在对应 Task 标注）：**
- `src/models/database.py` 的 async session factory 实际名字（Task 4 Step 3 注明先 grep 确认）
- conftest 里 async test client / db session fixture 的实际名字（Task 4、5 注明对齐）
- `published_workflow_instance` fixture 若 suite 没有需自建（Task 5 Step 2 给了组装指引）
- `instance_run` 拿 `runner_client` 是否加 `request: Request` 参数（Task 5 Step 5 给了二选一，推荐加参数）

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。所有代码完整给出。唯一的 `# TODO(Lane C)` 出现在 Task 5 Step 5 的「不推荐分支」里，且明确推荐走另一条无 TODO 的路径。

**类型一致性：** `node_exec_class` 返回 `Literal["inline","dispatch"]`；`WorkflowExecutor.__init__(workflow, on_progress, runner_client)` 三参数；`run_workflow_task(task_id, workflow_data, runner_client, channel_id)` 与两个端点的调用处参数顺序一致；`RunnerClient.run_node(node, inputs)` / `FakeRunnerClient.run_node(node, inputs)` / `_dispatch_node` 调用三处签名一致。
