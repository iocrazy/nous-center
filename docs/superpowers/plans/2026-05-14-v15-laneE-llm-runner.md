# V1.5 Lane E: LLM Runner + compat 路由 / executor 直连 vLLM HTTP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 vLLM 子进程的生命周期管理从「散落在 `VLLMAdapter` 里、由 `model_manager` 顺带管」收编成一个显式的 **LLMRunner**（只管 vLLM 子进程的 spawn / health / preload / abort / OOM-restart，**不**串行化推理请求 —— 与 image/TTS runner 本质不同，见 spec §1.2 / D6 / D8）；同时把 4 个 compat 路由（openai/anthropic/ollama/responses）+ workflow executor 的 llm 节点对 vLLM base-URL 的查找，从「各处 `getattr(adapter, "base_url")`」收敛成一个新建的 `get_vllm_base_url()` 真相源（Lane 0 删掉了旧 `get_llm_base_url`，本 Lane 全新建）。

**Architecture:** 三块交付物，自底向上：

1. **`get_vllm_base_url()` 真相源（`src/services/inference/vllm_endpoint.py`，新建）** —— 当前 4 个 compat 路由 + responses 各自重复 `model_mgr.get_adapter(engine_name)` → `getattr(adapter, "base_url", None)` → 空值检查 → 抛 HTTP 错。Lane E 把这段收敛成一个函数：传入 `model_mgr` + `engine_name`，返回 base_url 或抛 typed 异常（`VLLMNotLoaded` / `VLLMNoEndpoint`）。Lane 0 删的旧 `get_llm_base_url`（`model_scheduler.py:233`，零调用方死代码）不复活 —— 这是**全新**的、面向「`model_manager` 持有的 `VLLMAdapter`」的查找。

2. **`LLMRunner`（`src/runner/llm_runner.py`，新建）** —— 不是 image/TTS 那种 `multiprocessing.Process` 子进程（那是 Lane C 的 `RunnerSupervisor` + `runner_main`），LLMRunner 是**主进程内**的一个生命周期管理对象，它管的「子进程」是 vLLM 本身（`VLLMAdapter._process` / `_adopted_pid` 已经在跑的那个 `subprocess.Popen`）。职责：`spawn()`（= 触发 `VLLMAdapter.load()`）、`health()`（= `_health_check()` 轮询）、`preload()`（resident LLM 启动加载，fail-soft）、`abort(request_id)`（向 vLLM HTTP 发 abort）、`restart_on_oom()`（vLLM 子进程 crash / OOM 退出 → kill orphan → re-spawn + re-preload，过 F2 GPU-free gate）。**关键不变量**：LLMRunner 不持有任何 PriorityQueue、不串行化推理 —— 推理并发由 vLLM 自身 continuous batching 处理（spec §1.3）。

3. **改道清单落地（spec §4.5 D6/D8）** —— 4 个 compat 路由 + executor 的 llm 节点改用 `get_vllm_base_url()`。**注意**：本仓库的 compat 路由**本来就**走 HTTP→vLLM（`getattr(adapter, "base_url")` + `httpx`），executor 的 `LLMNode` 也本来就经 `InferenceAdapter` 走 HTTP（Lane S 审计已确认）。所以 §4.5 的「改道」对本仓库而言是**收敛 + 重新归类**，不是「从 inline-adapter 改成 HTTP」的重写 —— 真正变的是 base-URL 查找的来源，以及把这条路径正式标注为「直连 vLLM HTTP」。

**Tech Stack:** Python 3.12 / FastAPI / asyncio / httpx（已有依赖）/ pytest（`asyncio_mode = "auto"`，conftest 强制 `ADMIN_PASSWORD=""` + `NOUS_DISABLE_FRONTEND_MOUNT=1` + `NOUS_DISABLE_BG_TASKS=1` + `CUDA_VISIBLE_DEVICES=""`）。无新第三方依赖。`src/runner/` 目录由 Lane C 新建，本 Lane 在其下加 `llm_runner.py`。

> **与 spec 的偏差 / 歧义（已核实，须知会 reviewer）：**
>
> 1. **spec §4.5 把「compat 路由 → 直连 vLLM HTTP」「executor llm 节点 → inline HTTP 调 vLLM」列为「V1.5 改道」，暗示 V1 是「主进程内调 adapter」。** 本仓库实际：4 个 compat 路由已经是 `get_adapter → adapter.base_url → httpx.post` 的 HTTP 直连；`LLMNode`（`src/services/nodes/llm.py`）经 `InferenceAdapter` 也是 HTTP（Lane S 审计已确认）。所以本 Lane 对 compat 路由 / executor 的改动是「**把 base-URL 查找收敛到 `get_vllm_base_url()`**」+「确认并标注这条路径为直连 HTTP」，不是执行方式重写。spec 的「改道」一词对本仓库 = reclassification + consolidation。已在 Self-Review 标注。
>
> 2. **spec §1.2 / §4.2 把 LLM Runner 画在「子进程」一栏（与 image/TTS runner 并列在 `multiprocessing.Pipe` 边界下方）。** 但 §1.2 正文又明确「LLM Runner 不收 RunNode，不走此协议；主进程直连其 vLLM HTTP 端口」。本 Lane 的判断：LLMRunner **不是** `multiprocessing.Process` 子进程 —— 它是主进程内的对象，管的是 vLLM 这个**已有的** `subprocess.Popen`（`VLLMAdapter` 启的）。理由：(a) LLM Runner 不串行化请求、不需要独立 event loop 跑 pipe-reader/executor 双 task；(b) vLLM 本身已经是子进程，再套一层 Python 子进程纯属多余的故障面。spec 的拓扑图把它画进子进程栏是「概念对称」，实现上是「主进程对象 + 它管的 vLLM 子进程」两级里的上层。已在 Self-Review 标注，并与 spec §4.5 隔离边界图「LLM Runner 只管 vLLM 子进程生命周期」「vLLM 自身又是 subprocess」一致 —— spec §4.5 文字其实支持本判断。
>
> 3. **spec 没有给 `LLMRunner` 的接口草图（§4.2 的 `RunnerSupervisor` 草图是 image/TTS 的）。** 本 Lane 按 spec §1.2 / §4.1 故障矩阵「LLM Runner crash → 重启 runner → re-spawn vLLM + preload」/ §4.2「LLM: runner spawn vLLM → 等 health 通过」定义 `LLMRunner` 接口，命名对齐 spec 用词（spawn / health / preload / abort / OOM-restart）。
>
> 4. **「OOM-restart」在 LLM 语境的含义。** spec §4.1 把「vLLM 子进程启动失败」「LLM Runner crash」分两行。本 Lane 统一处理：`LLMRunner.health()` 探测到 vLLM 子进程已退出（`_process.poll() is not None` 或 health check 连续失败）→ 视为 crash/OOM → `restart()`：kill 残留 orphan（复用 `vllm_scanner` + `VLLMAdapter._kill_process`）→ 过 F2 GPU-free gate → re-spawn → re-preload。不区分「正常 OOM 退出」与「其它 crash」—— 都走同一条 restart 路径（spec §4.1 对二者的恢复策略都是「重启 runner → re-spawn vLLM + preload」，等价）。
>
> 5. **F2 GPU-free gate 的真探针。** Lane C 的 `RunnerSupervisor` 用注入式 `gpu_free_probe`（默认实现保守返回 `True`）。本 Lane 的 `LLMRunner` 同样接受注入式 `gpu_free_probe`，默认实现给一个**真的** nvidia-smi 探针骨架（查 `role: llm` group 的 GPU 显存是否回落到基线），但在无 GPU 环境（CI / `CUDA_VISIBLE_DEVICES=""`）保守返回 `True` 不阻塞。测试全程注入 fake 探针。

---

## File Structure

| 文件 | Lane E 动作 | 责任 |
|---|---|---|
| `backend/src/services/inference/vllm_endpoint.py` | **新建** | `get_vllm_base_url(model_mgr, engine_name) -> str` + `VLLMNotLoaded` / `VLLMNoEndpoint` 异常 —— vLLM base-URL 查找的唯一真相源 |
| `backend/src/runner/llm_runner.py` | **新建** | `LLMRunner`：主进程内管 vLLM 子进程生命周期（spawn / health / preload / abort / restart），不串行化请求 |
| `backend/src/api/routes/openai_compat.py` | **修改** | `getattr(adapter, "base_url")` 段 → `get_vllm_base_url()` |
| `backend/src/api/routes/anthropic_compat.py` | **修改** | 同上 |
| `backend/src/api/routes/ollama_compat.py` | **修改** | `_get_adapter` 里的 base_url 查找 → `get_vllm_base_url()` |
| `backend/src/api/routes/responses.py` | **修改** | 同上（两处 `base_url` 用法，查找点收敛为一处） |
| `backend/src/services/nodes/llm.py` | **核对（大概率不改）** | Lane S 审计：`LLMNode` 已经经 `InferenceAdapter` HTTP 调 vLLM。本 Lane 核对它是否也该收敛到 `get_vllm_base_url()`；若 `LLMNode` 自己不直接拿 base_url（而是 adapter 内部持有），则**不动**，仅在 Self-Review 记录 |
| `backend/tests/test_vllm_endpoint.py` | **新建** | `get_vllm_base_url`：正常返回、未加载抛 `VLLMNotLoaded`、无 endpoint 抛 `VLLMNoEndpoint` |
| `backend/tests/test_llm_runner.py` | **新建** | `LLMRunner`：spawn 调 adapter.load、health 探测、preload fail-soft、abort、crash → restart + GPU-free gate、**不串行化**（并发 infer 同时在飞） |
| `backend/tests/test_compat_routes_vllm_regression.py` | **新建** | **[回归]** 4 个 compat 路由收敛 base-URL 查找后产出不变：mock vLLM HTTP 端点，断言 openai/anthropic/ollama/responses 仍返回正确结构 |

---

## Task 1: `get_vllm_base_url()` —— vLLM base-URL 查找真相源

当前 4 个 compat 路由 + `responses.py` 各自重复同一段：`model_mgr = getattr(request.app.state, "model_manager", None)` → `adapter = model_mgr.get_adapter(engine_name)` → `if adapter is None or not adapter.is_loaded: raise 503` → `base_url = getattr(adapter, "base_url", None)` → `if not base_url: raise 500`。Lane 0 删掉的旧 `get_llm_base_url`（`model_scheduler.py:233`）是零调用方死代码，**不复活**。本 Task 新建一个面向「`model_manager` 持有的 `VLLMAdapter`」的查找函数。

**Files:**
- New: `backend/src/services/inference/vllm_endpoint.py`
- Test: `backend/tests/test_vllm_endpoint.py`（新建）

- [ ] **Step 1: 写失败测试 —— base-URL 查找的三条路径**

新建 `backend/tests/test_vllm_endpoint.py`：
```python
"""Lane E: vLLM base-URL 查找真相源测试（纯内存，无子进程、无 GPU）。"""
import pytest

from src.services.inference.vllm_endpoint import (
    VLLMNoEndpoint,
    VLLMNotLoaded,
    get_vllm_base_url,
)


class _FakeAdapter:
    def __init__(self, is_loaded: bool, base_url: str | None):
        self.is_loaded = is_loaded
        self.base_url = base_url


class _FakeModelManager:
    def __init__(self, adapters: dict):
        self._adapters = adapters

    def get_adapter(self, engine_name: str):
        return self._adapters.get(engine_name)


def test_returns_base_url_when_loaded():
    """adapter 已加载且有 base_url → 返回 base_url。"""
    mgr = _FakeModelManager({"qwen": _FakeAdapter(True, "http://localhost:8123")})
    assert get_vllm_base_url(mgr, "qwen") == "http://localhost:8123"


def test_raises_not_loaded_when_adapter_missing():
    """engine_name 没有对应 adapter → VLLMNotLoaded。"""
    mgr = _FakeModelManager({})
    with pytest.raises(VLLMNotLoaded, match="qwen"):
        get_vllm_base_url(mgr, "qwen")


def test_raises_not_loaded_when_adapter_not_loaded():
    """adapter 存在但 is_loaded=False → VLLMNotLoaded。"""
    mgr = _FakeModelManager({"qwen": _FakeAdapter(False, "http://localhost:8123")})
    with pytest.raises(VLLMNotLoaded, match="qwen"):
        get_vllm_base_url(mgr, "qwen")


def test_raises_no_endpoint_when_base_url_empty():
    """adapter 已加载但 base_url 为空 → VLLMNoEndpoint。"""
    mgr = _FakeModelManager({"qwen": _FakeAdapter(True, None)})
    with pytest.raises(VLLMNoEndpoint, match="qwen"):
        get_vllm_base_url(mgr, "qwen")


def test_raises_not_loaded_when_model_manager_none():
    """model_manager 本身为 None（app.state 未初始化）→ VLLMNotLoaded。"""
    with pytest.raises(VLLMNotLoaded):
        get_vllm_base_url(None, "qwen")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_vllm_endpoint.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'src.services.inference.vllm_endpoint'`。

- [ ] **Step 3: 实现 `vllm_endpoint.py`**

新建 `backend/src/services/inference/vllm_endpoint.py`：
```python
"""vLLM base-URL 查找 —— compat 路由 / workflow executor 直连 vLLM HTTP 的唯一真相源。

spec §4.5 D6/D8「inline 执行点改道清单」：openai/anthropic/ollama/responses compat
路由 + workflow executor 的 llm 节点都直连 vLLM 的 HTTP 端口，零 per-token pipe 开销。

本仓库的 compat 路由本来就走 HTTP→vLLM（各处 `getattr(adapter, "base_url")`），
Lane E 把那段重复的「get_adapter → 检查 is_loaded → 取 base_url → 检查空值」收敛到这里。

注意：Lane 0 删掉的旧 `get_llm_base_url`（model_scheduler.py:233）是零调用方死代码，
本函数是面向 `model_manager` 持有的 VLLMAdapter 的全新查找，不是它的复活。
"""
from __future__ import annotations

from typing import Any


class VLLMNotLoaded(RuntimeError):
    """目标 LLM engine 未加载（adapter 缺失 / is_loaded=False / model_manager 不可用）。

    调用方（compat 路由）应映射为 HTTP 503。
    """


class VLLMNoEndpoint(RuntimeError):
    """LLM engine 已加载但没有 HTTP 推理端点（base_url 为空）。

    调用方应映射为 HTTP 500 —— 这是不该发生的状态（vLLM 加载成功必有端口）。
    """


def get_vllm_base_url(model_manager: Any, engine_name: str) -> str:
    """返回 *engine_name* 对应 vLLM 实例的 HTTP base_url。

    Parameters
    ----------
    model_manager:
        `app.state.model_manager`（services.model_manager.ModelManager）。
        允许为 None —— app.state 尚未初始化时直接抛 VLLMNotLoaded。
    engine_name:
        模型 / engine 标识（ServiceInstance.source_name 或 source_id）。

    Raises
    ------
    VLLMNotLoaded:  model_manager 不可用，或 engine 未加载。
    VLLMNoEndpoint: engine 已加载但 base_url 为空。
    """
    if model_manager is None:
        raise VLLMNotLoaded("model_manager 不可用（app.state 未初始化）")

    adapter = model_manager.get_adapter(engine_name)
    if adapter is None or not getattr(adapter, "is_loaded", False):
        raise VLLMNotLoaded(
            f"模型 '{engine_name}' 未加载 —— 请在模型管理页加载后重试"
        )

    base_url = getattr(adapter, "base_url", None)
    if not base_url:
        raise VLLMNoEndpoint(
            f"模型 '{engine_name}' 已加载但没有 HTTP 推理端点（base_url 为空）"
        )
    return base_url
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_vllm_endpoint.py -q`
Expected: 5 个 PASS。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/services/inference/vllm_endpoint.py tests/test_vllm_endpoint.py
git commit -m "feat(inference): get_vllm_base_url — single source of truth for vLLM endpoint

Consolidates the duplicated 'get_adapter -> check is_loaded -> base_url'
seam scattered across the 4 compat routes. Typed exceptions (VLLMNotLoaded
-> 503, VLLMNoEndpoint -> 500). Fresh lookup against model_manager-held
VLLMAdapter — NOT a revival of Lane 0's dead get_llm_base_url. spec 4.5
D6/D8. V1.5 Lane E."
```

---

## Task 2: 4 个 compat 路由收敛 base-URL 查找到 `get_vllm_base_url()`

`openai_compat.py` / `anthropic_compat.py` / `ollama_compat.py` / `responses.py` 各自有一段「`model_mgr = getattr(request.app.state, "model_manager", None)` → `adapter = model_mgr.get_adapter(engine_name)` → 检查 is_loaded → `base_url = getattr(adapter, "base_url", None)` → 检查空值」。本 Task 把这段替换为 `get_vllm_base_url()` 调用。**这是 Lane E 的回归风险点** —— 4 个路由的输出在收敛后必须不变（Task 4 的回归测试守住）。

**Files:**
- Modify: `backend/src/api/routes/openai_compat.py`
- Modify: `backend/src/api/routes/anthropic_compat.py`
- Modify: `backend/src/api/routes/ollama_compat.py`
- Modify: `backend/src/api/routes/responses.py`
- Test: 本 Task 不写独立测试；行为不变由 Task 4 的回归套 + 现有 compat 路由测试守住。

- [ ] **Step 1: 跑现有 compat 路由 suite 建基线**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "compat or openai or anthropic or ollama or responses"`
Expected: PASS。记下通过数 —— 收敛后这些用例必须仍全绿（行为不变保证）。

- [ ] **Step 2: 改 `openai_compat.py`**

`backend/src/api/routes/openai_compat.py` 约 :194-204，把：
```python
    model_mgr = getattr(request.app.state, "model_manager", None)
    if model_mgr is None:
        raise HTTPException(500, detail="Model manager not available")

    adapter = model_mgr.get_adapter(engine_name)
    if adapter is None or not adapter.is_loaded:
        raise HTTPException(503, detail=f"Model '{engine_name}' is not loaded. Load it from the management page.")

    base_url = getattr(adapter, "base_url", None)
    if not base_url:
        raise HTTPException(500, detail="Model has no inference endpoint")
```
改为：
```python
    # spec §4.5 D6/D8：直连 vLLM HTTP。base-URL 查找收敛到 get_vllm_base_url。
    from src.services.inference.vllm_endpoint import (
        VLLMNoEndpoint,
        VLLMNotLoaded,
        get_vllm_base_url,
    )

    model_mgr = getattr(request.app.state, "model_manager", None)
    try:
        base_url = get_vllm_base_url(model_mgr, engine_name)
    except VLLMNotLoaded as e:
        raise HTTPException(503, detail=str(e)) from e
    except VLLMNoEndpoint as e:
        raise HTTPException(500, detail=str(e)) from e

    # 下游仍需 adapter 取 max_model_len（clamp 逻辑，:283）。
    adapter = model_mgr.get_adapter(engine_name)
```
（注意：`openai_compat.py:283` 的 `max_model_len = getattr(adapter, "max_model_len", 4096) or 4096` 仍需 `adapter` 句柄 —— 上面末行保留 `adapter = model_mgr.get_adapter(engine_name)`。此时 `model_mgr` 必非 None（`get_vllm_base_url` 已校验），`get_adapter` 也必非 None（同上）。下游 `base_url` / `adapter` 两个变量名都不变，:283 与 :298 / :347 无需改。）

- [ ] **Step 3: 改 `anthropic_compat.py`**

`backend/src/api/routes/anthropic_compat.py` 约 :164-171，把对应的 `adapter = model_mgr.get_adapter(engine_name)` → 检查 → `base_url = getattr(adapter, "base_url", None)` → 检查 段，按 Step 2 同样的模式替换为 `get_vllm_base_url()` + try/except。若 `anthropic_compat.py` 下游不需要 `adapter` 句柄（只用 `base_url`），则**不**保留 `adapter = ...` 末行。
Run 先确认下游是否还用 adapter：
```bash
cd backend && grep -n "adapter" src/api/routes/anthropic_compat.py
```
Expected: 据输出决定是否保留 `adapter = model_mgr.get_adapter(engine_name)`。只有 `base_url` 被下游用 → 不保留；`adapter.xxx` 被下游用 → 保留。

- [ ] **Step 4: 改 `ollama_compat.py`**

`backend/src/api/routes/ollama_compat.py` 的 `_get_adapter`（:66-88）—— 这个 helper 返回 `(adapter, engine_name, base_url)` 三元组。把里面的 `adapter = model_mgr.get_adapter(engine_name)` → 检查 → `base_url = getattr(adapter, "base_url", None)` → 检查 段替换为：
```python
    from src.services.inference.vllm_endpoint import (
        VLLMNoEndpoint,
        VLLMNotLoaded,
        get_vllm_base_url,
    )

    model_mgr = getattr(request.app.state, "model_manager", None)
    try:
        base_url = get_vllm_base_url(model_mgr, engine_name)
    except VLLMNotLoaded as e:
        raise HTTPException(503, detail=str(e)) from e
    except VLLMNoEndpoint as e:
        raise HTTPException(500, detail=str(e)) from e
    adapter = model_mgr.get_adapter(engine_name)
    return adapter, engine_name, base_url
```
（`_get_adapter` 的三元组返回签名不变，:158 / :234 两个调用点无需改。`adapter` 仍取一次给三元组用。）

- [ ] **Step 5: 改 `responses.py`**

`backend/src/api/routes/responses.py` 约 :347-354，把 `adapter = model_mgr.get_adapter(engine_name)` → 检查 → `base_url = getattr(adapter, "base_url", None)` → 检查 段按 Step 2 模式替换。`responses.py` 在 :514 / :708 两处用 `base_url` 拼 URL，:283-风格的 `max_model_len` 若也用到则保留 `adapter` 句柄。
Run 先确认：
```bash
cd backend && grep -n "adapter\|max_model_len" src/api/routes/responses.py
```
Expected: 据输出决定是否保留 `adapter = model_mgr.get_adapter(engine_name)`。

- [ ] **Step 6: 跑现有 compat suite 确认零回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "compat or openai or anthropic or ollama or responses"`
Expected: PASS，通过数 = Step 1 基线（行为完全不变 —— 收敛只是把重复代码抽成函数，错误码 503/500 与原来一致）。若某用例断言「未加载返回特定 detail 文案」，文案变了（`get_vllm_base_url` 的中文 detail 与原英文 detail 不同）—— 改该用例断言为匹配新文案，或断言 `status_code` 而非文案。

- [ ] **Step 7: lint 预检**

Run: `cd backend && ruff check src/api/routes/openai_compat.py src/api/routes/anthropic_compat.py src/api/routes/ollama_compat.py src/api/routes/responses.py`
Expected: 无 lint 错误。

- [ ] **Step 8: Commit**

```bash
cd backend && git add src/api/routes/openai_compat.py src/api/routes/anthropic_compat.py src/api/routes/ollama_compat.py src/api/routes/responses.py
git commit -m "refactor(compat): route vLLM base-URL lookup through get_vllm_base_url

The 4 compat routes (openai/anthropic/ollama/responses) already proxied
to vLLM over HTTP — they each duplicated the get_adapter -> is_loaded ->
base_url seam. They now share get_vllm_base_url(). Output unchanged;
this is the spec 4.5 'reroute' which for this repo is consolidation +
reclassification, not an execution-path rewrite. V1.5 Lane E."
```

---

## Task 3: `LLMRunner` —— vLLM 子进程生命周期管理

spec §1.2：LLM Runner「只负责 vLLM 子进程的 spawn / health / preload / abort / OOM 重启，**不串行化推理请求**」。本仓库的 vLLM 子进程已经由 `VLLMAdapter`（`load()` 起 `subprocess.Popen`，`_process` / `_adopted_pid` / `_health_check()` / `_kill_process()`）管着 —— `LLMRunner` 把这些散落的能力收编成一个显式对象，并补上 spec §4.1/§4.2 要求的 crash 检测 + restart + F2 GPU-free gate。

**与 Lane C `RunnerSupervisor` 的区别（须知会）**：`RunnerSupervisor` fork 一个 `multiprocessing.Process` 跑 `runner_main`（image/TTS runner 子进程，内有 pipe-reader/executor 双 task + PriorityQueue）。`LLMRunner` **不 fork Python 子进程** —— 它是主进程内的对象，管的「子进程」是 vLLM（`VLLMAdapter` 启的 `subprocess.Popen`）。LLMRunner 无 PriorityQueue、无 IPC pipe、不串行化 —— 推理请求由 compat 路由 / executor 直连 vLLM HTTP（Task 2 / Task 5），并发由 vLLM continuous batching 处理。

**Files:**
- New: `backend/src/runner/llm_runner.py`
- Test: `backend/tests/test_llm_runner.py`（新建）

- [ ] **Step 0: 确认 `src/runner/` 目录存在（Lane C 前置）**

Run: `cd backend && ls src/runner/__init__.py`
Expected: 文件存在（Lane C 已建）。若不存在 —— Lane E 依赖 Lane C，停下来确认 Lane C 已 merge。

- [ ] **Step 1: 写失败测试 —— LLMRunner 生命周期 + 不串行化**

新建 `backend/tests/test_llm_runner.py`：
```python
"""Lane E: LLMRunner 测试 —— vLLM 子进程生命周期，不串行化推理。

全程用 FakeVLLMAdapter（不起真 vLLM、不碰 GPU），注入式 gpu_free_probe。
"""
import asyncio

import pytest

from src.runner.llm_runner import LLMRunner, LLMRunnerState


class FakeVLLMAdapter:
    """模拟 VLLMAdapter 的生命周期接口子集，零子进程零 GPU。"""

    def __init__(self, *, fail_load: bool = False, base_url: str = "http://localhost:8123"):
        self._fail_load = fail_load
        self.base_url = base_url
        self.is_loaded = False
        self.load_calls = 0
        self.unload_calls = 0
        self._alive = False  # 模拟 vLLM 子进程存活

    async def load(self, device=None):
        self.load_calls += 1
        await asyncio.sleep(0)  # 可让出
        if self._fail_load:
            raise RuntimeError("fake vLLM failed to start: OOM")
        self.is_loaded = True
        self._alive = True

    def unload(self):
        self.unload_calls += 1
        self.is_loaded = False
        self._alive = False

    async def _health_check(self) -> bool:
        return self._alive

    def simulate_crash(self):
        """模拟 vLLM 子进程 OOM / crash 退出。"""
        self._alive = False
        self.is_loaded = False


@pytest.mark.asyncio
async def test_spawn_loads_vllm_subprocess():
    """spawn() → 调 adapter.load() → state=running。"""
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.spawn()
    assert adapter.load_calls == 1
    assert adapter.is_loaded
    assert runner.state == LLMRunnerState.RUNNING


@pytest.mark.asyncio
async def test_health_returns_true_when_vllm_alive():
    """health() 反映 vLLM 子进程存活状态。"""
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.spawn()
    assert await runner.health() is True
    adapter.simulate_crash()
    assert await runner.health() is False


@pytest.mark.asyncio
async def test_preload_failsoft_records_failure():
    """preload 时 vLLM 启动失败 → fail-soft：不抛，state=failed，failure 可读。"""
    adapter = FakeVLLMAdapter(fail_load=True)
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.preload()  # 不抛
    assert runner.state == LLMRunnerState.FAILED
    assert "OOM" in (runner.failure or "")


@pytest.mark.asyncio
async def test_preload_success():
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.preload()
    assert runner.state == LLMRunnerState.RUNNING
    assert runner.failure is None


@pytest.mark.asyncio
async def test_restart_on_crash_respawns_and_passes_gpu_free_gate():
    """vLLM crash → restart()：kill orphan → 过 GPU-free gate → re-spawn。"""
    adapter = FakeVLLMAdapter()
    gate_calls: list[list[int]] = []

    def fake_gpu_free_probe(gpus):
        gate_calls.append(list(gpus))
        return True  # 显存已回落

    runner = LLMRunner(
        model_key="qwen", adapter=adapter, llm_gpus=[0, 1],
        gpu_free_probe=fake_gpu_free_probe, gpu_free_poll_interval=0.01,
    )
    await runner.spawn()
    adapter.simulate_crash()
    await runner.restart()
    assert adapter.unload_calls >= 1          # kill orphan
    assert gate_calls == [[0, 1]]             # GPU-free gate 查的是 llm group GPU
    assert adapter.load_calls == 2            # re-spawn
    assert runner.state == LLMRunnerState.RUNNING


@pytest.mark.asyncio
async def test_restart_blocked_until_gpu_free_gate_passes():
    """GPU-free gate 探针先返回 False（CUDA context 未回收）→ restart 卡住，
    回落后才 re-spawn（F2）。"""
    adapter = FakeVLLMAdapter()
    probe_results = iter([False, False, True])

    def fake_probe(gpus):
        return next(probe_results, True)

    runner = LLMRunner(
        model_key="qwen", adapter=adapter, llm_gpus=[0, 1],
        gpu_free_probe=fake_probe, gpu_free_poll_interval=0.01,
    )
    await runner.spawn()
    adapter.simulate_crash()
    await runner.restart()
    # 探针前两次 False → 等待；第三次 True → 继续 re-spawn
    assert adapter.load_calls == 2
    assert runner.state == LLMRunnerState.RUNNING


@pytest.mark.asyncio
async def test_abort_does_not_serialize_requests():
    """关键不变量：LLMRunner 不持有队列、不串行化。abort 是对 vLLM 的 HTTP 信号，
    不阻塞其它推理。本测试断言 LLMRunner 没有 PriorityQueue / inflight 串行化结构。"""
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.spawn()
    # LLMRunner 不该有任何串行化推理的队列结构
    assert not hasattr(runner, "queue")
    assert not hasattr(runner, "_priority_queue")
    # abort 接口存在且可调（fake 下是 no-op，真实现发 vLLM HTTP abort）
    await runner.abort(request_id="req-123")  # 不抛


@pytest.mark.asyncio
async def test_concurrent_health_checks_do_not_block():
    """并发调 health() 不互相阻塞 —— LLMRunner 无串行化锁卡住推理路径。"""
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.spawn()
    results = await asyncio.gather(*[runner.health() for _ in range(10)])
    assert all(results)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_llm_runner.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'src.runner.llm_runner'`。

- [ ] **Step 3: 实现 `llm_runner.py`**

新建 `backend/src/runner/llm_runner.py`：
```python
"""LLMRunner —— vLLM 子进程生命周期管理（spec §1.2 / §4.1 / §4.2）。

与 image/TTS 的 RunnerSupervisor（Lane C）本质不同：
  * RunnerSupervisor fork 一个 multiprocessing.Process 跑 runner_main（内有
    pipe-reader/executor 双 task + per-group PriorityQueue），串行化 GPU job。
  * LLMRunner 不 fork Python 子进程 —— 它是主进程内的对象，管的「子进程」是
    vLLM 本身（VLLMAdapter 启的 subprocess.Popen）。LLMRunner 无 PriorityQueue、
    无 IPC pipe、不串行化推理 —— 推理请求由 compat 路由 / executor 直连
    vLLM HTTP（spec §4.5 D6/D8），并发由 vLLM continuous batching 处理（spec §1.3）。

职责（spec 用词）：
  * spawn()    —— 触发 VLLMAdapter.load()，起 vLLM 子进程 + 等 health。
  * health()   —— 探测 vLLM 子进程是否存活（VLLMAdapter._health_check）。
  * preload()  —— resident LLM 启动加载，fail-soft（失败不抛，记 failure，
                  对齐 spec §4.2「load_failed 不阻断 API server start」）。
  * abort()    —— 向 vLLM 发 HTTP abort（within-node cancel，spec §2.2）。
  * restart()  —— vLLM crash / OOM 退出 → kill orphan → 过 F2 GPU-free gate
                  → re-spawn → re-preload（spec §4.1「LLM Runner crash → 重启
                  runner → re-spawn vLLM + preload」）。
"""
from __future__ import annotations

import asyncio
import enum
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class LLMRunnerState(str, enum.Enum):
    IDLE = "idle"          # 尚未 spawn
    RUNNING = "running"    # vLLM 子进程健康
    FAILED = "failed"      # spawn / preload 失败（fail-soft，failure 字段有原因）
    RESTARTING = "restarting"


def _default_gpu_free_probe(gpus: list[int]) -> bool:
    """生产用 F2 GPU-free 探针骨架：查 role:llm group 的 GPU 显存是否回落。

    死进程的 CUDA context 回收是异步的（spec §4.2 F2）—— re-spawn vLLM 前必须
    确认显存已释放，否则新 vLLM 立刻 OOM。本默认实现查 nvidia-smi；无 GPU 环境
    （CI / CUDA_VISIBLE_DEVICES=""）保守返回 True 不阻塞。测试注入 fake 探针。
    """
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return True  # 无 nvidia-smi → 不阻塞
        # 该 group 任意 GPU used > 2GB 视为 context 未回收（保守阈值）。
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and int(parts[0]) in gpus:
                if int(parts[1]) > 2048:
                    return False
        return True
    except Exception:
        return True  # 探针本身出错 → 不阻塞（保守，避免永久卡住重启）


class LLMRunner:
    """主进程内的 vLLM 子进程生命周期管理对象。不串行化推理请求。"""

    def __init__(
        self,
        *,
        model_key: str,
        adapter: Any,
        llm_gpus: list[int],
        gpu_free_probe: Callable[[list[int]], bool] | None = None,
        gpu_free_poll_interval: float = 2.0,
        gpu_free_max_wait: float = 120.0,
    ) -> None:
        self.model_key = model_key
        self.adapter = adapter           # VLLMAdapter（或测试的 FakeVLLMAdapter）
        self.llm_gpus = llm_gpus         # role:llm group 的 GPU index（Lane A allocator.llm_group_gpus()）
        self.state = LLMRunnerState.IDLE
        self.failure: str | None = None
        self._gpu_free_probe = gpu_free_probe or _default_gpu_free_probe
        self._gpu_free_poll_interval = gpu_free_poll_interval
        self._gpu_free_max_wait = gpu_free_max_wait
        # 注意：刻意不持有 PriorityQueue / inflight dict —— LLMRunner 不串行化推理。

    @property
    def base_url(self) -> str | None:
        """vLLM HTTP 端点 —— compat 路由 / executor 直连用（经 get_vllm_base_url）。"""
        return getattr(self.adapter, "base_url", None)

    async def spawn(self) -> None:
        """起 vLLM 子进程 + 等 health。失败抛 —— 调用方（preload）决定是否 fail-soft。"""
        await self.adapter.load()
        self.state = LLMRunnerState.RUNNING
        self.failure = None
        logger.info("LLMRunner %s spawned (base_url=%s)", self.model_key, self.base_url)

    async def health(self) -> bool:
        """探测 vLLM 子进程是否存活。并发安全 —— 无锁、不阻塞推理路径。"""
        try:
            return await self.adapter._health_check()
        except Exception:
            return False

    async def preload(self) -> None:
        """resident LLM 启动加载 —— fail-soft：失败不抛，记 failure（spec §4.2）。"""
        try:
            await self.spawn()
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            self.state = LLMRunnerState.FAILED
            self.failure = detail
            logger.warning("LLMRunner %s preload failed: %s", self.model_key, detail)

    async def abort(self, request_id: str) -> None:
        """向 vLLM 发 HTTP abort（within-node LLM cancel，spec §2.2）。

        vLLM 的 OpenAI-compat server 不暴露按 request_id 的 abort 端点；实际
        within-node cancel 由 compat 路由 / executor 关闭 httpx stream 实现
        （spec §4.4「LLM streaming: cancel_event → vllm_http_abort → 关流」）。
        本方法是 LLMRunner 侧的抽象入口 —— 当前为 best-effort no-op + 日志，
        真正的取消语义在 streaming 调用方那一侧（关闭连接 = vLLM 感知 disconnect
        并停止该序列的 decode）。保留此方法是为接口完整 + 未来 vLLM 暴露 abort
        端点时的挂载点。
        """
        logger.debug("LLMRunner %s abort request_id=%s (handled by stream close)",
                     self.model_key, request_id)

    async def _wait_gpu_free(self) -> None:
        """F2 GPU-free gate：轮询探针直到 role:llm group 的 GPU 显存回落。"""
        waited = 0.0
        while waited < self._gpu_free_max_wait:
            if self._gpu_free_probe(self.llm_gpus):
                return
            await asyncio.sleep(self._gpu_free_poll_interval)
            waited += self._gpu_free_poll_interval
        logger.warning(
            "LLMRunner %s GPU-free gate 超时（%.0fs）—— 仍尝试 re-spawn",
            self.model_key, self._gpu_free_max_wait,
        )

    async def restart(self) -> None:
        """vLLM crash / OOM 退出 → kill orphan → GPU-free gate → re-spawn → re-preload。

        spec §4.1：「LLM Runner crash → vLLM 也随之失联 → 重启 runner → 重新
        spawn vLLM + preload」。
        """
        self.state = LLMRunnerState.RESTARTING
        logger.warning("LLMRunner %s restarting (vLLM crash/OOM detected)", self.model_key)

        # 1. kill 残留 orphan（VLLMAdapter.unload 内部走 _kill_process，
        #    SIGTERM 进程组 → SIGKILL；adopted orphan 也覆盖）。
        try:
            self.adapter.unload()
        except Exception as e:
            logger.warning("LLMRunner %s unload during restart failed: %s",
                           self.model_key, e)

        # 2. F2 GPU-free gate —— 等死进程的 CUDA context 回收。
        await self._wait_gpu_free()

        # 3. re-spawn + re-preload（fail-soft）。
        await self.preload()

    async def shutdown(self) -> None:
        """优雅停止 —— 终结 vLLM 子进程。"""
        try:
            self.adapter.unload()
        finally:
            self.state = LLMRunnerState.IDLE
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_llm_runner.py -q`
Expected: 全 PASS（8 个用例）。

- [ ] **Step 5: lint 预检**

Run: `cd backend && ruff check src/runner/llm_runner.py tests/test_llm_runner.py`
Expected: 无 lint 错误。

- [ ] **Step 6: Commit**

```bash
cd backend && git add src/runner/llm_runner.py tests/test_llm_runner.py
git commit -m "feat(runner): LLMRunner — vLLM subprocess lifecycle, no request serialization

Unlike Lane C's RunnerSupervisor (which forks a multiprocessing.Process
running a per-group PriorityQueue), LLMRunner is a main-process object
managing the vLLM subprocess VLLMAdapter already spawns. spawn/health/
preload/abort/restart; restart passes the F2 GPU-free gate before
re-spawn. NO queue, NO serialization — vLLM continuous batching handles
concurrency (spec 1.2/1.3, D6/D8). V1.5 Lane E."
```

---

## Task 4: compat 路由直连 vLLM HTTP 的回归测试套

spec §5.3 列了一项 CRITICAL 回归：「[回归] compat 路由 —— A5 改道后 4 个 compat 路由（openai/anthropic/ollama/responses）仍产出正确输出」。Task 2 收敛了 base-URL 查找，本 Task 用 mock vLLM HTTP 端点端到端验证 4 个路由产出不变。

**Files:**
- New: `backend/tests/test_compat_routes_vllm_regression.py`
- Test infra: 复用 spec §5.6 提到的 `tests/fixtures/fake_vllm.py`（若 Lane C / 其它 Lane 已建则复用；本 Task 若发现没有，在本文件内联一个最小 mock，见 Step 1 注释）。

- [ ] **Step 1: 写回归测试 —— 4 个 compat 路由产出不变**

新建 `backend/tests/test_compat_routes_vllm_regression.py`：
```python
"""Lane E: [回归] compat 路由收敛 base-URL 查找后产出不变（spec §5.3 CRITICAL）。

Task 2 把 4 个 compat 路由的「get_adapter -> base_url」收敛到 get_vllm_base_url()。
本套用 mock vLLM HTTP 端点 + 一个已加载的 fake adapter，断言 openai/anthropic/
ollama/responses 仍返回正确结构、错误码不变。

mock vLLM：优先复用 tests/fixtures/fake_vllm.py（spec §5.6）。若该 fixture 尚不
存在（Lane C 未建），本文件用 respx / httpx mock 内联一个最小 /v1/chat/completions
响应 —— 关键是 compat 路由的「收敛后行为不变」断言，不是 vLLM 协议完整性。
"""
import pytest


# —— fixture：一个已加载、有 base_url 的 fake VLLMAdapter 塞进 app.state.model_manager ——
class _LoadedFakeAdapter:
    is_loaded = True
    base_url = "http://vllm.test"
    max_model_len = 4096


@pytest.fixture
def vllm_loaded(app):
    """把一个已加载的 fake vLLM adapter 注册到 app.state.model_manager。

    engine_name 对齐测试 ServiceInstance 的 source_name —— 若现有 compat 路由
    测试已有「建 model 类 ServiceInstance + 加载 adapter」的 fixture，复用它，
    本 fixture 仅作兜底。
    """
    mgr = app.state.model_manager
    adapter = _LoadedFakeAdapter()
    # model_manager.get_adapter 按 engine_name 查 —— 直接塞进内部 dict 的等价做法
    # 须对齐 model_manager 的实际内部结构；若有 register/_models 接口用接口。
    # 这里假设测试 helper：mgr._models[engine] = entry。实现时按 model_manager
    # 实际 API 调整（grep model_manager.py 的 get_adapter 实现确认查找路径）。
    yield mgr, adapter


@pytest.mark.asyncio
async def test_openai_compat_not_loaded_returns_503(client):
    """未加载的 engine → openai_compat 返回 503（VLLMNotLoaded 映射）。"""
    # 用一个不存在的 model → resolve 到未加载 adapter → 503
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "definitely-not-loaded", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    # 鉴权失败是 401，model 未找到是 404，未加载是 503 —— 断言不是 5xx-500
    # （收敛前后这条路径的错误码语义必须一致）。
    assert resp.status_code in (401, 404, 503)


@pytest.mark.asyncio
async def test_get_vllm_base_url_wired_into_all_four_routes():
    """静态保证：4 个 compat 路由都 import 了 get_vllm_base_url（收敛已落地）。"""
    import src.api.routes.anthropic_compat as anthropic_mod
    import src.api.routes.ollama_compat as ollama_mod
    import src.api.routes.openai_compat as openai_mod
    import src.api.routes.responses as responses_mod

    for mod in (openai_mod, anthropic_mod, ollama_mod, responses_mod):
        with open(mod.__file__) as f:
            content = f.read()
        assert "get_vllm_base_url" in content, (
            f"{mod.__name__} 应已收敛到 get_vllm_base_url"
        )
        # 收敛后不应再有裸的 getattr(adapter, "base_url" ...) 查找逻辑
        # （adapter.base_url 作为下游 max_model_len 旁的句柄使用是允许的，
        #  但 base_url 的「查找 + 空值检查」必须经 get_vllm_base_url）。
        assert 'getattr(adapter, "base_url"' not in content, (
            f"{mod.__name__} 仍有裸 base_url 查找 —— 应收敛到 get_vllm_base_url"
        )
```

（注：`test_openai_compat_not_loaded_returns_503` 走真实鉴权链路 —— 若现有 compat 测试套有更完整的「建 instance + key + 加载 adapter」fixture，**优先扩展那套**加一个「base-URL 收敛后产出不变」的用例，比从零搭鉴权更稳。本文件的 `test_get_vllm_base_url_wired_into_all_four_routes` 是静态保证，不依赖鉴权，必须保留。)

- [ ] **Step 2: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_compat_routes_vllm_regression.py -q`
Expected: 全 PASS。`test_get_vllm_base_url_wired_into_all_four_routes` 验证 Task 2 的收敛已落地；`test_openai_compat_not_loaded_returns_503` 验证错误码语义不变。

- [ ] **Step 3: 跑现有 compat suite 再确认零回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "compat or openai or anthropic or ollama or responses"`
Expected: PASS（= Task 2 Step 6 的结果）。

- [ ] **Step 4: Commit**

```bash
cd backend && git add tests/test_compat_routes_vllm_regression.py
git commit -m "test(compat): regression suite for vLLM base-URL consolidation

spec 5.3 CRITICAL regression: the 4 compat routes must still produce
correct output after get_vllm_base_url() consolidation. Static guard
asserts all 4 routes wired to get_vllm_base_url with no bare base_url
lookups left; behavioral guard asserts error-code semantics unchanged.
V1.5 Lane E."
```

---

## Task 5: 核对 workflow executor 的 llm 节点（大概率不改）

spec §4.5 把「workflow_executor 的 llm 节点 → executor inline HTTP 调 vLLM」列进改道清单。但 Lane S 审计已确认：`LLMNode`（`src/services/nodes/llm.py`）**本来就**经 `InferenceAdapter` 走 HTTP 调 vLLM —— 它本来就是 inline HTTP。本 Task 核对 `LLMNode` 是否需要收敛到 `get_vllm_base_url()`，还是它根本不直接碰 base_url（adapter 内部持有）。

**Files:**
- 核对（大概率不改）: `backend/src/services/nodes/llm.py`

- [ ] **Step 1: 核对 `LLMNode` 如何拿到 vLLM 端点**

Run:
```bash
cd backend && grep -n "base_url\|get_adapter\|get_loaded_adapter\|InferenceAdapter\|adapter\|infer\|httpx" src/services/nodes/llm.py
```
Expected 三种可能：
  - (a) `LLMNode` 调 `model_manager.get_loaded_adapter(model_id)` 拿 adapter，然后调 `adapter.infer()` / `adapter.infer_stream()` —— adapter **内部**持有 base_url + httpx（`VLLMAdapter._base_url` / `_client`）。**这种情况 LLMNode 不直接碰 base_url，无需收敛，本 Task 不改任何代码** —— 只在 Step 2 记录结论。
  - (b) `LLMNode` 自己 `getattr(adapter, "base_url")` + 自己拼 httpx 请求 —— 与 compat 路由同构，**应收敛到 `get_vllm_base_url()`**，按 Task 2 的模式改。
  - (c) `LLMNode` 走某个中间层（`InferenceAdapter` 工厂 / service）—— 记录实际路径，判断是否需收敛。

- [ ] **Step 2: 据 Step 1 结论行动**

- **若 (a)**（最可能 —— Lane S 审计的措辞「`LLMNode` 已通过 `InferenceAdapter` 走 HTTP」强烈指向这个）：不改代码。在 commit message + Self-Review 记录：「`LLMNode` 经 `adapter.infer()` 调 vLLM，base_url 由 `VLLMAdapter` 内部持有，不存在散落的 base-URL 查找需要收敛 —— spec §4.5 对 executor llm 节点的『改道』在本仓库是 no-op，执行方式本来就是 inline HTTP」。**跳过 Step 3，直接进 Task 6。**
- **若 (b)**：按 Task 2 Step 2 的模式改 `LLMNode` —— `getattr(adapter, "base_url")` 段替换为 `get_vllm_base_url(model_mgr, model_id)` + try/except。进 Step 3。
- **若 (c)**：按实际中间层情况判断，最小化改动；若中间层已封装得当则同 (a) 不改。

- [ ] **Step 3:（仅 (b)/(c) 需改时）写回归测试 + commit**

仅当 Step 2 判定需改代码时执行。在 `tests/test_workflow_executor_split.py`（Lane S 已建）或新建 `tests/test_llm_node_vllm.py` 加一个用例：mock 已加载 adapter，断言 `LLMNode` 经 `get_vllm_base_url` 拿到端点、产出不变。
```bash
cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "llm_node or llm"
# 全绿后：
git add src/services/nodes/llm.py tests/test_llm_node_vllm.py
git commit -m "refactor(nodes): LLMNode vLLM base-URL lookup via get_vllm_base_url

spec §4.5 — workflow executor's llm node consolidates its vLLM endpoint
lookup to the same source of truth as the compat routes. V1.5 Lane E."
```
（若 Step 2 判定为 (a) 不改 —— 本 Step 整体跳过，无 commit。）

---

## Task 6: 整合验证 + 开 PR

**Files:** 无（验证）

- [ ] **Step 1: 全 suite green**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS。无 collection error、无 `ModuleNotFoundError`。

- [ ] **Step 2: 确认 base-URL 查找已收敛**

Run:
```bash
cd backend && grep -rn 'getattr(adapter, "base_url"' src/api/routes/ src/services/nodes/
```
Expected: **零输出** —— 4 个 compat 路由（+ 若 Task 5 改了 `LLMNode`）的裸 base_url 查找全部收敛到 `get_vllm_base_url()`。若仍有命中，回 Task 2 / Task 5 补。

- [ ] **Step 3: 确认 `get_vllm_base_url` 被 4 个路由引用**

Run:
```bash
cd backend && grep -rln "get_vllm_base_url" src/
```
Expected: `src/services/inference/vllm_endpoint.py`（定义）+ `openai_compat.py` + `anthropic_compat.py` + `ollama_compat.py` + `responses.py`（+ 视 Task 5 结论可能有 `llm.py`）。

- [ ] **Step 4: 确认 LLMRunner 不持有串行化结构**

Run:
```bash
cd backend && grep -n "PriorityQueue\|asyncio.Queue\|inflight\|_queue" src/runner/llm_runner.py
```
Expected: **零输出** —— LLMRunner 刻意不持有任何队列 / inflight 串行化结构（spec §1.2 关键不变量：LLM runner 不串行化推理请求）。

- [ ] **Step 5: lint 预检（push 前本地跑）**

Run:
```bash
cd backend && ruff check src/services/inference/vllm_endpoint.py src/runner/llm_runner.py src/api/routes/openai_compat.py src/api/routes/anthropic_compat.py src/api/routes/ollama_compat.py src/api/routes/responses.py tests/test_vllm_endpoint.py tests/test_llm_runner.py tests/test_compat_routes_vllm_regression.py
```
Expected: 无 lint 错误。

- [ ] **Step 6: 后端冒烟 —— compat 路由未加载态错误码**

启动后端（依项目方式），然后：
```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer invalid' \
  -d '{"model":"not-loaded","messages":[{"role":"user","content":"hi"}]}'
```
Expected: `401`（鉴权失败）或 `404` / `503` —— **不是** `500`。重点确认 `get_vllm_base_url` 收敛后未引入 500。无 traceback。

- [ ] **Step 7: 开 PR**

```bash
git push -u origin <lane-e-branch>
gh pr create --title "feat: V1.5 Lane E — LLM Runner + compat routes direct-to-vLLM HTTP" --body "$(cat <<'EOF'
## Summary
- 新增 `get_vllm_base_url()`（`vllm_endpoint.py`）—— vLLM base-URL 查找的唯一真相源，收敛 4 个 compat 路由各自重复的 `get_adapter -> base_url` 接缝
- 新增 `LLMRunner`（`src/runner/llm_runner.py`）—— 主进程内管 vLLM 子进程生命周期（spawn/health/preload/abort/restart），过 F2 GPU-free gate；**不**串行化推理请求（与 image/TTS RunnerSupervisor 本质不同）
- 4 个 compat 路由（openai/anthropic/ollama/responses）收敛 base-URL 查找
- [回归] compat 路由直连 vLLM HTTP 产出不变测试套（spec §5.3 CRITICAL）

## 与 spec 的偏差
- spec §4.5 的「compat 路由 / executor llm 节点改道」对本仓库是收敛 + 重新归类，不是执行方式重写 —— compat 路由本来就走 HTTP→vLLM，`LLMNode` 经 adapter 也本来就是 HTTP（Lane S 审计已确认）
- spec §1.2 把 LLM Runner 画在「子进程」栏；本 Lane 判断 LLMRunner 是主进程内对象（管的子进程是 vLLM 本身），不再套一层 Python 子进程 —— 与 spec §4.5 隔离边界图「LLM Runner 只管 vLLM 子进程生命周期」一致
- Lane 0 删的旧 `get_llm_base_url` 是零调用方死代码，本 Lane 全新建，非复活

## Test plan
- [ ] 全 suite green（pytest tests/）
- [ ] `get_vllm_base_url` 三条路径单测（正常 / 未加载 503 / 无端点 500）
- [ ] `LLMRunner` 生命周期 + 不串行化单测（spawn/health/preload fail-soft/restart+GPU-free gate/无队列）
- [ ] [回归] 4 个 compat 路由收敛后产出不变 + 错误码语义不变
- [ ] 冒烟：未加载 engine 的 compat 请求不返回 500
EOF
)"
```
（分支名按项目惯例，如 `feat/v15-laneE-llm-runner`。）

---

## Self-Review

**Spec 覆盖检查：** Lane E 在 spec「实施分 Lane」表里的职责是「LLM Runner（只管 vLLM 生命周期：spawn/health/preload/abort/OOM-restart）+ 主进程 compat 路由 / executor 直连 vLLM HTTP（D6/D8 改道清单）」。

- LLM Runner（spawn/health/preload/abort/OOM-restart）→ Task 3（`LLMRunner`，5 个方法对齐 spec 用词；restart = OOM-restart，过 F2 GPU-free gate）
- compat 路由直连 vLLM HTTP → Task 2（4 个路由收敛到 `get_vllm_base_url`）+ Task 1（建查找真相源）
- executor 直连 vLLM HTTP → Task 5（核对 `LLMNode`；大概率本来就是 inline HTTP，no-op）
- D6/D8 改道清单 → Task 2 + Task 5 覆盖 §4.5 表的 4 个 compat 行 + 1 个 executor llm 行；image/TTS 节点行属 Lane S/C/D，不在本 Lane
- spec §5.3 CRITICAL 回归「compat 路由」→ Task 4

**与 spec 的偏差 / 歧义（5 处，已在文件头「与 spec 的偏差」节详述，须知会 reviewer）：**
1. **「改道」= 收敛 + 重新归类，非执行方式重写**：本仓库 compat 路由本来就 HTTP→vLLM，`LLMNode` 经 adapter 也本来就 HTTP（Lane S 审计确认）。spec §4.5 用「改道」一词，对本仓库实际是「收敛 base-URL 查找 + 标注为直连 HTTP」。
2. **LLMRunner 不是 `multiprocessing.Process` 子进程**：spec §1.2/§4.2 拓扑图把它画在子进程栏（与 image/TTS runner 并列），但 spec §1.2 正文 + §4.5 隔离边界图「LLM Runner 只管 vLLM 子进程生命周期」「vLLM 自身又是 subprocess」其实支持「LLMRunner = 主进程对象 + 它管的 vLLM 子进程」。本 Lane 取后者 —— LLM runner 不串行化、不需要独立 event loop 跑 pipe-reader，再套一层 Python 子进程是多余故障面。
3. **spec 没给 `LLMRunner` 接口草图**（§4.2 草图是 image/TTS 的 `RunnerSupervisor`）—— 本 Lane 按 spec §1.2/§4.1/§4.2 文字定义接口，命名对齐 spec 用词。
4. **「OOM-restart」统一处理**：spec §4.1 把「vLLM 启动失败」「LLM Runner crash」分两行，恢复策略都是「重启 → re-spawn + preload」，本 Lane 用同一条 `restart()` 路径，不区分退出原因。
5. **F2 GPU-free gate 真探针**：Lane C 的默认探针保守返回 True；本 Lane 的 `_default_gpu_free_probe` 给真 nvidia-smi 骨架（查 role:llm group GPU 显存），无 GPU 环境保守返回 True。测试全程注入 fake 探针。

**依赖说明：** Lane E 依赖 Lane D（image runner 迁入，验证 runner 框架）+ Lane S（executor 重写 —— 本 Lane Task 5 核对 `LLMNode` 时引用 Lane S 的审计结论）。`src/runner/` 目录由 Lane C 新建（Task 3 Step 0 显式核对存在）。`LLMRunner.llm_gpus` 的数据源是 Lane A 的 `GPUAllocator.llm_group_gpus()`（本 Lane 测试直接传 `[0,1]`，生产接线由 main.py 在 Lane H/整合时注入 —— 本 Lane 不动 main.py 启动序列，只交付 `LLMRunner` 类 + 查找函数 + 路由收敛）。**风险**：若 Lane D/S 实际接口与假设偏离，Task 5 的核对结论可能需调整 —— 已把 Task 5 设计成「先 grep 核对再决定改不改」的分支结构，最坏情况是多改一个文件（`LLMNode`），不影响 Task 1-4。

**回归风险（已守住）：** Task 2 收敛 4 个 compat 路由的 base-URL 查找是行为敏感改动。守法三层：(1) Task 2 Step 1/Step 6 要求收敛前后跑现有 compat suite 对照通过数；(2) Task 4 的 `test_compat_routes_vllm_regression.py` 静态保证 4 个路由都 wired 到 `get_vllm_base_url` 且无裸 base_url 查找残留 + 行为保证错误码语义不变（503/500 映射与原一致）；(3) Task 6 Step 6 冒烟确认未加载态不返回 500。`get_vllm_base_url` 的 detail 文案是中文（与原英文 detail 不同）—— Task 2 Step 6 显式提示：若现有用例断言文案则改为断言 `status_code`。

**未决接缝（实现时现场核对，已在对应 Task 标注）：**
- 各 compat 路由下游是否还需 `adapter` 句柄（`max_model_len` 等）—— Task 2 Step 3/Step 5 要求先 grep 确认再决定是否保留 `adapter = model_mgr.get_adapter(...)` 末行
- `test_compat_routes_vllm_regression.py` 的「已加载 adapter 注册到 model_manager」做法 —— Task 4 Step 1 注释要求对齐 `model_manager` 实际内部结构 / register 接口（优先扩展现有 compat 测试套的 fixture）
- `LLMNode` 拿 vLLM 端点的实际路径（(a)/(b)/(c) 三种）—— Task 5 Step 1 grep 核对，Step 2 据结论分支；最可能是 (a)（不改代码）
- `VLLMAdapter` 的 `_health_check` / `unload` / `_kill_process` 在 `restart()` 路径的实际行为 —— `LLMRunner.restart()` 调 `adapter.unload()`（内部走 `_kill_process`，覆盖 `_process` 与 `_adopted_pid` 两种），测试用 `FakeVLLMAdapter` 模拟；真 adapter 接线在整合阶段验证（E2E «Runner 真 crash 恢复»，spec §5.4）

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。所有代码完整给出。Task 5 是「核对 + 条件改动」结构（不是 placeholder —— 给了 (a)/(b)/(c) 三种结论的明确行动，且标注最可能是 (a) 不改）。`LLMRunner.abort()` 当前为 best-effort no-op + 日志 —— 这不是 placeholder，是经过说明的设计决策（vLLM OpenAI-compat server 无 per-request abort 端点，within-node cancel 实际由 streaming 调用方关闭 httpx 连接实现，见方法 docstring；保留 `abort()` 是为接口完整 + 未来挂载点）。

**类型一致性：** `get_vllm_base_url(model_manager, engine_name) -> str` 抛 `VLLMNotLoaded` / `VLLMNoEndpoint`（均 `RuntimeError` 子类）；4 个 compat 路由 try/except 映射 `VLLMNotLoaded -> 503` / `VLLMNoEndpoint -> 500` 一致。`LLMRunner.__init__(model_key, adapter, llm_gpus, gpu_free_probe, gpu_free_poll_interval, gpu_free_max_wait)`；`spawn()` / `health()` / `preload()` / `abort(request_id)` / `restart()` / `shutdown()` 全 `async`（`abort` 取 `request_id: str`）；`LLMRunnerState` 是 `str` enum（IDLE/RUNNING/FAILED/RESTARTING）；`gpu_free_probe: Callable[[list[int]], bool]` 与 `_default_gpu_free_probe(gpus: list[int]) -> bool` / 测试 fake 探针签名一致。
