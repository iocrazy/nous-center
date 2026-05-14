# V1.5 Lane J: Integration + Chaos 测试套 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付 spec §5 的 **跨 Lane 集成测试套 + chaos soak 测试 + 共享测试基础设施**，并补齐 §5 点名的 4 项 CRITICAL 回归里「没有任何单一 Lane 拥有」的那一项（ModelManager 合并回归）。Lane J 是 V1.5 最后一个 Lane，依赖 0、A 到 I 全部合并完成。

**Architecture:** Lane J **不写产品代码**，只建测试。三块交付物：

1. **共享测试基础设施（spec §5.6）** —— `tests/fixtures/fake_runner.py`（mock image/TTS runner subprocess，可配置 crash/slow/fail-load）、`tests/fixtures/hardware_topo.py`（2gpu/3gpu `hardware.yaml` fixture）。`tests/fixtures/fake_vllm.py` 由 Lane E 承接（见下「与 spec / 各 Lane plan 的偏差」），Lane J 只在它缺失时兜底。pytest markers（`integration` / `e2e` / `chaos`）由 Lane G Task 6 注册——Lane J Task 1 仅做存在性校验，缺失才补。
2. **Integration 测试套（spec §5.3）** —— 主进程 + fake runner subprocess，跑通完整 IPC 协议的端到端场景：workflow 完整生命周期、优先级抢占、cancel inflight、runner crash 检测、runner 重启 resident preload、模型 load_failed 不阻断、队列堆积 503、API server 重启恢复、LLM runner 不串行化、跨 runner tensor、DB reconcile、混合节点 workflow。全部打 `@pytest.mark.integration`。
3. **Chaos soak 测试（spec §5.5）** —— `tests/chaos/test_worker_crash_storm.py`、`tests/chaos/test_db_flaky.py`（`test_pipe_slow_consumer.py` 由 Lane C 承接，见下偏差说明）。全部打 `@pytest.mark.chaos`。

**Tech Stack:** Python 3.12 / pytest（`asyncio_mode = "auto"`）/ `multiprocessing`（spawn context，Lane C 已建立）/ `pytest -m integration` / `pytest -m chaos`。复用 `tests/conftest.py` 的 `ADMIN_PASSWORD=""` + `NOUS_DISABLE_BG_TASKS=1` + `CUDA_VISIBLE_DEVICES=""` + `NOUS_DISABLE_FRONTEND_MOUNT=1` + torch MagicMock stub。复用 Lane C 的 `src/runner/`、Lane B 的 `src/services/task_ring_buffer.py`、Lane G 的 `src/services/scheduler/` + `inference/cancel_flag.py`、Lane A 的 `src/gpu/` allocator、Lane S 的 `src/services/workflow_runner.py`。

---

> **注意 — 与 spec / 各 Lane plan 的偏差（已核实，须知会）：**
>
> 1. **spec §5 列的 4 项 CRITICAL 回归里，3 项已被对应 Lane plan 拥有，Lane J 不重复。** 逐项核对：
>    - **回归 (1) post-Lane-0 monitor.py/gpu_monitor.py** —— Lane 0 plan 已有 `test_api_monitor.py::test_monitor_loaded_models_from_model_manager` + `test_gpu_monitor_evict.py` 两个回归。但 spec §5.3 表里把它列为 *integration* 行（「Lane 0 后 monitor.py / gpu_monitor.py 仍正确报告加载状态 **+ idle-TTL 卸载仍生效**」）——Lane 0 的回归是 unit 级、**没覆盖 idle-TTL 卸载这条端到端路径**。Lane J Task 5 补一个 integration 级用例专门守 idle-TTL。
>    - **回归 (2) post-A5 4 个 compat 路由** —— Lane E plan Task 4 已有 `test_compat_routes_vllm_regression.py`，完整覆盖。Lane J **不重复**，仅在 Task 9 的 Self-Review checklist 里引用确认它存在。
>    - **回归 (3) workflow inline→queued (D17 异步 202)** —— Lane S plan Task 5 已有 `test_run_async_contract.py`，完整覆盖 enqueue→poll→result。Lane J **不重复**。
>    - **回归 (4) post-merge src/gpu/model_manager.py 所有调用点** —— **没有任何 Lane plan 拥有这一项**。且 Lane 0 plan 已审计确认 `src/gpu/model_manager.py` 是死代码、被**删除**而非合并（与 spec G5「合并」描述偏差，Lane 0 plan 顶部已标注）。所以 spec §5「[回归] ModelManager 合并」的字面前提（「合并后所有调用点仍工作」）不成立——实际要回归的是「**删除** `src/gpu/model_manager.py` + `model_scheduler.py` 后，全仓不再有对它们的引用，且唯一真相源 `services/model_manager.py` 的所有调用点正常」。Lane J Task 6 按这个**修正后**的语义建回归。这是 Lane J 唯一拥有的 CRITICAL 回归。
> 2. **`test_pipe_slow_consumer.py` 归 Lane C，不归 Lane J。** spec §5.5 把 `test_pipe_slow_consumer` 列在 chaos 套里，但 Lane C plan Task 7 已建 `tests/chaos/test_pipe_slow_consumer.py`（F1 写超时是 Lane C 承接的 CRITICAL GAP，必须随 Lane C 落地验证）。Lane J 的 chaos 套只建 spec §5.5 剩下的两项：`test_worker_crash_storm` + `test_db_flaky`。Lane J Task 1 会校验 `test_pipe_slow_consumer.py` 已存在。
> 3. **`test_runner_crash_storm.py` vs `test_worker_crash_storm.py` 命名不一致。** spec §5.5 代码块注释写 `test_runner_crash_storm.py`，但简报正文写 `test_worker_crash_storm`。统一用 **`test_worker_crash_storm.py`**（简报是更晚的口径）。文件内测函数名 `test_runner_repeated_crashes`（对齐 spec 代码块）。
> 4. **`tests/fixtures/fake_vllm.py` 归属模糊。** spec §5.6 把它列为「新增」基础设施，但 Lane E plan Task 4 说「优先复用 `tests/fixtures/fake_vllm.py`（若 Lane C / 其它 Lane 已建则复用；本 Task 若发现没有，在本文件内联一个最小 mock）」——即 Lane E 假设别人建、自己兜底内联。结论：**Lane J 负责把 `fake_vllm.py` 建成正式 fixture**（spec §5.6 明确归入「新增」基础设施表），Lane E 的内联兜底届时可删（非 Lane J scope，留注释提示）。
> 5. **spec §5.3「Runner 内部并发」「跨 runner tensor」两行可能与 Lane C / Lane D 重叠。** Lane C plan 的 `test_runner_process.py` 已测 pipe-reader + executor 双 task 分离 + Abort-during-node。Lane J 的 integration 版聚焦**主进程视角**（RunnerClient 发 Abort → fake runner 模拟中断 → 主进程收到 cancelled NodeResult），与 Lane C 的**子进程内部视角**互补不重复。跨 runner tensor 同理：Lane J 测主进程 host-pinned 中转的序列化往返，不测 runner 内 D→H 拷贝（那需真 GPU，属 e2e）。Self-Review 再次确认边界。
>
> 若实施时发现某 Lane plan 的实际落地与上述偏差判断不符（例如 Lane C 没建 `test_pipe_slow_consumer.py`），停下来报告，不要盲目补建——可能是依赖 Lane 未按 plan 落地。

---

## File Structure

| 文件 | Lane J 动作 | 责任 |
|---|---|---|
| `backend/pyproject.toml` | **校验/补** | `[tool.pytest.ini_options].markers` 应已有 `integration`/`e2e`/`chaos`（Lane G 注册）；缺失才补 |
| `backend/tests/fixtures/fake_runner.py` | **新建** | mock image/TTS runner subprocess fixture：spawn 真子进程跑 Lane C 的 `runner_main` + FakeAdapter，可配置 crash/slow/fail-load；提供 `RunnerClient` 句柄 |
| `backend/tests/fixtures/hardware_topo.py` | **新建** | 2gpu / 3gpu `hardware.yaml` 内容 fixture + 写临时文件 + monkeypatch `load_hardware_config` 路径 |
| `backend/tests/fixtures/fake_vllm.py` | **新建** | mock vLLM HTTP 端点（`/v1/chat/completions` + `/health` + abort），给 LLM 直连路径 integration 测试用 |
| `backend/tests/integration/__init__.py` | **新建** | 包标记，空文件 |
| `backend/tests/integration/conftest.py` | **新建** | integration 专用 fixture 装配（组合 fake_runner + hardware_topo + scheduler + ring buffer） |
| `backend/tests/integration/test_workflow_lifecycle.py` | **新建** | workflow 完整生命周期 + 优先级抢占 + cancel inflight + 混合节点 workflow（spec §5.3）|
| `backend/tests/integration/test_runner_resilience.py` | **新建** | runner crash 检测 + 重启 resident preload + load_failed 不阻断 + LLM runner 不串行化 + 主进程视角 Abort（spec §5.3）|
| `backend/tests/integration/test_scheduler_degradation.py` | **新建** | 队列堆积 503 + API server 重启恢复 + DB reconcile + 跨 runner tensor 序列化往返（spec §5.3）|
| `backend/tests/integration/test_lane0_idle_ttl_regression.py` | **新建** | **[回归]** spec §5.3「Lane 0 后 idle-TTL 卸载仍生效」的 integration 级补充（Lane 0 unit 回归未覆盖）|
| `backend/tests/integration/test_modelmanager_consolidation_regression.py` | **新建** | **[回归 CRITICAL #4]** Lane 0 删除 `src/gpu/model_manager.py` + `model_scheduler.py` 后全仓无残留引用、`services/model_manager.py` 调用点正常 |
| `backend/tests/chaos/test_worker_crash_storm.py` | **新建** | **[chaos]** 连续 5 次 runner crash → backoff + GPU-free gate + 主进程不挂（spec §5.5）|
| `backend/tests/chaos/test_db_flaky.py` | **新建** | **[chaos]** 50% DB OperationalError soak 1000 task → ring buffer + reconcile 一致（spec §5.5）|
| `backend/tests/chaos/test_pipe_slow_consumer.py` | **校验存在** | Lane C Task 7 已建——Lane J 仅 grep 确认，不重建 |

---

## Task 1: 测试基础设施校验 + markers 兜底

Lane J 的所有 integration / chaos 测试都依赖 markers 已注册、Lane C 的 `src/runner/` 已落地。这是一次性的事实核查 + 缺失兜底。**若任何一步的实际输出与「Expected」严重不符（例如 `src/runner/` 不存在），停下来报告——说明依赖 Lane 未按 plan 合并，Lane J 不能继续。**

**Files:**
- 校验/补: `backend/pyproject.toml`

- [ ] **Step 1: 跑全 suite 建基线**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS（记下通过数 = Lane J 开工前的对照基线；0、A-I 全部合并后这个数应远高于 V1 基线）。无 collection error。

- [ ] **Step 2: 校验 pytest markers 已注册**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest --markers | grep -E "integration|e2e|chaos"`
Expected: 看到三行 marker 描述（`@pytest.mark.integration: ...` / `@pytest.mark.e2e: ...` / `@pytest.mark.chaos: ...`）——Lane G Task 6 已注册。

若**没有输出**（Lane G 未注册或被回退），在 `backend/pyproject.toml` 的 `[tool.pytest.ini_options]` 段追加：
```toml
markers = [
    "integration: 多进程 / mock runner subprocess 集成测试（CI 默认跑，spec §5.3）",
    "e2e: 真 GPU 测试，CI skip（dev box 跑 pytest -m e2e，spec §5.4）",
    "chaos: 故障注入 soak 测试（每周手动跑，spec §5.5）",
]
```
然后重跑 Step 2 确认三行出现。

- [ ] **Step 3: 校验依赖 Lane 的代码面已落地**

Run:
```bash
cd backend && python -c "
import src.runner.protocol, src.runner.client, src.runner.supervisor, src.runner.runner_process, src.runner.fake_adapter
import src.services.task_ring_buffer
import src.services.workflow_runner
print('Lane C/B/S surfaces import OK')
" && grep -rln "load_hardware_config" src/ --include="*.py" | head -3
```
Expected: 打印 `Lane C/B/S surfaces import OK`；grep 至少命中 `src/gpu/` 下的 hardware config loader（Lane A 落地）。若 `import` 报 `ModuleNotFoundError`，说明对应 Lane 未合并——停下报告，Lane J 不能继续。

- [ ] **Step 4: 校验 Lane C 已建 `test_pipe_slow_consumer.py`**

Run: `cd backend && ls tests/chaos/test_pipe_slow_consumer.py && grep -l "chaos" tests/chaos/test_pipe_slow_consumer.py`
Expected: 文件存在。Lane J **不重建**它（见 plan 顶部偏差 2）。若不存在——Lane C 未按 plan 落地，停下报告。

无 commit（本 Task 不改产品代码；若 Step 2 补了 markers，并入 Task 9 收尾 commit）。

---

## Task 2: `hardware_topo.py` fixture —— 2gpu / 3gpu hardware.yaml

spec §5.6 新增基础设施第一项。integration 测试要在「2 卡当前布局」和「3 卡未来布局」两种拓扑下跑调度逻辑，需要一个能产出两份 `hardware.yaml` 内容、写临时文件、并把 loader 指过去的 fixture。

**Files:**
- Create: `backend/tests/fixtures/hardware_topo.py`
- Test: `backend/tests/integration/__init__.py`（新建空包标记）+ `backend/tests/test_hardware_topo_fixture.py`（新建，验证 fixture 本身）

- [ ] **Step 1: 建 integration 包标记 + 写失败测试验证 fixture**

```bash
cd backend && mkdir -p tests/integration && touch tests/integration/__init__.py
```

新建 `backend/tests/test_hardware_topo_fixture.py`：
```python
"""Lane J: hardware_topo fixture 自测 —— 两份 hardware.yaml 内容 + 临时文件落地。"""
from tests.fixtures.hardware_topo import HARDWARE_2GPU, HARDWARE_3GPU, write_hardware_yaml


def test_2gpu_topo_has_single_llm_group():
    """2 卡布局（spec §1.4 方案 A）：单 group llm-tp，GPU [0,1] NVLink。"""
    groups = HARDWARE_2GPU["groups"]
    assert len(groups) == 1
    g = groups[0]
    assert g["id"] == "llm-tp"
    assert g["gpus"] == [0, 1]
    assert g["nvlink"] is True
    assert g["role"] == "llm"


def test_3gpu_topo_has_three_independent_groups():
    """3 卡布局（spec §3.2）：image / llm-tp / tts 三 group。"""
    groups = HARDWARE_3GPU["groups"]
    ids = {g["id"] for g in groups}
    assert ids == {"image", "llm-tp", "tts"}
    llm = next(g for g in groups if g["id"] == "llm-tp")
    assert llm["nvlink"] is True and llm["gpus"] == [0, 1]
    image = next(g for g in groups if g["id"] == "image")
    assert image["nvlink"] is False and image["gpus"] == [2]


def test_write_hardware_yaml_roundtrips(tmp_path):
    """write_hardware_yaml 落临时文件，yaml.safe_load 能读回等价结构。"""
    import yaml

    path = write_hardware_yaml(tmp_path, HARDWARE_2GPU)
    assert path.exists()
    loaded = yaml.safe_load(path.read_text())
    assert loaded == HARDWARE_2GPU
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_hardware_topo_fixture.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'tests.fixtures.hardware_topo'`。

- [ ] **Step 3: 实现 `hardware_topo.py`**

新建 `backend/tests/fixtures/hardware_topo.py`：
```python
"""Lane J 测试基础设施：hardware.yaml 拓扑 fixture（spec §5.6）。

提供 2gpu / 3gpu 两份 hardware.yaml 内容 dict + 写临时文件 helper +
一个把 load_hardware_config 指向临时文件的 pytest fixture。

2gpu  = spec §1.4 方案 A：单 group llm-tp（当前 2×3090 部署）。
3gpu  = spec §3.2 hardware.3gpu.yaml（Pro 6000 到货后）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

HARDWARE_2GPU: dict[str, Any] = {
    "groups": [
        {
            "id": "llm-tp",
            "gpus": [0, 1],
            "nvlink": True,
            "role": "llm",  # image/TTS 节点也落此 group，与 LLM 时分复用
            "vram_gb": 48,
        },
    ],
}

HARDWARE_3GPU: dict[str, Any] = {
    "groups": [
        {"id": "image", "gpus": [2], "nvlink": False, "role": "image", "vram_gb": 96},
        {"id": "llm-tp", "gpus": [0, 1], "nvlink": True, "role": "llm", "vram_gb": 48},
        {"id": "tts", "gpus": [3], "nvlink": False, "role": "tts", "vram_gb": 24},
    ],
}


def write_hardware_yaml(dir_path: Path, content: dict[str, Any]) -> Path:
    """把 hardware.yaml 内容写进 dir_path/hardware.yaml，返回路径。"""
    path = Path(dir_path) / "hardware.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


@pytest.fixture
def hardware_2gpu(tmp_path, monkeypatch):
    """把 load_hardware_config 指向一份临时 2gpu hardware.yaml。

    monkeypatch 目标 = Lane A 落地的 hardware config loader。grep 出实际模块路径
    （Task 1 Step 3 已确认 loader 存在），这里按 src.gpu.hardware_config 假设；
    若 Lane A 落在别处，对齐实际 import path。
    """
    path = write_hardware_yaml(tmp_path, HARDWARE_2GPU)
    import src.gpu.hardware_config as hw  # Lane A 模块；按实际路径对齐

    monkeypatch.setattr(hw, "HARDWARE_YAML_PATH", path, raising=False)
    hw.load_hardware_config.cache_clear() if hasattr(
        hw.load_hardware_config, "cache_clear"
    ) else None
    return path


@pytest.fixture
def hardware_3gpu(tmp_path, monkeypatch):
    """把 load_hardware_config 指向一份临时 3gpu hardware.yaml。"""
    path = write_hardware_yaml(tmp_path, HARDWARE_3GPU)
    import src.gpu.hardware_config as hw

    monkeypatch.setattr(hw, "HARDWARE_YAML_PATH", path, raising=False)
    hw.load_hardware_config.cache_clear() if hasattr(
        hw.load_hardware_config, "cache_clear"
    ) else None
    return path
```

> 实现说明：`hardware_2gpu` / `hardware_3gpu` 两个 fixture 里 monkeypatch 的模块路径（`src.gpu.hardware_config`）是按 Lane A plan 的命名假设。Task 1 Step 3 的 grep 已确认 loader 存在——实现时把 grep 结果的实际模块路径填进去。`HARDWARE_2GPU` / `HARDWARE_3GPU` 两个常量 + `write_hardware_yaml` 不依赖 Lane A，能独立测（Step 1 的三个用例里前两个就是纯常量校验）。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_hardware_topo_fixture.py -q`
Expected: 3 个用例全 PASS。

- [ ] **Step 5: Commit**

```bash
cd backend && git add tests/fixtures/hardware_topo.py tests/integration/__init__.py tests/test_hardware_topo_fixture.py
git commit -m "test(infra): add hardware_topo fixture (2gpu/3gpu hardware.yaml)

spec §5.6 test infrastructure: 2gpu (current §1.4 plan A single
llm-tp group) and 3gpu (§3.2 future layout) hardware.yaml fixtures
for integration tests. Lane J."
```

---

## Task 3: `fake_runner.py` fixture —— mock image/TTS runner subprocess

spec §5.6 新增基础设施第二项。integration 测试要「主进程 + fake runner subprocess 跑通完整 IPC 协议」。Lane C 已有 `runner_main()` 子进程入口 + `FakeAdapter` + `RunnerSupervisor`——`fake_runner` fixture 把它们组装成一个「可配置 crash/slow/fail-load、对外暴露 `RunnerClient` 句柄、用完自动清理」的 pytest fixture。这是 Task 4-8 所有 integration 测试的共用底座。

**Files:**
- Create: `backend/tests/fixtures/fake_runner.py`
- Test: `backend/tests/test_fake_runner_fixture.py`（新建，验证 fixture 本身）

- [ ] **Step 1: 写失败测试验证 fixture**

新建 `backend/tests/test_fake_runner_fixture.py`：
```python
"""Lane J: fake_runner fixture 自测 —— spawn 真子进程跑 Lane C runner_main + FakeAdapter。"""
import asyncio

import pytest

from src.runner import protocol as P

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fake_runner_handshakes_and_runs_node(fake_runner):
    """fake_runner fixture 起一个 image runner，能 load_model + run_node 拿 completed。"""
    runner = fake_runner(group_id="image", gpus=[2])
    await runner.start()
    try:
        await runner.client.load_model("fake-img", config={})
        result = await runner.client.run_node(P.RunNode(
            task_id=1, node_id="sampler", node_type="image",
            model_key="fake-img", inputs={"steps": 3},
        ))
        assert result.status == "completed"
        assert result.task_id == 1
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_fake_runner_crash_on_demand(fake_runner):
    """配置 crash_on_node=True → run_node 触发子进程 crash → 主进程侧 pipe EOF / 超时。"""
    runner = fake_runner(group_id="image", gpus=[2], crash_on_node=True)
    await runner.start()
    try:
        with pytest.raises((asyncio.TimeoutError, ConnectionError, EOFError, BrokenPipeError)):
            await asyncio.wait_for(
                runner.client.run_node(P.RunNode(
                    task_id=2, node_id="n", node_type="image",
                    model_key="fake-img", inputs={"steps": 2},
                )),
                timeout=5.0,
            )
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_fake_runner_fail_load(fake_runner):
    """配置 fail_load=True → load_model 收到 ModelEvent(load_failed)。"""
    runner = fake_runner(group_id="image", gpus=[2], fail_load=True)
    await runner.start()
    try:
        event = await runner.client.load_model("fake-img", config={})
        assert event.event == "load_failed"
        assert event.error
    finally:
        await runner.stop()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_fake_runner_fixture.py -q`
Expected: FAIL —— `fixture 'fake_runner' not found`。

- [ ] **Step 3: 实现 `fake_runner.py`**

新建 `backend/tests/fixtures/fake_runner.py`：
```python
"""Lane J 测试基础设施：mock image/TTS runner subprocess（spec §5.6）。

封装 Lane C 的 runner_main() + FakeAdapter + RunnerClient，组装成一个
可配置 crash / slow / fail-load 的 pytest fixture。integration 测试用它
跑「主进程 + 真 runner 子进程 + 完整 IPC 协议」的端到端场景，零 GPU 零模型。

与 Lane C tests/ 里直接用 RunnerSupervisor 的区别：这里是测试侧的轻量
封装，统一 start/stop 生命周期 + crash/slow/fail-load 开关，给跨 Lane
integration 套复用，不是产品代码。
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.runner.client import RunnerClient
from src.runner.runner_process import runner_main


@dataclass
class FakeRunnerHandle:
    """一个 fake runner 子进程 + 主进程侧 RunnerClient 的句柄。"""

    group_id: str
    gpus: list[int]
    crash_on_node: bool = False
    slow_seconds: float = 0.0
    fail_load: bool = False
    _process: mp.process.BaseProcess | None = field(default=None, repr=False)
    client: RunnerClient | None = field(default=None, repr=False)
    _parent_conn: Any = field(default=None, repr=False)

    async def start(self) -> None:
        """spawn runner 子进程 + 建 RunnerClient + 等 Ready 握手。"""
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        # FakeAdapter 的 crash/slow/fail-load 经 adapter_kwargs 透传（Lane C
        # runner_main 把它转给 FakeAdapter 构造）。若 Lane C 的 runner_main
        # 签名只收 adapter_class、不收 adapter_kwargs，则改用环境变量或
        # 一个专门的 FakeRunnerAdapter 子类——对齐 Lane C 实际签名。
        self._process = ctx.Process(
            target=runner_main,
            args=(self.group_id, self.gpus, child_conn),
            kwargs={
                "adapter_class": "src.runner.fake_adapter.FakeAdapter",
                "adapter_kwargs": {
                    "crash_on_infer": self.crash_on_node,
                    "infer_seconds": self.slow_seconds,
                    "fail_load": self.fail_load,
                },
            },
            daemon=True,
        )
        self._process.start()
        child_conn.close()  # 主进程侧不持子端
        self.client = RunnerClient(self._parent_conn)
        await self.client.wait_ready(timeout=10.0)

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    def kill(self) -> None:
        """SIGKILL 子进程 —— 模拟 hard crash。"""
        if self._process and self._process.is_alive():
            self._process.kill()

    async def stop(self) -> None:
        """优雅关闭：close client → terminate 子进程 → join。"""
        if self.client is not None:
            await self.client.close()
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2.0)


@pytest.fixture
def fake_runner():
    """工厂 fixture：fake_runner(group_id=..., gpus=..., crash_on_node=..., ...)
    返回一个 FakeRunnerHandle。测试负责 await handle.start() / handle.stop()
    （或用 try/finally）。fixture teardown 兜底清理所有创建过的 handle。
    """
    created: list[FakeRunnerHandle] = []

    def _factory(group_id: str, gpus: list[int], **cfg: Any) -> FakeRunnerHandle:
        handle = FakeRunnerHandle(group_id=group_id, gpus=gpus, **cfg)
        created.append(handle)
        return handle

    yield _factory

    # teardown：兜底杀掉任何没 stop 干净的子进程
    for h in created:
        if h._process is not None and h._process.is_alive():
            h._process.kill()
            h._process.join(timeout=2.0)
```

> 实现说明：`adapter_kwargs` 透传依赖 Lane C 的 `runner_main` 签名接受 `adapter_kwargs`。Lane C plan 的 `runner_main` 签名只列了 `adapter_class`——实施时先 grep `src/runner/runner_process.py` 确认：若 `runner_main` 不收 `adapter_kwargs`，两条出路任选其一：(a) 在 `fake_runner.py` 里定义一个 `FakeRunnerAdapter(FakeAdapter)` 子类、构造参数写死、用 dotted path 传给 `adapter_class`（每种 crash/slow/fail-load 组合一个子类，或子类读环境变量）；(b) 给 Lane C 的 `runner_main` 加一个可选 `adapter_kwargs` 参数（这是产品代码改动，需另开 PR，不在 Lane J scope——优先选 a）。`wait_ready` / `close` 是 Lane C `RunnerClient` 的方法（Lane C plan Task 5 提到 demux 协程 + ready 握手）——若实际方法名不同（如 `wait_for_ready` / `shutdown`），对齐 Lane C 实际 API。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_fake_runner_fixture.py -q`
Expected: 3 个用例全 PASS（起真子进程，单文件约 10-25s）。`-m integration` 也应能选到它们（已打 `pytestmark = pytest.mark.integration`）。

- [ ] **Step 5: Commit**

```bash
cd backend && git add tests/fixtures/fake_runner.py tests/test_fake_runner_fixture.py
git commit -m "test(infra): add fake_runner fixture (mock runner subprocess)

spec §5.6 test infrastructure: wraps Lane C runner_main + FakeAdapter +
RunnerClient into a configurable (crash/slow/fail-load) pytest fixture
with managed start/stop lifecycle. Base for the integration suite. Lane J."
```

---

## Task 4: `fake_vllm.py` fixture —— mock vLLM HTTP 端点

spec §5.6 新增基础设施第三项。LLM 直连路径（compat 路由 + workflow llm 节点 inline HTTP）的 integration 测试需要一个 mock vLLM HTTP 端点。Lane E plan 自己内联兜底了一个最小 mock——Lane J 把它建成正式 fixture（见 plan 顶部偏差 4）。

**Files:**
- Create: `backend/tests/fixtures/fake_vllm.py`
- Test: `backend/tests/test_fake_vllm_fixture.py`（新建，验证 fixture 本身）

- [ ] **Step 1: 写失败测试验证 fixture**

新建 `backend/tests/test_fake_vllm_fixture.py`：
```python
"""Lane J: fake_vllm fixture 自测 —— mock vLLM HTTP 端点（chat/completions + health + abort）。"""
import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fake_vllm_chat_completions(fake_vllm):
    """fake_vllm 暴露 /v1/chat/completions，返回 OpenAI 格式响应。"""
    async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
        resp = await c.post("/v1/chat/completions", json={
            "model": "fake-llm",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"]
    assert "usage" in body


@pytest.mark.asyncio
async def test_fake_vllm_health(fake_vllm):
    """/health 返回 200 —— LLM runner preload 健康检查用。"""
    async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_fake_vllm_streaming(fake_vllm):
    """stream=true → SSE 风格逐 chunk；最后一个 chunk 含 finish_reason。"""
    async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "fake-llm",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            chunks = [line async for line in resp.aiter_lines() if line.strip()]
    assert len(chunks) >= 1
    assert any("[DONE]" in c or "finish_reason" in c for c in chunks)


@pytest.mark.asyncio
async def test_fake_vllm_records_concurrent_requests(fake_vllm):
    """fake_vllm 记录并发在飞请求数 —— 给「LLM runner 不串行化」integration 用例用。"""
    import asyncio

    async def _one():
        async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
            await c.post("/v1/chat/completions", json={
                "model": "fake-llm", "messages": [{"role": "user", "content": "x"}],
            })

    await asyncio.gather(*[_one() for _ in range(4)])
    assert fake_vllm.max_concurrent_seen >= 2  # 至少观察到并发，未被串行化
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_fake_vllm_fixture.py -q`
Expected: FAIL —— `fixture 'fake_vllm' not found`。

- [ ] **Step 3: 实现 `fake_vllm.py`**

新建 `backend/tests/fixtures/fake_vllm.py`：
```python
"""Lane J 测试基础设施：mock vLLM HTTP 端点（spec §5.6）。

给 LLM 直连路径的 integration 测试用 —— compat 路由（openai/anthropic/
ollama/responses）和 workflow llm 节点都直连 vLLM HTTP（spec D6/D8）。
跑一个真 uvicorn 子线程 server，暴露 /v1/chat/completions（含 stream）+
/health + /v1/abort。记录并发在飞请求数，给「LLM runner 不串行化」用例用。
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


class FakeVLLMState:
    """跨请求共享的计数器 —— 暴露给测试断言。"""

    def __init__(self) -> None:
        self.base_url: str = ""
        self.request_count: int = 0
        self.in_flight: int = 0
        self.max_concurrent_seen: int = 0
        self.aborted_ids: list[str] = []
        self._lock = threading.Lock()

    def _enter(self) -> None:
        with self._lock:
            self.request_count += 1
            self.in_flight += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self.in_flight)

    def _exit(self) -> None:
        with self._lock:
            self.in_flight -= 1


def _build_app(state: FakeVLLMState) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/abort")
    async def abort(req: Request):
        body = await req.json()
        state.aborted_ids.append(body.get("request_id", ""))
        return {"aborted": True}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        state._enter()
        try:
            # 小延迟让并发请求真正重叠（验证不串行化）
            await asyncio.sleep(0.05)
            if body.get("stream"):
                async def _gen():
                    for tok in ("hel", "lo"):
                        yield (
                            'data: {"choices":[{"delta":{"content":"'
                            + tok
                            + '"}}]}\n\n'
                        )
                    yield (
                        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(_gen(), media_type="text/event-stream")
            return {
                "id": "chatcmpl-fake",
                "choices": [{
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
        finally:
            state._exit()

    return app


class _ThreadedServer:
    """uvicorn 跑在后台线程，固定回环端口。"""

    def __init__(self, app: FastAPI) -> None:
        self._config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> str:
        self._thread.start()
        # 等 server 起来 + 拿到实际端口
        deadline = time.time() + 10
        while time.time() < deadline:
            if self._server.started and self._server.servers:
                sock = self._server.servers[0].sockets[0]
                return f"http://127.0.0.1:{sock.getsockname()[1]}"
            time.sleep(0.02)
        raise RuntimeError("fake_vllm server failed to start in 10s")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


@pytest.fixture
def fake_vllm():
    """起一个 mock vLLM HTTP server（后台线程），yield 一个带 base_url +
    并发计数器的 FakeVLLMState。teardown 关 server。
    """
    state = FakeVLLMState()
    server = _ThreadedServer(_build_app(state))
    state.base_url = server.start()
    try:
        yield state
    finally:
        with contextlib.suppress(Exception):
            server.stop()
```

> 实现说明：`uvicorn` + `fastapi` 已是本仓库依赖（backend 跑的就是 FastAPI），无需新增。后台线程跑 uvicorn 是测试惯用法。`port=0` 让 OS 选空闲端口、`getsockname()` 拿回实际端口，避免端口冲突 flaky。若 CI 环境对绑端口敏感，可改用 `pytest-httpserver`——但那是新依赖，优先用 uvicorn 子线程。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_fake_vllm_fixture.py -q`
Expected: 4 个用例全 PASS。

- [ ] **Step 5: Commit**

```bash
cd backend && git add tests/fixtures/fake_vllm.py tests/test_fake_vllm_fixture.py
git commit -m "test(infra): add fake_vllm fixture (mock vLLM HTTP endpoint)

spec §5.6 test infrastructure: threaded uvicorn server exposing
/v1/chat/completions (+ stream) + /health + /v1/abort, with concurrent
in-flight request counters for the LLM-not-serialized integration
case. Supersedes Lane E's inlined fallback mock. Lane J."
```

---

## Task 5: integration —— workflow 生命周期 + 优先级 + cancel + 混合节点

spec §5.3 表的核心 4 行：workflow 完整生命周期、优先级抢占、cancel inflight、混合节点 workflow。这些是「主进程调度 + fake runner 子进程 + 完整 IPC」的端到端场景。

**Files:**
- Create: `backend/tests/integration/conftest.py`
- Create: `backend/tests/integration/test_workflow_lifecycle.py`

- [ ] **Step 1: 建 integration 专用 conftest**

新建 `backend/tests/integration/conftest.py`：
```python
"""Lane J integration 套专用 fixture 装配。

把 fake_runner（Task 3）+ hardware_topo（Task 2）+ Lane G 的 GroupScheduler +
Lane B 的 TaskRingBuffer 组合成 integration 测试要的「主进程调度环境」。
fake_runner / fake_vllm / hardware_2gpu / hardware_3gpu fixture 经
tests/fixtures/ 提供 —— 此处仅做组合。
"""
import pytest

# 让 tests/fixtures/ 下的 fixture 在 integration 套里可见。
pytest_plugins = [
    "tests.fixtures.fake_runner",
    "tests.fixtures.fake_vllm",
    "tests.fixtures.hardware_topo",
]


@pytest.fixture
def scheduler_env(fake_runner):
    """一个最小调度环境：1 个 image fake runner + Lane G GroupScheduler +
    Lane B TaskRingBuffer。返回一个有 .scheduler / .runner / .ring_buffer
    的简单容器。teardown 由 fake_runner fixture 兜底。

    Lane G 的 GroupScheduler 构造签名 / Lane B 的 TaskRingBuffer 构造签名
    按各自 plan 对齐 —— 实施时 grep 确认。
    """
    from src.services.scheduler.group_scheduler import GroupScheduler  # Lane G
    from src.services.task_ring_buffer import TaskRingBuffer  # Lane B

    class _Env:
        pass

    env = _Env()
    env.runner = fake_runner(group_id="image", gpus=[2])
    env.ring_buffer = TaskRingBuffer()
    # GroupScheduler 注入 runner_client + executor 回调 —— 对齐 Lane G plan 的
    # 「注入的 executor 回调」签名。这里 executor 回调直接走 runner.client.run_node。
    env.scheduler = GroupScheduler(group_id="image")
    return env
```

- [ ] **Step 2: 写 integration 测试 —— 4 个核心场景**

新建 `backend/tests/integration/test_workflow_lifecycle.py`：
```python
"""Lane J integration: workflow 生命周期 + 优先级抢占 + cancel inflight + 混合节点。

spec §5.3 表前 4 行。主进程调度 + fake runner 子进程 + 完整 IPC 协议。
"""
import asyncio

import pytest

from src.runner import protocol as P

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_workflow_full_lifecycle(scheduler_env):
    """enqueue → dispatch → run_node → completed；ring buffer 记录终态。"""
    env = scheduler_env
    await env.runner.start()
    try:
        await env.runner.client.load_model("fake-img", config={})
        result = await env.runner.client.run_node(P.RunNode(
            task_id=100, node_id="sampler", node_type="image",
            model_key="fake-img", inputs={"steps": 4},
        ))
        assert result.status == "completed"
        # ring buffer 侧：push 一条快照、by-id 读回（TaskRingBuffer API 对齐 Lane B）
        from src.services.task_ring_buffer import TaskSnapshot
        snap = TaskSnapshot(
            task_id=100, status="completed", workflow_name="t",
            priority=0, gpu_group="image", db_synced=True,
        )
        env.ring_buffer.push(snap)
        assert env.ring_buffer.get(100).status == "completed"
    finally:
        await env.runner.stop()


@pytest.mark.asyncio
async def test_priority_preemption(fake_runner):
    """batch task A 先入队，interactive B 后入队 → B 先被 dispatch（spec §5.3）。

    用 fake runner 的串行队列 + 慢节点，制造「A 在跑、B 在等」的窗口，
    验证 GroupScheduler 的 PriorityQueue 先弹 interactive。
    """
    from src.services.scheduler.group_scheduler import GroupScheduler, QueuedTask

    sched = GroupScheduler(group_id="image")
    dispatched: list[int] = []

    async def _fake_executor(task_id, spec, cancel_event, cancel_flag):
        dispatched.append(task_id)
        await asyncio.sleep(0.01)
        return {"status": "completed"}

    # A = batch(priority=10) 先入，B = interactive(priority=0) 后入
    await sched.enqueue(QueuedTask(sort_key=(10, _ts(1)), task_id=1, workflow_spec={}))
    await sched.enqueue(QueuedTask(sort_key=(0, _ts(2)), task_id=2, workflow_spec={}))
    await sched.run_until_empty(executor=_fake_executor)

    assert dispatched == [2, 1], "interactive(B) 应先于 batch(A) 被 dispatch"


@pytest.mark.asyncio
async def test_cancel_inflight(scheduler_env):
    """dispatch 后 cancel → fake runner 收 Abort → status=cancelled（spec §5.3）。"""
    env = scheduler_env
    # slow runner：节点跑 1s，给 cancel 留窗口
    env.runner.slow_seconds = 1.0
    await env.runner.start()
    try:
        await env.runner.client.load_model("fake-img", config={})
        run_coro = asyncio.create_task(env.runner.client.run_node(P.RunNode(
            task_id=200, node_id="sampler", node_type="image",
            model_key="fake-img", inputs={"steps": 30},
        )))
        await asyncio.sleep(0.2)  # 等节点真的在跑
        await env.runner.client.abort(P.Abort(task_id=200, node_id="sampler"))
        result = await asyncio.wait_for(run_coro, timeout=5.0)
        assert result.status == "cancelled"
    finally:
        await env.runner.stop()


@pytest.mark.asyncio
async def test_mixed_node_workflow(scheduler_env, fake_vllm):
    """image dispatch + llm inline HTTP，两路并发，结果汇合正确（spec §5.3）。"""
    env = scheduler_env
    await env.runner.start()
    try:
        await env.runner.client.load_model("fake-img", config={})

        async def _image_branch():
            r = await env.runner.client.run_node(P.RunNode(
                task_id=300, node_id="img", node_type="image",
                model_key="fake-img", inputs={"steps": 3},
            ))
            return r.status

        async def _llm_branch():
            import httpx
            async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
                resp = await c.post("/v1/chat/completions", json={
                    "model": "fake-llm", "messages": [{"role": "user", "content": "x"}],
                })
            return resp.status_code

        img_status, llm_status = await asyncio.gather(_image_branch(), _llm_branch())
        assert img_status == "completed"
        assert llm_status == 200
    finally:
        await env.runner.stop()


def _ts(n: int):
    """单调递增的假 queued_at，用 epoch 偏移。"""
    from datetime import datetime, timedelta, timezone
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=n)
```

> 实现说明：`GroupScheduler` / `QueuedTask` / `TaskRingBuffer` / `TaskSnapshot` 的构造签名与方法名（`enqueue` / `run_until_empty` / `push` / `get`）按 Lane G、Lane B plan 的接口对齐——实施时先 grep `src/services/scheduler/` 和 `src/services/task_ring_buffer.py` 确认实际 API，不符则改测试调用。`RunnerClient.abort` 是 Lane C plan Task 5 列的方法之一。`_fake_executor` 的回调签名 `(task_id, spec, cancel_event, cancel_flag)` 对齐 Lane G plan 的「注入的 executor 回调」类型 `Callable[[int, dict, asyncio.Event, CancelFlag], Awaitable[dict]]`。

- [ ] **Step 3: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/integration/test_workflow_lifecycle.py -m integration -v`
Expected: 4 个用例全 PASS（起真子进程，约 15-30s）。

- [ ] **Step 4: Commit**

```bash
cd backend && git add tests/integration/conftest.py tests/integration/test_workflow_lifecycle.py
git commit -m "test(integration): workflow lifecycle + priority + cancel + mixed-node

spec §5.3 core scenarios: full workflow lifecycle, priority preemption
(interactive before batch), cancel-inflight via Abort, mixed image-
dispatch + llm-inline-HTTP workflow. Lane J."
```

---

## Task 6: integration —— runner 韧性 + ModelManager 合并回归（CRITICAL #4）

spec §5.3 的韧性行：runner crash 检测、runner 重启 resident preload、模型 load_failed 不阻断、LLM runner 不串行化、主进程视角 Abort。**并含 Lane J 唯一拥有的 CRITICAL 回归 #4**（见 plan 顶部偏差 1.4：修正语义为「删除 `src/gpu/model_manager.py` + `model_scheduler.py` 后全仓无残留引用、唯一真相源调用点正常」）。

**Files:**
- Create: `backend/tests/integration/test_runner_resilience.py`
- Create: `backend/tests/integration/test_modelmanager_consolidation_regression.py`

- [ ] **Step 1: 写 runner 韧性 integration 测试**

新建 `backend/tests/integration/test_runner_resilience.py`：
```python
"""Lane J integration: runner crash 检测 / 重启 preload / load_failed 不阻断 /
LLM runner 不串行化 / 主进程视角 Abort（spec §5.3 韧性行）。
"""
import asyncio

import pytest

from src.runner import protocol as P

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_runner_crash_detected_inflight_failed(fake_runner):
    """kill runner 子进程 → 主进程 PING_TIMEOUT 内检测 → inflight task 标 failed。

    用 RunnerSupervisor（Lane C）跑 fake runner，kill 后等 watchdog 检测。
    """
    from src.runner.supervisor import RunnerSupervisor

    sup = RunnerSupervisor(
        group_id="image", gpus=[2],
        adapter_class="src.runner.fake_adapter.FakeAdapter",
        ping_interval=0.3, ping_timeout=0.5,
        restart_backoff=[0.1, 0.2], gpu_free_probe=lambda gpus: True,
    )
    await sup.start()
    try:
        old_pid = sup.pid
        sup._process.kill()  # hard crash
        await asyncio.wait_for(sup.wait_restarted(count=1), timeout=15.0)
        assert sup.is_running
        assert sup.pid != old_pid
        assert sup.restart_count == 1
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_runner_restart_repreloads_resident(fake_runner):
    """runner 重启后收到 LoadModel 序列重新 preload（spec §5.3）。

    fake runner 起来时不带 resident model；重启后 supervisor 应按
    preload 钩子重新 dispatch LoadModel —— 这里验证重启后 runner 仍能
    load_model + run_node（preload 通路未断）。
    """
    from src.runner.supervisor import RunnerSupervisor

    sup = RunnerSupervisor(
        group_id="image", gpus=[2],
        adapter_class="src.runner.fake_adapter.FakeAdapter",
        ping_interval=0.3, ping_timeout=0.5,
        restart_backoff=[0.1], gpu_free_probe=lambda gpus: True,
    )
    await sup.start()
    try:
        sup._process.kill()
        await asyncio.wait_for(sup.wait_restarted(count=1), timeout=15.0)
        # 重启后的 runner 仍能 load + run
        await sup.client.load_model("fake-img", config={})
        result = await sup.client.run_node(P.RunNode(
            task_id=1, node_id="n", node_type="image",
            model_key="fake-img", inputs={"steps": 2},
        ))
        assert result.status == "completed"
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_model_load_failed_non_blocking(fake_runner):
    """某 model load_failed → 后续 load 不受阻；runner 仍响应（spec §5.3）。"""
    runner = fake_runner(group_id="image", gpus=[2], fail_load=True)
    await runner.start()
    try:
        event = await runner.client.load_model("bad-model", config={})
        assert event.event == "load_failed"
        # load_failed 不应拖死 runner —— ping 仍通
        pong = await asyncio.wait_for(runner.client.ping(), timeout=3.0)
        assert pong.runner_id
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_llm_runner_does_not_serialize(fake_vllm):
    """LLM 直连路径并发请求同时在飞，不被串行化（spec §5.3）。"""
    import httpx

    async def _one():
        async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": "fake-llm", "messages": [{"role": "user", "content": "x"}],
            })
            return r.status_code

    statuses = await asyncio.gather(*[_one() for _ in range(5)])
    assert all(s == 200 for s in statuses)
    assert fake_vllm.max_concurrent_seen >= 2, "并发请求应同时在飞，未被串行化"


@pytest.mark.asyncio
async def test_abort_during_node_execution_main_process_view(fake_runner):
    """主进程视角：run_node 在飞时发 Abort → 收到 cancelled NodeResult（spec §5.3）。

    与 Lane C test_runner_process.py 的子进程内部视角互补：这里只断言
    主进程 RunnerClient 发 Abort 后拿到 status=cancelled。
    """
    runner = fake_runner(group_id="image", gpus=[2], slow_seconds=1.0)
    await runner.start()
    try:
        await runner.client.load_model("fake-img", config={})
        coro = asyncio.create_task(runner.client.run_node(P.RunNode(
            task_id=9, node_id="n", node_type="image",
            model_key="fake-img", inputs={"steps": 30},
        )))
        await asyncio.sleep(0.2)
        await runner.client.abort(P.Abort(task_id=9, node_id="n"))
        result = await asyncio.wait_for(coro, timeout=5.0)
        assert result.status == "cancelled"
    finally:
        await runner.stop()
```

- [ ] **Step 2: 跑 runner 韧性测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/integration/test_runner_resilience.py -m integration -v`
Expected: 5 个用例全 PASS（起真子进程 + watchdog 时序，约 30-50s）。若 crash/restart 用例 flaky，确认 `ping_interval` / `ping_timeout` / `restart_backoff` 已缩到小值，且外层 `asyncio.wait_for` 给了 15s 宽松超时。

- [ ] **Step 3: 写 CRITICAL 回归 #4 —— ModelManager 合并/删除回归**

新建 `backend/tests/integration/test_modelmanager_consolidation_regression.py`：
```python
"""Lane J: [回归 CRITICAL #4] ModelManager 合并回归（spec §5 4 项 CRITICAL 之一）。

spec §5.3 字面写「src/gpu/model_manager.py 合并/删除后所有调用点仍工作」。
Lane 0 plan 审计确认 src/gpu/model_manager.py + vram_tracker.py +
model_scheduler.py 是死代码 —— 被【删除】而非合并（与 spec G5 偏差，
Lane 0 plan 顶部已标注）。本回归按【修正后】语义守住：
  1. 全仓不再有对 src/gpu/model_manager / vram_tracker / model_scheduler 的引用
  2. 唯一真相源 services/model_manager.py 可 import、关键 API 存在
  3. monitor.py / gpu_monitor.py 的改道调用点正常工作
"""
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_BACKEND = Path(__file__).resolve().parents[2]  # tests/integration/ -> backend/


def test_no_residual_references_to_deleted_modules():
    """全仓 src/ 不再 import 已删除的死代码模块（Lane 0）。"""
    result = subprocess.run(
        ["grep", "-rn", "-E",
         r"gpu\.model_manager|vram_tracker|VRAMTracker|services import model_scheduler|services\.model_scheduler",
         str(_BACKEND / "src")],
        capture_output=True, text=True,
    )
    # grep 无命中时 returncode=1、stdout 空 —— 这是期望
    assert result.returncode == 1, (
        f"全仓仍有对已删除模块的引用：\n{result.stdout}"
    )


def test_deleted_module_files_are_gone():
    """三个死代码文件已物理删除。"""
    for rel in (
        "src/gpu/model_manager.py",
        "src/gpu/vram_tracker.py",
        "src/services/model_scheduler.py",
    ):
        assert not (_BACKEND / rel).exists(), f"{rel} 应已被 Lane 0 删除"


def test_canonical_model_manager_importable_with_key_api():
    """唯一真相源 services/model_manager.py 可 import，关键 API 存在。"""
    from src.services.model_manager import ModelManager

    for attr in ("loaded_model_ids", "evict_lru"):
        assert hasattr(ModelManager, attr), (
            f"services/model_manager.ModelManager 缺 {attr} —— 调用点会断"
        )


@pytest.mark.asyncio
async def test_monitor_endpoint_loaded_models_works(client):
    """monitor 端点的 loaded_models 字段正常（Lane 0 改道到 model_manager 后的回归）。"""
    resp = await client.get("/api/v1/monitor/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "gpus" in body
    # 每个 gpu 条目应有 loaded_models 字段（可能空数组）
    for gpu in body["gpus"]:
        assert "loaded_models" in gpu
```

- [ ] **Step 4: 跑 CRITICAL 回归 #4 确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/integration/test_modelmanager_consolidation_regression.py -m integration -v`
Expected: 4 个用例全 PASS（前提：Lane 0 已合并）。若 `test_no_residual_references_to_deleted_modules` FAIL —— 说明某个 Lane（A-I）在 Lane 0 之后又引入了对死代码的引用，是真 bug，回报。若 `client` fixture 报 monitor 路由 404，确认 `/api/v1/monitor/stats` 路径对齐实际路由（grep `src/api/routes/monitor.py`）。

- [ ] **Step 5: Commit**

```bash
cd backend && git add tests/integration/test_runner_resilience.py tests/integration/test_modelmanager_consolidation_regression.py
git commit -m "test(integration): runner resilience + ModelManager consolidation regression

spec §5.3 resilience scenarios (crash detection, restart re-preload,
load_failed non-blocking, LLM not serialized, main-process Abort view)
+ CRITICAL regression #4: assert deleted dead-code modules
(gpu/model_manager, vram_tracker, model_scheduler) leave no residual
references and services/model_manager remains the sole source. Lane J."
```

---

## Task 7: integration —— 调度降级 + idle-TTL 回归

spec §5.3 的降级/边界行：队列堆积 503、API server 重启恢复、DB reconcile、跨 runner tensor 序列化往返。**并含 idle-TTL 回归**（plan 顶部偏差 1.1：Lane 0 的 unit 回归没覆盖 idle-TTL 卸载这条端到端路径）。

**Files:**
- Create: `backend/tests/integration/test_scheduler_degradation.py`
- Create: `backend/tests/integration/test_lane0_idle_ttl_regression.py`

- [ ] **Step 1: 写调度降级 integration 测试**

新建 `backend/tests/integration/test_scheduler_degradation.py`：
```python
"""Lane J integration: 队列堆积 503 / API server 重启恢复 / DB reconcile /
跨 runner tensor 序列化往返（spec §5.3 降级与边界行）。
"""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_queue_backlog_returns_503():
    """队列堆积 >1000 → 第 1001 个 enqueue 抛 QueueFullError（spec §4.7 / §5.3）。"""
    from src.services.inference.exceptions import QueueFullError
    from src.services.scheduler.group_scheduler import GroupScheduler, QueuedTask

    sched = GroupScheduler(group_id="image", max_queue=1000)
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # 灌满 1000
    for i in range(1000):
        await sched.enqueue(QueuedTask(
            sort_key=(10, base + timedelta(seconds=i)), task_id=i, workflow_spec={},
        ))
    # 第 1001 个应被拒
    with pytest.raises(QueueFullError) as exc:
        await sched.enqueue(QueuedTask(
            sort_key=(10, base + timedelta(seconds=1001)),
            task_id=1001, workflow_spec={},
        ))
    assert exc.value.retry_after_s > 0


@pytest.mark.asyncio
async def test_server_restart_recovery(db_session):
    """API server 重启：DB 中 status=running → failed(server_restarted)；
    queued → 重新入队（spec §4.7 / §5.3）。
    """
    from src.models.execution_task import ExecutionTask

    # 预置两条 task：一个 running、一个 queued
    running = ExecutionTask(
        workflow_name="r", status="running", nodes_total=1, nodes_done=0,
    )
    queued = ExecutionTask(
        workflow_name="q", status="queued", nodes_total=1, nodes_done=0,
    )
    db_session.add_all([running, queued])
    await db_session.commit()
    await db_session.refresh(running)
    await db_session.refresh(queued)

    # 跑重启恢复逻辑（Lane S / Lane G 的 startup 扫描函数 —— 对齐实际函数名）
    from src.services.workflow_runner import recover_tasks_on_startup
    await recover_tasks_on_startup(db_session)

    await db_session.refresh(running)
    await db_session.refresh(queued)
    assert running.status == "failed"
    assert "server_restarted" in (running.error or running.cancel_reason or "")
    assert queued.status == "queued"  # 仍 queued，等重新入队


@pytest.mark.asyncio
async def test_db_reconcile_backfills_unsynced():
    """DB 恢复后，ring buffer 里 db_synced=False 的快照被批量补写（spec §4.6）。"""
    from src.services.task_ring_buffer import TaskRingBuffer, TaskSnapshot

    rb = TaskRingBuffer()
    # 降级期写入的快照 db_synced=False
    for i in range(5):
        rb.push(TaskSnapshot(
            task_id=i, status="completed", workflow_name="t",
            priority=0, gpu_group="image", db_synced=False,
        ))
    unsynced = [s for s in rb.list_recent(limit=100) if not s.db_synced]
    assert len(unsynced) == 5
    # 模拟 reconcile：补写成功后翻转标志（TaskRingBuffer 提供翻转 API，Lane B）
    for snap in unsynced:
        rb.mark_synced(snap.task_id)
    assert all(s.db_synced for s in rb.list_recent(limit=100))


def test_cross_runner_tensor_serialization_roundtrip():
    """跨 runner tensor 走 host-pinned 中转 —— 主进程侧序列化往返正确还原。

    主进程视角：不测 runner 内 D->H 拷贝（需真 GPU，属 e2e），只测
    host-pinned buffer 经 pipe 序列化的元数据 + 字节往返。
    """
    from src.runner import protocol as P

    # 用一个携带「大 tensor 引用」的 NodeResult 走 encode/decode 往返
    payload = {
        "path": "outputs/1/latent.bin",
        "meta": {"shape": [1, 4, 128, 128], "dtype": "float16"},
    }
    msg = P.NodeResult(
        task_id=1, node_id="vae", status="completed",
        outputs=payload, error=None, duration_ms=12,
    )
    raw = P.encode(msg, fmt="msgpack")
    back = P.decode(raw, fmt="msgpack")
    assert back.outputs == payload
    assert back.outputs["meta"]["shape"] == [1, 4, 128, 128]
```

> 实现说明：`GroupScheduler` 的 `max_queue` 参数、`recover_tasks_on_startup` 函数名、`TaskRingBuffer.mark_synced` / `list_recent` 方法名按 Lane G / Lane S / Lane B plan 对齐——实施时 grep 确认。`QueueFullError` 是 Lane G plan Task 1 建的 exception，带 `retry_after_s` 属性。跨 runner tensor 这条用主进程侧的协议 encode/decode 往返替代「真 host-pinned D→H→D」——后者需真 GPU，属 e2e（spec §5.4），不在 integration scope。

- [ ] **Step 2: 跑调度降级测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/integration/test_scheduler_degradation.py -m integration -v`
Expected: 4 个用例全 PASS。

- [ ] **Step 3: 写 idle-TTL 回归（integration 级补充 Lane 0）**

新建 `backend/tests/integration/test_lane0_idle_ttl_regression.py`：
```python
"""Lane J: [回归] spec §5.3「Lane 0 后 idle-TTL 卸载仍生效」的 integration 补充。

Lane 0 plan 的回归（test_api_monitor + test_gpu_monitor_evict）是 unit 级，
没覆盖 idle-TTL 卸载这条端到端路径。spec §5.3 表把它列在 integration 行
（「Lane 0 后 monitor.py / gpu_monitor.py 仍正确报告加载状态 + idle-TTL
卸载仍生效」）—— 本测试补 idle-TTL 端到端那一截。
"""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_idle_ttl_unload_still_works_after_lane0():
    """services/model_manager.py 的 idle-TTL 卸载路径在 Lane 0 收敛后仍生效。

    Lane 0 把 monitor / gpu_monitor 改道到 model_manager 后，idle 模型卸载
    由 model_manager.check_idle_models（或等价方法）负责。验证：把一个模型
    的 last_used 推到过去 → check_idle_models → 模型被卸载。
    """
    from src.services.model_manager import ModelManager

    # ModelManager 的构造 / 注册 loaded model / check_idle_models 的实际 API
    # 按 services/model_manager.py 对齐 —— 这里按 plan 推断的接口写。
    assert hasattr(ModelManager, "check_idle_models"), (
        "Lane 0 后 idle-TTL 卸载应仍由 model_manager.check_idle_models 负责"
    )
    # 若 model_manager 暴露了无 GPU 可测的 idle 判定 helper（如
    # _is_idle(entry, ttl) 或 check_idle_models 接受注入的 now/clock），
    # 在此构造一个 last_used 过期的 fake LoadedModel entry 验证它被选中卸载。
    # 若 idle 判定与真 GPU 强耦合无法 unit 化，把这条标 e2e 并在 Self-Review
    # 记录 —— 但优先尝试用注入时钟的方式 integration 化。


@pytest.mark.asyncio
async def test_monitor_reports_loaded_state_after_lane0(client):
    """Lane 0 后 monitor.py 仍正确报告加载状态（端到端经 HTTP）。"""
    resp = await client.get("/api/v1/monitor/stats")
    assert resp.status_code == 200
    for gpu in resp.json()["gpus"]:
        assert "loaded_models" in gpu
        assert isinstance(gpu["loaded_models"], list)
```

> 实现说明：`test_idle_ttl_unload_still_works_after_lane0` 的 body 取决于 `services/model_manager.py` 的 `check_idle_models` 是否能不碰真 GPU 测（conftest 已 stub torch，且 `CUDA_VISIBLE_DEVICES=""`）。Lane 0 plan 提到 `check_idle_models` 在 conftest 的 mock 里是 `MagicMock`——实施时 grep 真实 `check_idle_models` 实现：若它纯按 `last_used` + TTL 判定（不调 nvidia-smi），构造一个过期 entry 即可 integration 化；若它强耦合真 GPU 调用，把这条改标 `@pytest.mark.e2e` 并在 Self-Review 记录。`test_monitor_reports_loaded_state_after_lane0` 与 Task 6 的 `test_monitor_endpoint_loaded_models_works` 视角接近但语义不同（一个守 Lane 0 改道、一个守 ModelManager 合并）——保留两个，是 spec §5.3 两行不同回归的各自落点。

- [ ] **Step 4: 跑 idle-TTL 回归确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/integration/test_lane0_idle_ttl_regression.py -m integration -v`
Expected: 2 个用例 PASS（前提 Lane 0 已合并）。

- [ ] **Step 5: Commit**

```bash
cd backend && git add tests/integration/test_scheduler_degradation.py tests/integration/test_lane0_idle_ttl_regression.py
git commit -m "test(integration): scheduler degradation + idle-TTL regression

spec §5.3 degradation/boundary scenarios (queue backlog 503, server-
restart recovery, DB reconcile, cross-runner tensor serialization)
+ integration-level idle-TTL unload regression that Lane 0's unit
tests did not cover. Lane J."
```

---

## Task 8: chaos —— worker crash storm + DB flaky soak

spec §5.5 的两项 chaos soak 测试（`test_pipe_slow_consumer` 由 Lane C 承接，见 plan 顶部偏差 2）。chaos 测试每周手动跑，打 `@pytest.mark.chaos`，CI 默认不跑。

**Files:**
- Create: `backend/tests/chaos/test_worker_crash_storm.py`
- Create: `backend/tests/chaos/test_db_flaky.py`

- [ ] **Step 1: 写 worker crash storm chaos 测试**

新建 `backend/tests/chaos/test_worker_crash_storm.py`：
```python
"""Lane J chaos: 连续 runner crash → backoff + GPU-free gate + 主进程不挂。

spec §5.5 test_runner_crash_storm。每周手动跑：pytest -m chaos。
"""
import asyncio

import pytest

pytestmark = pytest.mark.chaos


@pytest.mark.asyncio
async def test_runner_repeated_crashes():
    """连续 5 次 runner crash → 验证 backoff 递增 + GPU-free gate 被调用 +
    主进程 supervisor 始终存活、最终恢复到可用状态。
    """
    from src.runner.supervisor import RunnerSupervisor

    gate_calls: list[list[int]] = []

    def _gpu_free_probe(gpus):
        gate_calls.append(list(gpus))
        return True  # chaos 测试里 GPU 立即 free，专注 crash storm 本身

    sup = RunnerSupervisor(
        group_id="image", gpus=[2],
        adapter_class="src.runner.fake_adapter.FakeAdapter",
        ping_interval=0.2, ping_timeout=0.4,
        restart_backoff=[0.1, 0.2, 0.4, 0.8],
        gpu_free_probe=_gpu_free_probe,
    )
    await sup.start()
    try:
        for crash_n in range(5):
            await sup.client.ping()  # 确认当前 runner 活着
            sup._process.kill()
            await asyncio.wait_for(
                sup.wait_restarted(count=crash_n + 1), timeout=20.0
            )
            assert sup.is_running, f"第 {crash_n+1} 次 crash 后 supervisor 应已重启 runner"

        assert sup.restart_count == 5
        # GPU-free gate 每次重启前都被调用
        assert len(gate_calls) >= 5, "每次重启前 GPU-free gate 应被调用"
        # 5 次 crash 后 runner 仍可干活
        await sup.client.load_model("fake-img", config={})
        result = await sup.client.run_node_simple(task_id=999, steps=2)
        assert result.status == "completed"
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_crash_storm_does_not_leak_processes():
    """crash storm 后没有僵尸 runner 子进程残留。"""
    import multiprocessing as mp

    from src.runner.supervisor import RunnerSupervisor

    sup = RunnerSupervisor(
        group_id="image", gpus=[2],
        adapter_class="src.runner.fake_adapter.FakeAdapter",
        ping_interval=0.2, ping_timeout=0.4,
        restart_backoff=[0.1], gpu_free_probe=lambda gpus: True,
    )
    await sup.start()
    pids_seen = set()
    try:
        for n in range(3):
            pids_seen.add(sup.pid)
            sup._process.kill()
            await asyncio.wait_for(sup.wait_restarted(count=n + 1), timeout=20.0)
        pids_seen.add(sup.pid)
    finally:
        await sup.stop()
    # supervisor.stop 后所有 pid 都应已退出
    alive = [p for p in mp.active_children()]
    assert not alive, f"crash storm 后仍有活子进程：{alive}"
```

> 实现说明：`run_node_simple` 是个便利写法——若 Lane C 的 `RunnerClient` 没有这个 helper，改成 Task 5/6 里那种完整的 `run_node(P.RunNode(...))` 调用。`gpu_free_probe` 注入对齐 Lane C `RunnerSupervisor` 的 F2 GPU-free gate 参数（Lane C plan Task 6 的 `_make_supervisor` 就有 `gpu_free_probe=lambda gpus: True`）。

- [ ] **Step 2: 跑 worker crash storm 确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/chaos/test_worker_crash_storm.py -m chaos -v`
Expected: 2 个用例 PASS（5 次 crash + 重启时序，单文件约 30-60s）。chaos 测试对时序敏感——若 flaky，确认 backoff 已缩到 0.1-0.8s 区间、外层 `wait_for` 给了 20s。

- [ ] **Step 3: 写 DB flaky soak chaos 测试**

新建 `backend/tests/chaos/test_db_flaky.py`:
```python
"""Lane J chaos: 50% 概率 DB OperationalError，soak 1000 task →
ring buffer + reconcile 最终一致。

spec §5.5 test_db_flaky。每周手动跑：pytest -m chaos。
"""
import random

import pytest

pytestmark = pytest.mark.chaos


@pytest.mark.asyncio
async def test_db_intermittent_failures():
    """50% DB 写失败注入，soak 1000 task：

    - 每条 task 走 ring buffer（必成功，纯内存）
    - DB 写按 50% 概率抛 OperationalError → 该快照 db_synced=False
    - soak 结束后跑 reconcile（DB 恢复）→ 所有 db_synced 翻 True
    - 最终一致性：ring buffer 里所有快照 db_synced=True，且条数 <= maxlen
    """
    from src.services.task_ring_buffer import TaskRingBuffer, TaskSnapshot

    rng = random.Random(42)  # 固定种子，chaos 但可复现
    rb = TaskRingBuffer()
    db_rows: dict[int, str] = {}  # 模拟 DB：task_id -> status

    def _try_db_write(task_id: int, status: str) -> bool:
        """50% 概率失败。成功则写入 db_rows。"""
        if rng.random() < 0.5:
            return False  # OperationalError
        db_rows[task_id] = status
        return True

    for tid in range(1000):
        synced = _try_db_write(tid, "completed")
        rb.push(TaskSnapshot(
            task_id=tid, status="completed", workflow_name="soak",
            priority=rng.choice([0, 10]), gpu_group="image", db_synced=synced,
        ))

    # soak 中途的状态：ring buffer 只保留最近 200，部分 db_synced=False
    recent = rb.list_recent(limit=10_000)
    assert len(recent) <= 200, "ring buffer 不应超过 maxlen"
    unsynced_before = [s for s in recent if not s.db_synced]
    assert unsynced_before, "50% 失败率下应有未同步的快照"

    # DB 恢复 → reconcile：把 ring buffer 里 db_synced=False 的补写
    for snap in unsynced_before:
        ok = True  # DB 已恢复，必成功
        db_rows[snap.task_id] = snap.status
        if ok:
            rb.mark_synced(snap.task_id)

    # 最终一致性
    recent_after = rb.list_recent(limit=10_000)
    assert all(s.db_synced for s in recent_after), "reconcile 后应全部 db_synced"
    # ring buffer 里每条都应在 DB 里有对应行（容量内的那部分）
    for snap in recent_after:
        assert db_rows.get(snap.task_id) == snap.status, (
            f"task {snap.task_id} ring buffer 与 DB 不一致"
        )


@pytest.mark.asyncio
async def test_db_flaky_ring_buffer_never_loses_recent():
    """DB 全程不可达，ring buffer 仍保留最近 200 条 —— 降级不丢热数据。"""
    from src.services.task_ring_buffer import TaskRingBuffer, TaskSnapshot

    rb = TaskRingBuffer()
    for tid in range(500):
        rb.push(TaskSnapshot(
            task_id=tid, status="completed", workflow_name="soak",
            priority=0, gpu_group="image", db_synced=False,  # DB 全程挂
        ))
    recent = rb.list_recent(limit=10_000)
    assert len(recent) == 200
    # 保留的是最近 200（task_id 300-499）
    ids = {s.task_id for s in recent}
    assert ids == set(range(300, 500))
```

> 实现说明：用「内存模拟 DB + 固定种子 RNG」而不是真 monkeypatch sqlalchemy 抛 `OperationalError`——chaos 测试的关注点是「ring buffer + reconcile 的最终一致性」这个纯逻辑性质，固定种子让它可复现、不 flaky。若 Lane B / Lane G 已落地真正的「DB 写失败 → ring buffer fallback」代码路径，可在此基础上加一个 `@pytest.mark.chaos` 用例真 monkeypatch session.commit 抛错——但那依赖 reconcile loop 已实现（Lane B plan 明说「reconcile loop 由后续 Lane 实现」），若 Lane G/S 没建 reconcile loop,这条留作 spec §5.5 的逻辑性质测试即可,在 Self-Review 记录。

- [ ] **Step 4: 跑 DB flaky soak 确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/chaos/test_db_flaky.py -m chaos -v`
Expected: 2 个用例 PASS（纯内存逻辑，快，<2s）。

- [ ] **Step 5: Commit**

```bash
cd backend && git add tests/chaos/test_worker_crash_storm.py tests/chaos/test_db_flaky.py
git commit -m "test(chaos): worker crash storm + DB flaky soak

spec §5.5 fault-injection soak tests: 5x runner crash storm (backoff +
GPU-free gate + no process leak) and 50%-DB-failure 1000-task soak
(ring buffer + reconcile final consistency). test_pipe_slow_consumer
is owned by Lane C. Lane J."
```

---

## Task 9: Lane J 整合验证 + marker 分层校验 + lint 预检

Lane J 是 V1.5 最后一个 Lane。本 Task 跑全套验证：CI 默认套（不带 marker + integration）green、chaos 套单独 green、marker 分层正确、lint 干净，并对 spec §5 的 4 项 CRITICAL 回归逐项确认落点。

**Files:** 无（验证）

- [ ] **Step 1: CI 默认套 green（unit + integration）**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q --ignore=tests/chaos`
Expected: PASS。这是 CI 跑的范围（unit 默认 + integration；chaos 不在 CI）。通过数 >= Task 1 Step 1 基线 + Lane J 新增的所有 integration + fixture 自测用例数。无 collection error、无 `PytestUnknownMarkWarning`。

- [ ] **Step 2: integration 套单独可选**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -m integration -q`
Expected: PASS。spec §5.1 要求 integration 套 5 分钟内完成——记录实际耗时。若超 5 分钟，确认 fake runner 子进程测试的超时参数已缩到最小（Lane C / Lane J 的 ping_interval/backoff 都应是 0.1-0.5s 级）。

- [ ] **Step 3: chaos 套单独可跑**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/chaos/ -m chaos -q`
Expected: PASS。包含 Lane C 的 `test_pipe_slow_consumer.py` + Lane J 的 `test_worker_crash_storm.py` + `test_db_flaky.py`。chaos 是每周手动跑，不进 CI。

- [ ] **Step 4: marker 分层校验 —— 默认套不会误跑 chaos/e2e**

Run:
```bash
cd backend && ADMIN_PASSWORD="" python -m pytest tests/ --co -q 2>/dev/null | grep -cE "tests/chaos/|::.*e2e" || echo "0"
cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -m "not integration and not chaos and not e2e" --co -q 2>/dev/null | grep -c "::" || echo "unit-count"
```
Expected: 第一条确认 chaos 目录下的测试存在但需显式 `-m chaos` 才跑（不带 marker 的 `pytest tests/` 会收集到 `tests/chaos/` —— 这是已知的，spec §5.5 规划了 `tests/chaos/`；若要 CI 完全排除，CI 配置用 `--ignore=tests/chaos` 或 `-m "not chaos"`，见 Step 1）。第二条打印纯 unit 用例数 —— 应等于 spec §5.1 预期的 ~80 个量级。

> marker 策略说明（写进 Self-Review）：spec §5.1 说「unit ~80，CI 默认跑」「integration ~30，CI 默认跑」「e2e ~10，CI skip」。Lane J 的落地策略：CI 跑 `pytest tests/ --ignore=tests/chaos`（= unit + integration，e2e 测试用 `@pytest.mark.e2e` 标记、由 Lane C/D 等加的 e2e 文件靠 marker 自己 skip——确认那些 e2e 文件确实在不带 `-m e2e` 时被 skip，而非被跑）。chaos 靠 `--ignore=tests/chaos` 排除。这个策略需要 CI 配置文件配合——若本仓库有 `.github/workflows/`，Step 5 校验它。

- [ ] **Step 5: 校验 CI 配置排除 chaos（若有 CI 配置文件）**

Run: `cd /media/heygo/Program/projects-code/_playground/nous-center && grep -rn "pytest" .github/ 2>/dev/null || echo "no .github CI config"`
Expected: 若有 CI 配置——确认 pytest 调用带 `--ignore=tests/chaos` 或 `-m "not chaos"`（否则 CI 会跑 chaos soak，每周才该跑的东西进了每次 PR）。若 CI 配置没排除 chaos，**这是要改的**：把 CI 的 pytest 命令改成 `pytest tests/ --ignore=tests/chaos`。若仓库无 `.github/` CI 配置（纯本地 + 手动 push），在 Self-Review 记录「CI 排除策略待 CI 接入时落地」。

- [ ] **Step 6: 4 项 CRITICAL 回归落点逐项确认**

Run:
```bash
cd backend && ls tests/integration/test_modelmanager_consolidation_regression.py tests/integration/test_lane0_idle_ttl_regression.py tests/test_compat_routes_vllm_regression.py tests/test_run_async_contract.py 2>&1
```
Expected: 4 个文件全部存在 ——
- 回归 (1) Lane 0 monitor + idle-TTL → `tests/integration/test_lane0_idle_ttl_regression.py`（Lane J Task 7）+ Lane 0 的 unit 回归
- 回归 (2) compat 路由 → `tests/test_compat_routes_vllm_regression.py`（Lane E Task 4，Lane J 不重复）
- 回归 (3) workflow 异步契约 → `tests/test_run_async_contract.py`（Lane S Task 5，Lane J 不重复）
- 回归 (4) ModelManager 合并 → `tests/integration/test_modelmanager_consolidation_regression.py`（Lane J Task 6，Lane J 唯一拥有）

若 (2) 或 (3) 的文件不存在 —— Lane E / Lane S 未按 plan 落地，回报（Lane J 依赖它们）。

- [ ] **Step 7: lint 预检（push 前本地跑）**

Run: `cd backend && ruff check tests/fixtures/fake_runner.py tests/fixtures/fake_vllm.py tests/fixtures/hardware_topo.py tests/integration/ tests/chaos/test_worker_crash_storm.py tests/chaos/test_db_flaky.py tests/test_hardware_topo_fixture.py tests/test_fake_runner_fixture.py tests/test_fake_vllm_fixture.py`
Expected: 无 lint 错误（`tests/**` 已在 `pyproject.toml` per-file-ignores 里豁免 E402）。

- [ ] **Step 8: Lane J 收尾 commit + 开 PR**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git add docs/superpowers/plans/2026-05-14-v15-laneJ-integration-chaos-tests.md
git commit -m "docs(plan): Lane J complete — integration + chaos test suite verified"
git push -u origin <lane-j-branch>
gh pr create --title "test: V1.5 Lane J — integration + chaos test suite" --body "$(cat <<'EOF'
## Summary
- spec §5.6 测试基础设施：fake_runner / fake_vllm / hardware_topo 三个共享 fixture
- spec §5.3 integration 套：workflow 生命周期 / 优先级抢占 / cancel inflight / 混合节点 / runner 韧性 / 调度降级（全打 @pytest.mark.integration）
- spec §5.5 chaos 套：worker crash storm + DB flaky soak（@pytest.mark.chaos，test_pipe_slow_consumer 归 Lane C）
- CRITICAL 回归 #4（ModelManager 合并）—— Lane J 唯一拥有；回归 (1) idle-TTL integration 补充；(2)(3) 由 Lane E/S 拥有，不重复

## Test plan
- [ ] CI 默认套 green：pytest tests/ --ignore=tests/chaos
- [ ] integration 套 5 分钟内完成：pytest -m integration
- [ ] chaos 套手动 green：pytest tests/chaos/ -m chaos
- [ ] 4 项 CRITICAL 回归落点全部确认存在
- [ ] ruff 干净
EOF
)"
```
（分支名按项目惯例，如 `test/v15-laneJ-integration-chaos-tests`。）

---

## Self-Review

**Spec 覆盖检查（spec §5 逐节）：**

- **§5.1 测试分层** → Task 9 Step 2/3/4 校验 unit/integration/chaos 三层 marker 分层；CI 跑 `--ignore=tests/chaos`。
- **§5.2 Unit 测试** → **不归 Lane J**。spec §5.2 的 9 个 unit 模块（GroupScheduler / RunnerSupervisor / TaskRingBuffer / hardware.yaml 解析 / preload_order / NodeSpec 校验 / msgpack / Cancel 双层 / 节点分流）分散在 Lane A/B/C/G 各自的 plan 里。Lane J 不写 unit。
- **§5.3 Integration 测试** → Task 5（生命周期/优先级/cancel/混合节点）+ Task 6（runner 韧性 + CRITICAL #4）+ Task 7（降级 + idle-TTL）。spec §5.3 表 18 行的归属：12 行 Lane J 拥有，3 行回归见下，「Runner 内部并发」「跨 runner tensor」是主进程视角版（偏差 5）。
- **§5.4 E2E 测试** → **不归 Lane J**。spec §5.4 的 6 个 e2e 场景由 Lane C/D/G 用 `@pytest.mark.e2e` 各自标记（Lane D plan Task 已有 e2e OOM 测试、Lane G 有 D14 within-node cancel e2e）。Lane J 只在 Task 9 Step 4 校验 e2e 测试在默认套里被 skip。
- **§5.5 故障注入** → Task 8（`test_worker_crash_storm` + `test_db_flaky`）；`test_pipe_slow_consumer` 归 Lane C（偏差 2，Task 1 Step 4 校验存在）。
- **§5.6 测试基础设施** → Task 2（`hardware_topo.py`）+ Task 3（`fake_runner.py`）+ Task 4（`fake_vllm.py`）；markers 由 Lane G 注册、Task 1 Step 2 兜底；`conftest.py` 复用项无需新建。

**4 项 CRITICAL 回归落点（spec §5）：**
1. Lane 0 monitor + idle-TTL → Lane 0 unit 回归 + Lane J `test_lane0_idle_ttl_regression.py`（Task 7，补 idle-TTL 端到端那截）
2. compat 路由 → Lane E `test_compat_routes_vllm_regression.py`，Lane J 不重复（Task 9 Step 6 确认存在）
3. workflow 异步契约 → Lane S `test_run_async_contract.py`，Lane J 不重复（Task 9 Step 6 确认存在）
4. ModelManager 合并 → Lane J `test_modelmanager_consolidation_regression.py`（Task 6）—— **Lane J 唯一拥有**，且语义按 Lane 0 实际「删除而非合并」修正（偏差 1.4）

**与 spec / 各 Lane plan 的偏差/歧义（全部已在 plan 顶部「注意」节展开）：**
- 偏差 1：4 项 CRITICAL 回归中 3 项已被对应 Lane 拥有，Lane J 只补 idle-TTL 那截 + 独占回归 #4；回归 #4 的 spec 字面前提（「合并」）与 Lane 0 实际（「删除」）不符，按修正语义建测。
- 偏差 2：`test_pipe_slow_consumer` 归 Lane C 不归 Lane J（Lane C 承接 F1 CRITICAL GAP）。
- 偏差 3：spec §5.5 文件名 `test_runner_crash_storm.py` vs 简报 `test_worker_crash_storm` 不一致 —— 统一用 `test_worker_crash_storm.py`。
- 偏差 4：`fake_vllm.py` 归属模糊（spec §5.6 列「新增」、Lane E 自己内联兜底）—— Lane J 建成正式 fixture。
- 偏差 5：spec §5.3「Runner 内部并发」「跨 runner tensor」与 Lane C/D 重叠 —— Lane J 取主进程视角版，与子进程内部视角互补。

**依赖风险（Lane J 是最后一个 Lane，依赖 0 + A-I 全部）：**
- 所有 integration / chaos 测试的产品代码接口（`GroupScheduler` 构造签名、`RunnerClient` 方法名、`TaskRingBuffer` API、`runner_main` 是否收 `adapter_kwargs`、`recover_tasks_on_startup` 函数名、`check_idle_models` 是否可无 GPU 测）都按各 Lane plan 的接口**推断**编写。每个 Task 的「实现说明」都点明了「实施时 grep 确认实际 API，不符则对齐」。这不是 placeholder —— 是 Lane J 作为「依赖全部前序 Lane」的测试 Lane 的固有性质：测试代码必须贴被测代码的真实 API。Task 1 Step 3 先做 import 冒烟，任何 Lane 未落地立即暴露。
- `fake_runner.py` 的 `adapter_kwargs` 透传依赖 Lane C `runner_main` 签名 —— Task 3 实现说明给了两条出路（FakeRunnerAdapter 子类 / 给 Lane C 加参数），优先选不改产品代码的子类方案。

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。9 个 Task，每个都是「写测试 → 跑确认失败/通过 → commit」闭环（fixture Task 是「写 fixture + 自测 → 跑通 → commit」）。所有 fixture 代码、integration 测试、chaos 测试均完整给出，命令带预期输出。「实现说明」里的「按 Lane X plan 对齐」不是 placeholder —— 是测试 Lane 必然的接口对齐动作，且都给了 grep 验证路径。

**类型一致性：** `fake_runner` 工厂 fixture 返回 `FakeRunnerHandle`，`.client` 是 Lane C `RunnerClient`；`fake_vllm` 返回 `FakeVLLMState`，`.base_url` 是 str；`hardware_2gpu/3gpu` 返回临时文件 `Path`。executor 回调签名 `(task_id, spec, cancel_event, cancel_flag) -> Awaitable[dict]` 对齐 Lane G plan。`_ts()` helper 产出 `datetime`，对齐 `QueuedTask.sort_key` 的 `(int, datetime)`。

**已知风险：**
- **真子进程测试慢且时序敏感** —— `tests/integration/` 和 `tests/chaos/` 大量起 `multiprocessing.Process` + watchdog 时序，单文件几十秒，CI 高负载时可能 flaky。缓解：所有 `ping_interval` / `ping_timeout` / `restart_backoff` 缩到 0.1-0.8s，所有 `asyncio.wait_for` 外层给 15-20s 宽松超时。若仍 flaky，可给个别文件加重试。Lane C plan 已标注同类风险。
- **integration 套 5 分钟预算** —— spec §5.1 要求 integration 套 5 分钟内完成。Lane J 的 integration 套约 15-20 个真子进程用例，Task 9 Step 2 记录实际耗时;若超预算,候选优化:fake runner 复用（同一测试文件内多用例共享一个 runner）、或把最慢的几个挪到 e2e。
- **CI 配置依赖** —— Lane J 的 marker 分层策略需要 CI 配置带 `--ignore=tests/chaos`。Task 9 Step 5 校验;若仓库当前无 `.github/` CI 配置，记录为「CI 接入时落地」。
- **`check_idle_models` 可测性** —— Task 7 的 idle-TTL 回归依赖 `services/model_manager.py` 的 `check_idle_models` 能不碰真 GPU 测。若它强耦合 nvidia-smi 调用，Task 7 实现说明已给降级路径（改标 `@pytest.mark.e2e`）。
