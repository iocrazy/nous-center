# V1.5 Lane A: hardware.yaml + GPUAllocator 重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 引入 `hardware.yaml`（manual-only 的 GPU 拓扑唯一真相源，含 2gpu / 3gpu 两份示例），把 `GPUAllocator` 从「只会按 free_mb 挑单卡」重构为「NVLink-group 感知、按 yaml `groups[]` 动态产出 group 列表与 runner 数」，为 Lane C（RunnerSupervisor 按 group 起 runner）与 Lane E（vLLM 按 `role: llm` group 选卡）提供拓扑数据源。

**Architecture:** `hardware.yaml` 落在 `backend/configs/`，与 `models.yaml` 同目录，复用 `config.py` 已有的 `_resolve_path` + `yaml.safe_load` + `lru_cache` 加载模式（新增 `load_hardware_config`）。`GPUAllocator` 保留现有 `get_best_gpu` / `get_free_mb`（`model_manager.py:319` 与 `test_gpu_allocator.py` 仍依赖，不能破），**新增** group 层 API：构造时读 `hardware.yaml` → 解析出 `GPUGroup` 列表（`id` / `gpus` / `nvlink` / `role` / `vram_gb`），暴露 `groups()` / `group_for_role(role)` / `runner_count()` / `llm_group_gpus()` / `group_by_id(id)`。group 内挑卡仍走 nvidia-smi free_mb（NVLink group 是「一组卡」，runner 占整组，但 image 这类单卡 group 的卡选择仍按实测显存）。`main.py` 实例化 `GPUAllocator()` 处不变签名——allocator 自己加载 `hardware.yaml`，加载失败 fail-soft 退化到「按 `detect_gpus()` 每卡一个单卡 group」，保证 Pro 6000 到货前/yaml 缺失时 API server 仍能起。

**Tech Stack:** Python 3 / FastAPI / pytest / PyYAML / `dataclass`。无新依赖。

> **注意 — 与 spec 的偏差 / 判断点（已核实，须知会）：**
>
> 1. **`hardware.yaml` 文件名 vs 两份示例的关系，spec 没说清。** spec §3.2 给了 `hardware.2gpu.yaml` 和 `hardware.3gpu.yaml` 两份「示例」，又说 `hardware.yaml`「手写、唯一真相源」。实际部署需要一个**确定的加载路径**。本 plan 决策：`load_hardware_config` 读 `configs/hardware.yaml`；仓库内提交 `hardware.2gpu.yaml` + `hardware.3gpu.yaml` 两份带注释的示例，**外加** 一份 `hardware.yaml` = 当前 2 卡布局的实际生效副本（内容等同 `hardware.2gpu.yaml`）。Pro 6000 到货后运维手动 `cp hardware.3gpu.yaml hardware.yaml`。两份 `*.Ngpu.yaml` 是只读参考样板，不被代码加载。
> 2. **`_detect_vllm_gpus*` 的「退化为读 hardware.yaml」spec 放在 §3.2，但改 vLLM 选卡属于 Lane E 的改道面。** 本 plan 只交付 allocator 侧的只读 API `llm_group_gpus()`（返回 `role: llm` group 的 `gpus`），**不动** `model_manager.py` 里 `_detect_vllm_gpus` / `_detect_vllm_gpus_for_adapter` 的调用点——那是 Lane E（vLLM 生命周期 + 改道）的职责。Lane A 只保证「数据源就绪」。已在 Self-Review 标注。
> 3. **3gpu 示例里 GPU 1 在 spec 硬件表写「24 GB」，§3.2 yaml 写 `vram_gb: 48`。** §3.2 的 `vram_gb` 是**整个 group 的合计**（GPU0+GPU1 = 24+24 = 48），不是单卡。本 plan 的 `GPUGroup.vram_gb` 字段语义 = group 合计 VRAM，与 §3.2 yaml 一致。Task 1 的示例文件注释里写明这一点。

---

## File Structure

| 文件 | Lane A 动作 | 责任 |
|---|---|---|
| `backend/configs/hardware.yaml` | **新建** | 实际生效的 GPU 拓扑文件，内容 = 当前 2 卡布局（§1.4 方案 A） |
| `backend/configs/hardware.2gpu.yaml` | **新建** | 只读参考样板：当前 2×3090 布局，带注释 |
| `backend/configs/hardware.3gpu.yaml` | **新建** | 只读参考样板：Pro 6000 到货后 3-group 布局，带注释 |
| `backend/src/config.py` | **修改** | 新增 `load_hardware_config(path="configs/hardware.yaml")`，复用 `_resolve_path` + `lru_cache` |
| `backend/src/services/gpu_allocator.py` | **修改** | 新增 `GPUGroup` dataclass + group 层 API；保留 `get_best_gpu` / `get_free_mb` 不变 |
| `backend/tests/test_hardware_config.py` | **新建** | `load_hardware_config` 解析 2gpu / 3gpu / 缺失 / 损坏的覆盖 |
| `backend/tests/test_gpu_allocator.py` | **修改** | 追加 group 层 API 测试；现有 4 个 free_mb 测试保持不动 |

---

## Task 1: 新建 hardware.yaml 三份配置文件

`hardware.yaml` 是 manual-only 的拓扑真相源（spec §3.2：V1.5 不解析 `nvidia-smi topo -m`，无 `detection.mode` 字段）。本 task 只新建文件，无测试——文件内容由 Task 2 的解析测试验证。

**Files:**
- Create: `backend/configs/hardware.yaml`
- Create: `backend/configs/hardware.2gpu.yaml`
- Create: `backend/configs/hardware.3gpu.yaml`

- [ ] **Step 1: 确认 configs 目录与 models.yaml 同级**

Run:
```bash
cd backend && ls configs/
```
Expected: 输出含 `models.yaml`。`hardware.yaml` 将放在同目录。

- [ ] **Step 2: 写 `configs/hardware.2gpu.yaml`（参考样板，当前布局）**

新建 `backend/configs/hardware.2gpu.yaml`：
```yaml
# hardware.2gpu.yaml — 参考样板（只读，代码不加载此文件）
#
# 当前部署：2×3090 NVLink 配对（spec §1.4 方案 A）。
# Pro 6000 未到货，image/TTS 节点与 LLM 时分复用同一 group 队列。
#
# 生效方式：把本文件内容同步到 configs/hardware.yaml（代码只加载 hardware.yaml）。
#
# 字段语义：
#   id       group 唯一标识，runner / scheduler 用它寻址
#   gpus     该 group 占用的 GPU index 列表（cuda:N 的 N）
#   nvlink   组内 GPU 是否 NVLink 互联；vLLM tensor-parallel 模型强约束 nvlink:true
#   role     image / llm / tts —— 节点按 role 找 group
#   vram_gb  整个 group 的合计 VRAM（多卡 group = 各卡之和），不是单卡值
#
# V1.5 不解析 nvidia-smi topo -m，拓扑全靠本文件手写。
groups:
  - id: llm-tp
    gpus: [0, 1]
    nvlink: true
    role: llm            # image/TTS 节点也落此 group，与 LLM 时分复用
    vram_gb: 48
```

- [ ] **Step 3: 写 `configs/hardware.3gpu.yaml`（参考样板，未来布局）**

新建 `backend/configs/hardware.3gpu.yaml`：
```yaml
# hardware.3gpu.yaml — 参考样板（只读，代码不加载此文件）
#
# 未来部署：Pro 6000 到货后的 3-group 独立布局（spec §3.2）。
# image 独占 96GB Pro 6000；LLM 用 2×3090 NVLink pair 跑 tensor-parallel；
# TTS 独占一张 24GB 卡。
#
# 生效方式：Pro 6000 到货后 `cp hardware.3gpu.yaml hardware.yaml`。
#
# 字段语义见 hardware.2gpu.yaml 头注释。vram_gb = group 合计 VRAM。
groups:
  - id: image
    gpus: [2]
    nvlink: false
    role: image
    vram_gb: 96

  - id: llm-tp
    gpus: [0, 1]
    nvlink: true
    role: llm
    vram_gb: 48          # GPU0(24) + GPU1(24) 合计

  - id: tts
    gpus: [3]
    nvlink: false
    role: tts
    vram_gb: 24
```

- [ ] **Step 4: 写 `configs/hardware.yaml`（实际生效，= 当前 2 卡布局）**

新建 `backend/configs/hardware.yaml`：
```yaml
# hardware.yaml — 实际生效的 GPU 拓扑（代码加载此文件，唯一真相源）
#
# 当前 = 2×3090 NVLink 配对布局，内容等同 hardware.2gpu.yaml。
# 切换布局：cp configs/hardware.3gpu.yaml configs/hardware.yaml （Pro 6000 到货后）。
#
# 字段语义见 hardware.2gpu.yaml 头注释。
# V1.5 manual-only：不解析 nvidia-smi topo -m，无 detection 字段。
groups:
  - id: llm-tp
    gpus: [0, 1]
    nvlink: true
    role: llm            # image/TTS 节点也落此 group，与 LLM 时分复用
    vram_gb: 48
```

- [ ] **Step 5: Commit**

```bash
cd backend && git add configs/hardware.yaml configs/hardware.2gpu.yaml configs/hardware.3gpu.yaml
git commit -m "feat(config): add hardware.yaml GPU topology config (manual-only)

hardware.yaml is the V1.5 single source of truth for GPU group / NVLink
topology (spec §3.2). configs/hardware.yaml is the loaded file (current
2x3090 layout); hardware.2gpu.yaml / hardware.3gpu.yaml are read-only
reference templates. V1.5 does not parse nvidia-smi topo -m. Lane A."
```

---

## Task 2: `config.py` 新增 `load_hardware_config`

`config.py` 已有 `load_model_configs(path="configs/models.yaml")` 模式：`_resolve_path` 把相对路径锚到 `backend/`，`yaml.safe_load` 读取。`load_hardware_config` 照同样套路，但要对「文件缺失」「yaml 损坏」「`groups` 缺失」做 fail-soft —— 返回空 `{"groups": []}` 而非抛异常（allocator 据此退化到 detect-based fallback，见 Task 3）。

**Files:**
- Modify: `backend/src/config.py`
- Test: `backend/tests/test_hardware_config.py`（新建）

- [ ] **Step 1: 写失败测试 — `load_hardware_config` 解析 hardware.yaml**

新建 `backend/tests/test_hardware_config.py`：
```python
"""Lane A: load_hardware_config 解析覆盖。"""
from src.config import load_hardware_config


def test_load_default_hardware_yaml():
    """configs/hardware.yaml 解析出 groups 列表。"""
    cfg = load_hardware_config()
    assert "groups" in cfg
    groups = cfg["groups"]
    assert isinstance(groups, list)
    assert len(groups) >= 1
    # 当前 2 卡布局：单个 llm-tp group
    llm = next(g for g in groups if g["id"] == "llm-tp")
    assert llm["gpus"] == [0, 1]
    assert llm["nvlink"] is True
    assert llm["role"] == "llm"
    assert llm["vram_gb"] == 48


def test_load_3gpu_template(tmp_path):
    """hardware.3gpu.yaml 样板解析出 3 个 group。"""
    cfg = load_hardware_config(path="configs/hardware.3gpu.yaml")
    ids = {g["id"] for g in cfg["groups"]}
    assert ids == {"image", "llm-tp", "tts"}
    image = next(g for g in cfg["groups"] if g["id"] == "image")
    assert image["gpus"] == [2]
    assert image["nvlink"] is False
    assert image["vram_gb"] == 96


def test_load_missing_file_returns_empty(tmp_path):
    """文件缺失 → fail-soft 返回 {'groups': []}，不抛异常。"""
    missing = tmp_path / "nope.yaml"
    cfg = load_hardware_config(path=str(missing))
    assert cfg == {"groups": []}


def test_load_corrupt_yaml_returns_empty(tmp_path):
    """yaml 损坏 → fail-soft 返回 {'groups': []}。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("groups: [ this is not: valid: yaml")
    cfg = load_hardware_config(path=str(bad))
    assert cfg == {"groups": []}


def test_load_missing_groups_key_returns_empty(tmp_path):
    """yaml 合法但无 groups 键 → 返回 {'groups': []}。"""
    nogroups = tmp_path / "nogroups.yaml"
    nogroups.write_text("detection:\n  mode: auto\n")
    cfg = load_hardware_config(path=str(nogroups))
    assert cfg == {"groups": []}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_hardware_config.py -v`
Expected: FAIL —— `ImportError: cannot import name 'load_hardware_config' from 'src.config'`。

- [ ] **Step 3: 在 `config.py` 实现 `load_hardware_config`**

`backend/src/config.py`，在 `load_model_configs` 函数之后追加：
```python
@lru_cache
def load_hardware_config(path: str = "configs/hardware.yaml") -> dict:
    """Load the manual GPU topology config (hardware.yaml).

    Returns a dict with a "groups" list. fail-soft: missing file, corrupt
    YAML, or missing "groups" key all return {"groups": []} so the
    GPUAllocator can degrade to detect-based single-card groups instead
    of crashing API server startup (spec §3.2, manual-only topology).
    """
    # path may be absolute (tests) or relative-to-backend (default).
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else _resolve_path(path)
    if not resolved.exists():
        return {"groups": []}
    try:
        with open(resolved) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return {"groups": []}
    groups = data.get("groups")
    if not isinstance(groups, list):
        return {"groups": []}
    return {"groups": groups}
```
（`lru_cache` 与 `load_model_configs` 不一致——后者没缓存——但 `hardware.yaml` 是 manual-only 配置，进程生命周期内不变，缓存安全且省去每次 allocator 实例化的磁盘 IO。`Path` 与 `yaml` 已在 `config.py` 顶部 import。)

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_hardware_config.py -v`
Expected: 5 个用例全 PASS。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/config.py tests/test_hardware_config.py
git commit -m "feat(config): add load_hardware_config with fail-soft parsing

Loads configs/hardware.yaml (the GPU topology source of truth). Missing
file / corrupt YAML / missing 'groups' key all return {'groups': []} so
GPUAllocator can fall back to detect-based single-card groups instead of
blocking startup. Lane A."
```

---

## Task 3: `GPUAllocator` 新增 `GPUGroup` dataclass + group 层 API

`GPUAllocator` 现在只有 `get_best_gpu` / `get_free_mb`（按 nvidia-smi free_mb 挑单卡）。Lane A 在**不动这两个方法**的前提下，让 allocator 在构造时加载 `hardware.yaml`，解析出 `GPUGroup` 列表，并暴露 group 层 API。`main.py:169` 的 `GPUAllocator()` 无参构造保持可用——allocator 内部默认调 `load_hardware_config()`；测试可注入 `hardware_config=` 覆盖。

fail-soft 退化：`hardware.yaml` 返回空 `groups` 时，allocator 用 `detect_gpus()` 给每张检测到的卡造一个单卡 group（`id=f"gpu{idx}"`, `nvlink=False`, `role="image"`, `vram_gb=` 检测值）。检测也为空（CI、无 GPU）→ `groups()` 返回空列表，`runner_count()` 返回 0。

**Files:**
- Modify: `backend/src/services/gpu_allocator.py`
- Test: `backend/tests/test_gpu_allocator.py`（追加，现有 4 个测试不动）

- [ ] **Step 1: 写失败测试 — group 层 API**

`backend/tests/test_gpu_allocator.py` 追加（保留文件顶部现有 4 个测试与 `_fake_stats`）：
```python
# ---- Lane A: group-aware API ----

_2GPU_CFG = {
    "groups": [
        {"id": "llm-tp", "gpus": [0, 1], "nvlink": True, "role": "llm", "vram_gb": 48},
    ]
}
_3GPU_CFG = {
    "groups": [
        {"id": "image", "gpus": [2], "nvlink": False, "role": "image", "vram_gb": 96},
        {"id": "llm-tp", "gpus": [0, 1], "nvlink": True, "role": "llm", "vram_gb": 48},
        {"id": "tts", "gpus": [3], "nvlink": False, "role": "tts", "vram_gb": 24},
    ]
}


def test_groups_parsed_from_hardware_config():
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config=_2GPU_CFG)
    groups = alloc.groups()
    assert len(groups) == 1
    g = groups[0]
    assert g.id == "llm-tp"
    assert g.gpus == [0, 1]
    assert g.nvlink is True
    assert g.role == "llm"
    assert g.vram_gb == 48


def test_runner_count_follows_groups():
    """runner 数 = groups 数量，不写死（spec §3.2）。"""
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=_2GPU_CFG).runner_count() == 1
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG).runner_count() == 3


def test_group_for_role():
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG)
    assert alloc.group_for_role("image").id == "image"
    assert alloc.group_for_role("llm").id == "llm-tp"
    assert alloc.group_for_role("tts").id == "tts"
    assert alloc.group_for_role("nonexistent") is None


def test_group_by_id():
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG)
    assert alloc.group_by_id("llm-tp").gpus == [0, 1]
    assert alloc.group_by_id("missing") is None


def test_llm_group_gpus():
    """Lane E 的 vLLM 选卡数据源：role:llm group 的 gpus。"""
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG).llm_group_gpus() == [0, 1]
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=_2GPU_CFG).llm_group_gpus() == [0, 1]
    # 无 llm group → 空列表
    img_only = {"groups": [{"id": "image", "gpus": [0], "nvlink": False,
                            "role": "image", "vram_gb": 24}]}
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=img_only).llm_group_gpus() == []


def test_nvlink_groups_only():
    """tensor-parallel 模型校验用：只返回 nvlink:true 的 group。"""
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG)
    nvlink_ids = {g.id for g in alloc.nvlink_groups()}
    assert nvlink_ids == {"llm-tp"}


def test_empty_hardware_config_falls_back_to_detected_gpus(monkeypatch):
    """hardware.yaml 为空 → 按 detect_gpus() 每卡一个单卡 group。"""
    from src.gpu.detector import GPUInfo

    monkeypatch.setattr(
        "src.services.gpu_allocator.detect_gpus",
        lambda: [
            GPUInfo(index=0, name="RTX 3090", vram_total_gb=24.0, compute_capability=(8, 6)),
            GPUInfo(index=1, name="RTX 3090", vram_total_gb=24.0, compute_capability=(8, 6)),
        ],
    )
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config={"groups": []})
    groups = alloc.groups()
    assert len(groups) == 2
    assert {g.id for g in groups} == {"gpu0", "gpu1"}
    assert all(g.nvlink is False for g in groups)
    assert groups[0].gpus == [0]
    assert alloc.runner_count() == 2


def test_empty_config_no_gpus_returns_empty(monkeypatch):
    """hardware.yaml 空 + 无 GPU（CI）→ groups 空，runner_count 0，不抛异常。"""
    monkeypatch.setattr("src.services.gpu_allocator.detect_gpus", lambda: [])
    alloc = GPUAllocator(poll_fn=lambda: [], hardware_config={"groups": []})
    assert alloc.groups() == []
    assert alloc.runner_count() == 0
    assert alloc.group_for_role("llm") is None
    assert alloc.llm_group_gpus() == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_gpu_allocator.py -v`
Expected: 现有 4 个 free_mb 测试 PASS；新增的 8 个 FAIL —— `TypeError: __init__() got an unexpected keyword argument 'hardware_config'`。

- [ ] **Step 3: 重写 `gpu_allocator.py` —— 加 `GPUGroup` + group API**

`backend/src/services/gpu_allocator.py` 整体替换为：
```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUGroup:
    """A scheduling unit: one runner owns exactly one GPUGroup.

    Parsed from hardware.yaml (spec §3.2). Multi-GPU groups (nvlink=True)
    are NVLink-paired cards used together for tensor-parallel models.
    """

    id: str
    gpus: list[int]
    nvlink: bool
    role: str          # image / llm / tts
    vram_gb: int       # group total VRAM (sum across cards), not per-card


class GPUAllocator:
    def __init__(
        self,
        poll_fn: Callable[[], list[dict]] | None = None,
        hardware_config: dict | None = None,
    ):
        if poll_fn is None:
            from src.services.gpu_monitor import poll_gpu_stats
            poll_fn = poll_gpu_stats
        self._poll = poll_fn

        if hardware_config is None:
            from src.config import load_hardware_config
            hardware_config = load_hardware_config()
        self._groups: list[GPUGroup] = self._build_groups(hardware_config)

    # ------------------------------------------------------------------
    # Group topology (Lane A)
    # ------------------------------------------------------------------

    def _build_groups(self, hardware_config: dict) -> list[GPUGroup]:
        """Parse hardware.yaml groups[]. fail-soft: empty config →
        one single-card group per detected GPU; no GPUs → []."""
        raw = hardware_config.get("groups") or []
        if raw:
            groups: list[GPUGroup] = []
            for entry in raw:
                groups.append(
                    GPUGroup(
                        id=entry["id"],
                        gpus=list(entry["gpus"]),
                        nvlink=bool(entry.get("nvlink", False)),
                        role=entry.get("role", "image"),
                        vram_gb=int(entry.get("vram_gb", 0)),
                    )
                )
            return groups

        # Fallback: hardware.yaml missing / empty. Build one single-card
        # group per detected GPU so the rest of V1.5 still has a topology.
        from src.gpu.detector import detect_gpus
        detected = detect_gpus()
        if not detected:
            logger.warning(
                "hardware.yaml has no groups and no GPUs detected — "
                "GPUAllocator.groups() will be empty."
            )
            return []
        logger.warning(
            "hardware.yaml has no groups — falling back to %d detect-based "
            "single-card group(s).", len(detected),
        )
        return [
            GPUGroup(
                id=f"gpu{g.index}",
                gpus=[g.index],
                nvlink=False,
                role="image",
                vram_gb=int(g.vram_total_gb),
            )
            for g in detected
        ]

    def groups(self) -> list[GPUGroup]:
        """All GPU groups parsed from hardware.yaml (or detect fallback)."""
        return list(self._groups)

    def runner_count(self) -> int:
        """How many GPU Runner subprocesses to spawn = number of groups
        (spec §3.2: runner count is not hard-coded)."""
        return len(self._groups)

    def group_by_id(self, group_id: str) -> GPUGroup | None:
        for g in self._groups:
            if g.id == group_id:
                return g
        return None

    def group_for_role(self, role: str) -> GPUGroup | None:
        """First group whose role matches. Used by node dispatch to find
        the GPU group for an image / llm / tts node."""
        for g in self._groups:
            if g.role == role:
                return g
        return None

    def nvlink_groups(self) -> list[GPUGroup]:
        """Groups with nvlink=True. tensor-parallel model spec validation
        requires the model land on one of these (spec §1.3)."""
        return [g for g in self._groups if g.nvlink]

    def llm_group_gpus(self) -> list[int]:
        """GPU indices of the role:llm group — the data source that
        replaces vLLM's self-detection (spec §3.2). Empty if no llm group.
        Lane E consumes this; Lane A only provides it."""
        g = self.group_for_role("llm")
        return list(g.gpus) if g is not None else []

    # ------------------------------------------------------------------
    # Free-VRAM probing (unchanged — model_manager.py:319 + tests depend
    # on this surface; do not break it)
    # ------------------------------------------------------------------

    def get_best_gpu(self, required_vram_mb: float) -> int:
        stats = self._poll()
        if not stats:
            return -1
        candidates = [(s["index"], s["free_mb"]) for s in stats if s["free_mb"] >= required_vram_mb]
        if not candidates:
            return -1
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_free_mb(self, gpu_index: int) -> int:
        stats = self._poll()
        for s in stats:
            if s["index"] == gpu_index:
                return s["free_mb"]
        return 0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_gpu_allocator.py -v`
Expected: 全部 12 个用例 PASS（现有 4 个 free_mb + 新增 8 个 group API）。

- [ ] **Step 5: 跑相关回归 — model_manager 仍能用 allocator**

`model_manager.py:319` 调 `get_best_gpu`，`test_model_manager_v2.py` / `test_lora_scanner.py` 用 `MagicMock()` 当 allocator。`main.py:169` 无参构造 `GPUAllocator()`。确认这些路径不受 group API 新增影响。

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_model_manager_v2.py tests/test_lora_scanner.py tests/test_config.py -q`
Expected: PASS。`MagicMock` allocator 没有 `groups()` 但也没人调它——Lane A 不改 `model_manager.py`，无回归。

- [ ] **Step 6: 跑全 suite 建信心**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS。无 import error、无 collection error。`GPUAllocator()` 无参构造在 lifespan 里会调 `load_hardware_config()`——但 `conftest.py` 设了 `NOUS_DISABLE_BG_TASKS=1` 且 `CUDA_VISIBLE_DEVICES=""`，lifespan 路径里 `GPUAllocator()` 仍会构造（加载真实 `configs/hardware.yaml`，解析出 `llm-tp` group），不触发 GPU 调用，安全。

- [ ] **Step 7: lint 预检**

Run: `cd backend && ruff check src/services/gpu_allocator.py src/config.py tests/test_gpu_allocator.py tests/test_hardware_config.py`
Expected: 无 lint 错误。

- [ ] **Step 8: Commit**

```bash
cd backend && git add src/services/gpu_allocator.py tests/test_gpu_allocator.py
git commit -m "refactor(gpu): make GPUAllocator NVLink-group aware

GPUAllocator now parses hardware.yaml into GPUGroup objects and exposes
groups() / runner_count() / group_for_role() / nvlink_groups() /
llm_group_gpus(). runner_count() drives how many GPU Runner subprocesses
Lane C spawns — not hard-coded. Empty hardware.yaml falls back to
detect-based single-card groups. get_best_gpu / get_free_mb unchanged.
Lane A."
```

---

## Task 4: 整合验证 + 开 PR

**Files:** 无（验证）

- [ ] **Step 1: 全 suite green**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS，无 skip 异常、无 collection error。

- [ ] **Step 2: 确认 hardware.yaml 被实际加载并解析正确**

Run:
```bash
cd backend && ADMIN_PASSWORD="" python -c "
from src.services.gpu_allocator import GPUAllocator
a = GPUAllocator(poll_fn=lambda: [])
print('runner_count:', a.runner_count())
print('groups:', [(g.id, g.gpus, g.nvlink, g.role) for g in a.groups()])
print('llm_group_gpus:', a.llm_group_gpus())
"
```
Expected: `runner_count: 1`；`groups: [('llm-tp', [0, 1], True, 'llm')]`；`llm_group_gpus: [0, 1]`。说明无参构造正确加载了 `configs/hardware.yaml`。

- [ ] **Step 3: lint 全量预检**

Run: `cd backend && ruff check src/ tests/`
Expected: 无新增 lint 错误。

- [ ] **Step 4: 开 PR**

```bash
git push -u origin <lane-A-branch>
gh pr create --title "feat: V1.5 Lane A — hardware.yaml + NVLink-aware GPUAllocator" --body "$(cat <<'EOF'
## Summary
- 新增 `configs/hardware.yaml`：manual-only 的 GPU 拓扑唯一真相源（当前 2x3090 NVLink 布局）+ 2gpu/3gpu 两份带注释参考样板
- `config.py` 新增 `load_hardware_config`，对文件缺失/损坏/无 groups 键 fail-soft 返回空
- `GPUAllocator` 重构为 NVLink-group 感知：新增 `GPUGroup` dataclass + `groups()` / `runner_count()` / `group_for_role()` / `nvlink_groups()` / `llm_group_gpus()`；按 yaml `groups[]` 动态决定 runner 数
- `get_best_gpu` / `get_free_mb` 保持不变，`model_manager.py` 与现有测试零回归

## Test plan
- [ ] 全 suite green（pytest tests/）
- [ ] `test_hardware_config.py`：2gpu/3gpu/缺失/损坏/无 groups 解析覆盖
- [ ] `test_gpu_allocator.py`：现有 4 个 free_mb 测试 + 新增 8 个 group API 测试全 PASS
- [ ] `GPUAllocator()` 无参构造冒烟：runner_count=1，llm-tp group 解析正确
EOF
)"
```
（分支名按项目惯例，如 `feat/v15-laneA-hardware-yaml-gpu-allocator`。）

---

## Self-Review

**Spec 覆盖检查：** Lane A 在 spec「实施分 Lane」表里的职责是「`hardware.yaml`（2gpu/3gpu 两份，manual-only）+ GPUAllocator 重构（NVLink-aware，按 yaml groups[] 动态决定 runner 数）」。

- `hardware.yaml` 2gpu/3gpu 两份，manual-only → Task 1（外加一份实际生效的 `hardware.yaml`，见偏差 1）
- GPUAllocator NVLink-aware → Task 3 `GPUGroup.nvlink` + `nvlink_groups()`
- 按 yaml `groups[]` 动态决定 runner 数 → Task 3 `runner_count()`
- spec §1.4 当前 2 卡布局（方案 A：单 `llm-tp` group = GPU [0,1]）→ `hardware.yaml` / `hardware.2gpu.yaml` 内容
- spec §3.2「`_detect_vllm_gpus*` 退化为读 `hardware.yaml` 的 `role: llm` group」→ **部分**：Lane A 只交付数据源 `llm_group_gpus()`，实际改 `model_manager.py` 的 vLLM 选卡调用点归 Lane E（见偏差 2）。
- spec §1.3「allocator 按 group 分配；tp 模型强制 `nvlink:true`」→ Lane A 提供 `nvlink_groups()` 供 Lane（spec 校验层）消费；强约束校验本身不在 Lane A 范围。

**偏差/判断点（已在 plan 头部详述）：**
1. `hardware.yaml` 是实际加载文件，`hardware.2gpu.yaml` / `hardware.3gpu.yaml` 是只读样板——spec 没明确这层关系，本 plan 做了决策。
2. `_detect_vllm_gpus*` 的改道留给 Lane E，Lane A 只给 `llm_group_gpus()` 只读 API。
3. `vram_gb` 语义 = group 合计 VRAM（spec §3.2 yaml 与硬件表的 24/48 出入已澄清）。

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。所有 yaml 文件内容、测试代码、`load_hardware_config` 与 `GPUAllocator` 实现全部给出完整代码。

**类型一致性：**
- `GPUGroup` 是 `@dataclass(frozen=True)`，字段 `id: str` / `gpus: list[int]` / `nvlink: bool` / `role: str` / `vram_gb: int`；`_build_groups` 构造时 `list(...)` / `bool(...)` / `int(...)` 强制类型，防 yaml 里 `vram_gb` 写成字符串。
- `load_hardware_config` 返回 `dict`，恒含 `"groups"` 键且值为 `list`（fail-soft 分支也返回 `{"groups": []}`），`_build_groups` 对 `.get("groups") or []` 安全。
- `GPUAllocator.__init__` 新增 `hardware_config: dict | None = None` 为关键字参数且有默认值——`main.py:169` 的 `GPUAllocator()` 无参构造、`test_gpu_allocator.py` 现有 `GPUAllocator(poll_fn=...)` 调用全部不破。
- `groups()` / `nvlink_groups()` 返回 `list[GPUGroup]`；`group_by_id` / `group_for_role` 返回 `GPUGroup | None`；`runner_count` 返回 `int`；`llm_group_gpus` 返回 `list[int]`——与测试断言一致。

**已知风险：**
- `load_hardware_config` 用了 `@lru_cache`，与 `load_model_configs`（无缓存）不一致。理由：`hardware.yaml` 是 manual-only 配置、进程内不变，缓存安全。但若未来加「热重载拓扑」功能需记得 `load_hardware_config.cache_clear()`——已在 Task 2 Step 3 注释说明。测试里 `test_load_missing_file_returns_empty` 等用不同 `path` 参数，`lru_cache` 按参数 key 缓存，不会串味。
- fallback 分支调 `detect_gpus()` 会 import torch 并可能触发 CUDA 探测。但只在 `hardware.yaml` 的 `groups` 为空时才走——仓库已提交非空的 `configs/hardware.yaml`，正常路径不触发。CI 里 `CUDA_VISIBLE_DEVICES=""` 使 `detect_gpus()` 返回 `[]`，`test_empty_config_no_gpus_returns_empty` 显式覆盖此路径。
- Lane A 不改 `model_manager.py` / `main.py` 的 allocator 使用方式——group API 是纯新增 surface，下游 Lane C/E 才消费。本 Lane 的回归面仅限「`GPUAllocator` 构造不再纯净（会读磁盘）」，Task 3 Step 6 的全 suite 已覆盖。
