# Image Component Multi-GPU — PR-4 (Workflow 节点 + 组件级 L1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 workflow 能用 4 个独立 loader 节点(unet/clip/vae/lora_apply)给 Flux2 三组件分别选文件+设备,经 image_generate 跨卡出图;老 model_key workflow 后端 inline 展开零改动跑通;组件级 L1 缓存让 clip/vae 跨 workflow 复用、改 LoRA 二跑只重 patch unet(<5s)。

**Architecture:** 4 个 loader 节点是**纯描述符 inline 节点**(主进程跑,不碰 GPU,输出 dict),描述符经 edges 落到 image_generate 的 unet/clip/vae 输入端口 → `merged_inputs` → `RunNode.inputs`。runner `_build_request` 见到 unet/clip/vae 三描述符即构造带 `components` 的 `ImageRequest`;`_node_executor` 走新入口 `ModelManager.get_or_load_image_adapter(components, pipeline_class)`,它解析 `auto` device、对**去 LoRA 的 base spec** 调 `get_or_load_component` 缓存 GPU 模块(clip/vae 跨 workflow 复用、transformer base 跨 LoRA 复用),再用 `DiffusersImageBackend.from_loaded_components` 把缓存模块拼成 pipe + ImageSampler、按 `(unet+clip+vae)` 组合 key 缓存整 adapter。LoRA 仍走现有 pipe 级 `_apply_loras`(描述符自带 abs path),`infer` 每次复用前**重申明 active adapter set**(同 runner 串行执行,base transformer 跨 adapter 共享安全)。老格式由 `workflow_executor` 在 dispatch 前把 model_key 翻成 3 个 `ComponentSpec`(device=`auto`)塞回 inputs。

**Tech Stack:** Python 3.12 / FastAPI / diffusers(Flux2KleinPipeline + 模型类)/ pydantic v2 / pytest + pytest-asyncio。runner = multiprocessing 子进程 + msgpack pipe。

**Branch:** `feat/image-component-multigpu-pr4`(从 master 切;每逻辑 PR 独立分支走 CI/CD)。

**前置(已 merged)**:PR-1 #112(`ComponentSpec`/`ComponentKey`/`QuantLoaderRegistry`/`ModelManager._components`)、PR-2 #113+#114(`ImageSampler`/`ModelArchAdapter`/`DiffusersImageBackend.from_components`+`load_from_components`)、PR-3 #115(`component_scanner`/`GET /api/v1/components`/`model_paths.yaml`)。

**Spec:** `docs/superpowers/specs/2026-05-19-image-component-multi-gpu-design.md`(rev 2)§4 / §5.4 / §5.5 / §7.4 / §9。

---

## 关键设计决策(写给执行者,先读)

1. **L1 缓存分两层**(用户选「组件级 L1 全做进 PR-4」):
   - **模块层** `ModelManager._components`(已存在,key=`to_component_key`=`(file,device,dtype,lora_set)`)。我们**总是用去 LoRA 的 base spec**(`loras=[]`)调它 → clip/vae/transformer 的 base GPU 模块按 `(file,device,dtype,∅)` 缓存,跨 workflow / 跨 LoRA 复用。`_load_component_impl` 从「返回 state_dict」改成「返回已 `.to(device)` 的 GPU 模块包」。
   - **整 adapter 层** `ModelManager._image_adapters`(新建,key=三组件**完整** key 的 tuple,**含 unet 的 lora_set**)。命中即直接复用 `DiffusersImageBackend`(含 pipe+sampler+已 patch 的 LoRA)。
2. **LoRA 走 pipe 级**(复用现有 `_apply_loras`,低风险),描述符自带 `path`(`LoRASpec.path`)。base transformer 跨多个 LoRA 组合共享:每个 LoRA 以独立 `adapter_name` 注册到共享 transformer(peft adapter 累积、不删),`infer` 复用前 `set_adapters([本组合])` 重申明。**只在 runner 串行执行下安全**(单 ModelManager 内无并发);多 runner 各自独立 ModelManager。
3. **跨进程不传张量**:loader 节点只输出 dict 描述符;真正 load 在 runner。
4. **老格式 inline 展开放后端 `workflow_executor`**(spec §7.4),runner `_build_request` 保留 model_key 兜底分支(防御,理论走不到)。
5. **`auto` device** 在 `get_or_load_image_adapter` 用 `allocator.get_best_gpu(vram_est)` 解析成 `cuda:N`,**在调 `get_or_load_component` 之前**解析(否则 base 模块 key 含 `"auto"` 撞不上缓存)。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `backend/src/services/inference/base.py` | Modify | `LoRASpec` 加 `path`;`ImageRequest` 加 `components` + `pipeline_class` |
| `backend/src/services/nodes/image_components.py` | Create | 4 个 loader inline 节点(纯描述符) |
| `backend/src/services/workflow_executor.py` | Modify | import 触发节点注册;`_dispatch_node` 设 `is_deterministic`;老格式 inline 展开 |
| `backend/src/services/inference/component_expand.py` | Create | `expand_legacy_image_node(spec) -> dict[str, ComponentSpec]`(model_key→三组件) |
| `backend/src/runner/runner_process.py` | Modify | `_build_request` components 分支;`_node_executor` components dispatch |
| `backend/src/services/model_manager.py` | Modify | `_load_component_impl` 改加载 GPU 模块;新增 `get_or_load_image_adapter` + auto 解析 + adapter combo 缓存 + OOM evict |
| `backend/src/services/inference/image_diffusers.py` | Modify | `from_loaded_components` 新 classmethod;`_apply_loras` 用 `spec.path`;`infer` 组件路径复用前重申明 LoRA;`load_from_components` 委托复用 |
| `backend/tests/test_*` | Create/Modify | 见各 Task |

执行顺序即下方 Task 顺序(类型 → 节点 → runner 协议 → 缓存重构 → 装配 → 编排入口 → 老格式 → 集成 → 真模型 smoke)。

---

## Task 1: 类型层 — `LoRASpec.path` + `ImageRequest.components` / `pipeline_class`

**Files:**
- Modify: `backend/src/services/inference/base.py`
- Test: `backend/tests/test_inference_request_components.py`(新建)

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_inference_request_components.py
"""PR-4: ImageRequest.components + LoRASpec.path round-trip."""
from __future__ import annotations

from src.services.inference.base import ImageRequest, LoRASpec
from src.services.inference.component_spec import ComponentSpec


def test_lora_spec_path_optional_default_none():
    assert LoRASpec(name="style", strength=0.8).path is None
    s = LoRASpec(name="style", strength=0.8, path="/m/loras/style.safetensors")
    assert s.path == "/m/loras/style.safetensors"


def test_image_request_components_default_none():
    req = ImageRequest(request_id="r1", prompt="a cat")
    assert req.components is None
    assert req.pipeline_class == "Flux2KleinPipeline"


def test_image_request_with_components():
    comps = {
        "unet": ComponentSpec(kind="unet", file="/m/u.safe", device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16", clip_arch="flux2"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }
    req = ImageRequest(request_id="r1", prompt="a cat", seed=42, components=comps)
    assert req.components["unet"].device == "cuda:1"
    assert req.components["vae"].kind == "vae"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_inference_request_components.py -q`
Expected: FAIL —`ImageRequest` 无 `components` / `LoRASpec` 无 `path`(AttributeError / ValidationError)。

- [ ] **Step 3: 改 `base.py`**

在 `LoRASpec`(base.py:73-78)加 `path` 字段:

```python
class LoRASpec(BaseModel):
    """LoRA reference by display name (ComfyUI-style)."""

    name: str
    strength: float = Field(1.0, ge=-2, le=2)
    # PR-4: component path carries the absolute LoRA file path so the runner
    # can load it without a name→path registry lookup (from_components sets
    # _lora_paths={}). Legacy yaml path leaves this None and resolves via
    # _lora_paths[name] as before.
    path: str | None = None
```

在 `ImageRequest`(base.py:80-90)末尾加两字段。`ComponentSpec` 在 `component_spec.py`,用 `TYPE_CHECKING` 避免循环 import(`component_spec` 已 import `base.LoRASpec`):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.inference.component_spec import ComponentSpec


class ImageRequest(InferenceRequest):
    modality: Literal[MediaModality.IMAGE] = MediaModality.IMAGE
    prompt: str
    negative_prompt: str = ""
    width: int = Field(1024, ge=64, le=4096)
    height: int = Field(1024, ge=64, le=4096)
    steps: int = Field(25, ge=1, le=200)
    seed: int | None = None
    cfg_scale: float = Field(7.0, ge=0, le=30)
    loras: list[LoRASpec] = Field(default_factory=list)
    # PR-4: component path. When set, the runner routes through
    # ModelManager.get_or_load_image_adapter instead of model_key. None ⇒
    # legacy model_key path (back-compat).
    components: dict[str, "ComponentSpec"] | None = None
    pipeline_class: str = "Flux2KleinPipeline"
```

`component_spec` import `base`,`base` 只在 `TYPE_CHECKING` 引用 `ComponentSpec` → 运行时无环。pydantic v2 解析前向引用需 `model_rebuild()`。在 `component_spec.py` 末尾(class 定义后)追加:

```python
# Resolve ImageRequest's forward ref to ComponentSpec now that both classes
# exist (base.py imports this module lazily-safe via TYPE_CHECKING).
from src.services.inference.base import ImageRequest as _ImageRequest  # noqa: E402
_ImageRequest.model_rebuild()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/test_inference_request_components.py -q`
Expected: PASS(3 passed)。

- [ ] **Step 5: 回归 — 确认没破坏既有 ImageRequest/ComponentSpec 用法**

Run: `cd backend && uv run pytest tests/test_component_spec.py tests/test_runner_protocol.py -q`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/inference/base.py backend/src/services/inference/component_spec.py backend/tests/test_inference_request_components.py
git commit -m "feat(image): PR-4 — LoRASpec.path + ImageRequest.components/pipeline_class"
```

---

## Task 2: 4 个 loader inline 节点(纯描述符)

**Files:**
- Create: `backend/src/services/nodes/image_components.py`
- Modify: `backend/src/services/workflow_executor.py:14`(import 触发 `@register`)
- Test: `backend/tests/test_image_component_nodes.py`(新建)

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_image_component_nodes.py
"""PR-4: 4 loader 节点输出描述符 dict (inline, 无 GPU)。"""
from __future__ import annotations

import pytest

from src.services.nodes.registry import get_node_class


@pytest.mark.asyncio
async def test_unet_load_emits_descriptor():
    node = get_node_class("image_unet_load")()
    out = await node.invoke(
        {"file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2"},
        {},
    )
    assert out == {"unet": {
        "kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
        "dtype": "bfloat16", "adapter_arch": "flux2", "loras": [],
    }}


@pytest.mark.asyncio
async def test_unet_load_defaults():
    node = get_node_class("image_unet_load")()
    out = await node.invoke({"file": "/m/u.safe"}, {})
    assert out["unet"]["device"] == "auto"
    assert out["unet"]["dtype"] == "bfloat16"
    assert out["unet"]["adapter_arch"] == "flux2"
    assert out["unet"]["loras"] == []


@pytest.mark.asyncio
async def test_clip_and_vae_load():
    clip = await get_node_class("image_clip_load")().invoke(
        {"file": "/m/c.safe", "device": "cuda:0", "clip_arch": "flux2"}, {})
    assert clip == {"clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0",
                             "dtype": "bfloat16", "clip_arch": "flux2"}}
    vae = await get_node_class("image_vae_load")().invoke(
        {"file": "/m/v.safe", "device": "cuda:2"}, {})
    assert vae == {"vae": {"kind": "vae", "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"}}


@pytest.mark.asyncio
async def test_lora_apply_appends():
    upstream = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
                "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []}
    out = await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "style", "lora_path": "/m/loras/style.safetensors", "strength": 0.8},
        {"unet": upstream},
    )
    assert out["unet"]["loras"] == [{"name": "style", "path": "/m/loras/style.safetensors", "strength": 0.8}]
    # upstream not mutated
    assert upstream["loras"] == []


@pytest.mark.asyncio
async def test_lora_apply_chains():
    base = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
            "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []}
    step1 = await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "a", "lora_path": "/m/loras/a.safetensors", "strength": 0.8}, {"unet": base})
    step2 = await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "b", "lora_path": "/m/loras/b.safetensors", "strength": 0.4}, {"unet": step1["unet"]})
    assert [l["name"] for l in step2["unet"]["loras"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_lora_apply_bypass_passthrough():
    upstream = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
                "dtype": "bfloat16", "adapter_arch": "flux2", "loras": [{"name": "x", "path": "/p/x", "strength": 1.0}]}
    out = await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "y", "lora_path": "/p/y", "strength": 0.5, "bypass": True}, {"unet": upstream})
    assert out["unet"] is upstream  # 完全透传(spec §4.4 bypass=True)


@pytest.mark.asyncio
async def test_lora_apply_missing_upstream_raises():
    from src.services.workflow_executor import ExecutionError
    with pytest.raises(ExecutionError, match="unet"):
        await get_node_class("image_lora_apply")().invoke(
            {"lora_file": "x", "lora_path": "/p/x"}, {})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_image_component_nodes.py -q`
Expected: FAIL —`get_node_class("image_unet_load")` 返回 None → `'NoneType' object is not callable`。

- [ ] **Step 3: 新建 `image_components.py`**

```python
# backend/src/services/nodes/image_components.py
"""PR-4 image component loader nodes — pure descriptor producers (inline).

These run in the backend event loop (node_routing default = inline, no GPU).
Each emits a plain descriptor dict; no tensors cross the wire. The runner
subprocess later materializes descriptors into ComponentSpec + ImageSampler
(spec §3.2). Output port names (unet/clip/vae) match image_generate's input
ports so WorkflowExecutor._get_inputs lands them in merged_inputs under those
keys (spec §5.4). Frontend palette + forms are PR-5.
"""
from __future__ import annotations

from src.services.nodes.registry import register

_AUTO = "auto"


@register("image_unet_load")
class ImageUnetLoadNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"unet": {
            "kind": "unet",
            "file": data["file"],
            "device": data.get("device") or _AUTO,
            "dtype": data.get("dtype") or "bfloat16",
            "adapter_arch": data.get("adapter_arch") or "flux2",
            "loras": [],
        }}


@register("image_clip_load")
class ImageClipLoadNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"clip": {
            "kind": "clip",
            "file": data["file"],
            "device": data.get("device") or _AUTO,
            "dtype": data.get("dtype") or "bfloat16",
            "clip_arch": data.get("clip_arch") or "flux2",
        }}


@register("image_vae_load")
class ImageVaeLoadNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"vae": {
            "kind": "vae",
            "file": data["file"],
            "device": data.get("device") or _AUTO,
            "dtype": data.get("dtype") or "bfloat16",
        }}


@register("image_lora_apply")
class ImageLoraApplyNode:
    """Chainable: input unet descriptor → output unet descriptor with one more
    LoRA appended. bypass=True passes the upstream descriptor straight through
    (spec §4.4)."""

    async def invoke(self, data: dict, inputs: dict) -> dict:
        from src.services.workflow_executor import ExecutionError

        upstream = inputs.get("unet")
        if not isinstance(upstream, dict) or upstream.get("kind") != "unet":
            raise ExecutionError("image_lora_apply 需要上游 unet 描述符输入(连 image_unet_load 或上一个 image_lora_apply)")
        if data.get("bypass"):
            return {"unet": upstream}
        appended = {
            "name": data["lora_file"],
            "path": data.get("lora_path"),
            "strength": float(data.get("strength", 1.0)),
        }
        return {"unet": {**upstream, "loras": [*upstream.get("loras", []), appended]}}
```

- [ ] **Step 4: 注册触发 — 改 `workflow_executor.py:14`**

```python
# Trigger @register side effects for all builtin nodes
from src.services.nodes import audio, image, image_components, llm, logic, text_io  # noqa: F401
```

- [ ] **Step 5: 跑测试确认通过 + 确认 4 节点是 inline(非 dispatch)**

Run: `cd backend && uv run pytest tests/test_image_component_nodes.py -q`
Expected: PASS。

补一条断言到同文件(确认路由),再跑:

```python
def test_loader_nodes_are_inline():
    from src.services.node_routing import node_exec_class
    for t in ("image_unet_load", "image_clip_load", "image_vae_load", "image_lora_apply"):
        assert node_exec_class(t) == "inline"
```

Run: `cd backend && uv run pytest tests/test_image_component_nodes.py -q` → PASS。

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/nodes/image_components.py backend/src/services/workflow_executor.py backend/tests/test_image_component_nodes.py
git commit -m "feat(image): PR-4 — 4 inline component loader nodes (unet/clip/vae/lora_apply)"
```

---

## Task 3: runner `_build_request` — components 分支

**Files:**
- Modify: `backend/src/runner/runner_process.py:147-181`
- Test: `backend/tests/test_runner_build_request.py`(新建)

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_runner_build_request.py
"""PR-4: _build_request 见 unet/clip/vae 三描述符 → ImageRequest.components。"""
from __future__ import annotations

from src.runner import protocol as P
from src.runner.runner_process import _build_request


def _img_node(inputs):
    return P.RunNode(task_id=1, node_id="g", node_type="image", model_key=None, inputs=inputs)


def test_build_request_components_branch():
    node = _img_node({
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
        "prompt": "a cat", "steps": 9, "seed": 42, "width": 768, "height": 768,
    })
    req = _build_request(node)
    assert req.components is not None
    assert req.components["unet"].device == "cuda:1"
    assert req.components["clip"].file == "/m/c.safe"
    assert req.prompt == "a cat"
    assert req.steps == 9
    assert req.seed == 42
    assert req.pipeline_class == "Flux2KleinPipeline"


def test_build_request_components_with_loras():
    node = _img_node({
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2",
                 "loras": [{"name": "style", "path": "/m/loras/style.safetensors", "strength": 0.8}]},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
        "prompt": "x",
    })
    req = _build_request(node)
    assert req.components["unet"].loras[0].name == "style"
    assert req.components["unet"].loras[0].path == "/m/loras/style.safetensors"


def test_build_request_legacy_no_components():
    node = _img_node({"prompt": "x", "steps": 25})
    req = _build_request(node)
    assert req.components is None
    assert req.prompt == "x"


def test_build_request_pipeline_class_override():
    node = _img_node({
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
        "prompt": "x", "pipeline_class": "Flux2Pipeline",
    })
    assert _build_request(node).pipeline_class == "Flux2Pipeline"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_runner_build_request.py -q`
Expected: FAIL —现 `_build_request` 不读 components,`req.components` 为 None。

- [ ] **Step 3: 改 `_build_request`(runner_process.py)**

把 image 分支替换为(spec §5.4):

```python
    if node.node_type == "image":
        # 转发 ImageRequest 全部字段 —— executor 已把 node.data 合并进 inputs，
        # steps/width/height/cfg_scale/seed/loras 都在这里能拿到。缺省走 pydantic
        # Field default(steps=25 / 1024x1024 / cfg_scale=7.0)。
        raw_seed = node.inputs.get("seed")
        seed = int(raw_seed) if raw_seed not in (None, "") else None
        base = dict(
            request_id=f"task-{node.task_id}",
            prompt=str(node.inputs.get("prompt", "")),
            negative_prompt=str(node.inputs.get("negative_prompt", "")),
            steps=int(node.inputs.get("steps") or 25),
            width=int(node.inputs.get("width") or 1024),
            height=int(node.inputs.get("height") or 1024),
            cfg_scale=float(node.inputs.get("cfg_scale") or 7.0),
            seed=seed,
        )
        # 新格式:三组件描述符齐全 → 走 components 路径(spec §5.4)。
        if all(k in node.inputs for k in ("unet", "clip", "vae")):
            from src.services.inference.component_spec import ComponentSpec
            return ImageRequest(
                **base,
                components={
                    "unet": ComponentSpec(**node.inputs["unet"]),
                    "clip": ComponentSpec(**node.inputs["clip"]),
                    "vae":  ComponentSpec(**node.inputs["vae"]),
                },
                pipeline_class=str(node.inputs.get("pipeline_class") or "Flux2KleinPipeline"),
            )
        # 老路径:无 components(workflow_executor 已 inline 展开过;走不到也安全)。
        loras_raw = node.inputs.get("loras") or []
        return ImageRequest(**base, loras=loras_raw if isinstance(loras_raw, list) else [])
```

> 注:`ComponentSpec(**dict)` 里 `loras` 是 `list[dict]`,pydantic 自动转 `list[LoRASpec]`(含新 `path` 字段)。device=`"auto"` 通过 `ComponentSpec` validator(regex 允许 `auto`)。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/test_runner_build_request.py -q`
Expected: PASS(4 passed)。

- [ ] **Step 5: 回归 protocol**

Run: `cd backend && uv run pytest tests/test_runner_protocol.py -q`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add backend/src/runner/runner_process.py backend/tests/test_runner_build_request.py
git commit -m "feat(image): PR-4 — runner _build_request components branch"
```

---

## Task 4: `workflow_executor._dispatch_node` — 设 `is_deterministic`(seed 非空)

**Files:**
- Modify: `backend/src/services/workflow_executor.py:230-236`
- Test: `backend/tests/test_workflow_executor_is_deterministic.py`(新建)

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_workflow_executor_is_deterministic.py
"""PR-4: 带 seed 的 image 节点 dispatch 时 is_deterministic=True (spec §3.3)。"""
from __future__ import annotations

import pytest

from src.runner import protocol as P
from src.services.workflow_executor import WorkflowExecutor


class _CapturingClient:
    def __init__(self):
        self.spec: P.RunNode | None = None

    async def run_node(self, spec, *, workflow_name=""):
        self.spec = spec
        return P.NodeResult(task_id=spec.task_id, node_id=spec.node_id, status="completed",
                            outputs={"image_url": "u"}, error=None, duration_ms=1)


def _exec(node_data):
    wf = {"nodes": [{"id": "g", "type": "image_generate", "data": node_data}], "edges": []}
    client = _CapturingClient()
    ex = WorkflowExecutor(wf, runner_clients={"image": client}, task_id=7)
    return ex, client


@pytest.mark.asyncio
async def test_seed_sets_deterministic():
    ex, client = _exec({"model_key": "flux2-klein-9b", "prompt": "x", "seed": 42})
    await ex._dispatch_node(ex._node_map["g"], {"prompt": "x", "seed": 42})
    assert client.spec.is_deterministic is True


@pytest.mark.asyncio
async def test_no_seed_not_deterministic():
    ex, client = _exec({"model_key": "flux2-klein-9b", "prompt": "x"})
    await ex._dispatch_node(ex._node_map["g"], {"prompt": "x"})
    assert client.spec.is_deterministic is False


@pytest.mark.asyncio
async def test_empty_seed_not_deterministic():
    ex, client = _exec({"model_key": "flux2-klein-9b", "prompt": "x", "seed": ""})
    await ex._dispatch_node(ex._node_map["g"], {"prompt": "x", "seed": ""})
    assert client.spec.is_deterministic is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_workflow_executor_is_deterministic.py -q`
Expected: FAIL —`is_deterministic` 恒 False。

- [ ] **Step 3: 改 `_dispatch_node`**(在构造 `P.RunNode` 前)

把 `spec = P.RunNode(...)` 改为先算 `is_deterministic`:

```python
        merged_inputs = {**{k: v for k, v in data.items() if not k.startswith("_")}, **inputs}

        # spec §3.3: seed 非空 ⇒ 确定性,runner / L2 cache 据此决定可缓存。
        is_deterministic = merged_inputs.get("seed") not in (None, "")

        spec = P.RunNode(
            task_id=task_id,
            node_id=node["id"],
            node_type=group_id,
            model_key=model_key,
            inputs=merged_inputs,
            is_deterministic=is_deterministic,
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/test_workflow_executor_is_deterministic.py -q`
Expected: PASS(3 passed)。

- [ ] **Step 5: 回归 executor**

Run: `cd backend && uv run pytest tests/test_workflow_executor.py tests/test_workflow_executor_dispatch.py -q`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/workflow_executor.py backend/tests/test_workflow_executor_is_deterministic.py
git commit -m "feat(image): PR-4 — set RunNode.is_deterministic from seed"
```

---

## Task 5: `ModelManager._load_component_impl` — 改加载 GPU 模块包

**Files:**
- Modify: `backend/src/services/model_manager.py:677-687`(+ 新增 `_load_component_module` 可注入 seam)
- Modify: `backend/tests/test_model_manager_components.py`(更新 stub 返回形状 + 加真 impl 的 seam 测试)
- Test: 同上文件

> 现 `_load_component_impl` 返回 `{"_state_dict": ...}`(PR-1 占位)。改为返回**已 `.to(device)` 的 GPU 模块包**:`{"module": <nn.Module>, "tokenizer": <or None>, "spec": spec, "device": spec.device, "loaded_at": ...}`。真加载逻辑抽到 `_load_component_module(spec)`(可 monkeypatch),让纯逻辑测试不碰 torch/diffusers。

- [ ] **Step 1: 写失败测试(更新既有 + 新增)**

在 `tests/test_model_manager_components.py` 末尾追加:

```python
@pytest.mark.asyncio
async def test_load_component_impl_returns_module_bundle(mm, monkeypatch):
    """_load_component_impl 经 _load_component_module seam 返回 GPU 模块包。"""
    spec = ComponentSpec(kind="vae", file="/m/v.safe", device="cuda:0", dtype="bfloat16")

    sentinel = object()

    def _fake_module(s):
        assert s is spec
        return {"module": sentinel, "tokenizer": None}

    monkeypatch.setattr(mm, "_load_component_module", _fake_module)
    bundle = await mm.get_or_load_component(spec)
    assert bundle["module"] is sentinel
    assert bundle["spec"] is spec
    assert bundle["device"] == "cuda:0"
    assert "loaded_at" in bundle
```

把既有 `test_get_or_load_component_marks_loaded` / `_cache_hit...` / `_distinguishes_lora_set` / `_unload...` / `_failed...` 里 monkeypatch 的目标从 `_load_component_impl` 改为 `_load_component_module`,返回值改成 `{"module": ..., "tokenizer": None}`(因为 `_load_component_impl` 现在会包一层 spec/device/loaded_at)。例如:

```python
async def test_get_or_load_component_cache_hit_does_not_call_loader_twice(mm, monkeypatch):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")
    calls = []

    def _counting(s):
        calls.append(s)
        return {"module": f"stub{len(calls)}", "tokenizer": None}

    monkeypatch.setattr(mm, "_load_component_module", _counting)
    r1 = await mm.get_or_load_component(spec)
    r2 = await mm.get_or_load_component(spec)
    assert r1 is r2
    assert len(calls) == 1
```

(对 `_marks_loaded` / `_distinguishes_lora_set` / `_unload` / `_failed` 同样把 seam 换成 `_load_component_module` 同步形状;`_failed` 的 `_broken_loader` 改成同步 `def` raise。)

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_model_manager_components.py -q`
Expected: FAIL —`mm._load_component_module` 不存在(AttributeError)。

- [ ] **Step 3: 改 `model_manager.py`**

把 `_load_component_impl` 改为薄包装 + 新增 `_load_component_module`(真加载,可注入):

```python
    async def _load_component_impl(self, spec):
        """Load a component's GPU-resident module bundle (PR-4 refactor).

        Returns: {"module": nn.Module(.to(device)),
                  "tokenizer": tokenizer | None,  # clip only
                  "spec": spec, "device": spec.device, "loaded_at": float}.

        The actual torch/diffusers work lives in `_load_component_module` so
        pure-logic tests can monkeypatch that seam without importing torch.
        NOTE: spec is the *base* identity here (caller strips LoRAs before
        calling — LoRAs are applied at adapter assembly, spec §5.5 / PR-4 L1).
        """
        bundle = await asyncio.to_thread(self._load_component_module, spec)
        return {
            "module": bundle["module"],
            "tokenizer": bundle.get("tokenizer"),
            "spec": spec,
            "device": spec.device,
            "loaded_at": time.monotonic(),
        }

    def _load_component_module(self, spec) -> dict:
        """Real GPU load (runs in a worker thread). unet→Flux2Transformer2DModel,
        clip→(AutoModelForCausalLM, AutoTokenizer), vae→AutoencoderKLFlux2.

        Reuses image_diffusers._load_component_module which encapsulates the
        from_pretrained(parent_dir) + quant_loaders fallback (the same loader
        load_from_components used in PR-2, now extracted)."""
        from src.services.inference.image_diffusers import load_component_module
        return load_component_module(spec)
```

> `load_component_module` 这个抽取函数在 **Task 6** 建(从 `load_from_components` 内的 `_load_module` 提取)。本 Task 只让 seam 测试过(monkeypatch `_load_component_module`,不触真函数)。为避免本 Task import 失败,`_load_component_module` 用**函数内 import**(上面已是),真函数 Task 6 才存在 —— 本 Task 的测试全 monkeypatch 该方法,不执行内部 import。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/test_model_manager_components.py -q`
Expected: PASS(全部,含改写的 + 新增)。

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/model_manager.py backend/tests/test_model_manager_components.py
git commit -m "refactor(image): PR-4 — _load_component_impl returns GPU module bundle via seam"
```

---

## Task 6: `DiffusersImageBackend` — `load_component_module` 抽取 + `from_loaded_components` + LoRA path + infer 重申明

**Files:**
- Modify: `backend/src/services/inference/image_diffusers.py`
- Test: `backend/tests/test_from_loaded_components.py`(新建,stub 模块)

> 目标:(1) 把 `load_from_components` 内嵌的 `_load_module` 提取成模块级 `load_component_module(spec)`(Task 5 引用它);(2) 新增 classmethod `from_loaded_components(modules, components, pipeline_class)` —— 用**已加载**的模块拼 pipe + sampler,不再自己 load;(3) `_apply_loras` 支持 `spec.path`;(4) 组件路径 `infer` 复用前重申明 LoRA;(5) 把现有 `load_from_components` 改成「自己 load base 模块 → 调 `from_loaded_components`」薄壳(保持 SSIM 测试与既有 from_components 调用方不破)。

- [ ] **Step 1: 写失败测试(用 stub 模块,不碰真 diffusers)**

```python
# backend/tests/test_from_loaded_components.py
"""PR-4: from_loaded_components 用预加载模块拼 pipe+sampler;LoRA path 应用。"""
from __future__ import annotations

import types

import pytest

from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_diffusers import DiffusersImageBackend


class _Mod:
    def __init__(self, dev):
        self._dev = dev
        self.applied = []  # set_adapters calls
        self.loaded = []   # load_lora_adapter / load_lora_weights paths

    @property
    def device(self):
        import torch
        return torch.device(self._dev)


def _modules():
    return {
        "transformer": _Mod("cuda:1"),
        "text_encoder": _Mod("cuda:0"),
        "tokenizer": object(),
        "vae": _Mod("cuda:2"),
    }


def _components(loras=None):
    return {
        "unet": ComponentSpec(kind="unet", file="/m/u.safe", device="cuda:1", dtype="bfloat16",
                              adapter_arch="flux2", loras=loras or []),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16", clip_arch="flux2"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }


def test_from_loaded_components_builds_sampler(monkeypatch):
    built = {}

    def _fake_assemble(self, modules):
        # stand in for real Flux2KleinPipeline construction (needs scheduler from disk)
        pipe = types.SimpleNamespace(
            transformer=modules["transformer"], text_encoder=modules["text_encoder"],
            tokenizer=modules["tokenizer"], vae=modules["vae"], scheduler=object())
        built["pipe"] = pipe
        return pipe

    monkeypatch.setattr(DiffusersImageBackend, "_assemble_pipe", _fake_assemble, raising=False)
    monkeypatch.setattr(
        "src.services.inference.image_diffusers.MODEL_ARCH_REGISTRY",
        {"Flux2KleinPipeline": object()}, raising=False)

    adapter = DiffusersImageBackend.from_loaded_components(_modules(), _components(), "Flux2KleinPipeline")
    assert adapter._sampler is not None
    assert adapter._sampler.pipe is built["pipe"]


def test_from_loaded_components_missing_kind_raises():
    with pytest.raises(ValueError, match="missing"):
        DiffusersImageBackend.from_loaded_components(_modules(), {"unet": _components()["unet"]}, "Flux2KleinPipeline")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_from_loaded_components.py -q`
Expected: FAIL —`from_loaded_components` / `_assemble_pipe` 不存在。

- [ ] **Step 3: 改 `image_diffusers.py`**

**3a.** 模块级提取 `load_component_module`(放在 `class DiffusersImageBackend` 之前,文件顶部 helpers 区,~line 300 前)。从现 `load_from_components` 内 `_load_module` + `_torch_dtype_from` 抽取:

```python
def _torch_dtype_from(dtype_str: str):
    import torch
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,
    }.get(dtype_str, torch.bfloat16)


def _load_hf_or_quant(spec, hf_class):
    """from_pretrained(parent_dir) first; quant_loaders fallback for single-file."""
    from pathlib import Path
    parent_dir = Path(spec.file).parent
    try:
        return hf_class.from_pretrained(parent_dir, torch_dtype=_torch_dtype_from(spec.dtype))
    except (OSError, ValueError) as primary:
        try:
            from src.services.inference.quant_loaders import QUANT_LOADERS
            sd = QUANT_LOADERS.dispatch(spec)
            module = hf_class.from_config(parent_dir / "config.json")
            module.load_state_dict(sd, strict=False)
            return module
        except Exception as quant_exc:
            raise RuntimeError(
                f"_load_hf_or_quant({spec.file}): from_pretrained failed "
                f"({type(primary).__name__}: {primary}); quant fallback also failed "
                f"({type(quant_exc).__name__}: {quant_exc})"
            ) from quant_exc


def load_component_module(spec) -> dict:
    """Load ONE component's GPU module(s) for spec (base identity — LoRAs applied
    later at assembly). Called by ModelManager._load_component_module in a worker
    thread. Returns {"module": .to(device), "tokenizer": tok|None}.

    HF-layout dirs: spec.file points at a .safetensors INSIDE the component dir
    (component_scanner emits abs_path to a real file), so Path(spec.file).parent
    is the HF component dir. tokenizer for clip lives at <root>/tokenizer where
    <root> = Path(clip.file).parent.parent (spec §4.6 layout)."""
    from pathlib import Path
    from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if spec.kind == "unet":
        mod = _load_hf_or_quant(spec, Flux2Transformer2DModel).to(spec.device)
        return {"module": mod, "tokenizer": None}
    if spec.kind == "clip":
        mod = _load_hf_or_quant(spec, AutoModelForCausalLM).to(spec.device)
        tok_dir = Path(spec.file).parent.parent / "tokenizer"
        if not tok_dir.is_dir():
            raise FileNotFoundError(
                f"tokenizer dir not found at {tok_dir} — clip component expects HF layout "
                f"<root>/{{text_encoder,tokenizer}}/")
        return {"module": mod, "tokenizer": AutoTokenizer.from_pretrained(tok_dir)}
    if spec.kind == "vae":
        mod = _load_hf_or_quant(spec, AutoencoderKLFlux2).to(spec.device)
        return {"module": mod, "tokenizer": None}
    raise ValueError(f"load_component_module: unknown kind {spec.kind!r}")
```

**3b.** 顶部 import `MODEL_ARCH_REGISTRY`(供 monkeypatch seam + 装配用)。现 `load_from_components` 是函数内 import;改成模块顶部:

```python
from src.services.inference.model_arch_adapter import MODEL_ARCH_REGISTRY
```

**3c.** 新增 `_assemble_pipe`(实例方法,从已加载模块拼 `Flux2KleinPipeline` + scheduler):

```python
    def _assemble_pipe(self, modules: dict):
        """Wire pre-loaded modules into a Flux2KleinPipeline. Scheduler is the
        only thing read from disk here (cheap); modules come from the L1 cache.
        scheduler_dir = <root>/scheduler where <root> = Path(unet.file).parent.parent."""
        from pathlib import Path
        from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler

        unet_file = self._components["unet"].file
        scheduler_dir = Path(unet_file).parent.parent / "scheduler"
        if not scheduler_dir.is_dir():
            raise FileNotFoundError(
                f"scheduler dir not found at {scheduler_dir} — unet component expects HF layout "
                f"<root>/{{transformer,scheduler}}/")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_dir)
        return Flux2KleinPipeline(
            transformer=modules["transformer"],
            text_encoder=modules["text_encoder"],
            tokenizer=modules["tokenizer"],
            vae=modules["vae"],
            scheduler=scheduler,
        )
```

**3d.** 新增 classmethod `from_loaded_components`:

```python
    @classmethod
    def from_loaded_components(
        cls,
        modules: dict,                    # {"transformer","text_encoder","tokenizer","vae"}
        components: dict[str, "ComponentSpec"],
        pipeline_class: str = "Flux2KleinPipeline",
    ) -> "DiffusersImageBackend":
        """Build an adapter from PRE-LOADED GPU modules (PR-4 L1 path). Unlike
        from_components (which loads inside load_from_components), this reuses
        cached modules from ModelManager and only assembles pipe + applies LoRAs
        + builds the sampler. LoRAs use spec.path (descriptor-carried abs path)."""
        required = {"unet", "clip", "vae"}
        missing = required - set(components.keys())
        if missing:
            raise ValueError(f"from_loaded_components: missing component kinds {sorted(missing)}")
        from src.services.inference.base import InferenceAdapter
        from src.services.inference.image_sampler import ImageSampler

        self = cls.__new__(cls)
        InferenceAdapter.__init__(self, paths={"_from_components": "true"}, device="multi")
        self._components = components
        self._pipeline_class = pipeline_class
        self._offload_strategy = "no_offload"
        self._lora_paths = {}
        self._torch_dtype = components["unet"].dtype
        self._loaded_loras = set()

        self._pipe = self._assemble_pipe(modules)
        if components["unet"].loras:
            self._apply_loras(components["unet"].loras)

        arch_adapter = MODEL_ARCH_REGISTRY.get(pipeline_class)
        if arch_adapter is None:
            raise RuntimeError(
                f"No ModelArchAdapter registered for {pipeline_class!r}. Known: {sorted(MODEL_ARCH_REGISTRY)}")
        self._sampler = ImageSampler(pipe=self._pipe, arch_adapter=arch_adapter)
        return self
```

**3e.** `_apply_loras` 支持 `spec.path`(image_diffusers.py:577 那行):

```python
                lora_path = getattr(spec, "path", None) or self._lora_paths.get(spec.name)
                if not lora_path:
                    raise ValueError(
                        f"LoRA {spec.name!r} has no path and is not in registered lora_paths "
                        f"(have: {sorted(self._lora_paths)})"
                    )
```

**3f.** 组件路径 `infer` 复用前重申明 active LoRA(image_diffusers.py:619-628)。base transformer 跨 adapter 共享 → 必须每次 sample 前 `set_adapters([本组合])`:

```python
    async def infer(
        self, req: InferenceRequest, cancel_flag: CancelFlag | None = None
    ) -> InferenceResult:
        """Dispatch to ImageSampler (component path) or legacy Pipeline.__call__."""
        if self._sampler is not None:
            # base transformer 可能被多个 (不同 LoRA 组合) adapter 共享 —— 复用前
            # 重申明本组合的 active adapter set(同 runner 串行,无并发竞态)。
            unet_loras = self._components["unet"].loras if self._components else []
            if unet_loras:
                self._apply_loras(unet_loras)
            if cancel_flag is not None:
                self._sampler.cancel_flag = cancel_flag
            return await self._sampler.sample(req)
        return await self._legacy_infer_impl(req, cancel_flag)
```

**3g.** 把现有 `load_from_components`(self 自己 load)改成委托:load base 模块(用 `load_component_module`)→ 复用装配。保持 `from_components` + `load()` 既有调用方与 SSIM 测试不破:

```python
    async def load_from_components(self) -> None:
        """Self-loading path (from_components + load()). Loads base modules then
        delegates to the shared assembly. ModelManager.get_or_load_image_adapter
        uses the cached path instead (Task 7); this remains for direct/test use
        and back-compat with PR-2 callers."""
        import torch
        unet, clip, vae = self._components["unet"], self._components["clip"], self._components["vae"]
        loaded_on_gpu = []
        try:
            t = load_component_module(unet); loaded_on_gpu.append(t["module"])
            c = load_component_module(clip); loaded_on_gpu.append(c["module"])
            v = load_component_module(vae);  loaded_on_gpu.append(v["module"])
            modules = {"transformer": t["module"], "text_encoder": c["module"],
                       "tokenizer": c["tokenizer"], "vae": v["module"]}
            self._pipe = self._assemble_pipe(modules)
            if unet.loras:
                self._apply_loras(unet.loras)
            from src.services.inference.image_sampler import ImageSampler
            arch_adapter = MODEL_ARCH_REGISTRY.get(self._pipeline_class)
            if arch_adapter is None:
                raise RuntimeError(
                    f"No ModelArchAdapter registered for {self._pipeline_class!r}. Known: {sorted(MODEL_ARCH_REGISTRY)}")
            self._sampler = ImageSampler(pipe=self._pipe, arch_adapter=arch_adapter)
        except Exception:
            for mod in loaded_on_gpu:
                try:
                    mod.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            self._pipe = None
            self._sampler = None
            raise
```

- [ ] **Step 4: 跑新测试确认通过**

Run: `cd backend && uv run pytest tests/test_from_loaded_components.py -q`
Expected: PASS(2 passed)。

- [ ] **Step 5: 回归 — SSIM / sampler / image_diffusers / flux2 组件 单测(stub 部分)**

Run: `cd backend && uv run pytest tests/test_image_sampler.py tests/test_image_diffusers.py tests/test_flux2_components_loaders.py tests/test_image_adapter_cancel.py -q`
Expected: PASS。`test_image_sampler_ssim.py` / `test_image_model_integration.py` 若标了真模型 marker(需 GPU)则在 smoke 阶段(Task 12)验,不在此跑 —— 确认它们的 marker(`grep -n "skipif\|mark" tests/test_image_sampler_ssim.py`),纯 stub 部分这里也跑。

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/inference/image_diffusers.py backend/tests/test_from_loaded_components.py
git commit -m "feat(image): PR-4 — from_loaded_components + load_component_module + LoRA path + infer re-assert"
```

---

## Task 7: `ModelManager.get_or_load_image_adapter` — auto 解析 + 组件级 L1 + adapter combo 缓存

**Files:**
- Modify: `backend/src/services/model_manager.py`(`__init__` 加 `_image_adapters`/锁;新增方法)
- Test: `backend/tests/test_get_or_load_image_adapter.py`(新建)

> 入口语义:给 `components`(含 `auto` device)+ `pipeline_class` → 返回 ready `DiffusersImageBackend`。流程:① 解析 `auto`→`cuda:N`;② 对**去 LoRA 的 base spec** 调 `get_or_load_component`(clip/vae/transformer base 跨调用复用);③ 命中 adapter combo 缓存(key=三组件**完整** key)直接返回;④ 未命中 → `from_loaded_components` 拼装 → 缓存。OOM 时 evict LRU 后重试一次。

- [ ] **Step 1: 写失败测试(全 stub,不碰 torch)**

```python
# backend/tests/test_get_or_load_image_adapter.py
"""PR-4: get_or_load_image_adapter — auto 解析 + 组件级 L1 + combo 缓存。"""
from __future__ import annotations

import pytest

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import LoRASpec
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


@pytest.fixture
def mm():
    return ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())


def _comps(unet_loras=None, unet_dev="cuda:1"):
    return {
        "unet": ComponentSpec(kind="unet", file="/m/u.safe", device=unet_dev, dtype="bfloat16",
                              adapter_arch="flux2", loras=unet_loras or []),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16", clip_arch="flux2"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }


@pytest.fixture
def stubbed(mm, monkeypatch):
    """stub 组件模块加载 + adapter 装配,记录调用。"""
    module_loads = []

    def _load_module(spec):
        module_loads.append((spec.kind, spec.file, spec.device, tuple((l.name, l.strength) for l in spec.loras)))
        return {"module": object(), "tokenizer": None}

    monkeypatch.setattr(mm, "_load_component_module", _load_module)

    assemble_calls = []

    def _fake_from_loaded(modules, components, pipeline_class):
        assemble_calls.append(components)
        return object()  # stand-in adapter

    monkeypatch.setattr(
        "src.services.inference.image_diffusers.DiffusersImageBackend.from_loaded_components",
        staticmethod(_fake_from_loaded))
    return mm, module_loads, assemble_calls


@pytest.mark.asyncio
async def test_same_combo_cache_hit(stubbed):
    mm, module_loads, assemble_calls = stubbed
    a1 = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    a2 = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    assert a1 is a2                       # combo 缓存命中
    assert len(assemble_calls) == 1       # 只装配一次
    assert len(module_loads) == 3         # unet/clip/vae 各 load 一次


@pytest.mark.asyncio
async def test_lora_change_reuses_clip_vae_reloads_unet(stubbed):
    mm, module_loads, assemble_calls = stubbed
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    await mm.get_or_load_image_adapter(
        _comps(unet_loras=[LoRASpec(name="s", path="/m/loras/s.safe", strength=0.8)]),
        "Flux2KleinPipeline")
    kinds = [m[0] for m in module_loads]
    # clip/vae 仍各只 load 一次(base spec 去 LoRA → key 不变);unet base 也只 load 一次
    # (LoRA 不进 base key) → 改 LoRA 不重 load 18GB transformer。
    assert kinds.count("clip") == 1
    assert kinds.count("vae") == 1
    assert kinds.count("unet") == 1
    assert len(assemble_calls) == 2       # 两个不同 LoRA 组合 → 两个 adapter


@pytest.mark.asyncio
async def test_auto_device_resolved(mm, monkeypatch):
    monkeypatch.setattr(mm._allocator, "get_best_gpu", lambda vram: 2)
    seen = []
    monkeypatch.setattr(mm, "_load_component_module",
                        lambda spec: (seen.append(spec.device), {"module": object(), "tokenizer": None})[1])
    monkeypatch.setattr(
        "src.services.inference.image_diffusers.DiffusersImageBackend.from_loaded_components",
        staticmethod(lambda modules, components, pc: object()))
    comps = _comps(unet_dev="auto")
    await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline")
    assert "auto" not in seen            # 解析过
    assert "cuda:2" in seen              # 解析成 allocator 选的卡
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_get_or_load_image_adapter.py -q`
Expected: FAIL —`get_or_load_image_adapter` 不存在。

- [ ] **Step 3: 改 `model_manager.py`**

`__init__`(在 `_component_failures` 后)加:

```python
        # PR-4: assembled image adapter cache (key = tuple of the 3 components'
        # FULL component keys, incl unet lora_set). Coexists with _models/_components.
        self._image_adapters: dict = {}
        self._image_adapter_locks: dict = {}
```

新增方法(放在组件 cache APIs 区附近):

```python
    # --- PR-4: assembled image adapter (component-level L1) ------------------

    _VRAM_EST_MB = {"unet": 18000, "clip": 6000, "vae": 1000}

    def _resolve_component_device(self, spec):
        """Resolve device='auto' → 'cuda:N' via allocator. Returns a NEW spec
        (ComponentSpec is mutable BaseModel; model_copy keeps validators)."""
        if spec.device != "auto":
            return spec
        idx = self._allocator.get_best_gpu(self._VRAM_EST_MB.get(spec.kind, 8000))
        resolved = f"cuda:{idx}" if idx >= 0 else "cpu"
        return spec.model_copy(update={"device": resolved})

    @staticmethod
    def _base_spec(spec):
        """Strip LoRAs → base module identity (LoRAs applied at assembly)."""
        return spec.model_copy(update={"loras": []}) if spec.loras else spec

    def _image_adapter_lock_for(self, key) -> asyncio.Lock:
        return self._image_adapter_locks.setdefault(key, asyncio.Lock())

    async def get_or_load_image_adapter(self, components: dict, pipeline_class: str = "Flux2KleinPipeline"):
        """PR-4 entry for the runner component path. Resolves auto devices,
        loads/reuses base modules via the component L1 cache, assembles (or
        reuses) a DiffusersImageBackend keyed by the full 3-component combo.

        OOM resilience: on first-load CUDA OOM, evict legacy LRU then retry once.
        """
        from src.services.inference.component_spec import to_component_key
        from src.services.inference.image_diffusers import DiffusersImageBackend

        resolved = {k: self._resolve_component_device(s) for k, s in components.items()}
        combo_key = (pipeline_class,) + tuple(
            to_component_key(resolved[k]) for k in ("unet", "clip", "vae"))

        async with self._image_adapter_lock_for(combo_key):
            cached = self._image_adapters.get(combo_key)
            if cached is not None:
                return cached

            for attempt in range(2):
                try:
                    # base modules (LoRA-stripped) → component L1 cache
                    base = {k: self._base_spec(resolved[k]) for k in ("unet", "clip", "vae")}
                    t = await self.get_or_load_component(base["unet"])
                    c = await self.get_or_load_component(base["clip"])
                    v = await self.get_or_load_component(base["vae"])
                    modules = {"transformer": t["module"], "text_encoder": c["module"],
                               "tokenizer": c["tokenizer"], "vae": v["module"]}
                    adapter = DiffusersImageBackend.from_loaded_components(
                        modules, resolved, pipeline_class)
                    self._image_adapters[combo_key] = adapter
                    return adapter
                except Exception as e:  # noqa: BLE001
                    if self._is_oom(e) and attempt == 0:
                        evicted = await self.evict_lru()
                        logger.warning("get_or_load_image_adapter OOM, evicted %r, retry", evicted)
                        continue
                    raise
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/test_get_or_load_image_adapter.py -q`
Expected: PASS(3 passed)。

- [ ] **Step 5: 回归 component cache**

Run: `cd backend && uv run pytest tests/test_model_manager_components.py -q`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/model_manager.py backend/tests/test_get_or_load_image_adapter.py
git commit -m "feat(image): PR-4 — ModelManager.get_or_load_image_adapter (component L1 + combo cache + auto)"
```

---

## Task 8: runner `_node_executor` — components dispatch 路径

**Files:**
- Modify: `backend/src/runner/runner_process.py:184-217`(adapter 获取分流)
- Test: `backend/tests/test_runner_components_dispatch.py`(新建,用 fake adapter)

> 现 `_node_executor` 用 `state.mm.get_or_load(node.model_key)` 取 adapter。改:先 `_build_request(node)`,若 `req.components` 非空 → `adapter = await state.mm.get_or_load_image_adapter(req.components, req.pipeline_class)`;否则保持 model_key 路径。`_build_request` 现在在取 adapter 之前调一次(注意现码是在 try 块里调 `_build_request` —— 需提前到 adapter 获取处,或两处协调。下面给出协调后的结构)。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_runner_components_dispatch.py
"""PR-4: _node_executor 走 components 路径 (get_or_load_image_adapter)。"""
from __future__ import annotations

import threading

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel
from src.runner.runner_process import _RunnerState, _node_executor


class _FakeAdapter:
    is_loaded = True

    async def infer(self, req, **kw):
        from src.services.inference.base import InferenceResult, UsageMeter
        return InferenceResult(media_type="image/png", data=b"\x89PNG\r\n",
                               metadata={"width": req.width, "height": req.height, "seed": req.seed},
                               usage=UsageMeter(image_count=1, latency_ms=1))


class _FakeMM:
    def __init__(self):
        self.calls = []

    async def get_or_load_image_adapter(self, components, pipeline_class):
        self.calls.append((tuple(sorted(components)), pipeline_class))
        return _FakeAdapter()

    async def get_or_load(self, key):  # legacy path — should NOT be hit here
        raise AssertionError("legacy get_or_load called on components path")


class _Collect(PipeChannel):
    def __init__(self):
        self.sent = []

    async def send_message(self, m):
        self.sent.append(m)


@pytest.mark.asyncio
async def test_components_path_uses_image_adapter(monkeypatch):
    mm = _FakeMM()
    state = _RunnerState("r", "image", [0, 1, 2], mm)
    ch = _Collect()
    node = P.RunNode(
        task_id=5, node_id="g", node_type="image", model_key=None,
        inputs={
            "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
            "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
            "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
            "prompt": "a cat", "seed": 42, "width": 256, "height": 256, "steps": 4,
        })
    state.cancel_flags[5] = threading.Event()
    state.run_queue.put_nowait(node)

    import asyncio
    task = asyncio.create_task(_node_executor(state, ch))
    await asyncio.sleep(0.2)
    state.shutdown.set()
    await asyncio.wait_for(task, timeout=2)

    results = [m for m in ch.sent if isinstance(m, P.NodeResult)]
    assert results and results[-1].status == "completed"
    assert results[-1].outputs["image_url"]            # write_image signed URL
    assert mm.calls == [(("clip", "unet", "vae"), "Flux2KleinPipeline")]
```

> 该测试需 `ADMIN_SESSION_SECRET` 才能签 URL(`write_image`)。conftest 已设;若无,断言改 `results[-1].status == "completed"` 即可(write_image url=None 时 outputs 无 image_url 但不 fail)。优先依赖 conftest 的 secret。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_runner_components_dispatch.py -q`
Expected: FAIL —现 `_node_executor` 调 `get_or_load(None)` → adapter None → status=failed("has no model_key")。

- [ ] **Step 3: 改 `_node_executor`(runner_process.py)**

把「取 adapter」段(:196-216)改为先 build req、按 components 分流。**关键**:`_build_request` 提前到这里调用,失败(ValueError)走原 ValueError 处理;下面 try 块里**不再**重复 `_build_request`(改为复用 `req`)。

```python
        cancel_flag = state.cancel_flags.get(node.task_id) or threading.Event()
        started = time.monotonic()

        # 先 build typed request —— components 路径据此分流 adapter 获取方式。
        try:
            req = _build_request(node)
        except ValueError as e:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=str(e),
                duration_ms=int((time.monotonic() - started) * 1000)))
            state.cancel_flags.pop(node.task_id, None)
            continue

        # adapter 获取:components 路径走 get_or_load_image_adapter(组件级 L1 +
        # combo 缓存);否则老 model_key 路径(get_or_load,含 OOM evict)。
        try:
            components = getattr(req, "components", None)
            if components:
                adapter = await state.mm.get_or_load_image_adapter(
                    components, getattr(req, "pipeline_class", "Flux2KleinPipeline"))
            else:
                adapter = await state.mm.get_or_load(node.model_key) if node.model_key else None
        except (ModelLoadError, ModelNotFoundError) as e:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=f"{type(e).__name__}: {e}",
                duration_ms=int((time.monotonic() - started) * 1000)))
            state.cancel_flags.pop(node.task_id, None)
            continue

        if adapter is None:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=f"node {node.node_id!r} has no model_key / components",
                duration_ms=int((time.monotonic() - started) * 1000)))
            state.cancel_flags.pop(node.task_id, None)
            continue
```

然后下面原 `try:` 块里删掉 `req = _build_request(node)` 那一行(req 已在上面 build 好),其余(tts 分支 / image signature 探测 / 发 result)不变。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/test_runner_components_dispatch.py -q`
Expected: PASS。

- [ ] **Step 5: 回归 runner 既有行为(model_key 路径 + cancel)**

Run: `cd backend && uv run pytest tests/ -q -k "runner" `
Expected: PASS(确认老 model_key 路径、cancel、load/unload 不破)。

- [ ] **Step 6: Commit**

```bash
git add backend/src/runner/runner_process.py backend/tests/test_runner_components_dispatch.py
git commit -m "feat(image): PR-4 — runner _node_executor components dispatch path"
```

---

## Task 9: `component_expand` — 老 model_key → 三 ComponentSpec

**Files:**
- Create: `backend/src/services/inference/component_expand.py`
- Test: `backend/tests/test_component_expand.py`(新建)

> 给定 `ModelSpec`(model_key 查出)+ 可选老格式 loras list,产 `{"unet","clip","vae"}` 三个 `ComponentSpec`(device=`auto`),unet 合并 loras。组件 file 从 `paths['main']`(HF root)推 `<root>/{transformer,text_encoder,vae}/` 的代表 .safetensors;`paths['quantized_transformer']` 覆盖 unet file。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_component_expand.py
"""PR-4: 老 model_key → 三 ComponentSpec inline 展开。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.services.inference.component_expand import expand_legacy_image_spec
from src.services.inference.registry import ModelSpec


def _write_layout(root: Path):
    for sub in ("transformer", "text_encoder", "vae", "scheduler", "tokenizer"):
        (root / sub).mkdir(parents=True)
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    (root / "text_encoder" / "model.safetensors").write_bytes(b"x")
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")


def test_expand_hf_layout(tmp_path, monkeypatch):
    root = tmp_path / "Flux2-klein-9B"
    _write_layout(root)
    monkeypatch.setenv("LOCAL_MODELS_PATH", str(tmp_path))
    spec = ModelSpec(id="flux2-klein-9b", model_type="image",
                     adapter_class="src.services.inference.image_diffusers.DiffusersImageBackend",
                     paths={"main": "Flux2-klein-9B"}, vram_mb=24000,
                     params={"accepts_lora_archs": ["flux2"]})
    comps = expand_legacy_image_spec(spec, loras=None)
    assert set(comps) == {"unet", "clip", "vae"}
    assert comps["unet"].device == "auto"
    assert comps["unet"].adapter_arch == "flux2"
    assert Path(comps["unet"].file).parent.name == "transformer"
    assert Path(comps["clip"].file).parent.name == "text_encoder"
    assert Path(comps["vae"].file).parent.name == "vae"


def test_expand_quantized_transformer_override(tmp_path, monkeypatch):
    root = tmp_path / "Flux2-klein-9B"
    _write_layout(root)
    (tmp_path / "qt").mkdir()
    qt = tmp_path / "qt" / "Flux2-fp8mixed.safetensors"
    qt.write_bytes(b"x")
    monkeypatch.setenv("LOCAL_MODELS_PATH", str(tmp_path))
    spec = ModelSpec(id="q", model_type="image",
                     adapter_class="src.services.inference.image_diffusers.DiffusersImageBackend",
                     paths={"main": "Flux2-klein-9B", "quantized_transformer": "qt/Flux2-fp8mixed.safetensors"},
                     vram_mb=18000, params={})
    comps = expand_legacy_image_spec(spec, loras=None)
    assert comps["unet"].file == str(qt)


def test_expand_merges_loras(tmp_path, monkeypatch):
    root = tmp_path / "Flux2-klein-9B"
    _write_layout(root)
    monkeypatch.setenv("LOCAL_MODELS_PATH", str(tmp_path))
    spec = ModelSpec(id="flux2-klein-9b", model_type="image",
                     adapter_class="x.DiffusersImageBackend",
                     paths={"main": "Flux2-klein-9B"}, vram_mb=24000, params={})
    comps = expand_legacy_image_spec(spec, loras=[{"name": "style", "strength": 0.7}])
    assert comps["unet"].loras[0].name == "style"
    assert comps["unet"].loras[0].strength == 0.7
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_component_expand.py -q`
Expected: FAIL —模块不存在。

- [ ] **Step 3: 新建 `component_expand.py`**

```python
# backend/src/services/inference/component_expand.py
"""PR-4 §7.4: legacy image model_key → 3 ComponentSpec (backend inline expand).

Old workflows reference image_generate by model_key only. The runner component
path needs unet/clip/vae descriptors, so before dispatch we translate the
model's yaml ModelSpec into 3 ComponentSpec (device='auto') and fold any
legacy LoRA list into the unet descriptor. File paths follow the HF layout
<root>/{transformer,text_encoder,vae}/ where <root> = paths['main'];
quantized_transformer overrides the unet file.
"""
from __future__ import annotations

import glob
from pathlib import Path

from src.services.inference.base import LoRASpec
from src.services.inference.component_spec import ComponentSpec


def _abs(rel_or_abs: str) -> Path:
    from src.config import get_settings
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return Path(get_settings().LOCAL_MODELS_PATH) / p


def _representative_file(component_dir: Path) -> str:
    """Pick a real .safetensors inside the HF component dir so Path(file).parent
    == the dir (load_component_module does from_pretrained(parent)). Falls back
    to a synthetic path under the dir if none on disk (load will then error
    clearly rather than silently mis-pathing)."""
    hits = sorted(glob.glob(str(component_dir / "*.safetensors")))
    if hits:
        return hits[0]
    return str(component_dir / "model.safetensors")


def expand_legacy_image_spec(spec, loras: list[dict] | None = None) -> dict[str, ComponentSpec]:
    main = _abs(spec.paths["main"])
    arch = (spec.params.get("accepts_lora_archs") or ["flux2"])[0]

    qt = spec.paths.get("quantized_transformer")
    unet_file = str(_abs(qt)) if qt else _representative_file(main / "transformer")
    clip_file = _representative_file(main / "text_encoder")
    vae_file = _representative_file(main / "vae")

    lora_specs = [
        LoRASpec(name=l["name"], strength=float(l.get("strength", 1.0)), path=l.get("path"))
        for l in (loras or []) if isinstance(l, dict) and l.get("name")
    ]
    return {
        "unet": ComponentSpec(kind="unet", file=unet_file, device="auto", dtype="bfloat16",
                              adapter_arch=arch, loras=lora_specs),
        "clip": ComponentSpec(kind="clip", file=clip_file, device="auto", dtype="bfloat16", clip_arch=arch),
        "vae":  ComponentSpec(kind="vae",  file=vae_file,  device="auto", dtype="bfloat16"),
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/test_component_expand.py -q`
Expected: PASS(3 passed)。

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/inference/component_expand.py backend/tests/test_component_expand.py
git commit -m "feat(image): PR-4 — component_expand (legacy model_key → 3 ComponentSpec)"
```

---

## Task 10: `workflow_executor` — 老格式 inline 展开接线

**Files:**
- Modify: `backend/src/services/workflow_executor.py`(`_dispatch_node` 内,merged_inputs 后)
- Test: `backend/tests/test_workflow_executor_legacy_expand.py`(新建)

> 在 `_dispatch_node` 算完 `merged_inputs` 后:若 `node.type == "image_generate"` 且 `model_key` 有值 且 `merged_inputs` 无 unet/clip/vae → 用 `expand_legacy_image_spec` 展开,把三 ComponentSpec 的 `model_dump()` 塞进 `merged_inputs["unet"/"clip"/"vae"]`,LoRA 取 `merged_inputs.get("loras")`。需 registry 取 ModelSpec —— 用全局 `_model_manager._registry`。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_workflow_executor_legacy_expand.py
"""PR-4 §7.4: 老 image_generate(model_key) dispatch 前 inline 展开成三组件。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.runner import protocol as P
from src.services import workflow_executor as we
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.registry import ModelRegistry, ModelSpec
from src.services.model_manager import ModelManager
from src.services.workflow_executor import WorkflowExecutor


class _Capturing:
    def __init__(self):
        self.spec = None

    async def run_node(self, spec, *, workflow_name=""):
        self.spec = spec
        return P.NodeResult(task_id=spec.task_id, node_id=spec.node_id, status="completed",
                            outputs={"image_url": "u"}, error=None, duration_ms=1)


class _Reg(ModelRegistry):
    def __init__(self, spec):
        self._config_path = ""
        self._specs = {spec.id: spec}


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root = tmp_path / "Flux2-klein-9B"
    for sub in ("transformer", "text_encoder", "vae"):
        (root / sub).mkdir(parents=True)
        (root / sub / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("LOCAL_MODELS_PATH", str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_legacy_image_node_expanded(layout, monkeypatch):
    spec = ModelSpec(id="flux2-klein-9b", model_type="image",
                     adapter_class="src.services.inference.image_diffusers.DiffusersImageBackend",
                     paths={"main": "Flux2-klein-9B"}, vram_mb=24000, params={"accepts_lora_archs": ["flux2"]})
    mm = ModelManager(registry=_Reg(spec), allocator=GPUAllocator())
    monkeypatch.setattr(we, "_model_manager", mm)

    wf = {"nodes": [{"id": "g", "type": "image_generate",
                     "data": {"model_key": "flux2-klein-9b", "prompt": "x", "seed": 1,
                              "loras": [{"name": "style", "strength": 0.6}]}}],
          "edges": []}
    client = _Capturing()
    ex = WorkflowExecutor(wf, runner_clients={"image": client}, task_id=3)
    await ex._dispatch_node(ex._node_map["g"], {})

    inp = client.spec.inputs
    assert set(("unet", "clip", "vae")) <= set(inp)
    assert inp["unet"]["loras"][0]["name"] == "style"
    assert inp["unet"]["device"] == "auto"
    assert Path(inp["clip"]["file"]).parent.name == "text_encoder"


@pytest.mark.asyncio
async def test_new_format_not_expanded(layout, monkeypatch):
    mm = ModelManager(registry=_Reg(ModelSpec(id="x", model_type="image", adapter_class="a",
                                              paths={"main": "Flux2-klein-9B"}, vram_mb=1)),
                      allocator=GPUAllocator())
    monkeypatch.setattr(we, "_model_manager", mm)
    wf = {"nodes": [{"id": "g", "type": "image_generate", "data": {"prompt": "x"}}], "edges": []}
    client = _Capturing()
    ex = WorkflowExecutor(wf, runner_clients={"image": client}, task_id=3)
    upstream_unet = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []}
    await ex._dispatch_node(ex._node_map["g"], {
        "unet": upstream_unet,
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
    })
    # 已有三组件 → 不展开,unet 原样透传
    assert client.spec.inputs["unet"] is upstream_unet
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_workflow_executor_legacy_expand.py -q`
Expected: FAIL —老节点 inputs 无 unet/clip/vae。

- [ ] **Step 3: 改 `_dispatch_node`**(merged_inputs 之后、算 is_deterministic 之前)

```python
        merged_inputs = {**{k: v for k, v in data.items() if not k.startswith("_")}, **inputs}

        # spec §7.4: 老格式 image_generate(只有 model_key、无 unet/clip/vae 边)→
        # 后端 inline 展开成三 ComponentSpec(device=auto),之后与新格式同路径。
        if (
            node["type"] == "image_generate"
            and model_key
            and not all(k in merged_inputs for k in ("unet", "clip", "vae"))
        ):
            from src.services.inference.component_expand import expand_legacy_image_spec
            if _model_manager is None:
                raise ExecutionError("老格式 image_generate 展开需要 ModelManager(_model_manager 未注入)")
            spec_obj = _model_manager._registry.get(model_key)
            if spec_obj is None:
                raise ExecutionError(f"老格式 image_generate 展开失败:model_key {model_key!r} 无 ModelSpec")
            comps = expand_legacy_image_spec(spec_obj, loras=merged_inputs.get("loras"))
            for kind in ("unet", "clip", "vae"):
                merged_inputs[kind] = comps[kind].model_dump()

        is_deterministic = merged_inputs.get("seed") not in (None, "")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/test_workflow_executor_legacy_expand.py -q`
Expected: PASS(2 passed)。

- [ ] **Step 5: 回归 executor + 老 image workflow publish 测试**

Run: `cd backend && uv run pytest tests/test_workflow_executor.py tests/test_workflow_publish_image.py -q`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/workflow_executor.py backend/tests/test_workflow_executor_legacy_expand.py
git commit -m "feat(image): PR-4 — workflow_executor legacy image_generate inline expansion (§7.4)"
```

---

## Task 11: 集成测试 — 全链路 fake_adapter + L1 缓存命中

**Files:**
- Test: `backend/tests/test_image_components_e2e.py`(新建)

> spec §9 集成(a):4 loader 节点 + image_generate 经 WorkflowExecutor(inline 跑 loader)+ FakeRunnerClient/真 runner client(fake_adapter)→ 描述符流转 + combo 缓存命中。这里用一个把 `RunnerClient` 替成「直接调一个共享 `_FakeMM`」的轻客户端,断言:① 三描述符正确到达 runner 侧 `_build_request`;② 同图二跑 combo 缓存命中(image adapter 只装配一次);③ 改 LoRA 二跑 clip/vae 复用。

- [ ] **Step 1: 写测试(直接驱动 _node_executor 两次,验缓存)**

```python
# backend/tests/test_image_components_e2e.py
"""PR-4 §9 集成(a): loader→image_generate 描述符流转 + 组件级 L1 缓存命中。"""
from __future__ import annotations

import asyncio
import threading

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel
from src.runner.runner_process import _RunnerState, _node_executor
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager
from src.services.nodes.registry import get_node_class


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""; self._specs = {}


class _Collect(PipeChannel):
    def __init__(self):
        self.sent = []

    async def send_message(self, m):
        self.sent.append(m)


async def _build_descriptors():
    """跑 inline loader 节点链产生三描述符(同 WorkflowExecutor 会做的)。"""
    unet = (await get_node_class("image_unet_load")().invoke(
        {"file": "/m/u.safe", "device": "cuda:1", "adapter_arch": "flux2"}, {}))["unet"]
    unet = (await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "style", "lora_path": "/m/loras/style.safe", "strength": 0.8}, {"unet": unet}))["unet"]
    clip = (await get_node_class("image_clip_load")().invoke(
        {"file": "/m/c.safe", "device": "cuda:0", "clip_arch": "flux2"}, {}))["clip"]
    vae = (await get_node_class("image_vae_load")().invoke({"file": "/m/v.safe", "device": "cuda:2"}, {}))["vae"]
    return unet, clip, vae


async def _run_once(state, ch, inputs, task_id):
    node = P.RunNode(task_id=task_id, node_id="g", node_type="image", model_key=None, inputs=inputs)
    state.cancel_flags[task_id] = threading.Event()
    state.run_queue.put_nowait(node)
    task = asyncio.create_task(_node_executor(state, ch))
    await asyncio.sleep(0.15)
    state.shutdown.set()
    await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_e2e_descriptor_flow_and_cache(monkeypatch):
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())

    module_loads, assemble = [], []
    monkeypatch.setattr(mm, "_load_component_module",
                        lambda spec: (module_loads.append((spec.kind, tuple((l.name, l.strength) for l in spec.loras))),
                                      {"module": object(), "tokenizer": None})[1])

    class _FakeAdapter:
        async def infer(self, req, **kw):
            from src.services.inference.base import InferenceResult, UsageMeter
            return InferenceResult(media_type="image/png", data=b"\x89PNG",
                                   metadata={"width": req.width, "height": req.height, "seed": req.seed},
                                   usage=UsageMeter(image_count=1, latency_ms=1))

    monkeypatch.setattr(
        "src.services.inference.image_diffusers.DiffusersImageBackend.from_loaded_components",
        staticmethod(lambda modules, components, pc: (assemble.append(pc), _FakeAdapter())[1]))

    unet, clip, vae = await _build_descriptors()
    base_inputs = lambda seed: {"unet": unet, "clip": clip, "vae": vae, "prompt": "a cat",
                                "seed": seed, "width": 256, "height": 256, "steps": 4}

    # 第一次:装配一次,三模块各 load 一次
    state = _RunnerState("r", "image", [0, 1, 2], mm); ch = _Collect()
    await _run_once(state, ch, base_inputs(42), 1)
    assert [m for m in ch.sent if isinstance(m, P.NodeResult)][-1].status == "completed"
    assert len(assemble) == 1
    assert len(module_loads) == 3

    # 第二次 同描述符 同 seed:combo 缓存命中,不再装配、不再 load
    state2 = _RunnerState("r", "image", [0, 1, 2], mm); ch2 = _Collect()
    await _run_once(state2, ch2, base_inputs(42), 2)
    assert len(assemble) == 1
    assert len(module_loads) == 3

    # 第三次 改 LoRA strength:combo 变 → 重装配;但 base 模块(去 LoRA)全命中,不再 load
    unet_b = {**unet, "loras": [{"name": "style", "path": "/m/loras/style.safe", "strength": 0.4}]}
    state3 = _RunnerState("r", "image", [0, 1, 2], mm); ch3 = _Collect()
    await _run_once(state3, ch3, {**base_inputs(42), "unet": unet_b}, 3)
    assert len(assemble) == 2          # 新 LoRA 组合 → 新 adapter
    assert len(module_loads) == 3      # 没有任何额外模块 load(clip/vae/unet base 全复用)
```

- [ ] **Step 2: 跑测试**

Run: `cd backend && uv run pytest tests/test_image_components_e2e.py -q`
Expected: PASS(若失败,按断言定位是描述符流转还是缓存键问题)。

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_image_components_e2e.py
git commit -m "test(image): PR-4 — e2e descriptor flow + component L1 cache hit (fake adapter)"
```

---

## Task 12: 真模型 smoke 验证(跨卡出图 + LoRA reswap + 同 seed)

> 遵循 [[feedback-verify-real-model]]:stub 过 ≠ 正确。本 Task 用真 Flux2-Klein-9B 在真卡上验核心假设。**非 TDD 编码** —— 是验收脚本 + 人工确认。需 GPU + 模型在盘。

**起 backend 前先处理端口(用户指令)**:`:8000` 上 PID `3529010` 是生产 backend(uvicorn,已跑 17h+)。**复用它**(直接打它的 API)或先停再起;**不要**再起一个撞 `:8000`。本 smoke 不必走 HTTP,可用 standalone 脚本直接构 `ModelManager` + runner 路径(见 [[dev-env-gotchas]]:真模型测试要 standalone、`uv` 不 load `.env`)。

**Files:**
- Create: `backend/scripts/smoke_pr4_components.py`(一次性验收脚本,可不提交或提交到 scripts/)

- [ ] **Step 1: 确认真模型在盘 + 三卡可见**

Run: `cd backend && ls "$LOCAL_MODELS_PATH/image/diffusers/Flux2-klein-9B/"{transformer,text_encoder,vae,scheduler,tokenizer}` (确认 HF layout 完整)
Run: `nvidia-smi --query-gpu=index,name,memory.free --format=csv`(确认 cuda:0=Pro6000?见 [[user-hardware]] CUDA 索引坑:cuda:1=Pro 6000)

- [ ] **Step 2: 写 standalone smoke 脚本**

```python
# backend/scripts/smoke_pr4_components.py
"""PR-4 真模型 smoke:跨卡三组件出图 + SSIM vs 单卡 baseline + LoRA reswap + 同 seed。

跑法(standalone,绕开 uv 不 load .env;见 dev-env-gotchas):
  cd backend && set -a && source .env && set +a && \
    .venv/bin/python scripts/smoke_pr4_components.py
"""
import asyncio, time
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import ImageRequest, LoRASpec
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


def _spec(kind, sub, dev, file_name, **kw):
    import glob, os
    base = os.path.expandvars("$LOCAL_MODELS_PATH/image/diffusers/Flux2-klein-9B")
    f = sorted(glob.glob(f"{base}/{sub}/*.safetensors"))[0]
    return ComponentSpec(kind=kind, file=f, device=dev, dtype="bfloat16", **kw)


async def main():
    mm = ModelManager(registry=ModelRegistry.__new__(ModelRegistry), allocator=GPUAllocator())
    mm._registry._specs = {}; mm._registry._config_path = ""
    comps = {
        "unet": _spec("unet", "transformer", "cuda:1", None, adapter_arch="flux2"),
        "clip": _spec("clip", "text_encoder", "cuda:0", None, clip_arch="flux2"),
        "vae":  _spec("vae",  "vae",          "cuda:2", None),
    }
    # 1) 跨卡出图
    t0 = time.monotonic()
    adapter = await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline")
    res = await adapter.infer(ImageRequest(request_id="s1", prompt="a red fox in snow, photorealistic",
                                           seed=42, steps=9, width=1024, height=1024))
    print(f"[1] cross-GPU first run: {time.monotonic()-t0:.1f}s, {len(res.data)} bytes")
    open("/tmp/pr4_crossgpu.png", "wb").write(res.data)

    # 2) 同 seed 二跑(combo 缓存命中 → 应极快;像素应一致)
    t0 = time.monotonic()
    res2 = await adapter.infer(ImageRequest(request_id="s2", prompt="a red fox in snow, photorealistic",
                                            seed=42, steps=9, width=1024, height=1024))
    print(f"[2] same-seed rerun: {time.monotonic()-t0:.1f}s (adapter reuse)")
    open("/tmp/pr4_crossgpu2.png", "wb").write(res2.data)

    # 3) 改 LoRA(若盘上有 flux2 lora)二跑:clip/vae 复用,只重 patch unet
    # comps_lora = {**comps, "unet": comps["unet"].model_copy(update={"loras":[LoRASpec(name="<x>", path="<abs>", strength=0.8)]})}
    # t0=time.monotonic(); a2=await mm.get_or_load_image_adapter(comps_lora,"Flux2KleinPipeline")
    # await a2.infer(ImageRequest(request_id="s3", prompt="...", seed=42, steps=9)); print(f"[3] lora reswap: {time.monotonic()-t0:.1f}s")

asyncio.run(main())
```

- [ ] **Step 3: 跑 smoke,人工核对**

Run: `cd backend && set -a && source .env && set +a && .venv/bin/python scripts/smoke_pr4_components.py`
Expected(成功标准 spec §2):
  - [1] 跨卡(unet→cuda:1 / clip→cuda:0 / vae→cuda:2)出图成功,≤ 35s,`/tmp/pr4_crossgpu.png` 是正常图(肉眼看 prompt 相关、无噪声/色块)
  - [2] 同 seed 二跑 adapter 复用、明显更快;两张 PNG 像素一致(`cmp /tmp/pr4_crossgpu.png /tmp/pr4_crossgpu2.png` 应一致或 SSIM≈1)
  - 若有 flux2 LoRA:[3] reswap < 5s 且无 18GB 重载(nvidia-smi 显存不大幅波动)

- [ ] **Step 4: SSIM vs 单卡 baseline(可选但 spec 要求 >0.99)**

复用既有 `tests/test_image_sampler_ssim.py` 的 baseline 思路:把三组件都 `.to("cuda:1")` 单卡跑一遍,跟跨卡输出算 SSIM。若 `test_image_sampler_ssim.py` 已有真模型 marker,直接:
Run: `cd backend && set -a && source .env && set +a && .venv/bin/python -m pytest tests/test_image_sampler_ssim.py -q -m <真模型marker>`
Expected: SSIM > 0.99(自写采样数学正确,跨卡不影响)。

- [ ] **Step 5: 老 workflow 回归(零改动跑通)**

用 master 上现有两个老 workflow(309542918354374656 / 308084173191516160)走一遍(经生产 backend `:8000` 或 standalone executor),确认 inline 展开后正常出图。
Expected: 出图成功,无报错(验 §7.4 inline 展开 + auto device 解析)。

- [ ] **Step 6: Commit(脚本可选提交)**

```bash
git add backend/scripts/smoke_pr4_components.py
git commit -m "test(image): PR-4 — real-model cross-GPU smoke script"
```

---

## 全套验证(完成所有 Task 后)

- [ ] **Lint / 类型 / 构建(push 前必跑,见 [[feedback-preflight-lint]]):**

```bash
cd backend && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
cd backend && uv run pytest -q          # 全后端套件绿
```

(本 PR 不含前端改动 —— 4 节点的 React 组件 + palette 是 PR-5;无需 tsc/vite build。)

- [ ] **完成开发分支:** 用 superpowers:finishing-a-development-branch — 确认全绿后开 PR 走 CI/CD(见 [[feedback-pr-per-change]] / [[feedback-auto-merge]]:CI 绿后直接 merge)。

---

## Self-Review(对 spec 核对)

- **§4.1-4.4 节点 schema**:Task 2 实现 4 节点的描述符输出(form 字段映射到 data;output port = kind 名)。前端表单/枚举/校验 = PR-5,本 PR 后端不做 schema 暴露。✓
- **§4.5 image_generate 改造**:多输入端口(unet/clip/vae)= 经 edges 落 merged_inputs(Task 3/4/10);端口未连的红色校验 = 前端 PR-5。✓(后端侧「三组件齐则走 components」覆盖)
- **§5.1 ComponentSpec / §5.5 ComponentKey + 跨字段约束**:PR-1 已建,本 PR 复用;device auto 解析 Task 7。✓
- **§5.4 runner protocol**:Task 3 _build_request components 分支(代码即 spec)。✓
- **§5.6 ImageSampler**:PR-2 已建,本 PR 经 from_loaded_components 复用(Task 6),不改采样数学(SSIM 回归 Task 6/12)。✓
- **§7.4 后端 inline 展开**:Task 9+10。✓
- **§3.3 L1 缓存**:组件级(模块层 + adapter combo 层)Task 5/6/7;clip/vae 复用 + LoRA reswap Task 7/11/12。✓ **L2 output cache = PR-6,不在本 PR。**
- **§2 成功标准**:跨卡出图/同 seed 二跑/改 LoRA 二跑/老 workflow 零改动 → Task 12 真模型验。Cancel mid-sampler(§G2)= PR-2 已接 cancel_flag,本 PR infer 透传(Task 6 3f),Task 12 可附带验。✓
- **类型一致性**:`from_loaded_components(modules, components, pipeline_class)` 签名在 Task 6 定义、Task 7 调用一致;`get_or_load_image_adapter(components, pipeline_class)` Task 7 定义、Task 8 调用一致;`load_component_module(spec)` Task 6 定义、Task 5 调用一致;`expand_legacy_image_spec(spec, loras)` Task 9 定义、Task 10 调用一致。✓
- **占位扫描**:无 TBD / "类似上文" / 无代码的步骤。✓

**已知缩减(明确非本 PR)**:GGUF(PR-7)、L2 output cache(PR-6)、前端节点/palette/useComponentState(PR-5)、组件/adapter LRU 淘汰(仅 OOM 时 evict legacy LRU;细粒度组件淘汰留后续)、批量推理 batch>1(spec 非目标)。
