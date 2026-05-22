# Image 细粒度图收敛 — PR-1(后端:编译→一次派发 + 整模型单卡)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development(推荐)或 superpowers:executing-plans 逐 Task 实施。步骤用 checkbox(`- [ ]`)。

**Goal:** 把 flux2 细粒度图从「inline 主进程吃 GPU」改成「inline 描述符产出 + 末端 VAE Decode 一次派发到 image runner」,整模型(transformer+clip+vae+lora)装到工作流所选的**单张卡**,复用已建好的 `get_or_load_image_adapter` + `ImageSampler` 执行引擎。Load Diffusion Model 补 `file`/`weight_dtype`/`device`,Load VAE 补 `file`/`weight_dtype`,Load CLIP 补 `file`/`weight_dtype`(单编码器;clip_stack 多编码器留 PR-3)。

**Architecture:** 细粒度图是线性链(Load* → Encode → KSampler → VAE Decode)。Load*/Encode/KSampler 作为 **inline 描述符产出节点**(主进程 event loop,不碰 GPU),只产/累积**嵌套 plain dict 描述符**(无张量):

```
Load Diffusion → unet 描述符 {_type:flux2_model, spec:{kind:unet,file,device,dtype,adapter_arch}, loras:[]}
Load LoRA      → 往 unet.loras append {name,path,strength}
Load CLIP      → clip 描述符 {_type:flux2_clip, type:"flux2", encoders:[{kind:clip,file,dtype}]}
Load VAE       → vae 描述符  {_type:flux2_vae, spec:{kind:vae,file,dtype}}
Encode Prompt  → conditioning 描述符 {_type:flux2_conditioning, clip:<clip bundle>, text, negative}
KSampler       → latent 描述符 {_type:flux2_latent, model:<unet bundle>, conditioning:<cond bundle>, width,height,steps,cfg_scale,seed}
VAE Decode  ★dispatch★  inputs={vae:<vae bundle>, latent:<latent bundle>}
```

`flux2_vae_decode` 进 `DISPATCH_NODE_TYPES`(role=image)→ `_dispatch_node` 把 `merged_inputs`(含嵌套 latent + vae)随 `RunNode` 投到 image runner。runner `_build_request` **walk 嵌套 latent → 摊平成 ImageRequest**:unet/clip/vae 三 ComponentSpec,**clip/vae 的 device 强制 = unet 的 device**(整模型单卡),prompt/参数从 latent 取。runner `_node_executor` 走 `get_or_load_image_adapter(components, pipeline_class)`(整模型装单卡;cross-device `.to()` 退化 no-op)→ `ImageSampler.sample()` → 出图。中间张量全在 runner 内,不跨进程。

**Tech Stack:** Python 3.12 / FastAPI / diffusers(Flux2KleinPipeline)/ pydantic v2 / pytest + pytest-asyncio。runner = multiprocessing 子进程 + msgpack pipe。flux2 节点 = plugin executor(`backend/nodes/flux2-components/`)。

**Branch:** `feat/image-granular-convergence-pr1`(从 master 切;每逻辑 PR 独立分支走 CI/CD)。

**Spec:** `docs/superpowers/specs/2026-05-21-image-granular-convergence-design.md`(rev 2)§2 / §3 / §4 / §7-PR1 / §8。

**前置(已 merged,复用)**:2026-05-19 PR-1..6 —— `ComponentSpec`/`to_component_key`、`get_or_load_image_adapter`/`get_or_load_component`、`DiffusersImageBackend.from_loaded_components`/`load_component_module`、`ImageSampler`、`ImageRequest.components`/`LoRASpec.path`、L1/L2 cache。

---

## 关键设计决策(写给执行者,先读)

1. **整模型单卡**:`device` 只在 Load Diffusion Model 上选(unet 的 device)。`_build_request` flatten 时把 clip/vae 的 `device` **覆盖成 unet 的 device** —— 三组件落同一张卡。`get_or_load_image_adapter` 照常逐组件 resolve(此时三者已同 device,combo cache 正常)。**不做**逐组件跨卡。
2. **VAE Decode 是唯一 dispatch 节点**;Load*/Encode/KSampler 全 inline 描述符。这复用 Family B 的 inline-loader + dispatch-terminal 模式(workflow_executor 已支持混跑)。`exec_vae_decode` 从 inline `EXECUTORS` **移除**(dispatch 路径不经 plugin executor)。
3. **描述符无张量**:Encode/KSampler **不再**在主进程 encode/sample(现状用 `_acquire_adapter` 在主进程跑 GPU —— 删掉)。它们只记计划,真正 encode/denoise/decode 在 runner 的 `ImageSampler.sample()` 一把跑完。
4. **Load CLIP 本 PR 单编码器**:widget = `file` + `weight_dtype`,产 `encoders:[{kind:clip,file,dtype}]`(1 条)+ `type:"flux2"`。多编码器 UI(clip_stack)+ gated 执行 = PR-3。`_build_request` 本 PR 只取 `encoders[0]`;`len(encoders)>1` 先抛"PR-3 未实现"占位错误(PR-3 替换成正式 gated 文案)。
5. **weight_dtype `default`**:= 文件原生精度。`ComponentSpec.dtype` 扩 `"default"` 字面量;`load_component_module` 的 dtype 映射遇 `default` 走 `from_pretrained` 不传 `torch_dtype`(原生)。
6. **LLM 卡保护**:`device` 命中常驻 LLM 占用的卡且空闲显存不足 → 装载前清晰报错(不静默 OOM)。本 PR 做"显存前置检查 + 清晰错误";精细的"哪张卡被 LLM 占"探测用现有 `gpu_free_probe` / allocator。
7. **Load Checkpoint**:产 ComponentSpec 形态 bundle(`model_key→三组件文件` resolver,三件同 device=auto)。若 resolver 比预期复杂(Task 7 实测),降级=本 PR 删 `flux2_load_checkpoint`(便捷节点;核心是三独立 loader)。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `backend/nodes/flux2-components/node.yaml` | Modify | Load Diffusion 加 file/weight_dtype/device/adapter_arch;Load CLIP 改 file/weight_dtype;Load VAE 加 file/weight_dtype;声明 componentRole |
| `backend/nodes/flux2-components/executor.py` | Modify | Load*/Encode/KSampler → 嵌套描述符(去 GPU);移除 exec_vae_decode(转 dispatch);Load Checkpoint resolver |
| `backend/src/services/node_routing.py` | Modify | `flux2_vae_decode` 进 `DISPATCH_NODE_TYPES` |
| `backend/src/services/workflow_executor.py` | Modify | `_NODE_TYPE_TO_GROUP_ID` 加 `flux2_vae_decode→"image"` |
| `backend/src/runner/runner_process.py` | Modify | `_build_request` 加 granular-terminal 分支(walk 嵌套 latent → ImageRequest,clip/vae device=unet device) |
| `backend/src/services/inference/component_spec.py` | Modify | `dtype` 字面量加 `"default"` |
| `backend/src/services/inference/image_diffusers.py` | Modify | `load_component_module` dtype=default → 原生(不传 torch_dtype) |
| `backend/src/services/model_manager.py` | Modify | `get_or_load_image_adapter` 装载前 LLM 卡/显存前置检查(清晰错误) |
| `backend/nodes/flux2-components/component_resolve.py` | Create | `resolve_checkpoint_components(model_key)→{unet,clip,vae 文件}`(Load Checkpoint 用) |
| `backend/tests/test_*` | Create/Modify | 见各 Task |

执行顺序 = 下方 Task 顺序(描述符节点 → dtype default → 路由 → runner flatten → LLM 卡保护 → Load Checkpoint → 集成 stub → 真模型 smoke)。

---

## Task 1: flux2 loader/中间节点 → 嵌套描述符(去 GPU)

**Files:**
- Modify: `backend/nodes/flux2-components/node.yaml`
- Modify: `backend/nodes/flux2-components/executor.py`
- Test: `backend/nodes/flux2-components/test_executor_descriptors.py`(新建)或 `backend/tests/test_flux2_descriptors.py`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_flux2_descriptors.py
"""PR-1: flux2 loader/中间节点产嵌套描述符(inline, 无张量/无 GPU)。"""
from __future__ import annotations

import pytest

import sys, pathlib
# 确保 plugin executor import 路径(沿用现有 conftest / nodes loader)
from nodes import get_all_executors

EX = get_all_executors()


@pytest.mark.asyncio
async def test_load_diffusion_descriptor():
    out = await EX["flux2_load_diffusion_model"](
        {"file": "/m/u.safe", "device": "cuda:1", "weight_dtype": "fp8_e4m3", "adapter_arch": "flux2"}, {})
    assert out["model"] == {
        "_type": "flux2_model",
        "spec": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
                 "dtype": "fp8_e4m3", "adapter_arch": "flux2"},
        "loras": [],
    }


@pytest.mark.asyncio
async def test_load_diffusion_defaults():
    out = await EX["flux2_load_diffusion_model"]({"file": "/m/u.safe"}, {})
    s = out["model"]["spec"]
    assert s["device"] == "auto" and s["dtype"] == "default" and s["adapter_arch"] == "flux2"


@pytest.mark.asyncio
async def test_load_clip_single_encoder():
    out = await EX["flux2_load_clip"]({"file": "/m/c.safe", "weight_dtype": "default"}, {})
    assert out["clip"] == {
        "_type": "flux2_clip", "type": "flux2",
        "encoders": [{"kind": "clip", "file": "/m/c.safe", "dtype": "default"}],
    }


@pytest.mark.asyncio
async def test_load_vae_descriptor():
    out = await EX["flux2_load_vae"]({"file": "/m/v.safe", "weight_dtype": "bfloat16"}, {})
    assert out["vae"] == {"_type": "flux2_vae", "spec": {"kind": "vae", "file": "/m/v.safe", "dtype": "bfloat16"}}


@pytest.mark.asyncio
async def test_load_lora_chains_on_descriptor():
    base = {"_type": "flux2_model", "spec": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
            "dtype": "fp8_e4m3", "adapter_arch": "flux2"}, "loras": []}
    s1 = await EX["flux2_load_lora"]({"lora_name": "a", "lora_path": "/m/loras/a.safe", "strength": 0.8}, {"model": base})
    s2 = await EX["flux2_load_lora"]({"lora_name": "b", "lora_path": "/m/loras/b.safe", "strength": 0.4}, {"model": s1["model"]})
    assert [l["name"] for l in s2["model"]["loras"]] == ["a", "b"]
    assert s2["model"]["loras"][0]["path"] == "/m/loras/a.safe"
    assert base["loras"] == []  # 上游不被改


@pytest.mark.asyncio
async def test_encode_prompt_descriptor_no_tensor():
    clip = {"_type": "flux2_clip", "type": "flux2", "encoders": [{"kind": "clip", "file": "/m/c.safe", "dtype": "default"}]}
    out = await EX["flux2_encode_prompt"]({"text": "a cat", "negative_prompt": ""}, {"clip": clip, "text": "a cat"})
    assert out["conditioning"] == {"_type": "flux2_conditioning", "clip": clip, "text": "a cat", "negative": ""}
    # 关键:无 prompt_embeds 张量(不在主进程 encode)
    assert "prompt_embeds" not in out["conditioning"]


@pytest.mark.asyncio
async def test_ksampler_descriptor_no_tensor():
    model = {"_type": "flux2_model", "spec": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
             "dtype": "fp8_e4m3", "adapter_arch": "flux2"}, "loras": []}
    cond = {"_type": "flux2_conditioning", "clip": {"_type": "flux2_clip", "type": "flux2", "encoders": []},
            "text": "x", "negative": ""}
    out = await EX["flux2_ksampler"](
        {"width": 768, "height": 768, "steps": 9, "cfg_scale": 4.0, "seed": 42},
        {"model": model, "conditioning": cond})
    lat = out["latent"]
    assert lat["_type"] == "flux2_latent" and lat["model"] is model and lat["conditioning"] is cond
    assert (lat["width"], lat["height"], lat["steps"], lat["cfg_scale"], lat["seed"]) == (768, 768, 9, 4.0, 42)
    assert "tensor" not in lat


@pytest.mark.asyncio
async def test_encode_missing_clip_raises():
    with pytest.raises(RuntimeError, match="CLIP"):
        await EX["flux2_encode_prompt"]({"text": "x"}, {})


@pytest.mark.asyncio
async def test_ksampler_missing_model_raises():
    with pytest.raises(RuntimeError, match="MODEL"):
        await EX["flux2_ksampler"]({}, {"conditioning": {"_type": "flux2_conditioning"}})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && uv run pytest tests/test_flux2_descriptors.py -q`
Expected: FAIL —现 executor 产旧 bundle(model_id)+ Encode/KSampler 跑 GPU 张量。

- [ ] **Step 3: 改 `executor.py`**

把 Load* 改成产 ComponentSpec-style 嵌套描述符;Encode/KSampler 去掉 `_acquire_adapter` GPU 调用,只产计划。替换文件顶部 `_bundle_*` / `_read_model_key` 与对应 exec:

```python
_DEFAULT_DTYPE = "default"
_AUTO = "auto"


def _spec_unet(data: dict) -> dict:
    return {
        "kind": "unet",
        "file": data["file"],
        "device": data.get("device") or _AUTO,
        "dtype": data.get("weight_dtype") or _DEFAULT_DTYPE,
        "adapter_arch": data.get("adapter_arch") or "flux2",
    }


async def exec_load_diffusion_model(data: dict, inputs: dict) -> dict:
    return {"model": {"_type": "flux2_model", "spec": _spec_unet(data), "loras": []}}


async def exec_load_clip(data: dict, inputs: dict) -> dict:
    # PR-1 单编码器;PR-3 换 clip_stack 产多条 + type 选择器
    enc = {"kind": "clip", "file": data["file"], "dtype": data.get("weight_dtype") or _DEFAULT_DTYPE}
    return {"clip": {"_type": "flux2_clip", "type": data.get("type") or "flux2", "encoders": [enc]}}


async def exec_load_vae(data: dict, inputs: dict) -> dict:
    spec = {"kind": "vae", "file": data["file"], "dtype": data.get("weight_dtype") or _DEFAULT_DTYPE}
    return {"vae": {"_type": "flux2_vae", "spec": spec}}


async def exec_load_lora(data: dict, inputs: dict) -> dict:
    upstream = inputs.get("model")
    if not isinstance(upstream, dict) or upstream.get("_type") != "flux2_model":
        raise RuntimeError("Load LoRA 的 MODEL 输入未连接,或上游不是 flux2_model")
    name = (data.get("lora_name") or "").strip()
    out = dict(upstream)
    out["loras"] = list(upstream.get("loras") or [])
    if name:
        out["loras"].append({
            "name": name,
            "path": data.get("lora_path") or None,
            "strength": float(data.get("strength", 1.0)),
        })
    return {"model": out}


async def exec_encode_prompt(data: dict, inputs: dict) -> dict:
    clip = inputs.get("clip")
    if not isinstance(clip, dict) or clip.get("_type") != "flux2_clip":
        raise RuntimeError("Encode Prompt 的 CLIP 端口未连接,或上游不是 flux2_clip")
    text = inputs.get("text") or data.get("text") or ""
    return {"conditioning": {
        "_type": "flux2_conditioning", "clip": clip,
        "text": text, "negative": data.get("negative_prompt", "") or "",
    }}


async def exec_ksampler(data: dict, inputs: dict) -> dict:
    model = inputs.get("model")
    if not isinstance(model, dict) or model.get("_type") != "flux2_model":
        raise RuntimeError("KSampler 的 MODEL 端口未连接,或上游不是 flux2_model")
    cond = inputs.get("conditioning")
    if not isinstance(cond, dict) or cond.get("_type") != "flux2_conditioning":
        raise RuntimeError("KSampler 的 CONDITIONING 端口未连接,或上游不是 flux2_conditioning")
    raw_seed = data.get("seed")
    seed = int(raw_seed) if raw_seed not in (None, "") else None
    return {"latent": {
        "_type": "flux2_latent", "model": model, "conditioning": cond,
        "width": int(data.get("width", 1024)), "height": int(data.get("height", 1024)),
        "steps": int(data.get("steps", 25)), "cfg_scale": float(data.get("cfg_scale", 4.0)),
        "seed": seed,
    }}
```

删掉 `exec_vae_decode` + `_acquire_adapter` + 旧 `_bundle_*`/`_read_model_key`/`_require`。`EXECUTORS` dict 去掉 `flux2_vae_decode`(Task 3 它走 dispatch);保留 `flux2_load_checkpoint`(Task 7 改)。**保留 `exec_load_checkpoint` 暂时产旧形态,Task 7 改正**(本 Task 不测它)。

- [ ] **Step 4: 改 `node.yaml`** widgets(Load Diffusion / CLIP / VAE)

```yaml
  flux2_load_diffusion_model:
    ...
    componentRole: unet
    widgets:
      - { name: file,         label: "文件", widget: component_select, role: unet }
      - { name: weight_dtype, label: "精度", widget: select, options: [default, bfloat16, float16, fp8_e4m3], default: default }
      - { name: device,       label: "显卡", widget: select, options: [auto, "cuda:0", "cuda:1", "cuda:2"], default: auto }
      - { name: adapter_arch, label: "架构", widget: select, options: [flux2, flux1], default: flux2 }

  flux2_load_clip:
    ...
    componentRole: clip
    widgets:
      - { name: file,         label: "文件", widget: component_select, role: clip }
      - { name: weight_dtype, label: "精度", widget: select, options: [default, bfloat16, fp8_e4m3], default: default }

  flux2_load_vae:
    ...
    componentRole: vae
    widgets:
      - { name: file,         label: "文件", widget: component_select, role: vae }
      - { name: weight_dtype, label: "精度", widget: select, options: [default, bfloat16, float16], default: default }
```

> `componentRole` 是新增的节点级字段;前端透传 PR-2 做,后端 node.yaml schema 若严格校验未知键需放行(确认 plugin loader 不因未知键报错;现有 image_*_load 的 componentRole 已是先例)。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd backend && uv run pytest tests/test_flux2_descriptors.py -q`
Expected: PASS。

- [ ] **Step 6: 回归 flux2 既有测试**

Run: `cd backend && uv run pytest tests/test_flux2_components_loaders.py -q`(若存在;不存在则跳过)
Expected: PASS 或按新描述符更新断言(旧测试若断言旧 bundle 形态,改成新描述符)。

- [ ] **Step 7: Commit**

```bash
git add backend/nodes/flux2-components/node.yaml backend/nodes/flux2-components/executor.py backend/tests/test_flux2_descriptors.py
git commit -m "feat(image): PR-1 — flux2 loader/中间节点产嵌套描述符(去主进程 GPU)"
```

---

## Task 2: `weight_dtype: "default"` 支持(原生精度)

**Files:**
- Modify: `backend/src/services/inference/component_spec.py`(`dtype` 字面量加 `default`)
- Modify: `backend/src/services/inference/image_diffusers.py`(`load_component_module` / `_load_hf_or_quant` dtype 映射)
- Test: `backend/tests/test_component_spec.py`(加 default 用例)+ `backend/tests/test_load_component_module_dtype.py`(新建,stub)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_component_spec.py 追加
def test_componentspec_accepts_default_dtype():
    from src.services.inference.component_spec import ComponentSpec
    s = ComponentSpec(kind="unet", file="/m/u.safe", device="cuda:0", dtype="default")
    assert s.dtype == "default"
```

```python
# backend/tests/test_load_component_module_dtype.py
"""PR-1: dtype=default → from_pretrained 不传 torch_dtype(原生精度)。"""
from __future__ import annotations
from src.services.inference.image_diffusers import _torch_dtype_from


def test_torch_dtype_default_is_none():
    # default ⇒ None ⇒ 调用方 from_pretrained 省略 torch_dtype
    assert _torch_dtype_from("default") is None


def test_torch_dtype_known():
    import torch
    assert _torch_dtype_from("bfloat16") == torch.bfloat16
```

- [ ] **Step 2: 跑确认失败** —`ComponentSpec` 拒 `default`;`_torch_dtype_from("default")` 现返回 bfloat16。

Run: `cd backend && uv run pytest tests/test_component_spec.py::test_componentspec_accepts_default_dtype tests/test_load_component_module_dtype.py -q`

- [ ] **Step 3: 改 `component_spec.py`** —`dtype` 字面量集合加 `"default"`(找到 `dtype:` 字段定义,Literal/正则放行 default)。

- [ ] **Step 4: 改 `image_diffusers.py`** —`_torch_dtype_from("default") → None`;`_load_hf_or_quant` 当 dtype 映射为 None 时 `from_pretrained(parent_dir)` 不传 `torch_dtype`:

```python
def _torch_dtype_from(dtype_str: str):
    import torch
    if dtype_str == "default":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "fp8_e4m3": torch.float8_e4m3fn}.get(dtype_str, torch.bfloat16)

# _load_hf_or_quant:
    td = _torch_dtype_from(spec.dtype)
    kwargs = {} if td is None else {"torch_dtype": td}
    return hf_class.from_pretrained(parent_dir, **kwargs)
```

- [ ] **Step 5: 跑确认通过 + 回归** `tests/test_component_spec.py tests/test_from_loaded_components.py`

- [ ] **Step 6: Commit** `feat(image): PR-1 — weight_dtype=default 走文件原生精度`

---

## Task 3: 路由 — `flux2_vae_decode` 进 dispatch

**Files:**
- Modify: `backend/src/services/node_routing.py`(`DISPATCH_NODE_TYPES`)
- Modify: `backend/src/services/workflow_executor.py`(`_NODE_TYPE_TO_GROUP_ID`)
- Test: `backend/tests/test_node_routing.py`(加断言)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_node_routing.py 追加
def test_flux2_vae_decode_is_dispatch():
    from src.services.node_routing import node_exec_class
    assert node_exec_class("flux2_vae_decode") == "dispatch"

def test_flux2_loaders_inline():
    from src.services.node_routing import node_exec_class
    for t in ("flux2_load_diffusion_model", "flux2_load_clip", "flux2_load_vae",
              "flux2_load_lora", "flux2_encode_prompt", "flux2_ksampler"):
        assert node_exec_class(t) == "inline"
```

```python
# tests/test_workflow_executor_dispatch.py 追加(或新建)
def test_flux2_vae_decode_maps_to_image_group():
    from src.services.workflow_executor import _NODE_TYPE_TO_GROUP_ID
    assert _NODE_TYPE_TO_GROUP_ID.get("flux2_vae_decode") == "image"
```

- [ ] **Step 2: 跑确认失败**

- [ ] **Step 3: 改两处**
  - `node_routing.py`: `DISPATCH_NODE_TYPES = frozenset({"image_generate", "tts_engine", "flux2_vae_decode"})`(PR-4 删 image_generate;本 PR 先加 flux2_vae_decode,保留 image_generate 直到 PR-4)。
  - `workflow_executor.py`: `_NODE_TYPE_TO_GROUP_ID` 加 `"flux2_vae_decode": "image"`。

- [ ] **Step 4: 跑确认通过 + 回归** `tests/test_node_routing.py tests/test_workflow_executor_dispatch.py`

- [ ] **Step 5: Commit** `feat(image): PR-1 — flux2_vae_decode 进 dispatch 路由(role=image)`

---

## Task 4: runner `_build_request` — granular terminal 摊平(整模型单卡)

**Files:**
- Modify: `backend/src/runner/runner_process.py`(`_build_request` image 分支)
- Test: `backend/tests/test_runner_build_request_granular.py`(新建)

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_runner_build_request_granular.py
"""PR-1: _build_request 见嵌套 latent+vae → ImageRequest;clip/vae device=unet device。"""
from __future__ import annotations
from src.runner import protocol as P
from src.runner.runner_process import _build_request


def _node(inputs):
    return P.RunNode(task_id=1, node_id="dec", node_type="image", model_key=None, inputs=inputs)


def _granular_inputs(unet_dev="cuda:1", loras=None):
    model = {"_type": "flux2_model",
             "spec": {"kind": "unet", "file": "/m/u.safe", "device": unet_dev, "dtype": "fp8_e4m3", "adapter_arch": "flux2"},
             "loras": loras or []}
    cond = {"_type": "flux2_conditioning",
            "clip": {"_type": "flux2_clip", "type": "flux2", "encoders": [{"kind": "clip", "file": "/m/c.safe", "dtype": "default"}]},
            "text": "a cat", "negative": ""}
    latent = {"_type": "flux2_latent", "model": model, "conditioning": cond,
              "width": 768, "height": 768, "steps": 9, "cfg_scale": 4.0, "seed": 42}
    vae = {"_type": "flux2_vae", "spec": {"kind": "vae", "file": "/m/v.safe", "dtype": "default"}}
    return {"latent": latent, "vae": vae, "url_ttl_seconds": "3600"}


def test_granular_flatten_single_card():
    req = _build_request(_node(_granular_inputs(unet_dev="cuda:1")))
    assert req.components is not None
    # 整模型单卡:clip/vae 的 device 被覆盖成 unet 的 device
    assert req.components["unet"].device == "cuda:1"
    assert req.components["clip"].device == "cuda:1"
    assert req.components["vae"].device == "cuda:1"
    assert req.components["clip"].file == "/m/c.safe"
    assert req.prompt == "a cat"
    assert (req.width, req.height, req.steps, req.seed) == (768, 768, 9, 42)
    assert req.pipeline_class == "Flux2KleinPipeline"


def test_granular_carries_loras():
    inp = _granular_inputs(loras=[{"name": "a", "path": "/m/loras/a.safe", "strength": 0.8}])
    req = _build_request(_node(inp))
    assert req.components["unet"].loras[0].name == "a"
    assert req.components["unet"].loras[0].path == "/m/loras/a.safe"


def test_granular_auto_device_passthrough():
    req = _build_request(_node(_granular_inputs(unet_dev="auto")))
    # auto 不在此解析(runner get_or_load_image_adapter 解析);三组件都带 auto
    assert req.components["unet"].device == "auto"
    assert req.components["vae"].device == "auto"


def test_granular_multi_encoder_not_yet():
    inp = _granular_inputs()
    inp["latent"]["conditioning"]["clip"]["encoders"].append({"kind": "clip", "file": "/m/c2.safe", "dtype": "default"})
    import pytest
    with pytest.raises(ValueError, match="多编码器|PR-3|encoder"):
        _build_request(_node(inp))


def test_legacy_flat_components_still_work():
    # PR-4 删 image_generate 前,旧 flat unet/clip/vae 路径不破
    flat = {
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
        "prompt": "x",
    }
    req = _build_request(_node(flat))
    assert req.components["clip"].device == "cuda:0"  # flat 路径保留各自 device(不强制单卡)
```

- [ ] **Step 2: 跑确认失败**

Run: `cd backend && uv run pytest tests/test_runner_build_request_granular.py -q`

- [ ] **Step 3: 改 `_build_request`**(image 分支,在 flat `unet/clip/vae` 判断**之前**加 granular 分支)

```python
    if node.node_type == "image":
        from src.services.inference.component_spec import ComponentSpec
        raw_seed = node.inputs.get("seed")
        # granular terminal:VAE Decode 派发,inputs 有嵌套 latent + vae
        latent = node.inputs.get("latent")
        vae_d = node.inputs.get("vae")
        if isinstance(latent, dict) and latent.get("_type") == "flux2_latent" \
                and isinstance(vae_d, dict) and vae_d.get("_type") == "flux2_vae":
            model_d = latent["model"]
            cond_d = latent["conditioning"]
            unet_spec = dict(model_d["spec"])
            device = unet_spec["device"]                 # 整模型单卡:这张卡定全局
            encoders = cond_d["clip"]["encoders"]
            if len(encoders) != 1:
                raise ValueError(
                    f"多编码器 CLIP({len(encoders)} 条)执行 PR-3 才支持;当前单编码器")
            clip_spec = dict(encoders[0]); clip_spec["device"] = device
            vae_spec = dict(vae_d["spec"]); vae_spec["device"] = device
            lseed = latent.get("seed")
            return ImageRequest(
                request_id=f"task-{node.task_id}",
                prompt=str(cond_d.get("text", "")),
                negative_prompt=str(cond_d.get("negative", "")),
                width=int(latent.get("width") or 1024),
                height=int(latent.get("height") or 1024),
                steps=int(latent.get("steps") or 25),
                cfg_scale=float(latent.get("cfg_scale") or 4.0),
                seed=int(lseed) if lseed not in (None, "") else None,
                components={
                    "unet": ComponentSpec(loras=model_d.get("loras") or [], **unet_spec),
                    "clip": ComponentSpec(**clip_spec),
                    "vae":  ComponentSpec(**vae_spec),
                },
                pipeline_class="Flux2KleinPipeline",
            )
        # ... 既有 flat unet/clip/vae 分支(PR-4 删 image_generate 后这段随老路径一起清理)...
```

> `ComponentSpec(loras=..., **unet_spec)`:unet_spec 含 kind/file/device/dtype/adapter_arch;loras 单独传。pydantic 自动把 `list[dict]` 转 `list[LoRASpec]`(含 path)。clip_spec 加 device 后含 kind/file/dtype/device;ComponentSpec 的 clip_arch 可选默认 None(本 PR 不传)。

- [ ] **Step 4: 跑确认通过**

Run: `cd backend && uv run pytest tests/test_runner_build_request_granular.py -q`
Expected: PASS。

- [ ] **Step 5: 回归** `tests/test_runner_build_request.py tests/test_runner_protocol.py`

- [ ] **Step 6: Commit** `feat(image): PR-1 — runner _build_request granular terminal 摊平(整模型单卡)`

---

## Task 5: LLM 卡保护 — 装载前显存前置检查

**Files:**
- Modify: `backend/src/services/model_manager.py`(`get_or_load_image_adapter` 装载前检查)
- Test: `backend/tests/test_image_adapter_card_guard.py`(新建,stub probe)

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_image_adapter_card_guard.py
"""PR-1: device 命中显存不足的卡 → 装载前清晰错误(不静默 OOM)。"""
from __future__ import annotations
import pytest
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    def __init__(self): self._config_path = ""; self._specs = {}


@pytest.fixture
def mm():
    return ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())


def _comps(dev="cuda:1"):
    return {
        "unet": ComponentSpec(kind="unet", file="/m/u.safe", device=dev, dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device=dev, dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device=dev, dtype="bfloat16"),
    }


@pytest.mark.asyncio
async def test_insufficient_vram_raises_clear_error(mm, monkeypatch):
    # 模拟 cuda:1 空闲显存严重不足(LLM 占着)
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: 500 if dev == "cuda:1" else 90000)
    with pytest.raises(RuntimeError, match="显存不足|cuda:1|LLM"):
        await mm.get_or_load_image_adapter(_comps("cuda:1"))
```

- [ ] **Step 2: 跑确认失败** —现无前置检查,直接进 load(stub 文件不存在会以别的错误失败 —— 用 monkeypatch 让检查在 load 前拦)。

- [ ] **Step 3: 改 `get_or_load_image_adapter`** —resolve device → **统一单卡** → 装载前显存检查:

```python
        resolved = {k: self._resolve_component_device(s) for k, s in components.items()}
        # PR-1 整模型单卡不变式:auto 可能让三组件各自 resolve 到不同卡 —— 以 unet
        # 解析出的卡为准,强制 clip/vae 落同一张卡(combo cache key 也据此稳定)。
        target = resolved["unet"].device
        for k in ("clip", "vae"):
            if resolved[k].device != target:
                resolved[k] = resolved[k].model_copy(update={"device": target})
        # LLM 卡保护:目标卡空闲显存不足 → 装载前清晰错误(不静默 OOM)
        need_mb = self._estimate_image_vram_mb(resolved)   # 粗估:三组件文件 bytes 之和 * 系数
        free_mb = self._free_vram_mb(target)
        if free_mb is not None and free_mb < need_mb:
            raise RuntimeError(
                f"{target} 空闲显存不足({free_mb}MB < 约需 {need_mb}MB)—— "
                f"该卡可能被常驻 LLM 占用。换张卡(device)或用更低精度(fp8),"
                f"或先释放该卡。")
```

> 注:统一单卡发生在 resolve 之后,所以即便用户传 `device=auto`,三组件也保证同卡。`test_get_or_load_image_adapter.py` 若有"三组件不同 device"的旧用例(Family B 跨卡),本 PR 它们语义变了 —— 改成断言"统一到 unet 卡",或标记为 cross-card 休眠用例 skip(cross-card 转 Future,见 spec §9)。

`_free_vram_mb(dev)` 复用 `gpu_free_probe` / nvidia-smi(无 GPU 环境返回 None → 跳过检查,不阻塞 CI)。`_estimate_image_vram_mb` 粗估(文件 bytes 求和 + 余量;fp8 文件本就小)。

- [ ] **Step 4: 跑确认通过 + 回归** `tests/test_get_or_load_image_adapter.py`(确认无 GPU 环境 free_mb=None 时旧测试不受影响)

- [ ] **Step 5: Commit** `feat(image): PR-1 — 图像装载前显存前置检查(LLM 卡保护)`

---

## Task 6: Load Checkpoint → ComponentSpec resolver(或删)

**Files:**
- Create: `backend/nodes/flux2-components/component_resolve.py`
- Modify: `backend/nodes/flux2-components/executor.py`(`exec_load_checkpoint`)
- Test: `backend/tests/test_flux2_checkpoint_resolve.py`(新建)

- [ ] **Step 1: 评估 resolver 可行性(15 分钟 spike)**
  - 看 `model_manager._registry.get(model_key)` 的 ModelSpec 是否含三组件文件路径(transformer/text_encoder/vae);或用 `component_expand.expand_legacy_image_spec`(2026-05-19 已有,Family B 老格式展开用的就是它!)。
  - **若 `expand_legacy_image_spec` 可直接复用** → resolver = 薄封装,继续 Step 2。
  - **若复杂** → 降级:从 `node.yaml` 删 `flux2_load_checkpoint`,从 `EXECUTORS` 删 `exec_load_checkpoint`,删本 Task 其余步骤,Commit `chore(image): PR-1 — 删 Load Checkpoint(便捷节点,核心是三独立 loader)`,跳到 Task 7。

- [ ] **Step 2: 写失败测试**(若继续)

```python
# backend/tests/test_flux2_checkpoint_resolve.py
import pytest
from nodes import get_all_executors

@pytest.mark.asyncio
async def test_checkpoint_emits_three_descriptors(monkeypatch):
    EX = get_all_executors()
    # monkeypatch resolver 返回三文件
    import nodes.__dict__ as _  # 占位:按实际 import 路径 monkeypatch resolve_checkpoint_components
    out = await EX["flux2_load_checkpoint"]({"model_key": "flux2-klein-9b-true-v2-fp8mixed",
                                             "device": "cuda:0", "weight_dtype": "default"}, {})
    assert out["model"]["_type"] == "flux2_model"
    assert out["clip"]["_type"] == "flux2_clip"
    assert out["vae"]["_type"] == "flux2_vae"
    # 三件同 device
    assert out["model"]["spec"]["device"] == "cuda:0"
```

- [ ] **Step 3–6**: 实现 `resolve_checkpoint_components(model_key) → {unet_file, clip_file, vae_file}`(复用 `expand_legacy_image_spec`);`exec_load_checkpoint` 用它产三描述符(device/weight_dtype 来自 widget,三件同 device);跑通;Commit `feat(image): PR-1 — Load Checkpoint 产 ComponentSpec 三描述符`。

---

## Task 7: 集成测试(stub)— granular workflow → 派发 ImageRequest

**Files:**
- Test: `backend/tests/test_granular_workflow_dispatch.py`(新建,FakeRunnerClient 捕获 RunNode)

- [ ] **Step 1: 写测试** —构造 Load Diffusion→LoRA→CLIP→Encode→KSampler + VAE→VAEDecode 的最小 workflow,用捕获式 runner client 跑 `WorkflowExecutor`,断言:
  - VAE Decode 走 dispatch(client 收到 RunNode,node_type="image")。
  - RunNode.inputs 含嵌套 latent(model.loras 有一条)+ vae。
  - inline 节点没碰 GPU(无 ModelManager 调用)。

```python
# 关键断言骨架
async def test_granular_workflow_dispatches_image_request():
    wf = {"nodes": [...7 节点...], "edges": [...]}
    client = _CapturingClient()  # run_node 记 spec,返回 completed
    ex = WorkflowExecutor(wf, runner_clients={"image": client}, task_id=7)
    await ex.execute()
    spec = client.spec
    assert spec.node_type == "image"
    latent = spec.inputs["latent"]
    assert latent["_type"] == "flux2_latent"
    assert latent["model"]["loras"][0]["name"] == "turbo"
    assert spec.inputs["vae"]["_type"] == "flux2_vae"
```

- [ ] **Step 2–4**: 跑通(可能要按现有 WorkflowExecutor 测试夹具调整 node/edge 结构);回归 `tests/test_workflow_executor.py`;Commit `test(image): PR-1 — granular workflow → image dispatch 集成`。

---

## Task 8: 真模型 smoke(standalone,需 GPU)

**Files:**
- Create: `backend/tests/manual/smoke_granular_pr1.py`(standalone,不进 pytest 默认套件)

> 遵守 `feedback_verify_real_model` + `dev_env_gotchas`(真模型测试 standalone 跑,不靠 stub)。本机三卡(cuda:0/2=3090, cuda:1=Pro 6000)。

- [ ] **Step 1: 写 standalone 脚本** —直接构造 granular ImageRequest(绕过前端),经 runner / get_or_load_image_adapter 出图。两组用例:
  - **A**:Flux2-fp8mixed,device=cuda:0(3090 24GB),期望**不 offload** 出图 ≤ ~25s,落盘可见。
  - **B**:Flux2-bf16,device=cuda:1(Pro 6000;**需先确认/释放 vLLM 占用**),出图。
  - **C**:A 基础上挂 1 条 LoRA(turbo),strength 0.8,出图且风格变化。
- [ ] **Step 2: 跑** `cd backend && uv run python tests/manual/smoke_granular_pr1.py`(按 dev_env_gotchas:`--noproxy` 等;真模型不走 CI)。
- [ ] **Step 3: 人工核**:三张图都出;A 不 offload(看显存/耗时);GPU 工作在 runner 子进程(非主进程);记录耗时贴 PR 描述。
- [ ] **Step 4: 文档** —把 smoke 结论(耗时 / 是否 offload / 子进程确认)写进 PR 描述;脚本 commit `test(image): PR-1 — granular 真模型 smoke 脚本 + 结论`。

---

## 收尾

- [ ] 全套后端测试:`cd backend && uv run pytest -q`(确认无回归)。
- [ ] 预检 lint(push 前):`cd backend && uv run ruff check .`(遵守 `feedback_preflight_lint`)。
- [ ] 开 PR `feat/image-granular-convergence-pr1` → CI 绿 → auto-merge(遵守 `feedback_auto_merge`)。
- [ ] 注意:本 PR 后细粒度图已能跑(后端),但前端 loader 还没 file/device/dtype 控件(PR-2)+ 还没删 Family B(PR-4)。过渡期两套并存正常。
