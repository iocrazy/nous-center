# PR-2 — comfy 量化模型桥接(fp8mixed/mxfp8/nvfp4)进 Modular 引擎 Plan

> REQUIRED SUB-SKILL: executing-plans。
> Spec: `2026-05-22-image-engine-modular-diffusers-design.md` §4(plan-eng-review D4)。前置:PR-1(#128/#129)。

**Goal**:让用户的 comfy 量化单文件(`Flux2-Klein-9B-True-v2-fp8mixed.safetensors` 等)在 **modular 引擎**下真能出图(spike v3 已验过原理,现固化进生产路径)。三步桥接:quant_loaders 反量化 → diffusers 转键 → `update_components`。

**Branch**:`feat/image-modular-quant-bridge-pr2`。

**关键设计(comfy 单文件只是 transformer 权重)**:
- clip/vae/scheduler/config **仍来自 HF repo**;comfy 单文件**只替换 transformer**。
- HF repo 推导:unet 是单文件(`_modular_repo_from_components` 抛错)时,从 **clip 或 vae 组件**(指向 HF text_encoder/vae)向上找 `model_index.json`。
- transformer override:`Flux2Transformer2DModel.from_config(repo/transformer)` + `load_state_dict(dequant_and_convert(unet_spec))` → 喂 `ModularImageBackend(transformer_override=...)` → `update_components`。
- ⚠️ 本次反量化到 bf16 → **不省显存**(spec §4 ⚠️;3090 仍 OOM,真 fp8 省显存是 future)。本 PR 只解「comfy 量化文件可用」。

---

## Task 1:D4 共享 guard helper `dequant_and_convert`

**Files**:`backend/src/services/inference/quant_loaders.py` + `backend/tests/test_quant_dequant_convert.py`

- [ ] **Step 1 失败测试**:`dequant_and_convert(spec) -> StateDict` = `QUANT_LOADERS.dispatch(spec)`(反量化)+ `convert_flux2_transformer_checkpoint_to_diffusers`(转键)。测:合成 comfy-key state_dict(`double_blocks...`)→ 输出全 `transformer.`... 的 diffusers 键(同 PR-1 LoRA 测套路,torch-free 部分 + 真 torch 部分 skipif)。guard:内部函数 import 失败 → 清晰错误(指向 diffusers 版本)。
- [ ] **Step 2 跑确认失败**
- [ ] **Step 3 实现**:helper 包 `from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers`(try/except ImportError → ValueError 提示 diffusers 版本/commit)。
- [ ] **Step 4 跑通**
- [ ] **Step 5 ruff + commit** `feat(image): dequant_and_convert 共享 helper(反量化+转键,D4 guard)`

## Task 2:现有栈 fp8 bug 顺带修(_load_hf_or_quant 缺转键)

**Files**:`backend/src/services/inference/image_diffusers.py` + 测试

- [ ] **Step 1 失败测试**:`_load_hf_or_quant` 对 comfy 单文件 quant spec → 用 `dequant_and_convert`(而非裸 dispatch+load)→ 键匹配(missing=0)。当前是 dispatch+load_state_dict(strict=False) 静默丢键。
- [ ] **Step 3 实现**:quant fallback 改调 `dequant_and_convert`。
- [ ] **Step 5 commit** `fix(image): _load_hf_or_quant comfy 单文件加转键(否则静默出垃圾)`(对 legacy + modular 都受益)

## Task 3:HF repo 推导(unet 单文件 fallback 到 clip/vae)

**Files**:`backend/src/services/model_manager.py` + `backend/tests/test_image_engine_selector.py`

- [ ] **Step 1 失败测试**:`_modular_repo_from_components` 当 unet 是单文件(无 model_index.json),从 clip / vae 组件向上找 repo;都找不到才报错。
- [ ] **Step 3 实现**:依次试 unet/clip/vae 组件文件向上找 `model_index.json`。
- [ ] **Step 5 commit** `feat(image): modular repo 推导 unet 单文件时 fallback clip/vae`

## Task 4:ModularImageBackend transformer override + 装配桥接

**Files**:`backend/src/services/inference/image_modular.py` + `backend/src/services/model_manager.py` + wiring 测

- [ ] **Step 1 失败测试**(wiring,mock):`ModularImageBackend(..., transformer_override=fake)` → `_ensure_pipe` 后 `pipe.update_components(transformer=fake)` 被调。`_get_or_load_modular_adapter`:unet 是 comfy quant(quant_type != none / 非 repo/transformer 下)→ 经 `dequant_and_convert` + `from_config` 建 override 传入。
- [ ] **Step 3 实现**:
  - `ModularImageBackend` 加 `transformer_override` 参数;`_ensure_pipe` 末尾若有 override 则 `update_components(transformer=override)`。
  - `_get_or_load_modular_adapter`:检测 unet comfy quant → 线程内 `dequant_and_convert` + `Flux2Transformer2DModel.from_config(repo/transformer)` + `load_state_dict` → override(diffusers/torch import 仍只经 image_modular)。
- [ ] **Step 4 跑通 + 回归**
- [ ] **Step 5 commit** `feat(image): ModularImageBackend transformer override + comfy 量化装配桥接`

## Task 5:真模型验证 + PR

- [ ] **standalone smoke**:`NOUS_IMAGE_ENGINE=modular` + unet=comfy `fp8mixed` 单文件 + clip/vae=HF → get_or_load_image_adapter → infer → **出正确狐狸图**(对比 PR-1 spike v3 的 spike_fp8_converted.png)。落 cuda:1。
- [ ] 后端全套 + ruff;PR → CI 全 pass(逐项确认)→ merge。
- [ ] PR 描述附:comfy fp8mixed 经生产 modular 路径出图结果 + 「不省显存」提醒。
