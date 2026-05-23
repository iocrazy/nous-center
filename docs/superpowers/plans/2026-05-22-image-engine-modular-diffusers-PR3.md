# PR-3 — LoRA 接 Modular 引擎(+ 多 CLIP 沿用 gated)Plan

> REQUIRED SUB-SKILL: executing-plans。
> Spec: `2026-05-22-image-engine-modular-diffusers-design.md`。前置 PR-1(#128/#129)+ PR-2(#130)。

**LoRA-API 未知已 spike 解除**:`Flux2KleinModularPipeline` 经 `Flux2LoraLoaderMixin`(MRO)有 `load_lora_weights`/`set_adapters`/`get_active_adapters` —— 与 `DiffusionPipeline` **同一 mixin**。所以:
- 同样的 **is_kohya dispatch bug**(把 ComfyUI/BFL LoRA 误路由)→ 本会话 #125 的 `_maybe_convert_comfy_flux2_lora` 修复**直接复用**。
- `ModularPipeline.from_pretrained` 返回模型专用子类(Flux2Klein…),有 LoRA API。

**Goal**:`ModularImageBackend` 支持 LoRA(含 ComfyUI 格式),与 legacy 行为对齐。多 CLIP 沿用收敛的 gated(单 CLIP 走;多编码器执行报清晰 gated,无模型可验)。

**Branch**:`feat/image-modular-lora-pr3`。

---

## Task 1:共享 ComfyUI-LoRA 转换 helper(从 image_diffusers 提到可复用处)

**Files**:`backend/src/services/inference/image_diffusers.py`(`_maybe_convert_comfy_flux2_lora` 已存)+ 视情况移到共享模块 + 测试

- [ ] **Step 1**:确认 `_maybe_convert_comfy_flux2_lora`(#125)可被 image_modular 复用(它内部 lazy import diffusers)。若耦合 image_diffusers 其它,提到 `quant_loaders` 旁或新 `lora_convert.py`。保持 diffusers import 仅在被 image_modular 经过的路径。
- [ ] **Step 2/3**:复用(import)或轻移;不改转换逻辑(#125 已真模型验过)。
- [ ] **Step 5 commit**(若有移动)

## Task 2:ModularImageBackend._apply_loras

**Files**:`backend/src/services/inference/image_modular.py` + wiring 测

- [ ] **Step 1 失败测试**(wiring,mock pipe):`ModularImageBackend` infer 带 `req.loras` → `_ensure_pipe` 后对 pipe(Flux2Klein)`load_lora_weights(...)` + `set_adapters(names, weights)`;ComfyUI 格式经 `_maybe_convert_comfy_flux2_lora` 预转换;无 lora → 不调。断言 load_lora_weights/set_adapters 调用 + 参数。
- [ ] **Step 3 实现**:`_apply_loras(loras)` 镜像 legacy `DiffusersImageBackend._apply_loras`:offload-safe 顺序、ComfyUI 转换、load_lora_weights(converted dict 或 path)、set_adapters(strength)、zero-match 防御报错。infer 前调。
- [ ] **Step 4 跑通 + 回归**
- [ ] **Step 5 ruff + commit** `feat(image): ModularImageBackend LoRA(复用 #125 ComfyUI 转换)`

## Task 3:多 CLIP 沿用 gated(确认,非新建)

- [ ] 确认 modular 路径:单 CLIP(clip_stack 1 个编码器)正常走;多编码器(len>1)沿用收敛的 runner gated(清晰报错,无多编码器模型可验,spec §9 future)。若 modular 装配未覆盖该 gate,补一致的 gated 报错。
- [ ] commit(若有改动)

## Task 4:真模型验证 + PR

- [ ] **standalone smoke**(cuda:1):`NOUS_IMAGE_ENGINE=modular` + Load LoRA(`klein_9B_Turbo_r128`,ComfyUI 格式)→ get_or_load_image_adapter → infer。验:① `get_active_adapters()` 含该 LoRA(非零匹配)② 出图**与无 LoRA 同 seed 有差异**(LoRA 真生效,turbo 用低步数)。对比 #125 legacy 的 turbo 出图。
- [ ] 后端全套 + ruff;PR → CI 全 pass(逐项)→ merge。
- [ ] PR 描述附:modular + ComfyUI LoRA 出图结果。
