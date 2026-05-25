# 量化图像模型按原生紧凑尺寸加载 Plan(无 offload)

> REQUIRED SUB-SKILL: executing-plans。
> spec:`docs/superpowers/specs/2026-05-25-image-fit-small-card-design.md`。

**Goal**:量化图像模型文件按**原生紧凑尺寸**加载进显存 + 正确出图(真省显存,大模型能上
3090);装不下报清晰错误。**无 offload、无 memory_mode 枚举**(用户拍板)。量化=文件属性
(scanner 已认),用户挑文件。

**核心已验**(PR-0 spike):fp8 存储省显存(33GB→16.7GB)成立;难点=量化组件在 modular pipe
里 noise dtype 传播 bug;走 quantization_config 路径(model.dtype 仍报 bf16)很可能绕开。

**Branch**:每 PR 独立分支。

---

## PR-1:fp8 真紧凑加载(先 spike 定方案)

### Task 1.0 — 加载方案 spike(真模型,gating)

**File**:`backend/tests/manual/spike_quant_compact.py`

- [ ] 在 3090(cuda:2)上验**三条候选加载方案**哪条能"紧凑驻留 + 出正确图 + model.dtype 报 bf16
  (绕开 noise bug)":
  - (A) `TorchAoConfig` float8 量化 **bf16 HF repo** 的 transformer(+text_encoder)at load
    (diffusers 原生,量化 linear 自报 compute dtype)。需 `torchao`。
  - (B) 保留 comfy fp8mixed 文件的 fp8 权重 + comfy scale,自定义 dequant-on-the-fly linear
    (不 dequant 到 bf16)。
  - (C) layerwise-casting(PR-0 已知 noise bug,作对照基线)。
- [ ] 记录每条:峰值显存、出图正确性(肉眼狐狸图)、推理延迟、是否撞 noise/latents dtype。
- [ ] 结论 → 定 PR-1 实现路径(预期 A 最干净);若 comfy fp8mixed 文件无法干净紧凑加载,
  记下取舍(可能 fp8 走 torchao-from-bf16,comfy 预量化文件单独处理/留 dequant)。
- [ ] 若方案涉及新依赖(torchao),评估装进 `pyproject.toml` 的 `image` extra(CI 不装,真模型 only)。

### Task 1.1 — 实现紧凑加载

**Files**:`src/services/inference/quant_loaders.py` / `image_modular.py` / `model_manager.py`

- [ ] 按 1.0 定的方案改:量化/降精度 transformer 按紧凑尺寸驻留(不再无条件 dequant 到 bf16)。
- [ ] 确保 modular pipe 出图正确(model.dtype 报 compute dtype;noise/latents 用 compute dtype)。

### Task 1.2 — fit-check 估算校准

**Files**:`model_manager.py`(`_estimate_image_vram_mb`)

- [ ] 估算反映**紧凑加载后**的真实占用(非文件字节;量化文件字节小但要按实际驻留估),
  装不下抛清晰错误(已有错误路径,校准数字 + 文案)。

### Task 1.3 — 测试

- [ ] 单元(CI,mock):紧凑加载分支选择 + 估算逻辑。
- [ ] 真模型 smoke:fp8 紧凑加载 Flux2 → 3090 出正确图 + 实测显存 < bf16 + 装不下报错。

### Task 1.4 — PR → CI → merge

---

## PR-2:GGUF 加载

**Files**:`quant_loaders.py`(移除 L92-97 eager reject)/ `image_modular.py`

- [ ] `from_single_file + GGUFQuantizationConfig(compute_dtype=bf16)` 加载 GGUF transformer
  (diffusers 原生);scanner 已认 `.gguf`(quant_type=gguf)。
- [ ] 真模型 smoke:GGUF(如 Q4)Flux2 → 3090 紧凑出图。
- [ ] fit-check 对 GGUF 估算。PR → CI → merge。

## PR-3:nvfp4 单文件 convert bug 修

**Files**:`quant_loaders.py`(nvfp4 路径 L156-224)

- [ ] 复现 "chunk expects ≥1-dim"(确切触发张量/行)+ 修(guard / reshape 修正)。
- [ ] nvfp4 也走紧凑加载(随 PR-1 方案)。真模型 smoke。PR → CI → merge。

---

## 不做(用户明确)

- ❌ CPU offload / group-offload(慢 8× + 脆弱 modular dtype 胶水)。
- ❌ `memory_mode` 枚举旋钮(量化=文件属性,挑文件即可)。
