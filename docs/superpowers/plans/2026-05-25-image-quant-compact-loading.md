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

## PR-1:fp8 真紧凑加载(weight-only,省显存)✅ MERGED #139(5e7e884)

> **Task 1.0 spike 已完成**(结论见 spec):方案 = **torchao `Float8WeightOnlyConfig` 量化 bf16 模型
> at load**(= ComfyUI weight_dtype 机制,用户拍板)。真模型验过:出正确图 + 进 24GB(峰值 23.3GB)+
> model.dtype 仍报 bf16(绕开 noise bug)。
>
> **⚠️ 速度更正(spike 证伪 bump 前提)**:3090 是 Ampere sm_86,**无 fp8 tensor core**,fp8 只省显存
> **不加速**;28.8s = 3090 跑 9B DiT @1024² 的原生速度(torch 2.10/2.11 字节一致)。所以 fp8 **与全栈
> bump 解耦,在当前 torch 2.10 即可落地**(bump 只为栈更新,不影响 fp8)。dynamic-activation fp8 在
> 3090 直接 assert 报错(sm<8.9)——只用 weight-only。comfy 预量化 fp8mixed 文件保持现状(dequant)。

### Task 1.1 — 实现紧凑加载 ✅

**Files**:`src/services/inference/image_modular.py` / `model_manager.py` / `pyproject.toml`

- [x] `weight_dtype` 选 fp8(fp8_e4m3,node.yaml 已有)→ `_ensure_pipe` bf16 加载后
  `_quantize_fp8_weight_only`(torchao `Float8WeightOnlyConfig`,transformer+text_encoder);override 优先。
- [x] torchao>=0.17 进 image extra。**1024² 峰值贴边 23.3GB —— 安全余量(分辨率/释放策略)留后续。**

### Task 1.2 — fit-check 估算校准 ✅

**Files**:`model_manager.py`(`_estimate_image_vram_mb`)— fp8 时 transformer/clip 估算减半。

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
