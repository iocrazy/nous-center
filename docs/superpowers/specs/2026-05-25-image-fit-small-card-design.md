# 量化图像模型按原生紧凑尺寸加载 — 显存设计(无 offload)

> 状态:**设计定稿,待 push + 写 plan**。
> 接 [[2026-05-22-image-engine-modular-diffusers-design]](引擎=Modular Diffusers)+
> 加载分离(#138)。依据 [[feedback-long-term-robustness]] [[feedback-verify-real-model]]。

## 目标与哲学(用户拍板 2026-05-25)

让大图像模型能跑在小卡(3090 24GB)上 —— **靠"用更量化的模型文件",不靠 offload。**

核心哲学(用户):
- **量化是模型文件的属性,不是运行时模式。** 用户挑的是文件(`…-bf16.safetensors` /
  `…-fp8mixed.safetensors` / `…-nvfp4mixed.safetensors` / `….gguf`),量化格式由文件决定
  (`component_scanner` 已从文件名认出 `quant_type`)。没有抽象的"选 fp8"。
- **能装就装,装不下就报清晰错误。** 不做 offload(又慢 8× 又依赖脆弱的 modular dtype 胶水)。
  塞不下 → 换更量化的文件,或换大卡(Pro 6000 96GB)。
- **不引入 `memory_mode` 枚举旋钮。** `weight_dtype` 下拉保留(bf16/fp16 原生 + 把 bf16 文件
  运行时降精度的次要便利),但主路径是"挑量化文件"。

## 当前 bug:量化文件没有按紧凑尺寸加载(这就是"能装就装"现在做不到的原因)

- `quant_loaders.py`:fp8mixed/mxfp8mixed/nvfp4mixed 都 **dequant 到 bf16**(L312/L265/L216)。
  17GB 的 fp8 文件 → 显存里又变回 ~17GB bf16 → 3090 照样 OOM。**没省显存。**
- `build_bridged_transformer`(image_modular.py:61)把反量化后的 bf16 state_dict 塞进
  `from_config` 的空 transformer → bf16 驻留。
- GGUF:`quant_loaders.py:92-97` eager reject(`UnsupportedQuantError`)。
- nvfp4 单文件 convert:`quant_loaders.py:219` `reshape(orig_shape)` 在某些张量上挂
  (memory 记的 "chunk expects ≥1-dim",待复现确认确切触发点)。
- fit-check 估算偏差:`get_or_load_image_adapter` 的 `_estimate_image_vram_mb` 按**文件字节数**估,
  fp8 文件字节小但实际加载成 bf16(2×)→ 估小了 → 先过检查再 OOM。修紧凑加载后估算才准。

## diffusers 钉死 commit(c8eba433 / 0.38.0.dev0)原生量化能力

- `GGUFQuantizationConfig` + `FromOriginalModelMixin.from_single_file` → 原生 GGUF 加载
  (Q4/Q5/Q8 dequant-on-the-fly,权重紧凑驻留)。
- `TorchAoConfig` / `QuantoConfig` / `BitsAndBytesConfig` / `PipelineQuantizationConfig`
  → 真量化加载(量化 linear 自报 compute dtype,forward 内部 dequant)。需装对应包
  (torchao / optimum-quanto / bitsandbytes)。
- `ModelMixin.enable_layerwise_casting` / `hooks.apply_layerwise_casting`(fp8 存储 / bf16 计算)。

**所以紧凑加载是"接官方量化能力",非自建。**

## PR-0 Spike 已验(真模型 cuda:2=3090 24GB,Flux2-klein-9B 33GB bf16)

`tests/manual/spike_fit_3090.py`。

- **fp8 存储确实省显存**:transformer+text_encoder layerwise-cast fp8 → **峰值 16.7GB**(33GB→16.7GB),
  轻松进 24GB。核心物理假设成立。
- **但 modular pipe 不感知组件的 storage dtype**:`Flux2PrepareLatentsStep`(before_denoise.py:300/312)
  的 noise device/dtype 跟组件走 —— transformer/text_encoder 变 fp8 → `block_state.dtype`
  = `prompt_embeds.dtype` = fp8 → `randn_tensor(dtype=fp8)` 崩 `normal_kernel_cuda not implemented`。
  试 `pipe(dtype=bf16)` **被 modular pipe 忽略**("Unexpected input … ignored")—— 无干净覆盖入口。
- offload 也验过能跑(group-offload,16GB 峰值,出正确图,但 53s ~8× 慢)—— **按用户决定砍掉。**

**结论**:紧凑加载省显存可行;难点 = 让量化/降精度的组件在 modular pipe 里正确出图(noise/latents
的 dtype 必须独立于组件 storage dtype)。

## PR-1 Task 1.0 spike 已验(torchao fp8 路径,`tests/manual/spike_quant_compact.py`)

**方案 A = diffusers/torchao `Float8WeightOnlyConfig` 验证通过(真模型 3090)**:
- `quantize_(transformer, Float8WeightOnlyConfig())` + 同样量化 text_encoder → **出正确狐狸图**;
- 量化后 `model.dtype` 仍报 **bf16**(量化权重包在 tensor subclass 里,对外报 compute dtype)
  → **天然绕开 PR-0 的 noise-dtype bug**(prepare_latents 用 bf16);
- 进 24GB 3090:resident 17.2GB / **推理峰值 23.3GB(贴边但 fit)**,1024² 活化大;
- transformer 单量化:17GB→8.7GB(微验证);
- **代价:推理 28.8s** —— ⚠️ **不是 torch 版本问题,是 3090 Ampere sm_86 无 fp8 tensor core**
  (torchao dynamic-activation fp8 直接 assert `需 CUDA>=8.9`)。fp8 在 3090 **只省显存不加速**,
  28.8s = 3090 跑 9B DiT @1024² 的原生速度(torch 2.10/2.11 字节一致,见
  [[2026-05-25-inference-stack-bump-torch211-design]] 的 spike)。fp8 真加速只在 sm≥8.9 卡。
- raw layerwise-casting(PR-0):model.dtype 变 fp8 → randn 崩,且 `pipe(dtype=)` 被忽略 → 弃。

**关键语义**:torchao 量化的是 **bf16 HF repo**(at load),不是读 comfy fp8mixed 文件。即 fp8 =
`weight_dtype` 选项(把加载的 bf16 模型量化)—— 与 ComfyUI UNETLoader 的 weight_dtype 机制一致。
comfy 预量化文件(fp8mixed/nvfp4)是另一格式,紧凑加载需 comfy-aware ops(单独评估)。

## PR-1 实现(已完成,本 PR)

`weight_dtype` 选 fp8_e4m3 → `ModularImageBackend._ensure_pipe`:bf16 加载 → torchao
`quantize_(transformer/text_encoder, Float8WeightOnlyConfig())`(model.dtype 仍 bf16,绕开 noise bug)
→ pipe.to(device)。`_estimate_image_vram_mb` 对 fp8 把 transformer/clip 估算减半(fit-check 准)。
torchao 进 image extra。**生产路径 smoke `smoke_fp8_compact.py` 验过**(3090 dtype=fp8_e4m3 出正确狐狸图,
峰值 23.3GB,28.9s)。无 offload、无 memory_mode 枚举。

## 设计

1. **量化 = 文件属性**(已有):scanner 认 `quant_type`;用户在 Load Diffusion Model 的文件下拉挑。
2. **紧凑加载**:量化文件按原生小尺寸装显存 + 正确出图。优先 quantization_config 路径
   (model.dtype 仍报 bf16 → 绕开 noise-dtype bug);层归一化等敏感层留 bf16。
3. **fit-check 报错**:装得下跑,装不下抛清晰错误(校准估算 = 紧凑加载后的真实占用,非文件字节)。
4. **无 offload、无 memory_mode 枚举。**

## PR 拆分(plan 细化)

- **PR-1**:fp8/comfy-量化 **紧凑加载**(改 `quant_loaders` / `build_bridged_transformer` 不再 dequant
  到 bf16;或改走 quantization_config)。先 real-model spike 验"紧凑 + 出正确图 + 估算准 + 装不下报错"。
- **PR-2**:**GGUF** 加载(`from_single_file + GGUFQuantizationConfig`;移除 eager reject;scanner 已认 .gguf)。
- **PR-3**:**nvfp4** 单文件 convert bug 修(复现 + guard/修 reshape)+ 紧凑加载。
- 每 PR:真模型 smoke(紧凑加载出正确图 + 显存实测)+ CI。
