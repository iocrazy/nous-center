# Image 引擎层迁移 — 自写 ImageSampler → Modular Diffusers

**Status**: Draft (rev 1,spike 验证中)
**Author**: heygo
**Date**: 2026-05-22
**Supersedes**: `2026-05-19-image-component-multi-gpu-design.md` 的**引擎实现**(§5.6 自写 `ImageSampler`、§5.2 `DiffusersImageBackend` 组件装配、§5.5 `get_or_load_image_adapter`)。**保留**:组件扫描(§4.6)、quant_loaders(§5.3,作量化桥接)、收敛后的细粒度图编辑器/runner/API(`2026-05-21-image-granular-convergence-design.md` rev 2 的前端 + runner 派发 + 计费/服务发布)。

## 0. 这版要做什么

用 **HuggingFace Modular Diffusers**(`ModularPipeline` + `ComponentsManager` + 官方 blocks)**替换我们 2026-05-19 自写的图像引擎**(ImageSampler 采样循环 + 组件装配 + 跨设备 .to()),并**同时解决「comfyui 量化模型(fp8mixed 等)怎么用」**(用户要求)。编辑器/runner/计费等上层不动。

**为什么现在**:我们当初自写 ImageSampler,是因为旧 diffusers 的 monolithic `Pipeline.__call__` 硬假设同 device、做不了分阶段(Task 0 实测)。Modular Diffusers(diffusers 0.38.0.dev0)现在官方解决了**而且更多**:flux2/ernie 官方 blocks、拆 stage 跑、`ComponentsManager` 动态 offload、per-component dtype/量化。这是 diffusers 原生升级,**与「继续自建 diffusers 框架、不转 ComfyUI」的方向一致**。

## 1. Spike 实测结论(2026-05-22,落 cuda:1 Pro 6000)

| 验证 | 结果 |
|---|---|
| `ModularPipeline.from_pretrained(HF-layout Flux2-klein)` + bf16 出图 | ✅ 输出**正确**(与自写 ImageSampler 同 seed 几乎同图),**推理 6.4s vs 自写 27s**(≈4×,待深究但至少不退化) |
| diffusers `from_single_file` 直接吃 comfy `fp8mixed` 单文件 | ❌ `chunk expects ≥1-dim tensor`(不认 comfy_quant 打包) |
| diffusers `from_single_file` 直接吃 `Q5_K.gguf` | ❌ `OSError: Unable to load weights`(不吃这个 GGUF) |
| **桥接**:quant_loaders 反量化 fp8mixed → `from_config` + `load_state_dict` **(无转键)** → 出图 | ❌ spike v2:`missing=233 unexpected=201`,**键不匹配出噪声图**(comfy `double_blocks` vs diffusers `transformer_blocks`) |
| **桥接(修正)**:quant_loaders 反量化 → **`convert_flux2_transformer_checkpoint_to_diffusers`(diffusers 自带)转键** → `from_config` + `load_state_dict` → `update_components` → 出图 | ✅ spike v3:`missing=0 unexpected=0`,**出正确狐狸图**(6.6s) |

**修正(spike 两次纠错)**:
1. 换 Modular Diffusers **不**自动白捡 comfy 量化/GGUF —— `from_single_file` 不认 comfy fp8 打包(chunk 崩)。
2. 即便我们 quant_loaders 反量化了,**还缺一步键转换**(comfy/BFL→diffusers,和 LoRA 那个同类)。diffusers 自带 `convert_flux2_transformer_checkpoint_to_diffusers`(single_file_utils.py:3773,from_single_file 内部用的就是它)可复用:**dequant(我们)→ 转键(diffusers 的)→ load**,顺序对了两个问题都解(spike v3 实测 missing=0,出正确图)。
3. **副发现(现有栈潜在 bug)**:`_load_hf_or_quant`(image_diffusers.py)的 quant fallback 是 `from_config` + `load_state_dict(strict=False)` **没有转键** —— 真去载 comfy 单文件会静默出垃圾。即收敛里的 fp8 支持是**名义上的/未真验**(收敛 smoke 用的是 HF-layout bf16)。此修法对新引擎和现有栈都适用。
4. **GGUF**:`from_single_file` 仍失败;套路应同(GGUF dequant + 同一转换器),单列 future。

## 2. 目标 / 非目标

### 目标
- runner 的 image 执行引擎从「`get_or_load_image_adapter` + 自写 `ImageSampler`」换成「`ModularPipeline`(Flux2Klein 官方 blocks)+ `ComponentsManager`」。
- **删除自写 `ImageSampler`**(采样循环 + 跨设备 .to() + dtype 对齐,本会话还修过 dtype bug)及其专属测试。
- **comfyui 量化模型可用**:fp8mixed / mxfp8mixed / nvfp4mixed 单文件 → quant_loaders 解包 → `from_config` 建 module → `update_components`。
- 细粒度图编辑器节点(Load Diffusion/CLIP/VAE/Encode/KSampler/VAE Decode)语义不变,executor 改为「装配/驱动 ModularPipeline」而非 ImageRequest→ImageSampler。
- 保留:组件扫描、quant_loaders、L2 输出缓存、四态 UI、runner 派发/计费/服务发布、LoRA(ComfyUI 格式转换)。

### 非目标(本 spec 不做)
- **GGUF 加载**:单列 future(需写 GGUF dequant,如 city96 那套,或等 diffusers 补 Flux2 GGUF)。
- **ComponentsManager 动态 offload(DisTorch 式落小卡)**:本次落 Pro 6000 不需要;接口接上但「3090 装大模型」的真验证留 future。
- 真 fp8 省显存(权重保持 fp8 + fp8 compute):本次反量化到 bf16 用(Pro 6000 不缺显存);省显存 fp8 留 future。
- ERNIE / 其它模型迁移:本次只迁 Flux2Klein(主力);ERNIE 官方 blocks 已在,后续同法接。

### 成功标准
- 细粒度图(bf16 HF-layout)经新引擎出图,SSIM ≈ 自写 ImageSampler(同架构 >0.99),耗时不劣于(预期更快)。
- comfyui fp8mixed 单文件经桥接出图正确。
- LoRA(klein turbo)经新引擎加载生效。
- 自写 ImageSampler 删除后,后端全套 + 真模型 smoke 绿。

## 3. 架构:替换什么 / 保留什么

| 现有(自建,2026-05-19) | 替换为 | 备注 |
|---|---|---|
| `image_sampler.py`(ImageSampler 采样循环) | `ModularPipeline`(Flux2Klein blocks,denoise/encode/decode 官方) | **删除** |
| `DiffusersImageBackend.from_loaded_components` / `_assemble_pipe` | ModularPipeline 装配 | **删除/精简** |
| `ModelManager.get_or_load_image_adapter` + 组件 L1(`_components`/`_image_adapters`) | `ComponentsManager`(跨 pipeline 共享 + 缓存) | runner 持一个 ComponentsManager 单例 |
| `load_component_module` / `_load_hf_or_quant` | ModularPipeline `load_components` + 量化桥接 | 量化桥接见 §4 |
| `quant_loaders`(fp8mixed/mxfp8/nvfp4 解包) | **保留**,作桥接 | comfy 量化 diffusers 不认 |
| 细粒度图 executors(产 ImageRequest) | 改为装配/驱动 ModularPipeline | 编辑器节点不变 |
| 组件扫描 / 编辑器 / runner 派发 / L2 / 四态 / 计费 / 服务发布 | **全保留** | 上层不动 |

**runner 执行流(新)**:
```
flux2_vae_decode dispatch → _node_executor
  → 取/建 ModularPipeline(Flux2Klein blocks,经 ComponentsManager)
  → load_components(per-component dtype);量化组件走桥接 update_components
  → 应用 LoRA(load_lora_weights,ComfyUI 格式转换沿用本会话修复)
  → pipe(prompt, steps, w/h, cfg, seed, [embeds if 拆 stage]) → image
  → write_image(签名 URL)→ L2 缓存
```

## 4. comfyui 量化模型桥接(用户要求重点)

diffusers `from_single_file` 不认 comfy 量化包(spike 实测)。桥接:

```python
# fp8mixed / mxfp8mixed / nvfp4mixed 单文件(spike v3 实测出正确图,missing=0)
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
spec = ComponentSpec(kind="unet", file=<单文件>, device=dev, dtype="bfloat16")
sd = QUANT_LOADERS.dispatch(spec)                      # ① 我们的反量化(解 comfy fp8 打包,已有)
conv = convert_flux2_transformer_checkpoint_to_diffusers(dict(sd))  # ② 转键 comfy→diffusers(关键!)
cfg = Flux2Transformer2DModel.load_config(<HF repo>/transformer)    # 架构 config
tr = Flux2Transformer2DModel.from_config(cfg).to(torch.bfloat16)
tr.load_state_dict(conv, strict=False)                # missing=0 unexpected=0
pipe.update_components(transformer=tr.to(dev))         # ③ 喂进 ModularPipeline
# 其余组件(clip/vae/scheduler)由 ModularPipeline.from_pretrained(HF repo) 提供
```

- **三步缺一不可**:① 反量化(解 fp8 打包)② 转键(comfy double_blocks→diffusers transformer_blocks)③ update_components。漏掉 ② 就是 spike v2 的噪声图。
- **单文件只是权重**;架构 config + clip/vae/scheduler/tokenizer 来自 HF repo。component_scanner 已能枚举单文件 + 探测 quant_type。
- 本次反量化到 bf16(Pro 6000 不缺显存,先证「文件可用」);真 fp8 省显存 future。
- **顺带修现有栈**:`_load_hf_or_quant` 的 quant fallback 也加 ② 转键(否则 comfy 单文件静默出垃圾)。
- **GGUF**:`reject` 改「未就绪」提示;实际加载 future(GGUF dequant + 同一转换器)。

## 5. PR 拆分

- **PR-1 spike 固化 + 引擎骨架**:把 spike 的 ModularPipeline 出图路径固化进 runner(新 `image_modular.py` 执行器),与现有 ImageSampler 路径**并存**(feature flag / 节点 data 选择),真模型 smoke 对比 SSIM/耗时。
- **PR-2 量化桥接**:fp8mixed/mxfp8/nvfp4 单文件 → quant_loaders → update_components,真模型验证 comfy 量化出图。
- **PR-3 LoRA + 多 CLIP 接新引擎**:ComfyUI LoRA 转换(本会话修复)接 ModularPipeline;多 CLIP 走官方 blocks。
- **PR-4 切换 + 删自写引擎**:细粒度图默认走 Modular 引擎,删 `image_sampler.py` + `get_or_load_image_adapter` + 相关测试;全套 + smoke 绿。
- **PR-5(future)**:GGUF 加载 / ComponentsManager offload 落小卡 / 真 fp8 省显存 / ERNIE 迁移。

## 6. 风险 / 取舍(诚实)

- **Modular Diffusers experimental(0.38.0.dev0,会破坏性变更)**:最大风险。缓解 —— 钉死 diffusers 版本;引擎层封装在 `image_modular.py` 一处,API 变只改一处;PR-1 并存灰度,验证够稳再 PR-4 删旧。
- **沉没成本**:删的是 2026-05-19 的 6-PR 引擎(ImageSampler 等);本会话刚做的收敛编辑器/runner/LoRA **全保留**。换来:删大量自维护采样/装配代码、更快、新模型官方 blocks 直接拿。
- **量化/GGUF 不免费**(spike 纠正):fp8 系列复用 quant_loaders 桥接;GGUF 单独。

## 7. Test Plan
- 每 PR:单测(桥接/装配 stub)+ **真模型 standalone smoke**(SSIM vs 自写 ImageSampler、comfy fp8mixed 出图、LoRA 生效),沿用 dev_env_gotchas standalone 测法。
- PR-4 删旧引擎后:后端全套 + 前端 tsc/vitest/build + grep 无 ImageSampler 残留。
- 遵守 feedback_verify_real_model:每个核心假设(Modular 出图正确、fp8 桥接、LoRA)真模型验。
