# 图像引擎收口 — modular 退役 + comfy 单文件自检架构(对齐 ComfyUI)— 设计

> 状态:**草稿,待 review + push 后写 plan**。
> 接 true-cfg 修复(plan 2026-05-25 PR-1)。依据 [[feedback-long-term-robustness]] [[feedback-verify-real-model]]
> [[project-image-model-layout]] [[project-image-sampling-progress]] [[project-image-engine-modular]]。
> 用户拍板(2026-05-26 对话):按存放位置定模型类别;ComfyUI 不用 diffusers modular;modular 该退役。

## 动机(根因 + 真模型已证)

true-cfg 修复(已开 PR)坐实了一件更大的事:**nous 图像引擎选错了抽象**。

- 用户实际跑的模型**全是 comfy 单文件**(`diffusion_models/` 下 True-v2、anima…);`diffusers/` 里只有
  2 个原生整模型(Flux2-klein-9B、ERNIE-Image),且只被"借 config"用,并不真出图。
- nous 现用 **Modular Diffusers** 单引擎(`ModularImageBackend`)。但 modular:
  1. 蒸馏 block(`Flux2KleinAutoBlocks`)把 cfg/negative 掐了 → 把 true-cfg 模型跑成蒸馏(质量根因,SSIM A/B 已证)。
  2. 无 `callback_on_step_end`(逐步进度要 hack `progress_bar`)。
  3. experimental API,blast radius 需隔离。
- **ComfyUI(comfy 单文件的事实标准)完全不用 diffusers modular**(grep `comfy/` = 0):它有自己的引擎
  (`comfy/ldm` + `k_diffusion`),从权重键自检架构(`model_detection.py`),cfg/negative 全交用户。
- 标准(非 modular)diffusers pipeline 在三件事上都更对:`is_distilled` 是构造参数、negative 自动 true-cfg、
  `callback_on_step_end` 内置——**等于"在 diffusers 里复刻 ComfyUI 的行为"**。modular 唯一独有的 ComponentsManager
  (offload)标准 pipeline 用 `enable_model_cpu_offload()` 也能做。

**结论:modular 退役,收口到「标准 diffusers pipeline + 按位置分类 + 自检架构」。**

## 模型两类(按存放位置定 —— 用户原则)

| 类别 | 位置 | 标志 | 加载 |
|---|---|---|---|
| 原生 diffusers 整模型 | `image/diffusers/<name>/` | 有 `model_index.json` | `Pipeline.from_pretrained(dir)`,**尊重其 config(含 is_distilled)** |
| ComfyUI 单文件 | `image/diffusion_models/`(+ `text_encoders/`、`vae/`) | 单文件无 config | 自检架构 → 标准 pipeline,`is_distilled=False`(cfg 控制) |

loader 节点已是两套(`flux2_load_checkpoint` 整模型 / `flux2_load_diffusion_model`+`load_clip`+`load_vae` 单文件)。

## 设计

### A. 引擎:标准 diffusers pipeline(modular 退役)
- 新 `DiffusersImageBackend`(或就改造 `ModularImageBackend` 去 modular 化):按架构选标准 pipeline 类
  (Flux2→`Flux2KleinPipeline`、Qwen-Image→`QwenImagePipeline`、AuraFlow→`AuraFlowPipeline`…)。
- `diffusers.modular*` import、`ComponentsManager`、modular blocks 全删;offload 用标准 pipeline 的
  `enable_model_cpu_offload()` / `enable_sequential_cpu_offload()`(future,塞小卡时启用)。
- LoRA / fp8(torchao)/ comfy LoRA 转换 **全是 pipeline 无关的**,原样保留(true-cfg PR 已证桥接组件喂标准 pipe 出图)。

### B. comfy 单文件轨:自检架构 + 内置 config(砍掉 per-model 参考库)
**现脆弱点**:单文件无 config → nous 现在要用户在 `diffusers/` 放一个**同架构原生整模型(18GB)**当 config 参考
(`_reference_repo_for_arch`)。换个架构(anima=Qwen-Image)就进不来。ComfyUI 不需要——它从权重自检 + 代码内置架构。

**改为**(对齐 ComfyUI + diffusers 已有能力):
- diffusers `from_single_file` 已能**从权重键自检架构 + 自带 config 映射**(`CHECKPOINT_KEY_NAMES` 含
  flux2/auraflow/qwenimage;`convert_*_checkpoint_to_diffusers`)。
- **plain 单文件**(bf16/fp16):直接 `from_single_file`(传 `config=<本地内置 config 目录>` 离线)。
- **comfy 量化单文件**(fp8mixed/nvfp4/gguf):保留 nous `dequant_comfy_mixed` 反量化,但 **config 取自内置
  per-arch config(几 KB json),不再要 18GB 参考整模型**。
- nous 仓库内 **bundle 每个支持架构的 diffusers config 目录**(transformer/text_encoder/vae/scheduler 的 config.json +
  tokenizer,**无权重**),或首次用时从 HF 缓存。`diffusers/` 不再被当 config 参考用(回归其本义:放原生整模型)。

### C. is_distilled / true-cfg(对齐 ComfyUI)
- comfy 单文件 → `is_distilled=False`,cfg 控制:cfg=1 退化无 CFG(蒸馏模型这样用),cfg>1+negative=true-CFG。
  **不做模型级 is_distilled 死开关**(ComfyUI 没有;用户用 cfg 表达)。`model_arch_adapter.py` 旧的
  `supports_cfg=False` 假设作废,改为 per-arch 的采样默认(shift、默认 cfg/steps)。
- 注:ComfyUI Flux2 `shift=2.02`、anima `ModelSamplingAuraFlow shift=3.0` —— shift 是 per-arch 采样默认,纳入注册表。

### D. per-arch 支持注册表(一次性加新架构,像 ComfyUI)
一个 `ImageArchSpec` 注册表(替代 `_reference_repo_for_arch` + `MODEL_ARCH_REGISTRY`):
```
arch → { pipeline_cls, config_dir(内置), text_encoder_arch, vae_cls, default_shift, default_steps, default_cfg }
```
- Flux2(已有):`Flux2KleinPipeline` + Qwen3-8B TE + `AutoencoderKLFlux2` + shift。
- Qwen-Image / AuraFlow(anima):新加(`QwenImagePipeline`/`AuraFlowPipeline` + qwen3 TE + `AutoencoderKLQwenImage`)。
- 架构检测:复用 diffusers `from_single_file` 的 `CHECKPOINT_KEY_NAMES`(权重键)→ 不靠文件名/位置猜架构。
**加新架构 = 注册表加一条 + bundle 它的 config**,不是为每个模型配置。

### E. loader 两套(已有,确认)
- `flux2_load_checkpoint`:原生整模型(`diffusers/`)。
- `flux2_load_diffusion_model` + `load_clip` + `load_vae`:comfy 单文件(`diffusion_models/` 等)。

## PR 拆分(plan 细化时定)
- **PR-A**:`DiffusersImageBackend` 收口(去 modular,标准 pipeline;Flux2 先行)+ 删 `diffusers.modular*` import/依赖
  收紧。真模型验:Flux2 单文件/整模型 cfg/negative/LoRA/fp8 全过。
- **PR-B**:架构自检 + 内置 config(砍 `_reference_repo_for_arch` 的 18GB 参考库依赖;`from_single_file` 路径 + bundle config)。
- **PR-C**:`ImageArchSpec` 注册表 + 加 Qwen-Image/AuraFlow(anima 真模型出图)。
- **PR-D**:offload(standard pipe `enable_*_cpu_offload`)落小卡(并入 [[project-image-fit-small-card]])。

## 不做 / future
- 不移植 ComfyUI 的 k-diffusion 采样栈(继续用 diffusers scheduler;见 sampling-control plan PR-2)。
- GGUF 反量化、nvfp4 单文件 convert bug 仍按 [[project-image-engine-modular]] future 排。
- ERNIE 迁移并入注册表(PR-C 后)。

## 验证(真模型,feedback-verify-real-model)
- Flux2 True-v2:cfg/negative/LoRA/fp8 经新 backend 出图正确(扩 smoke_true_cfg_prod / smoke_single_file_prod)。
- anima(Qwen-Image):丢进 `diffusion_models/` → 自检架构 → 出图(对照其自带 ComfyUI workflow 的 cfg/negative/shift)。
- 无参考库时单文件仍能加载(证明砍掉 per-model 参考库)。
