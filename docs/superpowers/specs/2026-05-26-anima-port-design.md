# Anima 模型移植到 nous — 设计 spec

> 状态:**草稿,待 review**。
> 接 plan `2026-05-26-image-engine-ux-consolidation.md` PR-C(原 plan 假设 anima = Qwen-Image,实测**错判**)。
> 依据 [[feedback-long-term-robustness]] [[feedback-verify-real-model]] [[project-image-model-layout]]。

## 背景:原 plan 判断失误

PR-C 原承诺「per-arch 注册表 + Qwen-Image/AuraFlow → **anima 能跑**」。读 anima 源码确认:

- **anima 不是 Qwen-Image 标准模型** —— CircleStone Labs + Comfy Org 合作训练的 2B 自定义 DiT。
- **anima 不是 AuraFlow** —— 工作流虽用 `ModelSamplingAuraFlow shift=3.0`,但权重结构完全不同。
- **anima 不在 diffusers 里** —— 只在 ComfyUI(自带 `comfy/ldm/anima/model.py`)。
- 权重键开头:`net.blocks.0.adaln_modulation_cross_attn.*` —— 这是 NVIDIA Cosmos Predict2 衍生的 DiT block 风格。

**实际架构**:
```
Anima = MiniTrainDIT (来自 cosmos/predict2.py)
      + LLMAdapter (anima 特有,t5xxl 嵌入→qwen 上下文桥接)
      + Wan21 latent format
      + qwen3-0.6b base text encoder
      + Qwen-Image VAE (diffusers 有 AutoencoderKLQwenImage)
```

## ComfyUI 实现盘点(精确代码量)

| 文件 | 行数 | 干啥 |
|---|---|---|
| `comfy/ldm/cosmos/predict2.py` | **899** | MiniTrainDIT(base DiT)+ PatchEmbed + Sample3DRoPE + AdaLNLora 等 building blocks |
| `comfy/ldm/anima/model.py` | 214 | Anima 类继承 MiniTrainDIT + LLMAdapter + RotaryEmbedding + Attention + TransformerBlock |
| `comfy/text_encoders/anima.py` | 63 | AnimaTokenizer + te() qwen3-0.6b wrapper |
| `comfy/supported_models.py` Anima entry | ~25 | metadata:sampling_settings shift=3.0,Wan21 latent,memory factor |
| `comfy/model_base.py` Anima class | ~25 | extra_conds:cross_attn + t5xxl_ids + t5xxl_weights |
| `comfy/latent_formats.py` Wan21 | ~30 | latent format(可能 8x downsample VAE)|
| `comfy/operations.py` | ~200 | 自定义 Linear/RMSNorm(manual_cast,fp8 兼容)|
| **总核心 LOC** | **~1450** | 都要 port(或 import comfy 作为 dep)|

## 移植路径(三选一)

### 选项 A:完整 Python port(deps free)
**做法**:把 comfy ldm + operations 移到 nous 自有代码,作为独立 nn.Module。
**优**:nous 完全自包含、无外部依赖、可独立优化(fp8/torchao/offload 整合)。
**缺**:~1450 行核心移植 + 适配 diffusers 风格的 pipeline 装配。**4-7 工作日纯实施**。
**推荐**:✅ 长远架构最干净;模型是 2B 不大,值得 self-contained。

### 选项 B:ComfyUI as library
**做法**:`pip install` ComfyUI 或 git submodule,`from comfy.ldm.anima.model import Anima`。
**优**:**无移植成本**(直接复用);ComfyUI 更新自动同步。
**缺**:nous 引入 ComfyUI 这个**巨大依赖**(几千文件 + 它自己的 deps + main entry 不能 import)+ ComfyUI 不是 PyPI 包,要 vendor。
**评价**:❌ 工程上反感(blast radius、版本锁死、跟 nous 单 admin infra 定位不符)。

### 选项 C:Subprocess bridge(像 vLLM 那样)
**做法**:nous 起一个 ComfyUI runner 子进程,经 HTTP/WS 投递任务,拿结果。
**优**:用 ComfyUI 现成全栈(模型 / 采样 / VAE);多 arch 可一并接(stable diffusion / flux1 / cosmos / lumina 等)。
**缺**:架构剧变(从 in-process diffusers 改 process-bridge);进度/cancel 链路重做;ComfyUI 启动慢、占内存。
**评价**:⚠️ 是个可考虑的「nous 的 LLM-vLLM 模式」类比,但变更面太大,**留 future spec 独立评估**(可能 6-12 月才值得)。

## 推荐方案 = 选项 A,拆 7 个 PR

```
PR-anima-1  Cosmos predict2 building blocks 移植   ← 最大块(MiniTrainDIT + 子组件,~600 LOC)
   ↓
PR-anima-2  Anima nn.Module(继承 MiniTrainDIT + LLMAdapter,~250 LOC)
   ↓
PR-anima-3  qwen3-0.6b text encoder wrapper       ← anima.py 独立
   ↓
PR-anima-4  comfy.operations 简化版(Linear/RMSNorm manual_cast)
   ↓
PR-anima-5  AnimaPipeline 类(组装 transformer + TE + VAE + FlowMatchEuler)
            + bundle config(backend/configs/image_arch/anima/)
   ↓
PR-anima-6  ImageArchSpec 注册 anima + 从权重键自检 arch
   ↓
PR-anima-7  真模型 smoke:anima-base-v1.0 出图(对照其自带 ComfyUI workflow 的 cfg/sampler/shift)
            + LoRA 接(anima Turbo LoRA 那种)
```

每 PR 独立分支、单元测试、ruff/tsc/build、最终 PR-anima-7 真模型 e2e 验。

## 关键技术决策(实施前 review)

1. **`comfy.operations` port 精简**:只要 `Linear` + `RMSNorm` 的 manual_cast 路径(fp8 兼容)。约 50-80 行即可,不要全 port。
2. **sampling 转换**:ComfyUI 用 k-diffusion 采样器;nous 用 diffusers `FlowMatchEulerDiscreteScheduler`。anima sampling_settings shift=3.0 已查证 = `time_snr_shift` 公式 = `FlowMatchEuler(shift=3.0)`,**1:1 等价**(读 ComfyUI 源码已证;见 #146 PR-2 plan 注释)。
3. **t5xxl 路径**:anima 的 LLMAdapter 收 t5xxl ids/weights(似乎为 t5xxl 文本嵌入桥接)。但工作流只用 qwen3-0.6b → 这个分支可能 inference 时**不走**。**首版可不实现 t5xxl 桥**,留 Turbo LoRA 或其它工况再启用。
4. **Wan21 latent format**:anima 用 Wan21(可能 8x↓ VAE,跟 Qwen-Image VAE 对齐?)。**实施时验确认 latent 通道数 + downsample 比**,关键!
5. **bridging 单文件**:`build_bridged_transformer` 现在 Flux2-specific(`Flux2Transformer2DModel.load_config + dequant_and_convert`)。anima 需 anima 版本(`AnimaModel.load_config + dequant_comfy_mixed`)。

## 验证(真模型,关键)

- PR-anima-7 必跑:anima-base-v1.0.safetensors + qwen_3_06b_base.safetensors + qwen_image_vae.safetensors → 同 prompt/seed,**对照 ComfyUI 原生工作流**出图(`split_files/example.png` 或 `anima_comparison.json` 的输入复用)。SSIM ≥ 0.95 视作通过(允许 sampling 微差异)。
- LoRA 路径:Turbo LoRA(civitai.com/models/2560840)接入 + 加速验证。

## 工作量估算

| 阶段 | 工时 |
|---|---|
| PR-anima-1(Cosmos predict2 port) | 1.5 天 |
| PR-anima-2(Anima nn.Module) | 0.5 天 |
| PR-anima-3(text encoder) | 0.5 天 |
| PR-anima-4(operations port) | 0.5 天 |
| PR-anima-5(pipeline + bundle config) | 1 天 |
| PR-anima-6(注册表 + arch 自检) | 0.5 天 |
| PR-anima-7(真模型验 + LoRA + 调优) | 1-2 天 |
| **总** | **5-6 工作日纯实施** |

加上 review / debug / 卡 GPU(共享 ComfyUI)等不可控,**真实交付 = 7-10 工作日**。

## 不做 / future

- 其它 Cosmos 衍生模型(CosmosT2IPredict2、Cosmos I2V 等)留独立 spec(虽然 MiniTrainDIT 基类是同一个,可能复用,但每个有自己的 conditioning + sampling)。
- ComfyUI subprocess bridge 方案(选项 C)留远期 spec —— 真要支持「ComfyUI 全栈」时再开。
- anima Turbo LoRA 单独 PR(基于 Flux2 LoRA 路径复用)。

## 决策点(用户)

实施 anima 移植值不值得这 7-10 天?三个判断维度:

1. **直接价值**:你需要在 nous 里跑 anima 吗?还是用 ComfyUI 直接跑就够?
2. **战略价值**:anima 是 Cosmos Predict2 系列首发,移植它解锁后续 NVIDIA Cosmos 系列(可能多模型)。
3. **机会成本**:这 7-10 天可用来:
   - PR-D(offload)/ PR-E(Topbar UX)/ PR-F(TAESD preview)+ 多个其它 image arch 适配。
   - 或者推进其它 nous 功能(LLM / TTS / Agent)。

**建议**:除非 anima 是核心需求,**优先做 PR-D/E/F**(投入回报更高 + 不卡 anima)。anima 留独立 sprint,值得就排上。
