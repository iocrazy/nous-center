# 单文件流水线装配(ComfyUI 式)— 设计

> 状态:**草稿,待 review + push 后写 plan**。
> 接 [[2026-05-22-image-engine-modular-diffusers-design]](引擎=Modular Diffusers)+ 加载分离(#138)
> + fp8 紧凑加载(#139)。依据 [[feedback-long-term-robustness]] [[feedback-verify-real-model]]。

## 目标(用户诉求)

让用户像 ComfyUI 那样**用单文件拼图像流水线**:单文件 transformer(diffusion_models/)+ 单文件
text encoder(text_encoders/)+ 单文件 vae(vae/),不被迫用 Load Checkpoint 整模型。

用户库 `~/models/nous/image/` 就是 ComfyUI 式拆分:
- `diffusers/`:整模型(ERNIE-Image、**Flux2-klein-9B**,带全套 config/scheduler/tokenizer)。
- `diffusion_models/flux|split_files/*.safetensors`:单文件 DiT(含 comfy fp8mixed 等量化)。
- `text_encoders/qwen_3_8b_fp8mixed.safetensors`:单文件 CLIP。
- `vae/flux2-vae.safetensors`:单文件 VAE。

## 现状与症结

- **modular(diffusers)引擎需 HF-layout repo** 的 `model_index.json` + 各组件 `config.json` + tokenizer
  才能装配流水线(scheduler/latent/各组件结构)。三个散单文件无 repo →
  `_modular_repo_from_components` 抛 `comfy 全单文件(无 HF clip/vae)暂不支持`(真机实测)。
- **comfy 单文件 transformer 支持已存在但只到 transformer**:`build_bridged_transformer`
  (image_modular.py)= dequant+转键(quant_loaders)+ 从 repo 的 `transformer/config.json` `from_config`
  + load_state_dict。它**依赖一个 repo**提供 config + clip + vae。verified smoke(`smoke_modular_comfy_fp8.py`)
  当年就是 comfy 单文件 transformer + **HF repo 的 clip/vae**。
- **Arc 8(#138)把 `diffusers/*/{text_encoder,vae}` 从 clip/vae 下拉移除了** → 原"comfy 单文件
  transformer + repo clip/vae"的 UI 入口也没了。所以现在只能 Load Checkpoint(整模型,无法换 comfy transformer)。

## 核心洞察:架构 → 参考整模型

用户**同时有整模型** `diffusers/Flux2-klein-9B`(全套 config)。所以单文件装配不用凭空造 config:
**单文件 unet/clip/vae + 架构=flux2 → 后端用 `diffusers/<arch 参考整模型>` 的 config/scheduler/tokenizer
搭组件骨架,把单文件权重(含 fp8 反量化)灌进去。** 把现在只对 transformer 的桥接扩到 clip+vae。

## 设计

1. **架构→参考整模型映射**:`flux2 → diffusers/Flux2-klein-9B`、`ernie → diffusers/ERNIE-Image`。
   实现:小配置表 + 自动发现(扫 `diffusers/` 找 model_index.json 的 `_class_name` 匹配架构)。
2. **桥接扩到 clip + vae**(仿 `build_bridged_transformer`):
   - `build_bridged_text_encoder(clip_spec, repo, device)`:repo 的 `text_encoder/config.json` 建
     Qwen3(或对应)+ tokenizer 取自 repo;dequant/load 单文件权重(fp8mixed→bf16)。
   - `build_bridged_vae(vae_spec, repo, device)`:repo 的 `vae/config.json` 建 AutoencoderKLFlux2 + 单文件权重。
3. **modular pipe 三组件 override**:`_ensure_pipe` 现只 `update_components(transformer=override)`;
   扩成 transformer + text_encoder + vae 三个 override(单文件路径)。repo 仍提供 scheduler/tokenizer/pipeline 类。
4. **repo 解析**:`_modular_repo_from_components` 找不到 HF 组件时 → fall back 到架构参考整模型。
5. **UI**:Load Diffusion Model / Load CLIP / Load VAE 单文件下拉照旧(各自文件夹);架构选项驱动参考整模型。
6. **fp8**:单文件若 comfy 量化(fp8mixed)→ 走 quant_loaders 反量化(现成);torchao weight-only(#139)是另一条
   (bf16 单文件运行时量化)。两者正交。
7. **fit-check / 真模型 smoke**:flux 单文件 + qwen fp8 clip + flux2-vae → 出正确图。

## PR-1 spike 结论(已验,2026-05-25,`tests/manual/spike_single_file_assembly.py`)

真模型(Pro6000)用用户库单文件验,精确定范围:
- **transformer 单文件 ✓**:现有 `build_bridged_transformer`(dequant+flux2 转键 + repo config)40s 建好,无误。
- **vae 单文件**:`vae/flux2-vae.safetensors` = **plain 标准 AutoencoderKLFlux2 键**(无量化无转键)→ 直接
  load_config + load_state_dict,trivial(spike 因 text_encoder 先挂未跑到,plain 风险极低)。
- **text_encoder 单文件 = 真障碍**:`qwen_3_8b_fp8mixed.safetensors` 是 **comfy 逐张量混合量化**(`.comfy_quant`
  JSON 标 format):**float8_e4m3fn ×141 + nvfp4 ×85 + 纯 bf16 ×172**。keys 是标准 Qwen3(无需转键),但
  现有 `load_fp8mixed` 只解 fp8、遇 nvfp4 张量原样传 uint8[*,2048] → load_state_dict size mismatch
  (file [1024,2048] uint8 vs model [1024,4096])。
  - **需逐张量 comfy 混合 dequantizer**:读每个权重的 `.comfy_quant.format` → 分发
    fp8(weight×weight_scale)/ nvfp4(int4 unpack + block scale [*,256])/ plain。**fp8+nvfp4 的 dequant
    数学 quant_loaders 已有**(load_fp8mixed / load_nvfp4mixed),但:① 现有 loader 假设整文件统一格式,
    需重构成 `.comfy_quant` 驱动的逐张量分发;② nvfp4 dequant 有已知 bug(见 image-quant-compact PR-3)需先修。
  - ComfyUI 自身 dequant 靠编译扩展 comfy_kitchen;我们走"dequant 到 bf16 再 bf16 跑",纯 torch 即可(不需 comfy_kitchen)。

### NVFP4 dequant 已解(真模型出干净狐狸图,spike 验证)

comfy NVFP4 精确格式(comfy/float.py:32-36 反推):
- 4-bit 码 `(sign<<3)|(exp<<1)|mantissa`;**bit3=符号**,`code&7` → E2M1 幅值 LUT `[0,0.5,1,1.5,2,3,4,6]`。
- **打包 nibble 顺序**:`packed=(even<<4)|odd` → **偶数元素=高 nibble、奇数元素=低 nibble**
  (一开始写反了 → 每对权重对调 → 出图能认但颗粒;修正顺序后干净)。
- scale:`dequant = e2m1 × block_scale(fp8 [out,in/16],repeat_interleave(16)) × global_scale(weight_scale_2 fp32 标量)`。
  (block scale 每 16 一值,对内交换不影响其对齐 → 唯一锅是 nibble 顺序。)
- 隔离验证:repo bf16 text encoder + 单文件 transformer/vae = 干净(证 transformer/vae 路径正确);
  单文件 text encoder nibble 修正后 = 干净 → NVFP4 算法已对。
- fp8(float8_e4m3fn 格式):`weight(fp8)×weight_scale → bf16`。plain:cast。
- text encoder 缺 `lm_head.weight`(tied,comfy 省):Flux2 文本编码不用 → 零初始化兜底。
- vae:plain 标准键,missing=0 完美 load。

## PR 拆分(plan 细化)

- **PR-1 spike**:真模型验 `build_bridged_text_encoder` + `build_bridged_vae`(用 Flux2-klein-9B repo config
  + 用户单文件 qwen fp8 clip / flux2-vae)能装配 + 出正确图;定架构→参考映射方式。
- **PR-2**:实现两个桥接 + 架构参考映射 + pipe 三组件 override + repo fallback。
- **PR-3**:UI/wiring 收尾 + 单元/真模型 smoke + 文档。

## 不做

- 不自研 ComfyUI 式"从权重纯检测架构"(diffusers 走 config;我们用参考整模型补 config,更省)。
- 整模型(Load Checkpoint)路径不变。

## 后续(用户点名,独立 arc)

参考 ComfyUI 补更多节点 —— **SeedVR2 放大**(视频/图像超分;库里 `SEEDVR2/` 已有模型;ComfyUI 节点:
SeedVR2 Load DiT + Load VAE + Video Upscaler)。新模态/新模型集成,本 arc 之后单列。
