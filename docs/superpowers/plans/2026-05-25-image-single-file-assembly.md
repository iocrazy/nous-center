# 单文件流水线装配(ComfyUI 式)Plan

> REQUIRED SUB-SKILL: executing-plans。
> spec:`docs/superpowers/specs/2026-05-25-image-single-file-assembly-design.md`。

**Goal**:单文件 transformer + text encoder + vae 像 ComfyUI 那样拼图;config/scheduler/tokenizer
取自「架构参考整模型」(diffusers/<arch>)。复用现有 comfy 桥接(transformer)并扩到 clip+vae。

**Branch**:每 PR 独立。

---

## PR-1:spike —— clip/vae 桥接出图(真模型,gating)

**File**:`backend/tests/manual/spike_single_file_assembly.py`

- [ ] 用 `diffusers/Flux2-klein-9B` 当参考 repo,**单文件**:transformer=diffusion_models/flux 单文件(bf16 或 comfy)、
  clip=`text_encoders/qwen_3_8b_fp8mixed.safetensors`、vae=`vae/flux2-vae.safetensors`。
- [ ] 原型 `build_bridged_text_encoder`(repo `text_encoder/config.json` 建模型 + tokenizer 取自 repo +
  dequant/load 单文件)、`build_bridged_vae`(repo `vae/config.json` + 单文件)。
- [ ] ModularPipeline.from_pretrained(repo) → load_components → `update_components(transformer=, text_encoder=, vae=)`
  三 override → 出图。验:**出正确狐狸图** + 三组件都来自单文件 + 峰值显存。
- [ ] 结论 → 定架构→参考映射方式 + 三桥接接口。任一组件桥接失败(config/tokenizer/键不符)→ 记录取舍。

## PR-2:实现单文件装配

**Files**:`backend/src/services/inference/image_modular.py`(+ `quant_loaders` 复用)/ `model_manager.py`

- [ ] 「架构→参考整模型」解析(配置表 + 自动发现 diffusers/ 匹配 `_class_name`)。
- [ ] `build_bridged_text_encoder` / `build_bridged_vae`(产品化 spike 原型)。
- [ ] `_ensure_pipe`:单文件路径 → 三组件 override;`_modular_repo_from_components` 找不到 HF 组件时
  fall back 到架构参考整模型。
- [ ] 单元(CI mock):桥接选择 + 架构映射 + override 装配接线。

## PR-3:UI / wiring 收尾 + 真模型 smoke

- [ ] Load Diffusion Model / Load CLIP / Load VAE 单文件路径端到端(架构选项驱动参考整模型)。
- [ ] 真模型 smoke:用户库单文件组合出图;fit-check 估算;后端全套 + 前端 build。
- [ ] PR → CI → merge。文档:CLAUDE.md / spec 标注单文件支持。

---

## 不做 / 后续

- 不自研"从权重检测架构"(用参考整模型补 config)。
- **后续独立 arc(用户点名)**:SeedVR2 放大(视频/图像超分)+ 参考 ComfyUI 补更多节点。
