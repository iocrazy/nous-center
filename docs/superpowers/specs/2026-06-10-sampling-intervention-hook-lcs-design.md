# 采样期 latent 干预挂钩 + LCS 复刻(设计)

**Status**: Draft
**Author**: heygo(设计 by Claude)
**Date**: 2026-06-10

## 0. 动机

用户写真工作流距 1:1 的最后一块引擎差距:`comfyui-lcs` 的 **LCSSharpnessIntervene / LCSColorAnchor**
(肤质锐化 / 保色)。逐节点比对(见 [[project_zimage_portrait_replication]])坐实:**这四个 LCS 节点不是
后处理(image→image),是 FLUX 采样期 post-CFG hook** —— 在去噪循环每步、沿预先标定的「锐化/保色方向」
推 `denoised`(x0 预测),依赖 sigma schedule;生成完再用无效。nous 现无「采样每步改 latent」的挂钩点。

本 spec 补这个**采样器基建**(通用 per-step latent 干预挂钩)+ vendor LCS 算法层复刻其干预。

## 1. 已钉的事实(读真代码,2026-06-10 调研)

### 1.1 nous 两个可用挂钩点
- **Z-Image 手写去噪循环**(`image_modular.py` `_run_zimage_segmented`,行 822-867):每步 `scheduler.step` /
  euler_ancestral 算完得 `latents`,**可在 step 后(行 857)或更准确地在 `denoised` 处插回调**。已有
  `pt.step(...)` 每步发进度(行 865),latent 已是可改 tensor。
- **Flux2 / 标准 diffusers 路径**(`image_modular.py` infer,`_step_cb` 行 1182-1200):**已用 diffusers
  `callback_on_step_end`**,`_step_cb(pipe, i, t, cb_kwargs)` 已从 `cb_kwargs["latents"]` 读 latent 做
  preview,**`return cb_kwargs` 把改后的 latents 交还 diffusers 续采**(diffusers 标准契约,nous 钉的
  commit `c8eba43` 的 Flux2KleinPipeline 支持)。改 latents 只需 `cb_kwargs["latents"] = modified`。

### 1.2 LCS 包(comfyui-lcs)
- **许可证 MIT**(`pyproject.toml` / README)—— 可 vendor(同 SeedVR2 方式,保留上游目录树)。
- `core/` = **1734 LOC 纯数学**,95% 零 comfy 耦合:`color_space.py`(380,双圆锥 LCS↔HSL)/`sharpness.py`
  (213,正弦光栅 PCA 标定锐化 PC1 方向,捕 ~94% 方差)/`calibration.py`(214,512 色样本过 VAE→PCA 色彩子空间)
  /`patchify.py`(93,latent patch 化)/`timestep.py`/`defaults.py`/`bilateral.py`/`relationships.py`/`adaptive.py`/`lcs_data.py`。
- **唯一耦合点 `core/sampling.py`(105)**:`denoised_to_raw`/`raw_to_denoised` 用 `model.latent_format.
  process_out/in`(VAE scale/shift)—— vendor 时改成接 `vae_scale,vae_shift` 参数(Flux1 16ch:scale=0.3611,
  shift=0.1159;各 VAE 不同,从 diffusers VAE config 取)。次要:`comfy.utils.ProgressBar`(删)/
  `comfy.model_management.intermediate_device`(改传 device 参数)/VAE encode(标定期由调用方传 vae)。
- 干预本质(`nodes/sharpen.py` `_build_sharpness_fn`):每步 `denoised → raw → patchify → patches += edit_vec
  → unpatchify → raw → denoised`,`edit_vec = strength*sign*pc1_dir`,且 `pc1_dir` 减去色彩子空间投影
  (`pc1_dir - B@(B.T@pc1_dir)`)→ **锐化与颜色正交,不串色**。只在 `start_step..end_step` 窗内生效。

### 1.3 关键语义:hook 作用在 `denoised`(x0)不是 latent 状态
ComfyUI `post_cfg_function` 拿的是 **CFG 合并后的 `denoised`(x0 估计)**,改完喂回 sampler step。nous 复刻
要在**同一语义点**挂:Z-Image 手写循环里 `denoised = latents - sigma*noise_pred`(euler_ancestral 分支已这么
算),应在 `denoised` 上干预再转回。**不是**简单改 step 后的 latents(那是不同语义,会错)。

## 2. 设计:统一干预契约 + 不落 tensor 进 msgpack

### 2.1 hook 契约(同时适配手写循环 + diffusers callback)
```python
LatentIntervene = Callable[[int, int, float, Tensor], Tensor]  # (step_idx, total_steps, sigma, denoised) -> denoised
```
引擎持 `list[LatentIntervene]`(可链多个干预,对齐 ComfyUI 链式 post_cfg)。每步在 **denoised(x0)语义点**
依次调用。Z-Image 循环 + Flux2 `_step_cb` 各自把 denoised 取出/算出 → 过 chain → 写回。

### 2.2 透传:描述符 + 标定数据落盘引用(不落 tensor 进 msgpack — 同 latent_ref 铁律)
- `ImageRequest` 加 `interventions: list[dict] | None`。每个描述符 = `{"_type":"lcs_sharpness"|"lcs_color_anchor",
  "strength":..,"start_step":..,"end_step":..,"calib_ref":{"path": <safetensors>}}`。
- 标定数据(basis/mean/pc1_dir,tensor)**落盘 safetensors**,描述符只带 `path`(同 init_latent_ref 落盘模式)。
- 引擎 `_build_interventions(req)` 读描述符 → 载标定 safetensors → 用 vendor 的 `_build_*_fn` 构造 `LatentIntervene`。

## 3. PR 拆分(独立分支 + CI 绿 + 引擎改真机 smoke)

### PR-1:per-step latent 干预挂钩基建(无 LCS,先打地基)
- 定 `LatentIntervene` 契约 + `ImageRequest.interventions` 字段;`runner_process._build_request` 透传;
  节点 `flux2_ksampler` 加可选 `interventions` 输入端口(描述符 list)。
- 两个挂钩点接入:① Z-Image `_run_zimage_segmented` 在 denoised(x0)语义点过 chain ② Flux2 `_step_cb`
  在 `cb_kwargs["latents"]`(注:Flux2 callback 给的是 latents 非 denoised,需按 scheduler 当前 sigma
  换算 denoised 或文档化语义差)过 chain 写回。
- **验证**:① 空 chain → golden 零回归(smoke_zimage_split SSIM 1.0 + smoke_image_ab Flux2 byte-identical)
  ② 平凡干预(denoised += k·常向量)→ 出图确定性改变 + 复现一致(证挂钩真生效、可改 latent)。
- **不含任何 LCS**;纯基建,blast radius 小。

### PR-2:vendor LCS core/ + 锐化标定/干预节点
- vendor `comfyui-lcs/core/` → `src/services/inference/lcs_vendor/`(保留目录树,MIT NOTICE)。改造:
  `sampling.py` denoised_to_raw/raw_to_denoised 接 vae_scale/shift;删 ProgressBar;device 传参;VAE encode 传入。
- 节点 `lcs_sharpness_calibrate`(过 VAE 跑光栅 PCA → 标定 safetensors,按 VAE 指纹缓存)+ `lcs_sharpness_intervene`
  (产 `{"_type":"lcs_sharpness",...,"calib_ref":path}` 描述符,接 ksampler interventions 端口)。
- **验证**:真机 e2e Z-Image/Flux2 挂锐化干预 → 出图更锐 + 颜色保持(vs strength=0 同 seed,锐度↑色差≈0);
  golden 零回归(strength=0 / 不挂 = byte-identical)。

### PR-3:色彩锚定节点(LCSColorAnchor)+ LCSLoadData
- vendor 的 `color_space.py`/`calibration.py`/`relationships.py`/`adaptive.py`/`bilateral.py` 接上;
  `lcs_load_data`(色彩基准标定)+ `lcs_color_anchor`(色漂纠正,auto/smooth/reference/self_anchor 4 模式)。
- **验证**:真机 e2e 色漂纠正 + 与锐化链式叠加;golden 零回归。

## 4. 不做 / 风险
- **不** bump diffusers。
- Flux2 callback 给 latents 非 denoised:PR-1 需明确语义(按 sigma 换算或文档化),错了干预方向反。
- 标定开销(VAE 跑光栅/色样本)首次慢 → 按 VAE 指纹缓存 safetensors(同 LCS 原设计)。
- vendor 的 `core/` 不归 nous lint(`pyproject.toml` extend-exclude,同 seedvr2_vendor)。
- LTXAV 视频解包路径(sampling.py)nous 用不上 → no-op fallback。

## 5. 范围
执行顺序:PR-1(挂钩基建)→ PR-2(锐化)→ PR-3(色彩锚定)。每 PR 独立分支 + CI 绿 + 引擎改真机 smoke
(smoke_zimage_split + smoke_image_ab golden 零回归是每个引擎 PR 的硬闸门)。
参见 [[project_zimage_portrait_replication]]、[[project_engine_smoke_verified]]、[[feedback_read_comfyui_source]]。
