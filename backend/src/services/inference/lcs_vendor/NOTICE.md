# Vendored: comfyui-lcs `core/`

Source: https://github.com/(upstream)/comfyui-lcs — **License: MIT**(见 LICENSE)。

`core/` 算法层(光栅锐化标定 / 色彩子空间 / patchify / PCA)整体 vendor 进 nous,保留上游目录树便于跟上游对照升级
(同 `seedvr2_vendor` 做法,不归 nous lint —— `pyproject.toml` ruff extend-exclude)。

## nous LOCAL MODIFICATIONS(偏离上游)
为脱离 ComfyUI 抽象(nous 无 comfy / 用 diffusers VAE),仅改耦合点,核心数学原样:
- `core/sharpness.py`:删 `import comfy.utils`(ProgressBar 去掉);`calibrate_sharpness(vae,…)` →
  `calibrate_sharpness(encode_fn,…)`(`vae.encode` ComfyUI BHWC API 换成调用方传 diffusers 适配的
  `encode_fn`);video VAE 逐图回退分支去掉。
- `core/sampling.py` / `core/calibration.py`:仍含 `import comfy.*`,**PR-2 不导入**(锐化只用 sharpness/patchify/
  lcs_data);PR-3 接色彩锚定时再 patch。

集成层(非 vendor)= `src/services/inference/lcs_integration.py`:diffusers VAE encode 适配 + 标定缓存 + per-step
锐化闭包(对照上游 `nodes/sharpen.py` `_build_sharpness_fn`,model.latent_format → 显式 vae scale/shift)。
spec `docs/superpowers/specs/2026-06-10-sampling-intervention-hook-lcs-design.md`。
