"""LCS 锐化干预集成层(spec 2026-06-10-sampling-intervention-hook,PR-2)。

把 vendor 的 comfyui-lcs core 算法(lcs_vendor/core,MIT)接到 nous 的 diffusers VAE + 采样挂钩:
- `_diffusers_encode_fn`:把 ComfyUI VAE.encode(BHWC/[0,1])适配成 diffusers VAE 的 raw-latent 编码
  (标定用),encode_fn(batch [B,3,H,W] in [0,1]) -> raw latent [B,C,h,w]。
- `get_sharpness_data`:惰性标定(正弦光栅 PCA,vendor calibrate_sharpness)+ **按 VAE 指纹缓存 safetensors**
  (标定 ~448 次 VAE encode,贵 → 缓存一次)。
- `build_sharpness_fn`:产 per-step 干预闭包 `fn(step_idx, total, sigma, denoised) -> denoised`,在 denoised(x0)
  上沿锐化 PC1 方向推 edit_vec(对照 vendor nodes/sharpen.py `_build_sharpness_fn`,改 comfy model.latent_format
  → 显式 vae scale/shift)。color 正交化(lcs_basis)是 PR-3,PR-2 只 DC 去除。
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _vae_scale_shift(vae: Any) -> tuple[float, float]:
    """diffusers VAE 的 scaling_factor / shift_factor(transformer 空间 ↔ raw VAE 空间)。
    denoised(transformer/process_in 空间)→ raw = denoised/scale + shift;raw → denoised =(raw-shift)*scale。
    Flux1/Z-Image AutoencoderKL:scale=0.3611,shift=0.1159。shift 缺省 0。"""
    cfg = getattr(vae, "config", None)
    scale = float(getattr(cfg, "scaling_factor", 1.0) or 1.0)
    shift = float(getattr(cfg, "shift_factor", 0.0) or 0.0)
    return scale, shift


def _vae_fingerprint(vae: Any) -> str:
    """VAE 指纹(标定缓存键):config 关键项 + 首参数张量 hash。同 VAE → 同标定。"""
    cfg = getattr(vae, "config", {})
    items = sorted((k, str(v)) for k, v in dict(cfg).items()
                   if k in ("scaling_factor", "shift_factor", "latent_channels", "block_out_channels"))
    h = hashlib.sha256(repr(items).encode())
    try:
        import torch  # noqa: PLC0415
        p = next(vae.parameters())
        h.update(p.detach().float().flatten()[:64].cpu().numpy().tobytes())
        _ = torch  # noqa: F841
    except Exception:  # noqa: BLE001 — best-effort
        pass
    return h.hexdigest()[:16]


def _cache_dir() -> Path:
    from src.config import get_settings  # noqa: PLC0415
    d = Path(get_settings().LOCAL_MODELS_PATH) / "image" / "lcs_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _diffusers_encode_fn(vae: Any, device: str) -> Any:
    """返回 encode_fn(batch [B,3,H,W] in [0,1]) -> raw latent。diffusers VAE:[0,1]→[-1,1] → encode →
    latent_dist.mean(raw,**不乘 scaling_factor** —— 与 nous 手写循环里 denoised 经 scale/shift 转 raw 对齐)。"""
    import torch  # noqa: PLC0415

    def encode_fn(batch: Any) -> Any:
        x = batch.to(device=device, dtype=next(vae.parameters()).dtype)
        x = x * 2.0 - 1.0  # [0,1] → [-1,1](diffusers VAE 输入约定)
        with torch.no_grad():
            posterior = vae.encode(x)
            latent = posterior.latent_dist.mean if hasattr(posterior, "latent_dist") else posterior[0]
        return latent.float().cpu()

    return encode_fn


def get_sharpness_data(vae: Any, device: str) -> Any:
    """惰性标定 + 缓存。返回 vendor SharpnessData(basis/mean/sign/lcs_basis)。"""
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    from src.services.inference.lcs_vendor.core.sharpness import (  # noqa: PLC0415
        SharpnessData,
        calibrate_sharpness,
    )

    fp = _vae_fingerprint(vae)
    cache = _cache_dir() / f"sharpness_{fp}.safetensors"
    if cache.exists():
        sd = load_file(str(cache))
        logger.info("LCS sharpness: 命中缓存 %s", cache.name)
        return SharpnessData(basis=sd["basis"], mean=sd["mean"], sign=float(sd["sign"].item()),
                             lcs_basis=sd.get("lcs_basis"))
    logger.info("LCS sharpness: 标定中(光栅 PCA,~448 VAE encode,首次约几十秒)…")
    data = calibrate_sharpness(_diffusers_encode_fn(vae, device))
    tensors = {"basis": data.basis.cpu().contiguous(), "mean": data.mean.cpu().contiguous(),
               "sign": torch.tensor(float(data.sign))}
    if data.lcs_basis is not None:
        tensors["lcs_basis"] = data.lcs_basis.cpu().contiguous()
    save_file(tensors, str(cache))
    logger.info("LCS sharpness: 标定完成 → 缓存 %s", cache.name)
    return data


def build_sharpness_fn(vae: Any, device: str, *, strength: float, start_step: int,
                       end_step: int) -> Any:
    """产 per-step 锐化干预闭包(对照 vendor nodes/sharpen.py `_build_sharpness_fn`)。
    契约 fn(step_idx, total, sigma, denoised) -> denoised。strength>0 更锐 / <0 更糊。"""
    import torch  # noqa: PLC0415

    from src.services.inference.lcs_vendor.core.patchify import patchify, unpatchify  # noqa: PLC0415

    data = get_sharpness_data(vae, device)
    scale, shift = _vae_scale_shift(vae)

    # edit_vec = strength*sign*pc1_dir(DC 去除;color 正交化 = PR-3)。CPU [64],用时移到 denoised device/dtype。
    pc1_dir = data.basis[:, 0].clone()
    pc1_dir = pc1_dir - pc1_dir.mean()
    edit_vec_cpu = (float(strength) * float(data.sign)) * pc1_dir  # [64]

    def _fn(step_idx: int, total: int, sigma: float, denoised: Any) -> Any:
        if strength == 0.0 or step_idx < start_step or step_idx > end_step:
            return denoised
        raw = denoised / scale + shift                      # denoised_to_raw(显式 scale/shift)
        patches, h_len, w_len, extra = patchify(raw)
        if patches is None:
            return denoised
        ev = edit_vec_cpu.to(device=patches.device, dtype=patches.dtype)
        patches = patches + ev
        raw_new = unpatchify(patches, h_len, w_len, extra)
        return (raw_new - shift) * scale                    # raw_to_denoised
    _ = torch  # noqa: F841
    return _fn
