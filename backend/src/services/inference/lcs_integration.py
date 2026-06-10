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

    def _fn(step_idx: int, total: int, sigma: float, denoised: Any, sigmas: Any = None) -> Any:
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


def get_lcs_data(vae: Any, device: str) -> Any:
    """惰性色彩标定(512 HSV 样本过 VAE → PCA 色彩子空间)+ 缓存。返回 vendor LCSData(basis[64,3]/mean/
    anchor_lcs/anchor_angles)。同 sharpness 缓存机制,key=VAE 指纹。"""
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    from src.services.inference.lcs_vendor.core.calibration import calibrate  # noqa: PLC0415
    from src.services.inference.lcs_vendor.core.lcs_data import LCSData  # noqa: PLC0415

    fp = _vae_fingerprint(vae)
    cache = _cache_dir() / f"lcsdata_{fp}.safetensors"
    if cache.exists():
        sd = load_file(str(cache))
        logger.info("LCS color: 命中缓存 %s", cache.name)
        return LCSData(basis=sd["basis"], mean=sd["mean"], anchor_lcs=sd["anchor_lcs"],
                       anchor_angles=sd["anchor_angles"])
    logger.info("LCS color: 标定中(512 HSV 样本 PCA,首次约几十秒)…")
    data = calibrate(_diffusers_encode_fn(vae, device))
    save_file({"basis": data.basis.cpu().contiguous(), "mean": data.mean.cpu().contiguous(),
               "anchor_lcs": data.anchor_lcs.cpu().contiguous(),
               "anchor_angles": data.anchor_angles.cpu().contiguous()}, str(cache))
    logger.info("LCS color: 标定完成 → 缓存 %s", cache.name)
    return data


def build_color_anchor_fn(vae: Any, device: str, *, mode: str = "self_anchor",
                          intensity: float = 0.8) -> Any:
    """产 per-step 色彩锚定干预闭包(忠实移植 vendor nodes/anchor.py `_build_adaptive_anchor_fn` 的
    self_anchor / smooth 模式;reference 模式需参考图,留后续)。契约 fn(step_idx,total,sigma,denoised,sigmas)。
    每步:denoised→raw→patchify→投影到色彩子空间→按 sigma 归一→检测色漂/邻域异常→纠正→反归一→denoised。
    mode=self_anchor:自适应检测色漂(EMA 邻域关系 + 异常)并拉回;smooth:双边滤波平滑色彩。"""
    import torch  # noqa: PLC0415

    from src.services.inference.lcs_vendor.core.adaptive import (  # noqa: PLC0415
        compute_step_phases,
        compute_strength_envelope,
    )
    from src.services.inference.lcs_vendor.core.bilateral import (  # noqa: PLC0415
        bilateral_filter_lcs,
        estimate_bilateral_params,
    )
    from src.services.inference.lcs_vendor.core.patchify import patchify, unpatchify  # noqa: PLC0415
    from src.services.inference.lcs_vendor.core.relationships import (  # noqa: PLC0415
        compute_local_relationships,
        detect_anomalies_adaptive,
        infer_color_from_neighbors,
    )
    from src.services.inference.lcs_vendor.core.timestep import (  # noqa: PLC0415
        denormalize_from_t50,
        get_alpha_beta,
        get_alpha_beta_t50,
        normalize_to_t50,
    )

    lcs_data = get_lcs_data(vae, device)
    scale, shift = _vae_scale_shift(vae)
    state: dict = {"phases": None, "envelope": None, "correction_index": 0, "r_ema": None,
                   "prev_c_mean": None}

    def _fn(step_idx: int, total: int, sigma: float, denoised: Any, sigmas: Any = None) -> Any:
        if state["phases"] is None:
            sched = sigmas if sigmas is not None else [sigma]
            state["phases"] = compute_step_phases(sched, mode)
            n_correct = sum(1 for p in state["phases"] if p == "correct")
            state["envelope"] = compute_strength_envelope(n_correct)
            state["correction_index"] = 0
        if step_idx >= len(state["phases"]):
            return denoised
        phase = state["phases"][step_idx]
        if phase == "skip":
            return denoised

        dev, dt = denoised.device, denoised.dtype
        ld = lcs_data.to(dev, dt)
        B_mat, mu = ld.basis, ld.mean
        raw = denoised / scale + shift
        patches, h_len, w_len, extra = patchify(raw)
        if patches is None:
            return denoised
        projection = (patches - mu) @ B_mat
        reconstruction = projection @ B_mat.T + mu
        residual = patches - reconstruction

        sigma_val = float(sigma)
        alpha_t, beta_t = get_alpha_beta(sigma_val, device=dev)
        alpha_t, beta_t = alpha_t.to(dt), beta_t.to(dt)
        alpha_50, beta_50 = get_alpha_beta_t50(device=dev)
        alpha_50, beta_50 = alpha_50.to(dt), beta_50.to(dt)
        c_norm = normalize_to_t50(projection, alpha_t, beta_t, alpha_50, beta_50)

        if phase == "observe":
            r_current = compute_local_relationships(c_norm, h_len, w_len)
            if state["r_ema"] is None:
                state["r_ema"] = r_current.detach().clone()
            else:
                state["r_ema"] = 0.8 * state["r_ema"] + 0.2 * r_current.detach()
            state["prev_c_mean"] = c_norm.detach().mean(dim=1, keepdim=True)
            return denoised

        ci = state["correction_index"]
        env = state["envelope"]
        step_strength = intensity * float(env[ci]) if ci < len(env) else intensity
        state["correction_index"] = ci + 1
        if mode == "self_anchor" and state["prev_c_mean"] is not None:
            delta = (c_norm.detach().mean(dim=1, keepdim=True) - state["prev_c_mean"]).abs().mean().item()
            step_strength *= min(delta / 0.1, 1.0)

        if mode == "smooth":
            sig_s, sig_c = estimate_bilateral_params(c_norm, h_len, w_len)
            c_filtered = bilateral_filter_lcs(c_norm, h_len, w_len, sig_s, sig_c)
            new_c_norm = c_norm + step_strength * (c_filtered - c_norm)
        else:  # self_anchor
            r_current = compute_local_relationships(c_norm, h_len, w_len)
            if state["r_ema"] is None:
                state["r_ema"] = r_current.detach().clone()
            anomaly_mag = detect_anomalies_adaptive(r_current, state["r_ema"])
            c_corrected = infer_color_from_neighbors(c_norm, anomaly_mag, h_len, w_len)
            new_c_norm = c_norm + step_strength * (c_corrected - c_norm)
            state["r_ema"] = 0.95 * state["r_ema"] + 0.05 * r_current.detach()
            state["prev_c_mean"] = c_norm.detach().mean(dim=1, keepdim=True)

        new_projection = denormalize_from_t50(new_c_norm, alpha_t, beta_t, alpha_50, beta_50)
        patches_new = new_projection @ B_mat.T + mu + residual
        raw_new = unpatchify(patches_new, h_len, w_len, extra)
        return (raw_new - shift) * scale
    _ = torch  # noqa: F841
    return _fn
