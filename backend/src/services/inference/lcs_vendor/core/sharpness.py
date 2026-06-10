"""Sharpness subspace calibration via sinusoidal grating stimuli.

Replaces the previous Gaussian blur approach with narrowband frequency
gratings, which achieve higher linearity (R²=0.94 vs 0.88) because each
stimulus contains a single spatial frequency — a purer probe of the VAE's
frequency encoding axis.

The two methods discover the same 1D subspace (|cos|=0.986, 9.7° apart),
but grating stimuli yield a cleaner PC1 direction.

--- nous-center LOCAL MODIFICATIONS(vs upstream comfyui-lcs,MIT)---
为脱离 ComfyUI 抽象(nous 无 comfy / 用 diffusers VAE),改 2 处:
  1. 删 `import comfy.utils`;`comfy.utils.ProgressBar` → 无(标定一次性,不需进度条)。
  2. `calibrate_sharpness(vae, ...)` → `calibrate_sharpness(encode_fn, ...)`:`vae.encode`(ComfyUI
     BHWC/[0,1] API)换成调用方传入的 `encode_fn(batch_bchw_01) -> raw_latent`(diffusers VAE 适配在
     调用方,见 lcs_integration.py)。其余 grating 生成 / PCA / sign 数学**原样保留**(vendor 的价值)。
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

import torch

from .patchify import patchify
from .lcs_data import LCSData


@dataclass
class SharpnessData:
    """Calibration data for the sharpness subspace.

    Produced by PCA on FLUX VAE-encoded sinusoidal gratings at varying
    spatial frequencies.  PC1 captures ~94% of variance with R²=0.94
    linearity vs log₂(frequency).
    """

    basis: torch.Tensor   # [64, K] PCA basis (columns), K typically 1-2
    mean: torch.Tensor    # [64] PCA mean (in color-removed space if lcs_data was used)
    sign: float           # +1 or -1: ensures positive strength = sharper
    lcs_basis: Optional[torch.Tensor] = None  # [64, 3] LCS basis used during calibration (for re-orthogonalization)

    def to(self, device, dtype=None):
        """Move all tensors to device/dtype."""
        kw = {"device": device}
        if dtype is not None:
            kw["dtype"] = dtype
        return SharpnessData(
            basis=self.basis.to(**kw),
            mean=self.mean.to(**kw),
            sign=self.sign,
            lcs_basis=self.lcs_basis.to(**kw) if self.lcs_basis is not None else None,
        )


def _generate_grating_batch(
    indices: List[int],
    angles: torch.Tensor,
    phases: torch.Tensor,
    frequencies: Tuple[float, ...],
    coord_x: torch.Tensor,
    coord_y: torch.Tensor,
) -> torch.Tensor:
    """Generate a batch of sinusoidal grating stimuli by flat index.

    Each flat index maps to (orientation, frequency) via divmod.
    Returns [len(indices), 3, H, W] tensor.
    """
    num_freqs = len(frequencies)
    batch = []
    for idx in indices:
        ori = idx // num_freqs
        freq = frequencies[idx % num_freqs]
        angle = angles[ori].item()
        phase = phases[ori].item()
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        coord = coord_x * cos_a + coord_y * sin_a
        grating = 0.5 + 0.3 * torch.sin(2 * math.pi * freq * coord + phase)
        batch.append(grating.unsqueeze(0).expand(3, -1, -1))
    return torch.stack(batch, dim=0)


def calibrate_sharpness(encode_fn, num_samples: int = 64, image_size: int = 512,
                        frequencies: Tuple[float, ...] = (1, 2, 4, 8, 16, 32, 64),
                        batch_size: int = 8,
                        lcs_data: LCSData = None,
                        # Legacy parameter — accepted but ignored
                        blur_levels: Optional[Tuple[float, ...]] = None,
                        ) -> SharpnessData:
    """Compute sharpness subspace data (PCA basis, mean, sign) from FLUX VAE.

    Generates sinusoidal gratings at varying spatial frequencies (one pure
    frequency per stimulus), VAE-encodes them, and runs PCA to find the
    sharpness/frequency direction in 64D patch space.

    Args:
        vae: ComfyUI VAE object
        num_samples: Number of orientations (each combined with all frequencies)
        image_size: Size of generated images
        frequencies: Spatial frequencies in cycles/image
        batch_size: Batch size for VAE encoding
        lcs_data: Optional LCS data for removing color component during calibration.
                  When provided, the sharpness PC1 will be orthogonal to the color subspace,
                  preventing color shifts during intervention.

    Returns: SharpnessData
    """
    if blur_levels is not None:
        warnings.warn(
            "blur_levels is deprecated and ignored; calibration now uses sinusoidal gratings",
            DeprecationWarning, stacklevel=2,
        )

    n_freqs = len(frequencies)
    total_images = num_samples * n_freqs

    print(f"\n[LCS Sharpness Calibration] Starting: {num_samples} orientations × {n_freqs} frequencies = {total_images} stimuli")
    print(f"[LCS Sharpness Calibration] Frequencies: {list(frequencies)} cycles/image")

    # Pre-compute shared state for grating generation
    gen = torch.Generator().manual_seed(42)
    angles = torch.rand(num_samples, generator=gen) * math.pi  # [0, π)
    phases = torch.rand(num_samples, generator=gen) * 2 * math.pi  # [0, 2π)
    y_coords = torch.linspace(-0.5, 0.5, image_size).unsqueeze(1)
    x_coords = torch.linspace(-0.5, 0.5, image_size).unsqueeze(0)
    coord_y = y_coords.expand(image_size, image_size)
    coord_x = x_coords.expand(image_size, image_size)

    # Build frequency labels for all stimuli (flat index → frequency)
    freq_labels = [frequencies[idx % n_freqs] for idx in range(total_images)]
    freq_labels_t = torch.tensor(freq_labels, dtype=torch.float32)
    log_freq = torch.log2(freq_labels_t.clamp(min=0.5))

    # Generate stimuli lazily per batch and VAE encode
    # nous LOCAL MOD:vae.encode(BHWC) → encode_fn(BCHW [0,1]) -> raw_latent(diffusers 适配在调用方);
    # 删 ComfyUI ProgressBar(标定一次性 + 缓存)。video VAE 逐图回退分支去掉(nous 图像 VAE 批量 OK)。
    vectors = []

    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        indices = list(range(batch_start, batch_end))
        batch = _generate_grating_batch(indices, angles, phases, frequencies, coord_x, coord_y)

        latent = encode_fn(batch)  # batch: [B,3,H,W] in [0,1] → raw latent [B,C,h,w]
        patches, _, _, _ = patchify(latent)
        avg = patches.mean(dim=1).cpu()
        vectors.extend(avg.unbind(0))

    # Stack all vectors: [N, 64]
    X = torch.stack(vectors, dim=0).float()
    print(f"[LCS Sharpness Calibration] Collected {X.shape[0]} vectors of dimension {X.shape[1]}")

    # Remove LCS color component FIRST, in the raw space where LCS was calibrated.
    # This must happen before per-vector DC removal, because the LCS basis has
    # significant DC components (PC1 ≈ brightness). Doing DC removal first would
    # shift vectors into a different space where B^T(x - mu) is incorrect.
    if lcs_data is not None:
        print("[LCS Sharpness Calibration] Removing LCS color component...")
        lcs_mean = lcs_data.mean.to(X.device, X.dtype)
        lcs_basis = lcs_data.basis.to(X.device, X.dtype)
        # Project out color: X' = X - B B^T (X - mu)
        centered = X - lcs_mean
        lcs_coords = centered @ lcs_basis  # [N, 3]
        X = X - lcs_coords @ lcs_basis.T
        print("[LCS Sharpness Calibration] Color component removed")

    # Remove per-vector DC AFTER color removal.
    # VAE encoding shifts the latent mean depending on stimulus content.
    # Per-vector zero-mean forces PCA to find patterns in the relative channel
    # structure, not in the absolute level.
    X = X - X.mean(dim=1, keepdim=True)

    # Step 3: PCA
    print("[LCS Sharpness Calibration] Computing PCA...")
    mean = X.mean(dim=0)  # [64]
    X_centered = X - mean
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    # Top 2 components
    basis = Vh[:2].T  # [64, 2]

    # Variance explained
    total_var = (S ** 2).sum()
    explained = (S[:2] ** 2) / total_var
    print(f"[LCS Sharpness Calibration] PC1: {explained[0]:.1%}, PC2: {explained[1]:.1%} ({(explained[0]+explained[1]):.1%} total)")

    # Step 4: Determine sign convention
    # Project all vectors onto PC1
    pc1_scores = X_centered @ basis[:, 0]  # [N]

    # Correlate PC1 score with log₂(frequency)
    # Higher frequency = sharper → if positive correlation, sign = +1
    correlation = torch.corrcoef(torch.stack([pc1_scores, log_freq]))[0, 1]
    sign = 1.0 if correlation > 0 else -1.0
    print(f"[LCS Sharpness Calibration] PC1-frequency correlation: {correlation:.3f} → sign = {sign:+.0f}")
    print(f"[LCS Sharpness Calibration] Complete! Basis shape: {basis.shape}")

    return SharpnessData(
        basis=basis,
        mean=mean,
        sign=sign,
        lcs_basis=lcs_data.basis.clone() if lcs_data is not None else None,
    )
