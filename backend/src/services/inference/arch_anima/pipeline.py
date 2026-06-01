"""AnimaPipeline — anima DiT + qwen3 TE + Qwen-Image VAE + FlowMatchEuler 装配。

接 PR-anima-1~4。spec PR-5 (part 2):把组件装成 diffusers 风格的 callable
pipeline:`pipe(prompt, ...)` → PIL Image。

不强制继承 `diffusers.DiffusionPipeline`(那个有重 hook / loader 框架);
ad-hoc class 跟 nous 现有 image_modular._build_klein_pipe 风格一致。

## 采样

ComfyUI Anima `sampling_settings`:`{"multiplier": 1.0, "shift": 3.0}`。
等价 diffusers `FlowMatchEulerDiscreteScheduler(shift=3.0)`。

## Latent 维度

Qwen-Image VAE:`z_dim=16`,8x spatial 下采样(3 阶段 temporal_compression [F,T,T])。
Anima 是单帧图像 → T=1,latent (B, 16, 1, H/8, W/8)。
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from PIL import Image


class AnimaPipeline:
    """anima DiT + AnimaTextEncoder + Qwen-Image VAE + FlowMatchEulerDiscreteScheduler。

    最简形态:`pipe(prompt)` → PIL Image。负面 prompt + CFG 走 PR-7 真模型阶段
    完善(本 PR 先 verify 装配 + 基本 denoise loop)。
    """

    # ComfyUI Anima sampling_settings 默认。
    DEFAULT_SHIFT = 3.0
    DEFAULT_STEPS = 30
    DEFAULT_CFG = 4.5  # README 推荐 4-5
    DEFAULT_SIZE = 1024

    def __init__(
        self,
        transformer: torch.nn.Module,   # nous Anima(继承 MiniTrainDIT)
        text_encoder,                    # nous AnimaTextEncoder(已 load)
        vae: torch.nn.Module,            # diffusers AutoencoderKLQwenImage
        scheduler,                       # diffusers FlowMatchEulerDiscreteScheduler
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_components(
        cls,
        anima_weights: str,
        qwen_weights: str,
        qwen_tokenizer_dir: str,
        vae_weights: Optional[str] = None,
        t5_tokenizer_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        shift: float = DEFAULT_SHIFT,
    ) -> "AnimaPipeline":
        """单文件路径全装配。

        - anima_weights:anima-base-v1.0.safetensors 单文件
        - qwen_weights:qwen_3_06b_base.safetensors
        - qwen_tokenizer_dir:ComfyUI 的 qwen25_tokenizer/(本地路径)
        - vae_weights:qwen_image_vae.safetensors 单文件(可选,None 时不创建 VAE)
        - t5_tokenizer_dir:可选 t5_tokenizer/,启用 LLMAdapter 桥接路径
        """
        from diffusers import (  # noqa: PLC0415
            AutoencoderKLQwenImage,
            FlowMatchEulerDiscreteScheduler,
        )

        from .load import load_anima_dit_from_single_file  # noqa: PLC0415
        from .text_encoder import AnimaTextEncoder  # noqa: PLC0415

        # 1) Anima DiT(继承 MiniTrainDIT)
        transformer = load_anima_dit_from_single_file(
            anima_weights, device=device, dtype=dtype,
        )

        # 2) AnimaTextEncoder(懒加载)
        te = AnimaTextEncoder(
            qwen_weights_path=qwen_weights,
            qwen_tokenizer_dir=qwen_tokenizer_dir,
            t5_tokenizer_dir=t5_tokenizer_dir,
            device=device,
            dtype=dtype,
        )
        te.load()

        # 3) Qwen-Image VAE(单文件 → diffusers AutoencoderKLQwenImage)
        vae: Optional[torch.nn.Module] = None
        if vae_weights is not None:
            from safetensors.torch import load_file  # noqa: PLC0415

            vae_sd = load_file(vae_weights, device="cpu")
            # 用 HF 上 Qwen/Qwen-Image 的 vae config(后续 PR 可 bundle)
            vae_cfg = AutoencoderKLQwenImage.load_config("Qwen/Qwen-Image", subfolder="vae")
            vae = AutoencoderKLQwenImage.from_config(vae_cfg)
            try:
                vae.load_state_dict(vae_sd, strict=False)
            except RuntimeError as e:
                # round-2026-06-01:防御纵深 —— 用户给 anima 接了非 Qwen-Image VAE(如 flux2-vae)
                # 时 size mismatch。strict=False 也救不了形状不符(2D conv vs Qwen 的 3D)。
                # 翻成人话,别甩裸 PyTorch state_dict 堆栈(用户看 size mismatch [128,3,3,3] 一脸懵)。
                raise RuntimeError(
                    f"Anima 需要 Qwen-Image VAE(qwen_image_vae.safetensors),"
                    f"当前 VAE 文件 '{vae_weights}' 权重形状不符(疑似接了 Flux2/其它架构的 VAE)。"
                    f"请在 Load VAE 选 qwen_image_vae。原始错误:{e}"
                ) from e
            vae = vae.to(device, dtype=dtype).eval()

        # 4) FlowMatchEulerDiscreteScheduler(shift=3.0 对齐 anima sampling_settings)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)

        return cls(
            transformer=transformer, text_encoder=te,
            vae=vae, scheduler=scheduler,
            device=device, dtype=dtype,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = DEFAULT_STEPS,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        seed: Optional[int] = None,
        guidance_scale: float = DEFAULT_CFG,
        step_callback: Optional[Callable[[int], None]] = None,
    ) -> Image.Image:
        """文本 → 图像(单帧)。

        - guidance_scale > 1 + negative_prompt → classifier-free guidance(true-CFG):
          双 forward(cond + uncond),noise_pred = uncond + scale * (cond - uncond)
        - guidance_scale == 1 或 negative_prompt 空 → cond-only(快一半)

        step_callback(done_1based):每个 denoise step 完成后调用(done = 1..num_inference_steps)。
        给 nous adapter 透传逐步进度(对齐 Flux2 的 callback_on_step_end);回调内可 raise
        中止 denoise。standalone 用不传(默认 None,行为不变)。
        """
        if self.vae is None:
            raise RuntimeError("AnimaPipeline.__call__ requires vae; init with vae_weights")

        do_cfg = guidance_scale > 1.0 and len(negative_prompt) > 0

        # 1) Text encode → context(可能两次:正/负)
        cond_enc = self.text_encoder.encode(prompt)
        cond_ctx = cond_enc["context"]
        cond_t5_ids = cond_enc.get("t5xxl_ids")
        cond_t5_w = cond_enc.get("t5xxl_weights")
        if do_cfg:
            uncond_enc = self.text_encoder.encode(negative_prompt)
            uncond_ctx = uncond_enc["context"]
            uncond_t5_ids = uncond_enc.get("t5xxl_ids")
            uncond_t5_w = uncond_enc.get("t5xxl_weights")

        # 2) Init latents(B=1, C=16, T=1, H/8, W/8)
        latent_h, latent_w = height // 8, width // 8
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        latents = torch.randn(
            (1, 16, 1, latent_h, latent_w),
            device=self.device, dtype=self.dtype, generator=gen,
        )

        # 3) Scheduler timesteps(FlowMatchEuler:无 init_noise_sigma / scale_model_input)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 4) Denoise loop(CFG = 双 forward)
        for i, t in enumerate(self.scheduler.timesteps):
            t_in = t.unsqueeze(0).to(self.device, dtype=self.dtype)
            cond_pred = self.transformer(
                latents, t_in, cond_ctx,
                t5xxl_ids=cond_t5_ids, t5xxl_weights=cond_t5_w,
            )
            if do_cfg:
                uncond_pred = self.transformer(
                    latents, t_in, uncond_ctx,
                    t5xxl_ids=uncond_t5_ids, t5xxl_weights=uncond_t5_w,
                )
                noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                noise_pred = cond_pred
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if step_callback is not None:
                step_callback(i + 1)  # done(1-based);total = total_steps

        # 5) VAE decode(latents 是 (1, 16, 1, h, w) — Qwen-Image VAE 接 5D)
        # 反归一化(若 config 有 latents_mean/std,做 inv-normalize)
        vae_cfg = self.vae.config
        if hasattr(vae_cfg, "latents_mean") and vae_cfg.latents_mean is not None:
            latents_mean = torch.tensor(vae_cfg.latents_mean, device=self.device, dtype=self.dtype).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(vae_cfg.latents_std, device=self.device, dtype=self.dtype).view(1, -1, 1, 1, 1) if hasattr(vae_cfg, "latents_std") else 1.0
            latents = latents * latents_std + latents_mean

        image_5d = self.vae.decode(latents).sample  # (B, 3, T, H, W)
        # 单帧:T=1 → squeeze
        image = image_5d[:, :, 0]  # (B, 3, H, W)
        image = (image.clamp(-1, 1) + 1) / 2  # [0, 1]
        image = (image[0].float().cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(image)
