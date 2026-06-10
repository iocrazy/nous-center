"""SeedVR2 超分上采样引擎 adapter —— 接 NumZ vendored 推理核心到 nous-center。

SeedVR2 = ByteDance 的 one-step diffusion 超分上采样器(DiT 7B + 专用 video VAE)。
推理核心 vendored 在 `seedvr2_vendor/`(NumZ src,Apache 2.0,不重写),本文件是**桥** ——
把 nous-center 的 InferenceAdapter 接口转成 NumZ 的 prepare_runner + 4 阶段
(encode → DiT one-step upscale → decode → post)。对标 ComfyUI 的 interfaces/ 桥层。

PR-3a:符合 InferenceAdapter ABC —— `__init__(paths, device, **params)` /
`async load(device)` / `async infer(UpscaleRequest)→InferenceResult`,让 ModelManager
能跟管 Flux2/anima 一样管 SeedVR2(LRU/显存/跨进程可见)。内核仍是 PR-2 的同步
`upscale(PIL)→PIL`(忠实复刻 NumZ CLI 的 4 阶段串法),infer 只是 decode 图 +
to_thread 包一层。project_seedvr2_pr3_design。
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any, ClassVar

# **必须在 import 任何 seedvr2_vendor 模块前** patch transformers 5.6-dev 的 flash_attn bug。
from src.services.inference.seedvr2_compat import apply_seedvr2_compat_patches

apply_seedvr2_compat_patches()

from src.services.inference.base import (  # noqa: E402 — 必须在 compat patch 之后
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UpscaleRequest,
    UsageMeter,
)

# 默认模型(CLI 白名单认的 7B fp8;HF 自动下到 model_dir)。
DEFAULT_DIT = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
DEFAULT_VAE = "ema_vae_fp16.safetensors"

# SeedVR2 DiT 白名单(我们暴露的子集 —— NumZ MODEL_REGISTRY 里 category=dit 的 safetensors)。
# **单一真相**:节点 widget(seedvr2_model_select)+ /components/seedvr2-dit 端点都读这,
# 避免 node.yaml 和后端各写一份漂掉。gguf 变体暂不列(GGUF 加载未接,见 project memory)。
# 纯数据(无 torch);不直接 import 钉死 torch 的 vendored model_registry(那个 import 链拉
# model 类 → torch,API 路由/CI 都不该付这代价)。filename 与 NumZ 白名单逐字对齐(已核)。
SEEDVR2_DIT_MODELS: list[dict[str, str]] = [
    {"filename": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
     "label": "7B fp8-mixed", "desc": "7B,fp8+block35 fp16 混精 — 显存友好,推荐(默认)"},
    {"filename": "seedvr2_ema_7b_fp16.safetensors",
     "label": "7B fp16", "desc": "7B 全 fp16 — 最高质量,显存占用大"},
    {"filename": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
     "label": "7B sharp fp8-mixed", "desc": "7B sharp 变体 fp8-mixed — 更锐利"},
    {"filename": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
     "label": "3B fp8", "desc": "3B fp8 — 更快/更省显存,质量略低"},
    {"filename": "seedvr2_ema_3b_fp16.safetensors",
     "label": "3B fp16", "desc": "3B 全 fp16"},
]


def seedvr2_dit_models_with_disk_status(model_dir: str | None = None) -> list[dict]:
    """SEEDVR2_DIT_MODELS + 每个是否已在磁盘(present)+ 大小。给 UI 下拉「混合」展示:
    盘上有的标已就绪,白名单其余标可下载(选了 NumZ 从 HF 自动下)。

    model_dir 缺省 = NAS_MODELS_PATH/image/SEEDVR2(与 runner get_or_load_seedvr2_adapter 一致)。
    """
    import os  # noqa: PLC0415

    if not model_dir:
        from src.config import get_settings  # noqa: PLC0415
        nas = (get_settings().NAS_MODELS_PATH or "").strip()
        model_dir = os.path.join(nas, "image", "SEEDVR2")
    out: list[dict] = []
    for m in SEEDVR2_DIT_MODELS:
        path = os.path.join(model_dir, m["filename"])
        present = os.path.isfile(path)
        size_mb = None
        if present:
            try:
                size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
            except OSError:
                size_mb = None
        out.append({**m, "present": present, "size_mb": size_mb, "is_default": m["filename"] == DEFAULT_DIT})
    return out


def _resolve_dev(val: Any, fallback: str) -> str:
    """归一设备串给 NumZ —— 它直接 `torch.device(str)`,**不认 'auto'**(会抛
    「device type at start of device string: auto」)。节点 widget device 默认 'auto',
    model_manager 解析出具体 cuda:N 传成 self.device 但没写回 config,故在此兜底:
    'auto'/'cuda'/空 → fallback(已解析的 cuda:N);'none'(offload 关)/'cpu'/'cuda:N' 原样。"""
    s = str(val or "").strip().lower()
    if s in ("", "auto", "cuda"):
        return fallback
    return s


def _clamp_seed(seed: int) -> int:
    """归一到 [0, 2**32-1] —— NumZ set_seed→np.random.seed 的要求(超范围 numpy 抛
    「Seed must be between 0 and 2**32 - 1」)。<2**32 不变,超范围(如 randomize 给的 2**53)
    才折叠;同输入同输出 → 复现一致。"""
    return int(seed) % (2**32)


def _decode_image(src: str) -> Any:
    """req.image(base64 data URI 或本地路径)→ PIL.Image(RGB)。"""
    from PIL import Image  # noqa: PLC0415

    if src.startswith("data:"):
        # "data:image/png;base64,...." → 取逗号后的 base64 体
        _, _, b64 = src.partition(",")
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(src).convert("RGB")


class SeedVR2UpscaleBackend(InferenceAdapter):
    """SeedVR2 超分引擎(InferenceAdapter)。输入图 + 目标分辨率 → 超分图。

    paths 接口(由 ModelManager._get_or_load_seedvr2_adapter 装填):
      paths["model_dir"] = SEEDVR2 模型目录(DiT/VAE 所在;缺模型时 HF 自动下到此)
      paths["dit"]       = DiT 文件名(可选,默认 DEFAULT_DIT;NumZ 白名单语义,非全路径)
      paths["vae"]       = VAE 文件名(可选,默认 DEFAULT_VAE)

    内核(load 装 runner;upscale 跑 NumZ 4 阶段)忠实复刻 NumZ CLI 的 process_single_file
    串法(setup_generation_context → prepare_runner → encode/upscale/decode/postprocess
    → ctx['final_video'])。
    """

    modality: ClassVar[MediaModality] = MediaModality.IMAGE
    # 7B fp8 DiT + video VAE;Pro 6000 上跑绰绰有余,给保守上界供 VRAM 守卫/估算用。
    estimated_vram_mb: ClassVar[int] = 24000

    def __init__(self, paths: dict[str, str], device: str = "cuda", **params: Any) -> None:
        super().__init__(paths, device, **params)
        self.model_dir = paths.get("model_dir") or params.get("model_dir")
        if not self.model_dir:
            raise RuntimeError("SeedVR2UpscaleBackend 需要 paths['model_dir'](SEEDVR2 模型目录)")
        # 三节点对齐 ComfyUI:DiT / VAE 各一份 config dict(device/offload/blockswap/tiling/attention)。
        # 缺省 = 空 dict(走默认),向后兼容单节点(paths["dit"]/["vae"] + 单 device)。
        # dit_config 形状:{model,device,offload_device,blocks_to_swap,swap_io_components,attention_mode}
        # vae_config 形状:{model,device,offload_device,encode_tiled,encode_tile_size,encode_tile_overlap,
        #                   decode_tiled,decode_tile_size,decode_tile_overlap,tile_debug}
        self.dit_cfg: dict[str, Any] = dict(params.get("dit_config") or {})
        self.vae_cfg: dict[str, Any] = dict(params.get("vae_config") or {})
        self.dit_model = self.dit_cfg.get("model") or paths.get("dit") or params.get("dit_model") or DEFAULT_DIT
        self.vae_model = self.vae_cfg.get("model") or paths.get("vae") or params.get("vae_model") or DEFAULT_VAE
        # 增强阶段 tensor offload(对齐 ComfyUI enhance 节点 offload_device;默认 cpu,峰值显存友好)。
        self.tensor_offload = str(params.get("tensor_offload", "cpu") or "cpu")
        self.enable_debug = bool(params.get("enable_debug", False))
        self._runner: Any | None = None
        self._ctx_base: dict[str, Any] | None = None
        self._debug: Any | None = None

    @staticmethod
    def vendored_import_ok() -> bool:
        """PR-1 自检:vendored 推理核心能 import(兼容 patch 生效)。"""
        from src.services.inference.seedvr2_vendor.src.core import (  # noqa: PLC0415
            generation_phases,
            generation_utils,
        )
        return hasattr(generation_utils, "prepare_runner") and hasattr(
            generation_phases, "upscale_all_batches"
        )

    async def load(self, device: str) -> None:
        """ABC 入口:重 GPU 装载丢 to_thread,不阻塞 runner 事件循环(否则 supervisor
        ping 超时误判 crash,见 image_anima 同款处理)。"""
        self.device = device
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        """装 DiT + VAE runner(NumZ prepare_runner)+ 备好 ctx(cache_context + text_embeds)。
        缺模型时 prepare_runner 从 HF 自动下到 model_dir。对齐 NumZ CLI process_single_file。"""
        from src.services.inference.seedvr2_vendor.src.core.generation_utils import (  # noqa: PLC0415
            load_text_embeddings,
            prepare_runner,
            setup_generation_context,
        )
        from src.services.inference.seedvr2_vendor.src.utils.constants import (  # noqa: PLC0415
            get_script_directory,
        )
        from src.services.inference.seedvr2_vendor.src.utils.debug import Debug  # noqa: PLC0415

        import torch  # noqa: PLC0415

        self._debug = Debug(enabled=self.enable_debug)

        # DiT / VAE 各自 device(config 优先,回退 self.device);offload device("none"→None)。
        # **关键**:节点 widget device 默认 "auto",但 NumZ 直接 torch.device(str) 不认 "auto"
        # (model_manager 解析出 target=cuda:N 传给 self.device,却没写回 config)→ 必须在此把
        # "auto"/"cuda"/空 归一到 self.device(已解析的具体卡),否则 torch.device("auto") 抛
        # RuntimeError「device type at start of device string: auto」。
        dit_device = _resolve_dev(self.dit_cfg.get("device"), self.device)
        vae_device = _resolve_dev(self.vae_cfg.get("device"), self.device)
        dit_offload_str = _resolve_dev(self.dit_cfg.get("offload_device", "none"), self.device)
        vae_offload_str = _resolve_dev(self.vae_cfg.get("offload_device", "none"), self.device)
        dit_offload = torch.device(dit_offload_str) if dit_offload_str != "none" else None
        vae_offload = torch.device(vae_offload_str) if vae_offload_str != "none" else None

        # BlockSwap:7B 塞小卡(N>0 或 swap_io 才启;需 offload_device != device)。忠实复刻
        # video_upscaler.execute 的构法。
        blocks_to_swap = int(self.dit_cfg.get("blocks_to_swap", 0) or 0)
        swap_io = bool(self.dit_cfg.get("swap_io_components", False))
        block_swap_config = None
        if blocks_to_swap > 0 or swap_io:
            block_swap_config = {"blocks_to_swap": blocks_to_swap, "swap_io_components": swap_io}
            if dit_offload is not None:
                block_swap_config["offload_device"] = dit_offload

        # VAE tiling:大图分块不爆显存。tile_size/overlap 是 (H,W) tuple(prepare_runner 契约)。
        encode_tiled = bool(self.vae_cfg.get("encode_tiled", False))
        decode_tiled = bool(self.vae_cfg.get("decode_tiled", False))
        enc_ts = int(self.vae_cfg.get("encode_tile_size", 512) or 512)
        enc_to = int(self.vae_cfg.get("encode_tile_overlap", 64) or 64)
        dec_ts = int(self.vae_cfg.get("decode_tile_size", 512) or 512)
        dec_to = int(self.vae_cfg.get("decode_tile_overlap", 64) or 64)
        tile_debug = str(self.vae_cfg.get("tile_debug", "false") or "false")
        # attention:config 优先,默认 sdpa(flash_attn 装不上,SeedVR2 支持 SDPA 回退)。
        attention_mode = str(self.dit_cfg.get("attention_mode") or "sdpa")

        # tensor offload:增强节点 offload_device;"none"→保持 GPU(传 None 让 NumZ 不挪)。
        # "auto" 同样归一到 self.device(不能直接 torch.device("auto"))。
        _tensor_off_str = _resolve_dev(self.tensor_offload, self.device)
        tensor_offload = torch.device(_tensor_off_str) if _tensor_off_str not in ("none", "") else None
        ctx = setup_generation_context(
            dit_device=dit_device,
            vae_device=vae_device,
            dit_offload_device=dit_offload or "cpu",
            vae_offload_device=vae_offload or "cpu",
            tensor_offload_device=tensor_offload or "cpu",
            debug=self._debug,
        )
        # prepare_runner 改/装 ctx in-place,返回 (runner, cache_context)。串入 blockswap + tiling + attention。
        self._runner, cache_context = prepare_runner(
            dit_model=self.dit_model,
            vae_model=self.vae_model,
            model_dir=self.model_dir,
            debug=self._debug,
            ctx=ctx,
            block_swap_config=block_swap_config,
            encode_tiled=encode_tiled,
            encode_tile_size=(enc_ts, enc_ts),
            encode_tile_overlap=(enc_to, enc_to),
            decode_tiled=decode_tiled,
            decode_tile_size=(dec_ts, dec_ts),
            decode_tile_overlap=(dec_to, dec_to),
            tile_debug=tile_debug,
            attention_mode=attention_mode,
        )
        # NumZ CLI 在 prepare_runner 后、encode 前手动补的两步(否则 encode 撞 KeyError):
        ctx["cache_context"] = cache_context
        ctx["text_embeds"] = load_text_embeddings(
            get_script_directory(), ctx["dit_device"], ctx["compute_dtype"], self._debug,
        )
        self._ctx_base = ctx
        self._model = self._runner  # ABC is_loaded → True

    def unload(self) -> None:
        """释放 DiT+VAE runner + 显存。base.unload 只置 _model=None,但 SeedVR2 还持
        _runner(DiT 7B + video VAE)+ ctx,不显式拆 + empty_cache 显存不降(同 anima/modular)。"""
        self._runner = None
        self._ctx_base = None
        self._model = None
        try:
            import gc  # noqa: PLC0415

            import torch  # noqa: PLC0415
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — best-effort
            pass

    async def infer(
        self,
        req: InferenceRequest,
        *,
        progress_callback: Any | None = None,
        cancel_flag: Any | None = None,
    ) -> InferenceResult:
        """UpscaleRequest → InferenceResult(image/png)。decode 输入图 → to_thread 跑
        同步 upscale 内核(CUDA 阻塞,不能占 runner 事件循环)→ PNG bytes。

        progress_callback / cancel_flag 暂未细化(SeedVR2 是 one-step,无 per-step 进度;
        PR-3b runner 接入时按需桥接 stage 级进度)。
        """
        if not isinstance(req, UpscaleRequest):
            raise TypeError(f"SeedVR2UpscaleBackend 只接受 UpscaleRequest,收到 {type(req).__name__}")

        pil = _decode_image(req.image)
        # NumZ set_seed → np.random.seed 要求 [0, 2**32-1];randomize 给的 2**53 超范围会抛。
        # 引擎边界归一;metadata 报折叠后的有效 seed。
        seed = _clamp_seed(req.seed if req.seed is not None else 42)

        def _run() -> Any:
            return self.upscale(
                pil,
                resolution=req.resolution,
                seed=seed,
                batch_size=req.batch_size,
                color_correction=req.color_correction,
                latent_noise_scale=req.latent_noise_scale,
                input_noise_scale=req.input_noise_scale,
                max_resolution=req.max_resolution,
                temporal_overlap=req.temporal_overlap,
                prepend_frames=req.prepend_frames,
                uniform_batch_size=req.uniform_batch_size,
            )

        t = time.monotonic()
        out = await asyncio.to_thread(_run)
        latency_ms = int((time.monotonic() - t) * 1000)

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return InferenceResult(
            media_type="image/png",
            data=buf.getvalue(),
            metadata={
                "width": out.width,
                "height": out.height,
                "resolution": req.resolution,
                "seed": seed,
                "engine": "seedvr2",
            },
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )

    def upscale(
        self,
        image: "Any",  # noqa: UP037 — PIL.Image.Image,惰性 import 避免顶层依赖
        resolution: int = 1080,
        seed: int = 42,
        batch_size: int = 1,
        color_correction: str = "lab",
        latent_noise_scale: float = 0.0,
        input_noise_scale: float = 0.0,
        max_resolution: int = 0,
        temporal_overlap: int = 0,
        prepend_frames: int = 0,
        uniform_batch_size: bool = False,
    ) -> "Any":
        """单图超分。image(PIL.Image)→ 超分后 PIL.Image。忠实复刻 NumZ 4 阶段。

        resolution = 目标短边(SeedVR2 语义:输出最短边像素;非倍数)。
        max_resolution = 长边上限(0=不限);temporal_overlap/prepend_frames = 视频帧间(单图=0)。
        """
        if self._runner is None:
            raise RuntimeError("SeedVR2 未 load")
        seed = _clamp_seed(seed)  # 直接调 upscale 的路径(smoke)也归一
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        from src.services.inference.seedvr2_vendor.src.core.generation_phases import (  # noqa: PLC0415
            decode_all_batches,
            encode_all_batches,
            postprocess_all_batches,
            upscale_all_batches,
        )

        # 输入图 → frames_tensor [1,H,W,C] float16 [0,1] RGB(对齐 NumZ extract_frames_from_image)。
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        frames = torch.from_numpy(rgb[None, ...]).to(torch.float16)

        # ctx 每次 upscale 重置(NumZ ctx 阶段间 in-place 累积,复用 runner 但 ctx 要新)。
        ctx = dict(self._ctx_base) if self._ctx_base else {}

        ctx = encode_all_batches(
            self._runner, ctx=ctx, images=frames, debug=self._debug,
            batch_size=batch_size, uniform_batch_size=uniform_batch_size, seed=seed,
            progress_callback=None, temporal_overlap=temporal_overlap,
            resolution=resolution, max_resolution=max_resolution,
            input_noise_scale=input_noise_scale, color_correction=color_correction,
        )
        ctx = upscale_all_batches(
            self._runner, ctx=ctx, debug=self._debug, progress_callback=None,
            seed=seed, latent_noise_scale=latent_noise_scale, cache_model=False,
        )
        ctx = decode_all_batches(
            self._runner, ctx=ctx, debug=self._debug, progress_callback=None,
            cache_model=False,
        )
        ctx = postprocess_all_batches(
            ctx=ctx, debug=self._debug, progress_callback=None,
            color_correction=color_correction, prepend_frames=prepend_frames,
            temporal_overlap=temporal_overlap, batch_size=batch_size,
        )

        out = ctx["final_video"]  # [N,H,W,C] [0,1]
        if out.is_cuda:
            out = out.cpu()
        if out.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
            out = out.to(torch.float32)
        arr = (out[0].clamp(0, 1).float().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
