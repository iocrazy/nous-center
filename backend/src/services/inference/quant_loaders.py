"""Quant loader registry for image-component-multi-gpu PR-1.

Per spec §5.3: registry maps (ComponentSpec) → loaded weights (state_dict).
First-match-wins; register specific formats (fp8mixed / mxfp8mixed / nvfp4mixed)
before plain safetensors fallback.

Each loader returns:
  dict[str, Tensor]      (state_dict — caller's responsibility to wrap into a module)

GGUF is rejected eagerly with UnsupportedQuantError; V2 PR-7 will add it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, NoReturn

import torch

from src.services.inference.component_spec import ComponentSpec

logger = logging.getLogger(__name__)


# Loaders all return the same shape: caller wraps into a torch.nn.Module.
StateDict = dict[str, torch.Tensor]


def _load_file(path: str) -> StateDict:
    """Lazy safetensors load —— 模块被 import 不该强依赖 safetensors(在 image extra,
    CI backend job 不装);只有真加载量化文件时才需。让 quant_loaders 在 CI 可被 import。"""
    from safetensors.torch import load_file as _lf  # noqa: PLC0415

    return _lf(path, device="cpu")


class UnsupportedQuantError(RuntimeError):
    """Raised when no registered loader matches a ComponentSpec."""


class QuantLoaderRegistry:
    """First-match-wins registry. Register specific formats before generic fallbacks."""

    def __init__(self) -> None:
        self._loaders: list[tuple[Callable[[ComponentSpec], bool], Callable[[ComponentSpec], Any]]] = []

    def register(self, *, match: Callable[[ComponentSpec], bool]) -> Callable[[Callable], Callable]:
        """Decorator. `match(spec)` → bool decides if this loader handles the spec."""
        def deco(fn: Callable[[ComponentSpec], Any]) -> Callable[[ComponentSpec], Any]:
            self._loaders.append((match, fn))
            return fn
        return deco

    def dispatch(self, spec: ComponentSpec) -> StateDict:
        for matcher, fn in self._loaders:
            if matcher(spec):
                logger.debug("quant_loaders: dispatching %s to %s", spec.file, fn.__name__)
                return fn(spec)
        raise UnsupportedQuantError(f"no quant loader matches {spec.file!r}")


QUANT_LOADERS = QuantLoaderRegistry()


def dequant_and_convert(spec: ComponentSpec) -> StateDict:
    """comfy 量化单文件 Flux2 transformer → diffusers-key state_dict(D4 共享桥接)。

    两步缺一不可(spike v2/v3 实证):
      ① `QUANT_LOADERS.dispatch(spec)` 反量化(解 comfy fp8/mxfp8/nvfp4 打包)
      ② diffusers `convert_flux2_transformer_checkpoint_to_diffusers` 转键
         (comfy `double_blocks.*` → diffusers `transformer_blocks.*`)
    漏掉 ② → load_state_dict 静默丢键(missing=233)→ 噪声图。

    新 modular 引擎 + legacy `_load_hf_or_quant` 都调本 helper。转换器是 diffusers
    **内部函数**(loaders.single_file_utils)→ guard import,失败报清晰错误。
    仅适用 Flux2 transformer(caller 保证 kind=unet/adapter_arch=flux2)。
    """
    sd = QUANT_LOADERS.dispatch(spec)
    try:
        from diffusers.loaders.single_file_utils import (
            convert_flux2_transformer_checkpoint_to_diffusers,
        )
    except ImportError as e:  # diffusers 版本/commit 不符
        raise ValueError(
            "diffusers 缺 convert_flux2_transformer_checkpoint_to_diffusers"
            "(loaders.single_file_utils)—— diffusers 版本与 pyproject 钉的 commit 不符,"
            "无法转 comfy 量化键。检查 diffusers 安装。"
        ) from e
    return convert_flux2_transformer_checkpoint_to_diffusers(dict(sd))


# Reject GGUF eagerly — V2 PR-7 work, not in scope for PR-1.
@QUANT_LOADERS.register(match=lambda spec: spec.file.lower().endswith(".gguf"))
def reject_gguf(spec: ComponentSpec) -> NoReturn:
    raise UnsupportedQuantError(
        f"GGUF quantization is V2 PR-7 follow-up; cannot load {spec.file!r} in PR-1"
    )


_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp8_e4m3": torch.float8_e4m3fn,
}


def _dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    """Map ComponentSpec.dtype string → torch.dtype.

    Raises UnsupportedQuantError on unknown dtype rather than silently falling
    back to bfloat16 (which would miscast user-loaded weights). PR-3+ adding
    a new format must register here.
    """
    try:
        return _DTYPE_MAP[dtype_str]
    except KeyError:
        raise UnsupportedQuantError(
            f"unknown target dtype {dtype_str!r}; expected one of {sorted(_DTYPE_MAP)}"
        )


def _has_comfy_quant_metadata(file_path: str) -> bool:
    """Sniff a safetensors header for any `.comfy_quant` suffixed key (cheap — no full read).

    Fails soft (returns False) on any error — caller falls through to plain loader.
    Operators can grep logs at DEBUG level to diagnose misdispatch.
    """
    try:
        from safetensors import safe_open
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".comfy_quant"):
                    return True
    except Exception as exc:  # noqa: BLE001 — fail-soft on header read error
        logger.debug("comfy_quant header sniff failed for %s: %s", file_path, exc)
        return False
    return False


def _unpack_int4_to_int8(packed: torch.Tensor) -> torch.Tensor:
    """Two 4-bit signed values per byte → expand to (-8..7) int8.

    Layout (community nvfp4 quants): low nibble = first weight, high nibble = second.
    Each nibble is signed 4-bit (range -8..7) following two's complement on 4 bits.
    """
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    # Sign-extend 4-bit → 8-bit: values >= 8 are negative
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    # Interleave: [low_0, high_0, low_1, high_1, ...]
    interleaved = torch.stack([low, high], dim=-1).reshape(-1)
    return interleaved


@QUANT_LOADERS.register(match=lambda spec: "nvfp4mixed" in Path(spec.file).name.lower())
def load_nvfp4mixed(spec: ComponentSpec) -> StateDict:
    """NVIDIA FP4 → unpack 2-per-byte → block-wise fp32 scale → target dtype.

    Real file: Flux2-Klein-9B-True-v2-nvfp4mixed.safetensors (5.6GB — about 1/3 the
    size of bf16, half of fp8mixed).

    Algorithm:
      1. Load uint8 packed weights + fp32 per-16-block scales + int32 original shape
      2. Unpack each byte into two signed 4-bit values (range -8..7)
      3. Per 16-element block, multiply by fp32 scale
      4. Reshape to original shape, cast to target dtype
    """
    target = _dtype_str_to_torch(spec.dtype)
    raw = _load_file(spec.file)
    BLOCK_SIZE = 16

    clean: dict[str, torch.Tensor] = {}
    unpacked_count = 0

    # Group keys by base name (so we can find weight + weight_scale + weight_shape together)
    bases: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in raw.items():
        if ".comfy_quant" in k:
            continue
        if k.endswith("_scale"):
            base = k[: -len("_scale")]
            bases.setdefault(base, {})["scale"] = v
        elif k.endswith("_shape"):
            base = k[: -len("_shape")]
            bases.setdefault(base, {})["shape"] = v
        else:
            bases.setdefault(k, {})["weight"] = v

    for base, parts in bases.items():
        weight = parts.get("weight")
        if weight is None:
            continue
        if weight.dtype != torch.uint8:
            # Not a packed nvfp4 weight — pass-through (only cast fp32/fp16 like fp8mixed branch)
            if weight.dtype in (torch.float32, torch.float16):
                clean[base] = weight.to(target)
            else:
                clean[base] = weight
            continue

        scale = parts.get("scale")
        shape = parts.get("shape")
        if scale is None or shape is None:
            logger.warning("nvfp4: %s packed but missing scale/shape; loading raw uint8", base)
            clean[base] = weight
            continue

        # 1. Unpack 2 4-bit weights per byte → flat int8
        unpacked = _unpack_int4_to_int8(weight)
        # 2. Block-wise scale (one fp32 per 16-elem block)
        flat = unpacked.to(torch.float32)
        assert flat.numel() % BLOCK_SIZE == 0
        assert scale.numel() * BLOCK_SIZE == flat.numel()
        block_scales = scale.to(torch.float32).repeat_interleave(BLOCK_SIZE)
        dequant = (flat * block_scales).to(target)
        # 3. Reshape to original shape
        orig_shape = tuple(int(x) for x in shape.tolist())
        clean[base] = dequant.reshape(orig_shape)
        unpacked_count += 1

    logger.info("quant_loaders.nvfp4mixed: %d nvfp4 tensors unpacked, %d total keys (%s)",
                unpacked_count, len(clean), Path(spec.file).name)
    return clean


@QUANT_LOADERS.register(match=lambda spec: "mxfp8mixed" in Path(spec.file).name.lower())
def load_mxfp8mixed(spec: ComponentSpec) -> StateDict:
    """Microscaling fp8 → dequant by block-wise E8M0 scale → target dtype.

    Format: per-32-element blocks, each with a uint8 E8M0 exponent in `.weight_scale`.
    Real file: Flux2-Klein-9B-True-v2-mxfp8mixed.safetensors (9.7GB).

    Algorithm:
      1. Load fp8 weight + uint8 scale tensor (1 byte per 32-element block)
      2. For each block: scale_fp32 = 2.0 ** (uint8_scale - 127)   # E8M0 with bias 127
      3. fp32_weight = fp8_weight × scale (broadcast within block)
      4. Cast to target dtype, drop metadata keys
    """
    target = _dtype_str_to_torch(spec.dtype)
    raw = _load_file(spec.file)
    BLOCK_SIZE = 32

    clean: dict[str, torch.Tensor] = {}
    dequant_count = 0
    for key, tensor in raw.items():
        if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale"
            scale_uint8 = raw.get(scale_key)
            if scale_uint8 is None:
                logger.warning("mxfp8: tensor %s missing %s; using fp8 cast only", key, scale_key)
                clean[key] = tensor.to(target)
                continue
            # E8M0: scale = 2^(uint8 - 127)
            scale_fp32 = torch.pow(2.0, scale_uint8.to(torch.float32) - 127.0)
            # Broadcast block-wise: flatten weights, repeat each scale BLOCK_SIZE times
            flat = tensor.flatten().to(torch.float32)
            assert flat.numel() % BLOCK_SIZE == 0, \
                f"mxfp8: {key} numel {flat.numel()} not divisible by block {BLOCK_SIZE}"
            assert scale_fp32.numel() * BLOCK_SIZE == flat.numel(), \
                f"mxfp8: {key} scale count {scale_fp32.numel()} × block {BLOCK_SIZE} ≠ weight numel {flat.numel()}"
            block_scales = scale_fp32.repeat_interleave(BLOCK_SIZE)
            dequant = (flat * block_scales).to(target).reshape(tensor.shape)
            clean[key] = dequant
            dequant_count += 1
        else:
            # Same pass-through policy as load_fp8mixed: only cast fp32/fp16
            if tensor.dtype in (torch.float32, torch.float16):
                clean[key] = tensor.to(target)
            else:
                clean[key] = tensor

    logger.info("quant_loaders.mxfp8mixed: %d block-quant tensors dequant'd, %d total keys (%s)",
                dequant_count, len(clean), Path(spec.file).name)
    return clean


@QUANT_LOADERS.register(match=lambda spec: (
    "fp8mixed" in Path(spec.file).name.lower()
    or _has_comfy_quant_metadata(spec.file)
))
def load_fp8mixed(spec: ComponentSpec) -> StateDict:
    """Wikeeyang comfy_quant fp8 → dequant by `.weight_scale` companion → target dtype.

    Algorithm (preserved from image_diffusers.py:105 load_quantized_transformer):
      1. safetensors_load_file → state dict with fp8 weights + .weight_scale + .comfy_quant
      2. For each fp8 tensor, multiply by float32 scale, cast to target dtype
      3. Drop .weight_scale and .comfy_quant marker keys
      4. Return clean state dict ready for caller's load_state_dict

    Reference fixture: /media/heygo/program/models/nous/image/diffusion_models/
    Flux2-Klein-9B-True-v2-fp8mixed.safetensors
    """
    target = _dtype_str_to_torch(spec.dtype)
    raw = _load_file(spec.file)

    clean: dict[str, torch.Tensor] = {}
    fp8_count = 0
    for key, tensor in raw.items():
        if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
            continue  # metadata key, drop
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale"
            scale = raw.get(scale_key)
            if scale is None:
                logger.warning("fp8 tensor %s has no companion %s scale; loading at fp8 dtype", key, scale_key)
                clean[key] = tensor.to(target)
                continue
            # dequant: fp8 × scale → fp32 → target
            clean[key] = (tensor.to(torch.float32) * scale.to(torch.float32)).to(target)
            fp8_count += 1
        else:
            # Preserve pre-refactor behavior: only cast fp32/fp16 tensors to target;
            # other dtypes (uint8/int/bool marker tensors) pass through unchanged.
            # On Flux2 fp8mixed in practice the only non-fp8 tensors are bf16 weights,
            # so this is a no-op vs the unconditional cast — but it makes the
            # "byte-identical to pre-PR-1" claim in image_diffusers.py:129 honest.
            if tensor.dtype in (torch.float32, torch.float16):
                clean[key] = tensor.to(target)
            else:
                clean[key] = tensor

    logger.info("quant_loaders.fp8mixed: %d fp8 weights dequant'd, %d total keys (%s)",
                fp8_count, len(clean), Path(spec.file).name)
    return clean


# Plain bf16/fp16 safetensors — uniform state_dict loader. Caller (PR-2's
# DiffusersImageBackend or test) decides whether to wrap into a module.
@QUANT_LOADERS.register(match=lambda spec: spec.file.endswith(".safetensors"))
def load_safetensors_plain(spec: ComponentSpec) -> StateDict:
    """Plain bf16/fp16 safetensors → state_dict, target dtype applied.

    Note: this is the FALLBACK matcher. It MUST stay last in this module —
    PR-3+ fp8mixed / mxfp8mixed / nvfp4mixed loaders register ABOVE this function
    so their filename-substring matchers run first (`_loaders` is iterated in
    registration order; first match wins). Loads to CPU regardless of
    `spec.device` — caller is responsible for the subsequent `.to(device)`.
    """
    target = _dtype_str_to_torch(spec.dtype)
    sd = _load_file(spec.file)
    return {k: v.to(target) for k, v in sd.items()}


# E2M1(NVFP4 4-bit 浮点)码 → 幅值:code&7,bit3=符号。comfy/float.py 反推。
_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_NVFP4_BLOCK = 16


def dequant_comfy_mixed(spec: ComponentSpec) -> StateDict:
    """comfy 逐张量混合量化 → target dtype state_dict(标准键,**不转架构键**)。

    `.comfy_quant`(JSON utf-8 bytes)逐张量标 format:
      - float8_e4m3fn: weight(fp8) × weight_scale → target
      - nvfp4: uint8 unpack(偶数元素=高 nibble/奇数=低)→ E2M1 解码 × block_scale(fp8 [out,in/16])
               × global_scale(weight_scale_2 fp32 标量)→ target
      - 无 comfy_quant 的权重: plain cast(fp32/fp16/bf16)或原样(int/bool 标记)
    真模型验过(spike_single_file_assembly:Qwen3 fp8×141+nvfp4×85+plain×172 → 干净狐狸图)。
    用于 comfy 单文件 text encoder / vae(键已是目标架构,无需 flux2 那种转键)。
    """
    import json  # noqa: PLC0415

    target = _dtype_str_to_torch(spec.dtype)
    raw = _load_file(spec.file)
    e2m1 = torch.tensor(_E2M1_MAG, dtype=torch.float32)

    def _fmt(base: str) -> str | None:
        cq = raw.get(base + ".comfy_quant")
        if cq is None:
            return None
        return json.loads(bytes(cq.tolist()).decode("utf-8")).get("format")

    clean: StateDict = {}
    n_fp8 = n_nvfp4 = n_plain = 0
    for key, t in raw.items():
        if key.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
            continue  # companion/metadata — drop
        base = key[: -len(".weight")] if key.endswith(".weight") else None
        fmt = _fmt(base) if base else None
        if fmt == "float8_e4m3fn":
            # round3 #5:用 .get 守卫(对齐 load_fp8mixed)—— 残缺/截断单文件可能丢
            # weight_scale companion。fp8 缺 scale 降级按 fp8 cast(幅度不对但不崩,
            # 与兄弟 loader 一致),不抛裸 KeyError。
            scale_t = raw.get(base + ".weight_scale")
            if scale_t is None:
                logger.warning(
                    "dequant_comfy_mixed: %s 标 fp8 但缺 %s.weight_scale,降级 cast",
                    key, base)
                clean[key] = t.to(target)
                n_fp8 += 1
                continue
            scale = scale_t.to(torch.float32)
            clean[key] = (t.to(torch.float32) * scale).to(target)
            n_fp8 += 1
        elif fmt == "nvfp4":
            # nvfp4 是 packed uint8,缺 block/global scale 无法解码(plain-cast 是垃圾)
            # → fail-loud 指名张量,别静默出错图(round3 #5)。
            bs_t = raw.get(base + ".weight_scale")
            gs_t = raw.get(base + ".weight_scale_2")
            if bs_t is None or gs_t is None:
                raise UnsupportedQuantError(
                    f"nvfp4 张量 {key} 缺 scale("
                    f"weight_scale={'有' if bs_t is not None else '缺'}/"
                    f"weight_scale_2={'有' if gs_t is not None else '缺'})"
                    f" —— 单文件可能截断/转换器漏写")
            bs = bs_t.to(torch.float32)       # [out, in/16]
            gs = gs_t.to(torch.float32)     # fp32 标量
            out = t.shape[0]
            low = (t & 0x0F).to(torch.long)
            high = ((t >> 4) & 0x0F).to(torch.long)
            dec = lambda c: torch.where(c >= 8, -1.0, 1.0) * e2m1[c & 0x7]  # noqa: E731
            # comfy 打包 packed=(even<<4)|odd → 偶数元素=高 nibble、奇数=低 nibble
            vals = torch.stack([dec(high), dec(low)], dim=-1).reshape(out, -1)
            blk = bs.repeat_interleave(_NVFP4_BLOCK, dim=1)          # [out, in]
            clean[key] = (vals * blk * gs).to(target)
            n_nvfp4 += 1
        else:
            clean[key] = (t.to(target)
                          if t.dtype in (torch.float32, torch.float16, torch.bfloat16) else t)
            n_plain += 1
    logger.info("quant_loaders.comfy_mixed: fp8=%d nvfp4=%d plain=%d (%s)",
                n_fp8, n_nvfp4, n_plain, Path(spec.file).name)
    return clean
