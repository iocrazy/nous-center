"""SeedVR2 节点 executor —— DiT/VAE 加载节点(inline,产配置 dict)。

三节点对齐 ComfyUI:`seedvr2_load_dit` / `seedvr2_load_vae` 是 **inline 节点**(主进程,无 GPU,
只把 widget 值 bundle 成配置 dict,像 flux2 loader 产描述符;真加载在增强节点 runner 里)。
`seedvr2_upscale` 是 **dispatch 节点,不在此 EXECUTORS**(runner_process._node_executor 执行,
读上游 dit/vae 配置 → get_or_load_seedvr2_adapter → infer)。

配置 dict 形状对齐 NumZ video_upscaler 消费的契约(image_seedvr2.SeedVR2UpscaleBackend 串进
prepare_runner)。spec 2026-06-02-seedvr2-three-node。
"""
from __future__ import annotations

from typing import Any


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


async def exec_load_dit(data: dict, inputs: dict) -> dict:
    """widget → SEEDVR2_DIT 配置 dict(device/blockswap/offload/attention)。"""
    return {
        "dit": {
            "model": data.get("dit_model") or "",
            "device": data.get("device") or "auto",
            "blocks_to_swap": _int(data.get("blocks_to_swap"), 0),
            "swap_io_components": bool(data.get("swap_io_components", False)),
            "offload_device": data.get("offload_device") or "none",
            "attention_mode": data.get("attention_mode") or "sdpa",
        }
    }


async def exec_load_vae(data: dict, inputs: dict) -> dict:
    """widget → SEEDVR2_VAE 配置 dict(device/tiling)。"""
    return {
        "vae": {
            "model": data.get("vae_model") or "",
            "device": data.get("device") or "auto",
            "encode_tiled": bool(data.get("encode_tiled", False)),
            "encode_tile_size": _int(data.get("encode_tile_size"), 512),
            "encode_tile_overlap": _int(data.get("encode_tile_overlap"), 64),
            "decode_tiled": bool(data.get("decode_tiled", False)),
            "decode_tile_size": _int(data.get("decode_tile_size"), 512),
            "decode_tile_overlap": _int(data.get("decode_tile_overlap"), 64),
            "offload_device": data.get("offload_device") or "none",
        }
    }


EXECUTORS = {
    "seedvr2_load_dit": exec_load_dit,
    "seedvr2_load_vae": exec_load_vae,
}
