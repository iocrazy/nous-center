"""PR-0 spike — 大图像模型塞进 24GB 3090:layerwise-fp8 cast vs group-offload。standalone,需 GPU。

验核心假设(feedback_verify_real_model)——在一张 3090(~24GB)上跑 Flux2-klein-9B(~25GB bf16):
  - baseline  : 纯 bf16(预期 OOM —— 这就是要解决的问题)
  - fp8cast   : transformer.enable_layerwise_casting(fp8_e4m3 存储 / bf16 计算)→ 省 ~½ 权重显存,不掉速
  - offload   : transformer.enable_group_offload(cpu offload + cuda stream 重叠)→ 任何模型塞下,速度代价

每模式记录:能否加载、峰值显存(MB)、出图延迟、出图是否正确(肉眼/对比 Pro6000 已知狐狸图)。

用法(每模式独立进程,避免显存残留):
    cd backend
    SPIKE_MODE=fp8cast  SPIKE_DEVICE=cuda:2 uv run python tests/manual/spike_fit_3090.py
    SPIKE_MODE=offload  SPIKE_DEVICE=cuda:2 uv run python tests/manual/spike_fit_3090.py
    SPIKE_MODE=baseline SPIKE_DEVICE=cuda:2 uv run python tests/manual/spike_fit_3090.py
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

# 与生产一致(uv 不 load .env);PCI_BUS_ID 下 cuda:0/2=3090, cuda:1=Pro6000。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = "/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B"
DEVICE = os.environ.get("SPIKE_DEVICE", "cuda:2")  # 一张 3090(约束卡)
MODE = os.environ.get("SPIKE_MODE", "fp8cast")
PROMPT = "a photo of a red fox sitting in autumn leaves, sharp focus, detailed"
OUT_DIR = Path(__file__).parent / "_smoke_out"


def _get_transformer(pipe):
    """从 ModularPipeline 取 transformer 组件(experimental API,多路兜底)。"""
    for getter in (
        lambda: pipe.transformer,
        lambda: pipe.get_component("transformer"),
        lambda: pipe.components["transformer"],
    ):
        try:
            t = getter()
            if t is not None:
                return t
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"取不到 transformer 组件;pipe attrs={[a for a in dir(pipe) if 'transf' in a.lower()]}")


def main() -> int:
    import torch

    from src.services.inference.image_modular import _import_modular, _torch_dtype

    dev = torch.device(DEVICE)
    props = torch.cuda.get_device_properties(dev)
    print(f"[spike:{MODE}] device={DEVICE} ({props.name} {props.total_memory/1024**3:.1f}GB)")
    torch.cuda.reset_peak_memory_stats(dev)

    modular_pipeline_cls, components_manager_cls = _import_modular()
    cm = components_manager_cls()
    pipe = modular_pipeline_cls.from_pretrained(REPO, components_manager=cm)

    t_load = time.monotonic()
    try:
        pipe.load_components(torch_dtype=_torch_dtype("bfloat16"))
        transformer = _get_transformer(pipe)
        print(f"[spike:{MODE}] transformer={type(transformer).__name__}")

        text_encoder = getattr(pipe, "text_encoder", None)
        # 两个大件:transformer 17GB + text_encoder(Qwen3)16GB —— 必须一起处理才进 24GB。
        big = [("transformer", transformer)] + ([("text_encoder", text_encoder)] if text_encoder is not None else [])
        print(f"[spike:{MODE}] big components: {[n for n, _ in big]}")

        if MODE == "fp8cast":
            from diffusers.hooks import apply_layerwise_casting
            for name, mod in big:
                apply_layerwise_casting(
                    mod, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
                print(f"[spike:{MODE}] layerwise-cast {name} → fp8 storage / bf16 compute")
            pipe.to(dev)
        elif MODE == "offload":
            from diffusers.hooks import apply_group_offloading
            for name, mod in big:
                apply_group_offloading(
                    mod, onload_device=dev, offload_device=torch.device("cpu"),
                    offload_type="block_level", num_blocks_per_group=1, use_stream=True)
                print(f"[spike:{MODE}] group-offload {name} (block_level, stream)")
            # 大件由 group-offload 管放置;小件(vae)显式上卡。
            vae = getattr(pipe, "vae", None)
            if vae is not None and hasattr(vae, "to"):
                vae.to(dev)
            # group-offload 用 diffusers hook(非 accelerate _hf_hook),pipe._execution_device
            # 退回 self.device=cpu → latents 落 cpu → denoise 设备冲突。覆盖它指向 onload 卡。
            type(pipe)._execution_device = property(lambda self: dev)
            print(f"[spike:{MODE}] patched _execution_device → {dev}")
        elif MODE == "baseline":
            pipe.to(dev)
        else:
            raise SystemExit(f"未知 SPIKE_MODE={MODE}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[spike:{MODE}] OOM at load/place: {e}")
        print(f"[spike:{MODE}] RESULT mode={MODE} loaded=NO oom=YES")
        return 0
    load_ms = int((time.monotonic() - t_load) * 1000)
    mem_after_load = torch.cuda.max_memory_allocated(dev) / 1024**2
    print(f"[spike:{MODE}] loaded in {load_ms}ms, peak-after-load={mem_after_load:.0f}MB")

    # _execution_device 已指向卡(offload 也 patch 过),latents 在卡上 → cuda generator。
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    # fp8cast:transformer/text_encoder.dtype 变 fp8 会污染 prepare_latents 的 noise dtype
    # (block_state.dtype=prompt_embeds.dtype)→ randn fp8 崩。显式传 dtype=bf16 试覆盖。
    extra = {"dtype": torch.bfloat16} if MODE == "fp8cast" else {}
    t_inf = time.monotonic()
    try:
        out = pipe(prompt=PROMPT, num_inference_steps=20, width=1024, height=1024,
                   generator=gen, **extra)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[spike:{MODE}] OOM during inference: {e}")
        print(f"[spike:{MODE}] RESULT mode={MODE} loaded=YES infer=OOM")
        return 0
    inf_ms = int((time.monotonic() - t_inf) * 1000)
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2

    img = out.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    OUT_DIR.mkdir(exist_ok=True)
    out_png = OUT_DIR / f"spike_fit_3090_{MODE}.png"
    out_png.write_bytes(buf.getvalue())
    print(f"[spike:{MODE}] saved → {out_png} ({len(buf.getvalue())} bytes)")
    print(f"[spike:{MODE}] RESULT mode={MODE} loaded=YES peak_vram={peak:.0f}MB "
          f"load_ms={load_ms} infer_ms={inf_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
