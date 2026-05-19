"""PR-2 manual smoke — cancel ImageSampler mid-denoise, verify <= 500ms.

Spec §8 PR-2 acceptance criterion: 25-step sampling cancelled at step 10
must return within 500ms of the cancel.set() call.

Schedules an async cancel via asyncio.create_task; fires it ~15 seconds in
(~step 10 of 25 at Pro 6000 step time ~1.5s); asserts SamplerCancelled
raised within 500ms.

Run:
    cd backend && NOUS_DISABLE_RUNNER_SPAWN=1 .venv/bin/python scripts/smoke_pr2_cancel.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: F401 — ensures CUDA init before component load

from src.services.inference.base import ImageRequest
from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_diffusers import DiffusersImageBackend
from src.services.inference.image_sampler import SamplerCancelled

FLUX2_KLEIN_DIR = Path("/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B")


async def main() -> int:
    # Single-device for this smoke — cancel timing is the dimension under test,
    # not cross-device assembly (covered by smoke_pr2_cross_device.py).
    components = {
        "unet": ComponentSpec(kind="unet", adapter_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "transformer/diffusion_pytorch_model.safetensors"),
            device="cuda:1", dtype="bfloat16"),
        "clip": ComponentSpec(kind="clip", clip_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "text_encoder/model.safetensors"),
            device="cuda:1", dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae",
            file=str(FLUX2_KLEIN_DIR / "vae/diffusion_pytorch_model.safetensors"),
            device="cuda:1", dtype="bfloat16"),
    }

    adapter = DiffusersImageBackend.from_components(components, pipeline_class="Flux2KleinPipeline")
    await adapter.load()

    cancel_flag = CancelFlag()
    cancel_time: dict[str, float | None] = {"set_at": None}

    async def _cancel_after_seconds(secs: float) -> None:
        await asyncio.sleep(secs)
        cancel_time["set_at"] = time.monotonic()
        cancel_flag.set("smoke_test_cancel")

    req = ImageRequest(request_id="smoke-cancel", prompt="cancel test",
                       seed=1, steps=25, width=512, height=512)

    # Step ~1.5s on Pro 6000 -> step 10 ~15s. Schedule cancel at 15s.
    cancel_task = asyncio.create_task(_cancel_after_seconds(15.0))

    t0 = time.monotonic()
    try:
        await adapter.infer(req, cancel_flag=cancel_flag)
        print("FAILED: infer returned without cancellation")
        return 1
    except SamplerCancelled as e:
        cancelled_at = time.monotonic()
        set_at = cancel_time["set_at"]
        latency = (cancelled_at - set_at) * 1000 if set_at is not None else 0
        print(f"SamplerCancelled raised (reason={e.reason!r}) {latency:.0f}ms after cancel.set()")
        print(f"Total elapsed: {cancelled_at - t0:.1f}s")
        if latency > 500:
            print(f"FAILED: cancel latency {latency:.0f}ms exceeds 500ms target")
            return 1
        print("Cancel-mid-sampler smoke passed (<=500ms)")
        return 0
    finally:
        if not cancel_task.done():
            cancel_task.cancel()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
