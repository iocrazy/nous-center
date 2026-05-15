"""Lane G: image adapter callback_on_step_end 接入测试（fake pipe，integration）.

用 fake diffusers pipe（纯 Python，模拟 num_inference_steps 步、每步 invoke
callback_on_step_end），不需要真 GPU。覆盖三条路径：
  1. 正常完成 —— callback 每步返回，跑满 steps
  2. within-node cancel —— 第 N 步 CancelFlag 被 set，callback 抛 NodeCancelled
  3. timeout —— wait_for 超时，set flag，下一步 callback 抛，infer 抛 NodeTimeout
  4. 外部 cancel_flag —— infer 接受外部 flag，set 后 infer 抛 NodeCancelled

这些用例的 infer()/sample() 路径都会 import torch，所以无 torch 时整文件 skip。
"""
import asyncio

import pytest

torch = pytest.importorskip("torch")  # 全文件 skip if torch unavailable

from src.services.inference.cancel_flag import CancelFlag  # noqa: E402
from src.services.inference.exceptions import (  # noqa: E402
    NodeCancelled,
    NodeTimeout,
)


# ---- fake pipe ------------------------------------------------------------


class _FakeLatent:
    """最小 latent stub：sample() 的 unpack 路径只读 .shape / .device。"""

    def __init__(self, batch=1):
        self.shape = (batch, 16, 32, 32)
        self.device = "cpu"


class _FakePipe:
    """模拟 diffusers pipe.__call__ 的采样循环：跑 num_inference_steps 步，
    每步 invoke callback_on_step_end。完全同步、纯 Python、不碰 CUDA。"""

    def __init__(self, step_sleep_s: float = 0.0):
        self.vae_scale_factor = 8
        self._step_sleep_s = step_sleep_s
        self.steps_run = 0

    def __call__(self, *, num_inference_steps, callback_on_step_end=None, **kw):
        import time
        for step in range(num_inference_steps):
            if callback_on_step_end is not None:
                callback_on_step_end(self, step, 1000 - step, {})
            self.steps_run += 1
            if self._step_sleep_s:
                time.sleep(self._step_sleep_s)
        # 返回带 .images 属性的对象，模仿 diffusers PipelineOutput
        return _FakePipelineOutput()

    # sample() 的 unpack 尾段需要这两个 —— 直接透传，本测试不验证 unpack 正确性
    def _prepare_latent_ids(self, shape_proxy):
        return shape_proxy

    def _unpack_latents_with_ids(self, packed, latent_ids):
        return packed


class _FakeImage:
    """模拟 PIL Image：infer 收尾 image.save(buf, format='PNG')。"""

    def save(self, buf, format=None):
        buf.write(b"\x89PNG\r\n\x1a\n" + b"fake-png-data")


class _FakePipelineOutput:
    """模拟 diffusers PipelineOutput：infer 末尾读 out.images[0]。"""

    def __init__(self):
        self.images = [_FakeImage()]


# 用于 sample() 路径测试（output_type="latent"）的 pipe —— 返回 (latent,) tuple
class _FakeSamplePipe(_FakePipe):
    def __call__(self, *, num_inference_steps, callback_on_step_end=None, **kw):
        import time
        for step in range(num_inference_steps):
            if callback_on_step_end is not None:
                callback_on_step_end(self, step, 1000 - step, {})
            self.steps_run += 1
            if self._step_sleep_s:
                time.sleep(self._step_sleep_s)
        return (_FakeLatent(),)


# ---- sample() 集成测试 ----------------------------------------------------


async def test_sample_runs_all_steps_when_not_cancelled():
    """正常路径：cancel_flag 未 set，sample 跑满 num_inference_steps。"""
    from src.services.inference.image_diffusers import sample

    pipe = _FakeSamplePipe()
    flag = CancelFlag()
    conditioning = {"prompt_embeds": "fake", "text_ids": "fake"}
    result = await asyncio.to_thread(
        sample, pipe, conditioning,
        width=512, height=512, num_inference_steps=8,
        guidance_scale=3.5, cancel_flag=flag,
    )
    assert pipe.steps_run == 8
    assert result is not None


async def test_sample_interrupts_when_flag_set_mid_run():
    """within-node cancel：跑到中段时另一线程 set flag，sample 在下一步抛
    NodeCancelled，steps_run 远小于请求的 30。"""
    from src.services.inference.image_diffusers import sample

    pipe = _FakeSamplePipe(step_sleep_s=0.01)  # 给取消线程一个介入窗口
    flag = CancelFlag()

    async def cancel_after_delay():
        await asyncio.sleep(0.05)  # ~5 步后
        flag.set("user requested")

    conditioning = {"prompt_embeds": "fake", "text_ids": "fake"}
    cancel_task = asyncio.create_task(cancel_after_delay())
    with pytest.raises(NodeCancelled):
        await asyncio.to_thread(
            sample, pipe, conditioning,
            width=512, height=512, num_inference_steps=30,
            guidance_scale=3.5, cancel_flag=flag,
        )
    await cancel_task
    assert pipe.steps_run < 30  # 没跑满 —— 被中断了
    assert pipe.steps_run >= 1


# ---- infer() 集成测试 ----------------------------------------------------


async def test_infer_timeout_sets_flag_and_raises_node_timeout():
    """timeout 路径：infer 把采样包进 wait_for(to_thread(...))，超时后 set
    flag + 抛 NodeTimeout；在飞的 fake pipe 在下一步因 flag 中断（不挂死）。"""
    from src.services.inference.base import ImageRequest
    from src.services.inference.image_diffusers import DiffusersImageBackend

    adapter = DiffusersImageBackend(paths={"main": "/fake"}, device="cpu")
    # 直接注入 fake pipe，跳过 load()
    adapter._pipe = _FakePipe(step_sleep_s=0.05)  # 每步 50ms，30 步 = 1.5s

    req = ImageRequest(
        request_id="t-timeout", prompt="x",
        width=512, height=512, steps=30, cfg_scale=3.5,
        timeout_s=0.2,  # 0.2s 远小于 1.5s —— 必超时
    )
    with pytest.raises(NodeTimeout) as ei:
        await adapter.infer(req)
    assert ei.value.timeout_s == 0.2


async def test_infer_external_cancel_flag_is_honored():
    """infer 接受外部传入的 cancel_flag（runner pipe-reader 持同一引用）；
    外部 set 后，在飞的采样在下一步中断，infer 抛 NodeCancelled。"""
    from src.services.inference.base import ImageRequest
    from src.services.inference.image_diffusers import DiffusersImageBackend

    adapter = DiffusersImageBackend(paths={"main": "/fake"}, device="cpu")
    adapter._pipe = _FakePipe(step_sleep_s=0.02)

    external_flag = CancelFlag()
    req = ImageRequest(
        request_id="t-ext-cancel", prompt="x",
        width=512, height=512, steps=30, cfg_scale=3.5,
    )

    async def cancel_soon():
        await asyncio.sleep(0.1)
        external_flag.set("aborted by runner")

    cancel_task = asyncio.create_task(cancel_soon())
    with pytest.raises(NodeCancelled) as ei:
        await adapter.infer(req, cancel_flag=external_flag)
    await cancel_task
    assert ei.value.reason == "aborted by runner"
