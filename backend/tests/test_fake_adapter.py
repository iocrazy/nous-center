"""Lane C: FakeAdapter 测试 —— 零 GPU 的 InferenceAdapter，给 runner 框架测试用。"""
import asyncio
import threading

import pytest

from src.runner.fake_adapter import FakeAdapter, FakeLoadError
from src.services.inference.base import ImageRequest, InferenceAdapter, MediaModality


def _img_req(steps: int = 4) -> ImageRequest:
    return ImageRequest(request_id="r1", prompt="a cat", steps=steps)


def test_fake_adapter_is_inference_adapter():
    a = FakeAdapter(paths={"main": "/fake"})
    assert isinstance(a, InferenceAdapter)
    assert a.modality == MediaModality.IMAGE


@pytest.mark.asyncio
async def test_load_and_infer_happy_path():
    a = FakeAdapter(paths={"main": "/fake"}, infer_seconds=0.01)
    assert not a.is_loaded
    await a.load("cpu")
    assert a.is_loaded
    result = await a.infer(_img_req())
    assert result.media_type == "image/png"
    assert result.data  # 非空 bytes
    assert result.usage.image_count == 1


@pytest.mark.asyncio
async def test_fail_load_raises():
    a = FakeAdapter(paths={"main": "/fake"}, fail_load=True)
    with pytest.raises(FakeLoadError):
        await a.load("cpu")
    assert not a.is_loaded


@pytest.mark.asyncio
async def test_crash_on_infer_raises_runtime_error():
    """crash_on_infer=True —— infer 抛异常，模拟节点执行期 native fault。"""
    a = FakeAdapter(paths={"main": "/fake"}, crash_on_infer=True)
    await a.load("cpu")
    with pytest.raises(RuntimeError):
        await a.infer(_img_req())


@pytest.mark.asyncio
async def test_infer_reports_per_step_progress():
    """多 step 时，progress_callback 每 step 被调一次，参数单调递增。"""
    a = FakeAdapter(paths={"main": "/fake"}, infer_seconds=0.0)
    await a.load("cpu")
    seen: list[tuple[int, int]] = []
    await a.infer(_img_req(steps=5), progress_callback=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


@pytest.mark.asyncio
async def test_cancel_flag_interrupts_infer():
    """传入一个已 set 的 threading.Event，infer 在下一 step 边界抛 asyncio.CancelledError.

    对应 spec §4.4：within-node cancel 信号穿过 to_thread 边界（这里 fake 用
    asyncio.sleep 模拟，真 adapter 用 callback_on_step_end —— 接口形状一致）。
    """
    a = FakeAdapter(paths={"main": "/fake"}, infer_seconds=0.02)
    await a.load("cpu")
    flag = threading.Event()
    flag.set()  # 一开始就取消
    with pytest.raises(asyncio.CancelledError):
        await a.infer(_img_req(steps=10), cancel_flag=flag)


@pytest.mark.asyncio
async def test_infer_yields_to_event_loop():
    """infer 必须可让出 —— 否则 runner 的 pipe-reader 收不到调度（spec §4.4）.

    起一个并发的 sleep(0)，infer 跑 3 step（每 step sleep）期间它应能完成。
    """
    a = FakeAdapter(paths={"main": "/fake"}, infer_seconds=0.05)
    await a.load("cpu")
    other_ran = asyncio.Event()

    async def _other():
        await asyncio.sleep(0)
        other_ran.set()

    await asyncio.gather(a.infer(_img_req(steps=3)), _other())
    assert other_ran.is_set()
