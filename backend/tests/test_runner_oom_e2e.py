"""Lane D: 真 GPU OOM 路径 e2e 测试（spec §5.4）。

@pytest.mark.e2e —— CI skip。dev box 手动跑：
    cd backend && python -m pytest tests/test_runner_oom_e2e.py -m e2e -v

需要：真 GPU + 一份配了「故意超 VRAM」模型的 models.yaml（见下方 SETUP）。
这条测试验证 ModelManager.get_or_load 在真 CUDA OOM 下的 evict-retry-fail 行为
—— 单测用 FakeOOMError 模拟过，但真 torch.cuda.OutOfMemoryError 的类名 / 行为
要在真硬件上确认一次。

SETUP:
准备一份 models.yaml，含三个条目 —— `filler`（一个能占住目标 GPU 大部分显存的真模型）、
`oversized`（剩余显存绝对放不下的模型）、`normal`（正常尺寸 image 模型）。
`export NOUS_OOM_TEST_YAML=/path/to/oom-test-models.yaml` 后 `pytest -m e2e`。
"""
import os
from pathlib import Path

import pytest

# CI 环境 CUDA_VISIBLE_DEVICES="" —— 整个文件在无 GPU 时 skip。
pytestmark = pytest.mark.e2e

_OOM_YAML = os.getenv("NOUS_OOM_TEST_YAML", "")


@pytest.mark.skipif(
    not _OOM_YAML or not Path(_OOM_YAML).exists(),
    reason="set NOUS_OOM_TEST_YAML to a models.yaml with an oversized model",
)
@pytest.mark.asyncio
async def test_real_oom_evict_retry_then_load_failed():
    """真 GPU：先占满显存，再 load 一个放不下的模型 →
    get_or_load evict 一次重试 → 仍 OOM → load_failed."""
    from src.errors import ModelLoadError
    from src.runner.runner_modelmanager import build_runner_model_manager

    # fake_adapter=False —— 走真 adapter / 真权重
    mm = build_runner_model_manager(
        group_id="image", gpus=[0], models_yaml_path=_OOM_YAML, fake_adapter=False,
    )
    # SETUP 约定：yaml 里 'filler' 模型先占住显存，'oversized' 放不下。
    await mm.get_or_load("filler")
    assert "filler" in mm.loaded_model_ids

    with pytest.raises(ModelLoadError):
        await mm.get_or_load("oversized")
    # filler 被 evict（get_or_load 的第一次 OOM 触发）
    assert "filler" not in mm.loaded_model_ids
    # oversized 落 _load_failures
    assert "oversized" in mm._load_failures


@pytest.mark.skipif(
    not _OOM_YAML or not Path(_OOM_YAML).exists(),
    reason="set NOUS_OOM_TEST_YAML to a models.yaml with a normal-sized model",
)
@pytest.mark.asyncio
async def test_real_model_loads_and_infers_in_runner_mm():
    """真 GPU sanity：runner ModelManager 能 load 一个真 image 模型并 infer。"""
    from src.runner.runner_modelmanager import build_runner_model_manager
    from src.services.inference.base import ImageRequest

    mm = build_runner_model_manager(
        group_id="image", gpus=[0], models_yaml_path=_OOM_YAML, fake_adapter=False,
    )
    adapter = await mm.get_or_load("normal")  # yaml 约定的正常尺寸模型
    assert adapter.is_loaded
    result = await adapter.infer(ImageRequest(
        request_id="e2e-1", prompt="a red cube", steps=4, width=512, height=512,
    ))
    assert result.media_type.startswith("image/")
    assert result.data
