"""Lane D: build_runner_model_manager 工厂 —— runner 子进程内构造独立 ModelManager."""
from pathlib import Path

import pytest

from src.runner.fake_adapter import FakeAdapter
from src.runner.runner_modelmanager import build_runner_model_manager
from src.services.model_manager import ModelManager

_FIXTURE = str(Path(__file__).parent / "fixtures" / "runner_models.yaml")


def test_factory_returns_model_manager():
    mm = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    assert isinstance(mm, ModelManager)


def test_factory_registry_has_fixture_specs():
    """registry 从 yaml fixture 读到两个 fake image spec。"""
    mm = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    assert mm._registry.get("fake-img-a") is not None
    assert mm._registry.get("fake-img-b") is not None


@pytest.mark.asyncio
async def test_factory_fake_mode_loads_fake_adapter():
    """fake_adapter=True —— load_model 出来的 adapter 是 FakeAdapter（不碰真权重）。"""
    mm = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    adapter = await mm.get_or_load("fake-img-a")
    assert isinstance(adapter, FakeAdapter)
    assert adapter.is_loaded


@pytest.mark.asyncio
async def test_factory_each_call_is_independent_instance():
    """spec §4.5：每个 runner 一个独立 ModelManager —— 两次 build 互不共享状态。"""
    mm1 = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    mm2 = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    assert mm1 is not mm2
    await mm1.get_or_load("fake-img-a")
    assert "fake-img-a" in mm1.loaded_model_ids
    assert "fake-img-a" not in mm2.loaded_model_ids  # 状态不共享
