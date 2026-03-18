import time
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from src.services.model_scheduler import (
    get_model_dependencies,
    add_reference,
    remove_reference,
    get_status,
    get_llm_base_url,
    check_idle_models,
    load_model,
    unload_model,
    _references,
    _loaded_models,
    _last_used,
    IDLE_TIMEOUT_SECONDS,
)


@pytest.fixture(autouse=True)
def clear_state():
    _references.clear()
    _loaded_models.clear()
    _last_used.clear()
    yield
    _references.clear()
    _loaded_models.clear()
    _last_used.clear()


def test_get_model_dependencies():
    with patch("src.services.model_scheduler.load_model_configs") as mock:
        mock.return_value = {
            "cosyvoice2": {"type": "tts", "gpu": 1, "vram_gb": 3},
            "qwen3-4b": {"type": "llm", "gpu": 0, "vram_gb": 4},
        }
        wf = {
            "nodes": [
                {"type": "tts_engine", "data": {"engine": "cosyvoice2"}},
                {"type": "llm", "data": {"model_key": "qwen3-4b"}},
            ]
        }
        deps = get_model_dependencies(wf)
    assert len(deps) == 2
    keys = {d["key"] for d in deps}
    assert keys == {"cosyvoice2", "qwen3-4b"}


def test_get_model_dependencies_dedup():
    """Duplicate model_key in multiple nodes should only appear once."""
    with patch("src.services.model_scheduler.load_model_configs") as mock:
        mock.return_value = {
            "cosyvoice2": {"type": "tts", "gpu": 1, "vram_gb": 3},
        }
        wf = {
            "nodes": [
                {"type": "tts_engine", "data": {"engine": "cosyvoice2"}},
                {"type": "tts_engine", "data": {"engine": "cosyvoice2"}},
            ]
        }
        deps = get_model_dependencies(wf)
    assert len(deps) == 1


def test_get_model_dependencies_unknown_model_skipped():
    """Models not in config are skipped."""
    with patch("src.services.model_scheduler.load_model_configs") as mock:
        mock.return_value = {}
        wf = {"nodes": [{"type": "llm", "data": {"model_key": "nonexistent"}}]}
        deps = get_model_dependencies(wf)
    assert len(deps) == 0


def test_reference_counting():
    add_reference("model_a", "wf_1")
    add_reference("model_a", "wf_2")
    assert len(_references["model_a"]) == 2

    remove_reference("model_a", "wf_1")
    assert len(_references["model_a"]) == 1

    remove_reference("model_a", "wf_2")
    assert len(_references["model_a"]) == 0


def test_reference_counting_idempotent():
    """Adding same reference twice is a no-op (set semantics)."""
    add_reference("model_a", "wf_1")
    add_reference("model_a", "wf_1")
    assert len(_references["model_a"]) == 1

    remove_reference("model_a", "wf_1")
    assert len(_references["model_a"]) == 0


def test_remove_nonexistent_reference():
    """Removing a reference that doesn't exist should not raise."""
    remove_reference("model_a", "wf_99")
    assert len(_references["model_a"]) == 0


def test_get_status():
    _loaded_models.add("model_x")
    add_reference("model_x", "wf_1")
    _last_used["model_x"] = 12345.0

    status = get_status()
    assert "model_x" in status["loaded"]
    assert "wf_1" in status["references"]["model_x"]
    assert status["last_used"]["model_x"] == 12345.0


def test_get_status_empty():
    status = get_status()
    assert status["loaded"] == []
    assert status["references"] == {}
    assert status["last_used"] == {}


def test_get_llm_base_url_not_loaded():
    with patch("src.workers.llm_engines.registry._ENGINE_INSTANCES", {}):
        assert get_llm_base_url("nonexistent") is None


def test_get_llm_base_url_loaded():
    mock_engine = MagicMock()
    mock_engine.is_loaded = True
    mock_engine.base_url = "http://127.0.0.1:9999"
    with patch(
        "src.workers.llm_engines.registry._ENGINE_INSTANCES",
        {"test_model": mock_engine},
    ):
        assert get_llm_base_url("test_model") == "http://127.0.0.1:9999"


async def test_check_idle_models_unloads_expired():
    """Models past idle timeout with no references should be unloaded."""
    _loaded_models.add("idle_model")
    _last_used["idle_model"] = time.time() - IDLE_TIMEOUT_SECONDS - 10

    with patch("src.services.model_scheduler.load_model_configs") as mock_cfg:
        mock_cfg.return_value = {
            "idle_model": {"type": "tts", "resident": False},
        }
        with patch("src.services.model_scheduler.unload_model", new_callable=AsyncMock) as mock_unload:
            await check_idle_models()
            mock_unload.assert_called_once_with("idle_model")


async def test_check_idle_models_skips_referenced():
    """Models with active references should not be unloaded."""
    _loaded_models.add("referenced_model")
    _last_used["referenced_model"] = time.time() - IDLE_TIMEOUT_SECONDS - 10
    add_reference("referenced_model", "wf_1")

    with patch("src.services.model_scheduler.load_model_configs") as mock_cfg:
        mock_cfg.return_value = {
            "referenced_model": {"type": "tts", "resident": False},
        }
        with patch("src.services.model_scheduler.unload_model", new_callable=AsyncMock) as mock_unload:
            await check_idle_models()
            mock_unload.assert_not_called()


async def test_check_idle_models_skips_resident():
    """Resident models should not be unloaded."""
    _loaded_models.add("resident_model")
    _last_used["resident_model"] = time.time() - IDLE_TIMEOUT_SECONDS - 10

    with patch("src.services.model_scheduler.load_model_configs") as mock_cfg:
        mock_cfg.return_value = {
            "resident_model": {"type": "tts", "resident": True},
        }
        with patch("src.services.model_scheduler.unload_model", new_callable=AsyncMock) as mock_unload:
            await check_idle_models()
            mock_unload.assert_not_called()


async def test_load_model_already_loaded():
    """Loading an already-loaded model should just update last_used."""
    _loaded_models.add("already_loaded")
    old_time = time.time() - 100
    _last_used["already_loaded"] = old_time

    await load_model("already_loaded")
    assert _last_used["already_loaded"] > old_time


async def test_load_model_unknown():
    """Loading an unknown model should raise ValueError."""
    with patch("src.services.model_scheduler.load_model_configs", return_value={}):
        with pytest.raises(ValueError, match="Unknown model"):
            await load_model("nonexistent")


async def test_unload_model_skips_referenced():
    """Unloading a model with active references (non-force) should skip."""
    _loaded_models.add("ref_model")
    add_reference("ref_model", "wf_1")

    with patch("src.services.model_scheduler.load_model_configs") as mock_cfg:
        mock_cfg.return_value = {"ref_model": {"type": "tts"}}
        await unload_model("ref_model", force=False)

    assert "ref_model" in _loaded_models  # still loaded


async def test_unload_model_force():
    """Force-unloading a model should work even with references."""
    _loaded_models.add("force_model")
    _last_used["force_model"] = time.time()
    add_reference("force_model", "wf_1")

    with patch("src.services.model_scheduler.load_model_configs") as mock_cfg:
        mock_cfg.return_value = {"force_model": {"type": "tts"}}
        with patch("src.workers.tts_engines.registry._ENGINE_INSTANCES", {}):
            await unload_model("force_model", force=True)

    assert "force_model" not in _loaded_models
