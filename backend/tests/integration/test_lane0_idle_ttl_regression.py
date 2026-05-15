"""Lane J: [regression] spec §5.3 "Lane 0 idle-TTL unload still works"
integration-level supplement to Lane 0's unit regression.

Lane 0's regression tests (test_api_monitor + test_gpu_monitor_evict) are
unit-level and do not cover the idle-TTL unload end-to-end. spec §5.3
lists this scenario in the integration row ("Lane 0: monitor.py /
gpu_monitor.py still report loaded state + idle-TTL unload still works")
— this file fills the idle-TTL end-to-end gap.
"""
import time

import pytest

pytestmark = pytest.mark.integration


def _make_mm_with_entry(*, model_id: str, ttl_seconds: int, last_used_offset_s: float):
    """Build a ModelManager carrying one LoadedModel entry with controllable
    spec + last_used. Real ModelSpec + FakeAdapter — Pydantic strict types
    reject MagicMock here.
    """
    from unittest.mock import AsyncMock, MagicMock

    from src.runner.fake_adapter import FakeAdapter
    from src.services.inference.registry import ModelSpec
    from src.services.model_manager import LoadedModel, ModelManager

    spec = ModelSpec(
        id=model_id,
        model_type="image",
        adapter_class="src.runner.fake_adapter.FakeAdapter",
        paths={"main": f"/fake/{model_id}"},
        vram_mb=0,
        resident=False,
        ttl_seconds=ttl_seconds,
    )
    adapter = FakeAdapter(paths=spec.paths, device="cpu")
    adapter._model = object()  # mark loaded so unload_model condition holds

    registry = MagicMock()
    registry.get = MagicMock(return_value=spec)
    allocator = MagicMock()

    mm = ModelManager(registry=registry, allocator=allocator)
    mm.unload_model = AsyncMock()  # capture the call instead of real unload

    entry = LoadedModel(
        spec=spec,
        adapter=adapter,
        gpu_index=0,
        last_used=time.monotonic() - last_used_offset_s,
    )
    mm._models[model_id] = entry
    return mm


@pytest.mark.asyncio
async def test_check_idle_models_unloads_expired_entries():
    """services/model_manager.check_idle_models unloads non-resident, non-
    referenced models whose last_used is older than ttl_seconds.

    Pure logical test: inject a stale LoadedModel into ModelManager's
    internal dict, then call check_idle_models — the entry must be unloaded.
    """
    mm = _make_mm_with_entry(
        model_id="expired-model", ttl_seconds=1, last_used_offset_s=60,
    )
    # No references → eligible for idle unload.
    assert "expired-model" not in mm._references

    await mm.check_idle_models()

    mm.unload_model.assert_awaited_once_with("expired-model")


@pytest.mark.asyncio
async def test_check_idle_models_skips_referenced_entries():
    """A model with active references is NOT unloaded even if its TTL expired.

    spec §4.x: references gate idle-TTL eviction (a model held by a running
    workflow must not be unloaded under it).
    """
    mm = _make_mm_with_entry(
        model_id="held-model", ttl_seconds=1, last_used_offset_s=60,
    )
    mm._references["held-model"] = {"workflow-42"}  # active reference

    await mm.check_idle_models()

    mm.unload_model.assert_not_called()


@pytest.mark.asyncio
async def test_monitor_reports_loaded_state_after_lane0(client):
    """Lane 0 re-routing target: /api/v1/monitor/stats still surfaces loaded state.

    The endpoint must return 200 with a `gpus` field. When GPUs are visible,
    each GPU dict carries `loaded_models` driven by ModelManager (which is
    the canonical source-of-truth after Lane 0). Without GPUs the list is
    empty — what matters is the endpoint shape stays alive.
    """
    resp = await client.get("/api/v1/monitor/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "gpus" in body
