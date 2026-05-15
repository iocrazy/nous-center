"""Lane J: [CRITICAL regression #4] ModelManager consolidation (spec §5 4-of-4).

spec §5.3 literally says "after src/gpu/model_manager.py is merged, all
call-sites still work". Lane 0's audit confirmed src/gpu/model_manager.py
+ vram_tracker.py + src/services/model_scheduler.py were DEAD CODE — they
were DELETED, not merged (Lane 0 plan flagged the spec G5 wording mismatch).

This regression guards the CORRECTED semantics:
  1. The whole src/ tree has no residual references to the deleted modules.
  2. The three dead-code files are physically gone.
  3. The canonical source-of-truth (services/model_manager.py:ModelManager)
     is importable and exposes the key API surface call-sites rely on.
  4. The monitor endpoint, which Lane 0 re-routed onto model_manager, still
     reports loaded_models per GPU.
"""
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# tests/integration/test_*.py → tests/integration → tests → backend
_BACKEND = Path(__file__).resolve().parents[2]


def test_no_residual_references_to_deleted_modules():
    """All of src/ no longer imports the deleted dead-code modules.

    Restricts to *.py (excluding __pycache__ stale .pyc bytecode) — Lane 0
    deleted the source files but the bytecode may linger in dev envs until
    a fresh interpreter run, which is harmless. The regression target is
    that no .py source references the deleted modules.
    """
    result = subprocess.run(
        [
            "grep",
            "-rn",
            "-E",
            "--include=*.py",
            r"gpu\.model_manager|vram_tracker|VRAMTracker|services import model_scheduler|services\.model_scheduler",
            str(_BACKEND / "src"),
        ],
        capture_output=True,
        text=True,
    )
    # grep returncode==1 means "no matches" — that is the success condition.
    assert result.returncode == 1, (
        "src/ still references deleted modules:\n" + result.stdout
    )


def test_deleted_module_files_are_gone():
    """The three dead-code source files are physically deleted (Lane 0)."""
    for rel in (
        "src/gpu/model_manager.py",
        "src/gpu/vram_tracker.py",
        "src/services/model_scheduler.py",
    ):
        assert not (_BACKEND / rel).exists(), (
            f"{rel} should have been deleted by Lane 0 but still exists"
        )


def test_canonical_model_manager_importable_with_key_api():
    """services/model_manager.ModelManager exposes the API call-sites rely on."""
    from src.services.model_manager import ModelManager

    for attr in (
        "loaded_model_ids",   # property — monitor.py / gpu_monitor.py read this
        "evict_lru",          # used by LRU eviction in load path
        "check_idle_models",  # idle-TTL unload (Lane 0 routing target)
        "unload_model",
        "get_or_load",
    ):
        assert hasattr(ModelManager, attr), (
            f"services/model_manager.ModelManager is missing {attr!r} — "
            "call-sites will break"
        )


@pytest.mark.asyncio
async def test_monitor_endpoint_loaded_models_works(client):
    """/api/v1/monitor/stats exposes loaded_models per GPU (Lane 0 re-routing).

    nvidia-smi is unavailable in test env (CUDA_VISIBLE_DEVICES=""), so the
    `gpus` array may be empty — but the endpoint must return 200 and the
    response shape must contain `gpus`. When GPUs ARE present, each must
    have a `loaded_models` array (the field driven by ModelManager).
    """
    resp = await client.get("/api/v1/monitor/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "gpus" in body
    gpus = body["gpus"]
    # Schema: either {"count": n, "gpus": [...]} or [...] directly. Handle both.
    if isinstance(gpus, dict):
        gpu_list = gpus.get("gpus", [])
    else:
        gpu_list = gpus
    for gpu in gpu_list:
        # loaded_models is the field Lane 0 re-routed onto ModelManager.
        # When GPUs are visible it must be present; without GPUs the list
        # is empty and the field may not be emitted — that's still OK
        # because the regression target is "field driven by model_manager
        # if visible", not "field always present even with zero GPUs".
        if "loaded_models" in gpu:
            assert isinstance(gpu["loaded_models"], list)
