"""Unified model scheduler — reference counting + idle timeout."""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import load_model_configs, get_settings
from src.gpu.detector import get_device_for_engine

logger = logging.getLogger(__name__)

IDLE_TIMEOUT_SECONDS = 300  # 5 minutes

# Reference counts: model_key -> set of workflow_ids
_references: dict[str, set[str]] = defaultdict(set)

# Last used timestamp for idle timeout
_last_used: dict[str, float] = {}

# Track which models are loaded
_loaded_models: set[str] = set()


def get_model_dependencies(workflow: dict) -> list[dict]:
    """Extract model dependencies from workflow nodes."""
    configs = load_model_configs()
    deps = []
    seen: set[str] = set()
    for node in workflow.get("nodes", []):
        node_type = node.get("type", "")
        data = node.get("data", {})

        model_key = None
        if node_type == "tts_engine":
            model_key = data.get("engine")
        elif node_type == "llm":
            model_key = data.get("model_key")  # references models.yaml key
        elif node_type == "agent":
            # Agent's model resolved later
            pass

        if model_key and model_key in configs and model_key not in seen:
            seen.add(model_key)
            deps.append({"key": model_key, **configs[model_key]})

    return deps


async def load_model(model_key: str) -> None:
    """Load a model by key (TTS or LLM)."""
    if model_key in _loaded_models:
        _last_used[model_key] = time.time()
        return

    configs = load_model_configs()
    cfg = configs.get(model_key)
    if not cfg:
        raise ValueError(f"Unknown model: {model_key}")

    settings = get_settings()
    model_type = cfg.get("type", "")
    device = get_device_for_engine(cfg)

    if model_type == "tts":
        # Ensure engine classes are registered
        import src.workers.tts_engines.cosyvoice2  # noqa: F401
        import src.workers.tts_engines.indextts2  # noqa: F401
        import src.workers.tts_engines.qwen3_tts  # noqa: F401
        import src.workers.tts_engines.moss_tts  # noqa: F401
        from src.workers.tts_engines.registry import get_engine

        local_path = cfg.get("local_path", "")
        model_path = Path(settings.LOCAL_MODELS_PATH) / local_path
        engine = get_engine(model_key, model_path=str(model_path), device=device)
        if not engine.is_loaded:
            await asyncio.to_thread(engine.load)

    elif model_type == "llm":
        from src.workers.llm_engines.registry import get_engine

        local_path = cfg.get("local_path", "")
        model_path = Path(settings.LOCAL_MODELS_PATH) / local_path
        engine = get_engine(
            model_key,
            model_path=str(model_path),
            device=device,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            gpu=cfg.get("gpu"),
        )
        if not engine.is_loaded:
            await asyncio.to_thread(engine.load)

    else:
        raise ValueError(f"Unsupported model type: {model_type} (only tts/llm supported)")

    _loaded_models.add(model_key)
    _last_used[model_key] = time.time()
    logger.info("Model loaded: %s on %s", model_key, device)


async def unload_model(model_key: str, force: bool = False) -> None:
    """Unload a model if no references remain."""
    if not force and _references[model_key]:
        logger.info(
            "Model %s still referenced by %s, skipping unload",
            model_key,
            _references[model_key],
        )
        return

    configs = load_model_configs()
    cfg = configs.get(model_key, {})

    if not force and cfg.get("resident", False):
        logger.info("Model %s is resident, skipping unload", model_key)
        return

    model_type = cfg.get("type", "")

    if model_type == "tts":
        from src.workers.tts_engines import registry

        engine = registry._ENGINE_INSTANCES.get(model_key)
        if engine and engine.is_loaded:
            engine.unload()
    elif model_type == "llm":
        from src.workers.llm_engines import registry

        engine = registry._ENGINE_INSTANCES.get(model_key)
        if engine and engine.is_loaded:
            engine.unload()

    _loaded_models.discard(model_key)
    _last_used.pop(model_key, None)
    logger.info("Model unloaded: %s", model_key)


def add_reference(model_key: str, workflow_id: str) -> None:
    """Track that a workflow references this model."""
    _references[model_key].add(workflow_id)


def remove_reference(model_key: str, workflow_id: str) -> None:
    """Remove workflow reference to model."""
    _references[model_key].discard(workflow_id)


def get_status() -> dict:
    """Return current model scheduler status."""
    return {
        "loaded": list(_loaded_models),
        "references": {k: list(v) for k, v in _references.items() if v},
        "last_used": {k: v for k, v in _last_used.items()},
    }


async def check_idle_models() -> None:
    """Unload models that have been idle too long and have no references."""
    now = time.time()
    to_unload = []
    for model_key in list(_loaded_models):
        if model_key not in _last_used:
            continue
        if _references[model_key]:
            continue  # Still referenced
        configs = load_model_configs()
        cfg = configs.get(model_key, {})
        if cfg.get("resident", False):
            continue
        if now - _last_used[model_key] > IDLE_TIMEOUT_SECONDS:
            to_unload.append(model_key)

    for key in to_unload:
        logger.info("Idle timeout: unloading %s", key)
        await unload_model(key)


def get_llm_base_url(model_key: str) -> str | None:
    """Get the base URL for a loaded LLM model."""
    from src.workers.llm_engines import registry

    engine = registry._ENGINE_INSTANCES.get(model_key)
    if engine and engine.is_loaded:
        return engine.base_url
    return None
