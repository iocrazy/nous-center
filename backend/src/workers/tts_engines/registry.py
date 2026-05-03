from src.workers.tts_engines.base import TTSEngine

# Engine class registry - maps engine name to its class
_ENGINE_CLASSES: dict[str, type[TTSEngine]] = {}

# Loaded engine instances - singleton per engine
_ENGINE_INSTANCES: dict[str, TTSEngine] = {}


def register_engine(cls: type[TTSEngine]) -> type[TTSEngine]:
    """Decorator to register a TTS engine class."""
    # Create a temporary instance to get the name, then discard
    # Or we use a class-level attribute
    name = cls.ENGINE_NAME  # type: ignore[attr-defined]
    _ENGINE_CLASSES[name] = cls
    return cls


def get_engine(name: str, model_path: str, device: str = "cuda") -> TTSEngine:
    """Get or create a TTS engine instance.

    `model_path` is the legacy single-component path; wrapped into
    paths={"main": model_path} for the v2 adapter contract.
    """
    if name in _ENGINE_INSTANCES:
        return _ENGINE_INSTANCES[name]

    if name not in _ENGINE_CLASSES:
        raise ValueError(
            f"Unknown TTS engine: {name}. Available: {list(_ENGINE_CLASSES.keys())}"
        )

    engine = _ENGINE_CLASSES[name](paths={"main": model_path}, device=device)
    _ENGINE_INSTANCES[name] = engine
    return engine


def list_engines() -> list[str]:
    """List all registered engine names."""
    return list(_ENGINE_CLASSES.keys())
