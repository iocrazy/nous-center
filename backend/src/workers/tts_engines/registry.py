"""TTS 引擎注册表 —— 薄绑定到泛型 EngineRegistry(见 workers/engine_registry.py)。

模块级 register_engine/get_engine/list_engines + _ENGINE_CLASSES/_ENGINE_INSTANCES
别名保持既有 import 面不变(engines.py/tts.py 直接读 _ENGINE_INSTANCES)。
"""
from src.workers.engine_registry import EngineRegistry
from src.workers.tts_engines.base import TTSEngine

# TTS 实例化契约:model_path 包成 paths={"main": ...}(v2 adapter),忽略额外 kwargs。
_registry: EngineRegistry[TTSEngine] = EngineRegistry(
    "TTS",
    lambda cls, model_path, device, **kwargs: cls(
        paths={"main": model_path}, device=device
    ),
)

# 模块级别名(共享同一 dict/方法)。
_ENGINE_CLASSES = _registry.classes
_ENGINE_INSTANCES = _registry.instances
register_engine = _registry.register
get_engine = _registry.get
list_engines = _registry.list
