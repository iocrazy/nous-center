"""LLM 引擎注册表 —— 薄绑定到泛型 EngineRegistry(见 workers/engine_registry.py)。

模块级 register_engine/get_engine/list_engines + _ENGINE_CLASSES/_ENGINE_INSTANCES
别名保持既有 import 面不变(engines.py/tts.py/openai_compat.py 直接读 _ENGINE_INSTANCES)。
"""
from src.workers.engine_registry import EngineRegistry
from src.workers.llm_engines.base import LLMEngine

# LLM 实例化契约:cls(model_path=..., device=..., **kwargs)。
_registry: EngineRegistry[LLMEngine] = EngineRegistry(
    "LLM",
    lambda cls, model_path, device, **kwargs: cls(
        model_path=model_path, device=device, **kwargs
    ),
)

# 模块级别名(共享同一 dict/方法)。
_ENGINE_CLASSES = _registry.classes
_ENGINE_INSTANCES = _registry.instances
register_engine = _registry.register
get_engine = _registry.get
list_engines = _registry.list
