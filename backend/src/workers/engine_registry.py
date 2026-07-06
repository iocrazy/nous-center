"""泛型引擎注册表 —— 合并 llm/tts 双胞胎 registry(架构 #4 冗余去重)。

llm_engines/registry.py 与 tts_engines/registry.py 原是近乎逐行相同的两份拷贝:
同样的 classes/instances dict + register 装饰器 + get 单例 + list。唯一区别是
`get` 里的实例化契约(LLM 传 model_path=+kwargs;TTS 包成 paths={"main":...})。
抽成泛型,实例化用可注入 callable 覆盖两种契约。两个模块留薄绑定 + 模块级别名
(_ENGINE_CLASSES/_ENGINE_INSTANCES),兼容 engines.py 等直接读内部 dict 的代码。
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

E = TypeVar("E")


class EngineRegistry(Generic[E]):
    def __init__(self, kind: str, instantiate: Callable[..., E]) -> None:
        self.kind = kind
        # name → 引擎类 / 单例实例(模块级别名指向这两个 dict,共享同一对象)。
        self.classes: dict[str, type[E]] = {}
        self.instances: dict[str, E] = {}
        # (cls, model_path, device, **kwargs) -> E —— 覆盖 LLM/TTS 两种实例化契约。
        self._instantiate = instantiate

    def register(self, cls: type[E]) -> type[E]:
        """装饰器:按 cls.ENGINE_NAME 注册引擎类。"""
        name = cls.ENGINE_NAME  # type: ignore[attr-defined]
        self.classes[name] = cls
        return cls

    def get(self, name: str, model_path: str, device: str = "cuda", **kwargs) -> E:
        """取或创建引擎单例(每 name 一个)。"""
        if name in self.instances:
            return self.instances[name]
        if name not in self.classes:
            raise ValueError(
                f"Unknown {self.kind} engine: {name}. Available: {list(self.classes.keys())}"
            )
        engine = self._instantiate(self.classes[name], model_path, device, **kwargs)
        self.instances[name] = engine
        return engine

    def list(self) -> list[str]:
        return list(self.classes.keys())
