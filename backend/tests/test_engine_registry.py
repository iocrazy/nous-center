"""泛型 EngineRegistry —— 合并 llm/tts 双胞胎 registry(架构 #4 冗余)。
两边只有实例化那行不同,其余(classes/instances/register/get 单例/list)完全重复。"""
import pytest

from src.workers.engine_registry import EngineRegistry


class _FakeEngine:
    ENGINE_NAME = "fake"
    def __init__(self, **kw):
        self.kw = kw


def test_register_get_singleton_list():
    reg = EngineRegistry("TEST", lambda cls, mp, dev, **kw: cls(model_path=mp, device=dev, **kw))
    reg.register(_FakeEngine)
    assert reg.list() == ["fake"]
    e1 = reg.get("fake", "/m", "cuda")
    e2 = reg.get("fake", "/m", "cuda")
    assert e1 is e2  # 单例
    assert e1.kw == {"model_path": "/m", "device": "cuda"}


def test_unknown_raises():
    reg = EngineRegistry("TEST", lambda cls, mp, dev, **kw: cls())
    with pytest.raises(ValueError, match="Unknown TEST engine"):
        reg.get("nope", "/m")


def test_instantiate_callable_shapes_tts_vs_llm():
    """TTS 把 model_path 包成 paths={'main':...},LLM 直接传 model_path= —— 实例化
    可注入,证明泛型能覆盖两种契约。"""
    captured = {}
    tts = EngineRegistry("TTS", lambda cls, mp, dev, **kw: captured.update(paths={"main": mp}, device=dev) or cls())
    tts.register(_FakeEngine)
    tts.get("fake", "/tts")
    assert captured == {"paths": {"main": "/tts"}, "device": "cuda"}


def test_module_aliases_share_dict():
    """两个 registry 模块暴露的 _ENGINE_INSTANCES/_ENGINE_CLASSES 必须是 registry
    内部同一 dict(engines.py 等直接读它)。"""
    from src.workers.llm_engines import registry as llm_reg
    from src.workers.tts_engines import registry as tts_reg
    assert llm_reg._ENGINE_CLASSES is llm_reg._registry.classes
    assert llm_reg._ENGINE_INSTANCES is llm_reg._registry.instances
    assert tts_reg._ENGINE_CLASSES is tts_reg._registry.classes
