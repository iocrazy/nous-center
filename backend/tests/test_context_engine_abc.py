import inspect
import pytest

from src.services.context.base import (
    ContextEngine,
    ContextOverflowError,
)
from src.services.memory.base import PluginBase


def test_context_engine_subclass_of_plugin_base():
    assert issubclass(ContextEngine, PluginBase)


def test_compress_is_async():
    assert inspect.iscoroutinefunction(ContextEngine.compress)


def test_should_compress_is_sync():
    assert not inspect.iscoroutinefunction(ContextEngine.should_compress)


def test_context_overflow_error_exists():
    assert issubclass(ContextOverflowError, Exception)
    with pytest.raises(ContextOverflowError):
        raise ContextOverflowError("too big")
