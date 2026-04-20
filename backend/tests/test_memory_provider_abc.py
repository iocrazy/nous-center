# backend/tests/test_memory_provider_abc.py
import inspect

import pytest

from src.services.memory.base import (
    MemoryEntry,
    MemoryProvider,
    MemoryProviderClientError,
    MemoryProviderError,
    MemoryProviderInternalError,
    PluginBase,
    StoredMemoryEntry,
)


def test_plugin_base_is_abstract():
    from abc import ABC
    assert issubclass(PluginBase, ABC)


def test_memory_provider_subclass_of_plugin_base():
    assert issubclass(MemoryProvider, PluginBase)


def test_system_prompt_block_is_async():
    sig = inspect.iscoroutinefunction(PluginBase.system_prompt_block)
    assert sig is True


def test_plugin_base_default_system_prompt_block_returns_empty():
    class _Impl(PluginBase):
        async def initialize(self):
            pass
    import asyncio
    result = asyncio.run(_Impl().system_prompt_block(instance_id=1))
    assert result == ""


def test_error_hierarchy():
    assert issubclass(MemoryProviderClientError, MemoryProviderError)
    assert issubclass(MemoryProviderInternalError, MemoryProviderError)
    assert not issubclass(MemoryProviderClientError, MemoryProviderInternalError)


def test_memory_entry_typed_dict_keys():
    entry: MemoryEntry = {
        "category": "preference",
        "content": "user likes brief replies",
        "context_key": None,
    }
    assert entry["category"] == "preference"
