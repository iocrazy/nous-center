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


from datetime import datetime, timezone


class _FakeMemoryProvider(MemoryProvider):
    """In-memory MemoryProvider for contract test runner. Not production-ready."""
    name = "fake"

    def __init__(self):
        self._store: list[dict] = []
        self._next_id = 1
        self._simulate_fail = False

    async def initialize(self):
        pass

    async def add_entries(self, *, instance_id, api_key_id, entries, context_key=None):
        if not entries:
            return []
        if len(entries) > 100:
            raise MemoryProviderClientError("batch > 100")
        for i, e in enumerate(entries):
            if len(e["content"].encode()) > 10_240:
                raise MemoryProviderClientError(f"entries[{i}].content > 10KB")
        if self._simulate_fail:
            raise MemoryProviderInternalError("simulated db fail")
        ids: list[int] = []
        for e in entries:
            row = {
                "id": self._next_id,
                "instance_id": instance_id,
                "api_key_id": api_key_id,
                "category": e["category"],
                "content": e["content"],
                "context_key": e.get("context_key") if e.get("context_key") is not None else context_key,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._store.append(row)
            ids.append(self._next_id)
            self._next_id += 1
        return ids

    async def prefetch(self, *, instance_id, query, limit=10, context_key=None):
        if self._simulate_fail:
            return []  # best-effort swallow
        results = [r for r in self._store if r["instance_id"] == instance_id]
        if context_key:
            results = [r for r in results if r["context_key"] == context_key]
        if query:
            results = [r for r in results if query in r["content"]]
        return [StoredMemoryEntry(**r) for r in results[:limit]]


class AbstractMemoryProviderTests:
    """Contract tests any MemoryProvider must pass.

    Subclass and override `provider_factory()` to return an instance.
    """

    def provider_factory(self) -> MemoryProvider:
        raise NotImplementedError

    @pytest.fixture
    async def provider(self):
        p = self.provider_factory()
        await p.initialize()
        yield p
        await p.shutdown()

    @pytest.mark.asyncio
    async def test_add_entries_empty_list_idempotent(self, provider):
        assert await provider.add_entries(
            instance_id=1, api_key_id=None, entries=[]
        ) == []

    @pytest.mark.asyncio
    async def test_add_entries_returns_ids(self, provider):
        ids = await provider.add_entries(
            instance_id=1,
            api_key_id=None,
            entries=[{"category": "preference", "content": "short replies", "context_key": None}],
        )
        assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_add_entries_batch_over_100_raises(self, provider):
        big = [{"category": "fact", "content": "x", "context_key": None}] * 101
        with pytest.raises(MemoryProviderClientError):
            await provider.add_entries(instance_id=1, api_key_id=None, entries=big)

    @pytest.mark.asyncio
    async def test_add_entries_content_over_10kb_raises(self, provider):
        big_content = "x" * 11_000
        with pytest.raises(MemoryProviderClientError):
            await provider.add_entries(
                instance_id=1, api_key_id=None,
                entries=[{"category": "fact", "content": big_content, "context_key": None}],
            )

    @pytest.mark.asyncio
    async def test_prefetch_returns_matching(self, provider):
        await provider.add_entries(
            instance_id=1, api_key_id=None,
            entries=[
                {"category": "preference", "content": "simple replies", "context_key": None},
                {"category": "fact", "content": "lives in Tokyo", "context_key": None},
            ],
        )
        results = await provider.prefetch(instance_id=1, query="Tokyo", limit=10)
        assert len(results) == 1
        assert "Tokyo" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_cross_instance_isolation(self, provider):
        await provider.add_entries(
            instance_id=1, api_key_id=None,
            entries=[{"category": "preference", "content": "foo", "context_key": None}],
        )
        results = await provider.prefetch(instance_id=2, query="foo")
        assert results == []

    @pytest.mark.asyncio
    async def test_context_key_filters_prefetch(self, provider):
        await provider.add_entries(
            instance_id=1, api_key_id=None,
            entries=[
                {"category": "fact", "content": "project alpha note", "context_key": "alpha"},
                {"category": "fact", "content": "project beta note", "context_key": "beta"},
            ],
        )
        r = await provider.prefetch(instance_id=1, query="project", context_key="alpha")
        assert len(r) == 1
        assert r[0]["context_key"] == "alpha"

    @pytest.mark.asyncio
    async def test_prefetch_swallows_internal_error(self, provider):
        # Test that best-effort swallow works when implementations inject fail
        if hasattr(provider, "_simulate_fail"):
            provider._simulate_fail = True
            results = await provider.prefetch(instance_id=1, query="anything")
            assert results == []
            provider._simulate_fail = False


class TestFakeMemoryProviderContract(AbstractMemoryProviderTests):
    """Verify the abstract contract test itself works with the Fake impl."""

    def provider_factory(self):
        return _FakeMemoryProvider()
