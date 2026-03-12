from unittest.mock import AsyncMock

from src.services.tts_cache import make_cache_key, TTSCacheService


def test_make_cache_key_deterministic():
    key1 = make_cache_key(text="hello", engine="cosyvoice2", voice="default", speed=1.0, sample_rate=24000)
    key2 = make_cache_key(text="hello", engine="cosyvoice2", voice="default", speed=1.0, sample_rate=24000)
    assert key1 == key2
    assert key1.startswith("tts:")


def test_make_cache_key_differs_on_text():
    key1 = make_cache_key(text="hello", engine="cosyvoice2", voice="default", speed=1.0, sample_rate=24000)
    key2 = make_cache_key(text="world", engine="cosyvoice2", voice="default", speed=1.0, sample_rate=24000)
    assert key1 != key2



async def test_cache_service_get_miss():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    svc = TTSCacheService(mock_redis, ttl=3600)
    result = await svc.get("tts:nonexistent")
    assert result is None
    mock_redis.get.assert_called_once_with("tts:nonexistent")


async def test_cache_service_set():
    mock_redis = AsyncMock()
    svc = TTSCacheService(mock_redis, ttl=3600)
    await svc.set("tts:key", "base64data")
    mock_redis.set.assert_called_once_with("tts:key", "base64data", ex=3600)
