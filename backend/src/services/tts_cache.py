"""TTS result caching via Redis.

Cache key = SHA-256 of (text + engine + voice + speed + sample_rate + emotion).
Value = base64-encoded audio bytes.
"""

import hashlib
import json

from src.config import get_settings


def make_cache_key(
    text: str,
    engine: str,
    voice: str = "default",
    speed: float = 1.0,
    sample_rate: int = 24000,
    emotion: str | None = None,
) -> str:
    payload = json.dumps(
        {"text": text, "engine": engine, "voice": voice, "speed": speed, "sr": sample_rate, "emotion": emotion},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"tts:{digest}"


class TTSCacheService:
    """Thin wrapper around Redis for TTS audio caching."""

    def __init__(self, redis_client, ttl: int | None = None):
        self._redis = redis_client
        self._ttl = ttl or get_settings().CACHE_TTL_SECONDS

    async def get(self, key: str) -> str | None:
        """Return cached base64 audio or None."""
        return await self._redis.get(key)

    async def set(self, key: str, audio_base64: str) -> None:
        """Cache base64 audio with TTL."""
        await self._redis.set(key, audio_base64, ex=self._ttl)
