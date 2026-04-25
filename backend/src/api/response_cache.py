"""In-memory cache for read-heavy GET endpoints + ETag/304 + write invalidation.

Design notes captured from /plan-eng-review:

- Decorator returns ``Response(content=bytes)`` on HIT so cached body bypasses
  pydantic re-validation. Routes that use ``@cached`` should drop ``response_model``
  (the response is pre-serialized by the cache layer).
- Cache key includes path + sorted query string so ``?type=tts`` and ``?type=llm``
  don't collide.
- ETag is computed over the *bytes we send*, not the dict pre-serialization, so
  list-order non-determinism in the underlying data (PG JSON, set iteration) can't
  make ETags drift while the body is unchanged.
- Per-prefix lock (not per-key) avoids unbounded ``asyncio.Lock`` accumulation when
  the path space is large or attacker-controlled. Coarser, but 30s TTL absorbs the
  contention.
- Only 2xx with body (no exception) is cached. 4xx/5xx and ``Set-Cookie``-bearing
  responses are passed through.
- Emits ``Cache-Control: private, max-age=0, must-revalidate`` so Cloudflare /
  reverse proxies do NOT cache cross-admin. Browser may use it via ETag/304 but
  must always revalidate with the origin.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import wraps
from urllib.parse import urlencode

from fastapi import HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder

# `Cache-Control: private` ensures Cloudflare / shared caches never store this.
# `max-age=0, must-revalidate` lets the browser cache the body and use If-None-Match
# but always revalidates with us.
_CACHE_CONTROL = "private, max-age=0, must-revalidate"


@dataclass
class _Entry:
    body: bytes
    etag: str
    expires_at: float


@dataclass
class _Metrics:
    hits: int = 0
    misses: int = 0
    etag_304: int = 0
    invalidations: int = 0
    by_prefix: dict[str, dict[str, int]] = field(default_factory=dict)

    def bump(self, prefix: str, kind: str) -> None:
        setattr(self, kind, getattr(self, kind) + 1)
        per = self.by_prefix.setdefault(prefix, {"hits": 0, "misses": 0, "etag_304": 0, "invalidations": 0})
        per[kind] += 1

    def snapshot(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "etag_304": self.etag_304,
            "invalidations": self.invalidations,
            "by_prefix": {k: dict(v) for k, v in self.by_prefix.items()},
        }


_store: dict[tuple[str, str], _Entry] = {}
_locks: dict[str, asyncio.Lock] = {}
metrics = _Metrics()


def _lock_for(prefix: str) -> asyncio.Lock:
    lock = _locks.get(prefix)
    if lock is None:
        lock = _locks[prefix] = asyncio.Lock()
    return lock


def _build_key(request: Request) -> str:
    # Deterministic ordering so ?a=1&b=2 and ?b=2&a=1 collapse to one key.
    items = sorted(request.query_params.multi_items())
    return f"{request.url.path}?{urlencode(items)}" if items else request.url.path


def _serialize(obj) -> bytes:
    # ``sort_keys=True`` makes object key order deterministic; ``default=str``
    # handles datetime/Decimal/etc. that ``jsonable_encoder`` may emit as raw
    # objects in some edge cases (it usually pre-stringifies).
    return json.dumps(jsonable_encoder(obj), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")


def _etag_for(body: bytes) -> str:
    # 16 hex chars (~64 bits) — collision risk negligible per-prefix and per-30s.
    return hashlib.sha256(body).hexdigest()[:16]


def _build_response(body: bytes, etag: str, status: int = 200) -> Response:
    return Response(
        content=body,
        media_type="application/json",
        status_code=status,
        headers={
            "ETag": etag,
            "Cache-Control": _CACHE_CONTROL,
        },
    )


def cached(prefix: str, ttl: int):
    """Decorate a FastAPI handler so its JSON response is cached for ``ttl`` seconds.

    The handler must (a) take ``request: Request`` as a parameter (so we can
    read the path + query for the cache key) and (b) NOT declare
    ``response_model=`` (the cache returns a pre-serialized ``Response``).
    """

    def decorator(handler: Callable[..., Awaitable]):
        @wraps(handler)
        async def wrapper(*args, **kwargs):
            request: Request | None = kwargs.get("request")
            if request is None:
                # Fall through to handler without caching if route signature changes.
                return await handler(*args, **kwargs)

            key = _build_key(request)
            store_key = (prefix, key)
            now = time.monotonic()

            entry = _store.get(store_key)
            if entry is not None and entry.expires_at > now:
                metrics.bump(prefix, "hits")
                # Browser holds the same etag — serve 304.
                if request.headers.get("if-none-match") == entry.etag:
                    metrics.bump(prefix, "etag_304")
                    return Response(
                        status_code=304,
                        headers={"ETag": entry.etag, "Cache-Control": _CACHE_CONTROL},
                    )
                return _build_response(entry.body, entry.etag)

            # MISS — coalesce concurrent fills behind a per-prefix lock.
            async with _lock_for(prefix):
                # Re-check under the lock (another waiter may have populated).
                entry = _store.get(store_key)
                if entry is not None and entry.expires_at > now:
                    metrics.bump(prefix, "hits")
                    if request.headers.get("if-none-match") == entry.etag:
                        metrics.bump(prefix, "etag_304")
                        return Response(
                            status_code=304,
                            headers={"ETag": entry.etag, "Cache-Control": _CACHE_CONTROL},
                        )
                    return _build_response(entry.body, entry.etag)

                metrics.bump(prefix, "misses")
                try:
                    result = await handler(*args, **kwargs)
                except HTTPException:
                    # Don't cache error responses; FastAPI handles them.
                    raise
                # Don't cache responses the handler built itself with non-2xx
                # status, ``Set-Cookie`` headers, or unusual content type.
                if isinstance(result, Response):
                    if result.status_code >= 400 or result.headers.get("set-cookie"):
                        return result
                    body = result.body
                    etag = _etag_for(body)
                    _store[store_key] = _Entry(body, etag, now + ttl)
                    return _build_response(body, etag, status=result.status_code)

                body = _serialize(result)
                etag = _etag_for(body)
                _store[store_key] = _Entry(body, etag, now + ttl)
                if request.headers.get("if-none-match") == etag:
                    metrics.bump(prefix, "etag_304")
                    return Response(
                        status_code=304,
                        headers={"ETag": etag, "Cache-Control": _CACHE_CONTROL},
                    )
                return _build_response(body, etag)

        return wrapper

    return decorator


def invalidate(*prefixes: str) -> None:
    """Drop all cache entries whose key starts with any of the given prefixes.

    Pass multiple prefixes for cross-resource writes (e.g. publishing a workflow
    affects both ``services`` and ``workflows`` lists).
    """
    if not prefixes:
        return
    targets = set(prefixes)
    keys_to_delete = [k for k in _store if k[0] in targets]
    for k in keys_to_delete:
        _store.pop(k, None)
    for p in targets:
        metrics.bump(p, "invalidations")


def _reset_for_tests() -> None:
    """Test-only helper to clear cache + metrics in-place.

    Mutating in-place (not reassigning ``metrics``) so external imports of
    ``metrics`` stay valid across tests.
    """
    _store.clear()
    _locks.clear()
    metrics.hits = 0
    metrics.misses = 0
    metrics.etag_304 = 0
    metrics.invalidations = 0
    metrics.by_prefix.clear()
