"""Lightweight in-memory runtime metrics — gzip codec / context engine / cache.

Process-wide counters with thread-safe increments. Re-zeros on process
restart (this is intentional — the dashboard "运行时观测" card shows
current-process activity, not historical). Persistent metrics belong in
Prometheus / DB; this is the cheap "is anything happening?" view.

All public APIs are pure dict reads / atomic adds — no async, no IO. Safe
to call from request handlers.

Tracks three things:
  * gzip codec：raw / compressed bytes 累计 + 调用次数（→ 平均压缩比）
  * context engine compaction：调用次数、被丢的 turn 数、最终是否仍超限
  * context cache lookup：lookups / hits（→ 命中率）+ 当前缓存条目计数（由
    路由侧填，本模块只负责加减）

Reset 用于测试隔离。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TypedDict


@dataclass
class _GzipStats:
    calls: int = 0
    raw_bytes: int = 0
    compressed_bytes: int = 0


@dataclass
class _CompactionStats:
    calls: int = 0
    turns_dropped: int = 0
    truncated: int = 0  # 调用了 compress 但仍触发 ContextOverflowError 的次数


@dataclass
class _CacheStats:
    lookups: int = 0
    hits: int = 0


@dataclass
class _RuntimeMetrics:
    gzip: _GzipStats = field(default_factory=_GzipStats)
    compaction: _CompactionStats = field(default_factory=_CompactionStats)
    cache: _CacheStats = field(default_factory=_CacheStats)
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = _RuntimeMetrics()


# ---------- public mutators ----------


def record_gzip(raw_bytes: int, compressed_bytes: int) -> None:
    with _state.lock:
        _state.gzip.calls += 1
        _state.gzip.raw_bytes += raw_bytes
        _state.gzip.compressed_bytes += compressed_bytes


def record_compaction(turns_dropped: int, truncated: bool = False) -> None:
    with _state.lock:
        _state.compaction.calls += 1
        _state.compaction.turns_dropped += turns_dropped
        if truncated:
            _state.compaction.truncated += 1


def record_cache_lookup(hit: bool) -> None:
    with _state.lock:
        _state.cache.lookups += 1
        if hit:
            _state.cache.hits += 1


def reset_for_tests() -> None:
    """Test fixture helper — zero all counters."""
    global _state
    _state = _RuntimeMetrics()


# ---------- snapshot ----------


class GzipSnapshot(TypedDict):
    calls: int
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float | None  # raw / compressed; None when calls==0


class CompactionSnapshot(TypedDict):
    calls: int
    turns_dropped: int
    avg_turns_dropped: float | None  # turns_dropped / calls
    truncated: int


class CacheSnapshot(TypedDict):
    lookups: int
    hits: int
    hit_rate: float | None  # hits / lookups in [0, 1]; None when lookups==0


class ResponseCacheSnapshot(TypedDict):
    hits: int
    misses: int
    etag_304: int
    invalidations: int
    hit_rate: float | None  # hits / (hits + misses); None when both are 0
    by_prefix: dict[str, dict[str, int]]


class RuntimeSnapshot(TypedDict):
    gzip: GzipSnapshot
    compaction: CompactionSnapshot
    cache: CacheSnapshot
    response_cache: ResponseCacheSnapshot


def snapshot() -> RuntimeSnapshot:
    """Atomic snapshot — copies under lock so reads can't tear."""
    # Imported lazily to avoid circular import (response_cache imports
    # nothing from this module today, but pinning the order keeps it safe).
    from src.api.response_cache import metrics as _rc_metrics

    with _state.lock:
        g = _state.gzip
        c = _state.compaction
        ca = _state.cache
        ratio = (g.raw_bytes / g.compressed_bytes) if g.compressed_bytes else None
        avg_drop = (c.turns_dropped / c.calls) if c.calls else None
        hit_rate = (ca.hits / ca.lookups) if ca.lookups else None
        rc = _rc_metrics.snapshot()
        rc_total = rc["hits"] + rc["misses"]
        rc_hit_rate = (rc["hits"] / rc_total) if rc_total else None
        return {
            "gzip": {
                "calls": g.calls,
                "raw_bytes": g.raw_bytes,
                "compressed_bytes": g.compressed_bytes,
                "compression_ratio": round(ratio, 3) if ratio is not None else None,
            },
            "compaction": {
                "calls": c.calls,
                "turns_dropped": c.turns_dropped,
                "avg_turns_dropped": (
                    round(avg_drop, 2) if avg_drop is not None else None
                ),
                "truncated": c.truncated,
            },
            "cache": {
                "lookups": ca.lookups,
                "hits": ca.hits,
                "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
            },
            "response_cache": {
                "hits": rc["hits"],
                "misses": rc["misses"],
                "etag_304": rc["etag_304"],
                "invalidations": rc["invalidations"],
                "hit_rate": round(rc_hit_rate, 3) if rc_hit_rate is not None else None,
                "by_prefix": rc["by_prefix"],
            },
        }
