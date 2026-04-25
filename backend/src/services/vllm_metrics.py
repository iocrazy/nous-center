"""Pull KV-cache + scheduler metrics from running vLLM instances.

vLLM exposes Prometheus-style metrics on each instance's HTTP port at /metrics.
We scrape the active LLM instances (whose ports we know via ModelManager) and
parse the gauges we care about for the m04 dashboard.

This is read-only; the caller (observability route) wraps it in the runtime
snapshot. We don't ship a Prometheus client library — the gauges we care about
have one number each and a tiny regex parses them in microseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

# Per-instance gauges we surface on the dashboard. Keys here are the suffix
# AFTER ``vllm:``, values are display-friendly names. Add to this list as new
# panels are needed.
_GAUGES = {
    "num_requests_running": "running",
    "num_requests_waiting": "waiting",
    "kv_cache_usage_perc": "kv_cache_usage_perc",
    "prefix_cache_queries_total": "prefix_cache_queries_total",
    "prefix_cache_hits_total": "prefix_cache_hits_total",
    "num_preemptions_total": "num_preemptions_total",
}

# `cache_config_info{...} 1.0` carries config as label key=value pairs.
_CONFIG_LABELS = {
    "block_size",
    "cache_dtype",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "num_gpu_blocks",
}

_GAUGE_LINE = re.compile(
    r"^vllm:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{(?P<labels>[^}]*)\}\s+(?P<value>[-+0-9.eE]+)\s*$"
)
_LABEL_KV = re.compile(r'(\w+)="([^"]*)"')


@dataclass
class VLLMInstanceSnapshot:
    name: str
    port: int
    healthy: bool
    config: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def parse_metrics_text(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (config, gauges) parsed from a vLLM /metrics body."""
    config: dict[str, Any] = {}
    gauges: dict[str, Any] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _GAUGE_LINE.match(line)
        if m is None:
            continue
        name = m.group("name")
        if name == "cache_config_info":
            for k, v in _LABEL_KV.findall(m.group("labels")):
                if k in _CONFIG_LABELS:
                    config[k] = v
            continue
        if name in _GAUGES:
            try:
                gauges[_GAUGES[name]] = float(m.group("value"))
            except ValueError:
                pass
    return config, gauges


async def fetch_instance(name: str, port: int, *, timeout: float = 1.5) -> VLLMInstanceSnapshot:
    """Hit one vLLM /metrics. Short timeout — UI poll mustn't block on a slow vLLM."""
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
        if r.status_code != 200:
            return VLLMInstanceSnapshot(
                name=name, port=port, healthy=False, error=f"HTTP {r.status_code}"
            )
        config, stats = parse_metrics_text(r.text)
        # Derived: prefix cache hit rate.
        q = stats.get("prefix_cache_queries_total")
        h = stats.get("prefix_cache_hits_total")
        if q and q > 0 and h is not None:
            stats["prefix_cache_hit_rate"] = round(h / q, 3)
        return VLLMInstanceSnapshot(name=name, port=port, healthy=True, config=config, stats=stats)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPError) as exc:
        return VLLMInstanceSnapshot(
            name=name, port=port, healthy=False, error=type(exc).__name__
        )


async def snapshot_all(instances: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """Scrape every (name, port) pair concurrently. Returns plain dicts."""
    import asyncio

    if not instances:
        return []
    snaps = await asyncio.gather(*(fetch_instance(n, p) for n, p in instances))
    return [
        {
            "name": s.name,
            "port": s.port,
            "healthy": s.healthy,
            "config": s.config,
            "stats": s.stats,
            "error": s.error,
        }
        for s in snaps
    ]
