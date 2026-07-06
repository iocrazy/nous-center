"""响应缓存命中/失效计数器 —— 下沉到 services 层打破 services→api 反向依赖。

`_Metrics`/`metrics` 原在 api/response_cache.py。response_cache 写(bump),
services/runtime_metrics 只读(snapshot)。此前 runtime_metrics 函数内 lazy import
`from src.api.response_cache import metrics`(层级倒置)。移到中性模块后两边都向下 import。
"""
from __future__ import annotations

from dataclasses import dataclass, field


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


metrics = _Metrics()
