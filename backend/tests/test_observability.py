"""runtime metrics 模块 + /api/v1/observability/runtime 端点测试。"""

from __future__ import annotations

import pytest

from src.services import runtime_metrics as rm


@pytest.fixture(autouse=True)
def _reset_metrics():
    rm.reset_for_tests()
    yield
    rm.reset_for_tests()


def test_gzip_compression_ratio():
    rm.record_gzip(raw_bytes=1000, compressed_bytes=200)
    rm.record_gzip(raw_bytes=2000, compressed_bytes=500)
    snap = rm.snapshot()
    assert snap["gzip"]["calls"] == 2
    assert snap["gzip"]["raw_bytes"] == 3000
    assert snap["gzip"]["compressed_bytes"] == 700
    # ratio = 3000 / 700 ≈ 4.286
    assert snap["gzip"]["compression_ratio"] == pytest.approx(4.286, abs=0.01)


def test_gzip_zero_calls_returns_none_ratio():
    snap = rm.snapshot()
    assert snap["gzip"]["calls"] == 0
    assert snap["gzip"]["compression_ratio"] is None


def test_compaction_avg_dropped():
    rm.record_compaction(turns_dropped=3, truncated=False)
    rm.record_compaction(turns_dropped=5, truncated=False)
    rm.record_compaction(turns_dropped=2, truncated=True)
    snap = rm.snapshot()
    assert snap["compaction"]["calls"] == 3
    assert snap["compaction"]["turns_dropped"] == 10
    assert snap["compaction"]["avg_turns_dropped"] == pytest.approx(3.33, abs=0.01)
    assert snap["compaction"]["truncated"] == 1


def test_cache_hit_rate():
    for _ in range(7):
        rm.record_cache_lookup(hit=True)
    for _ in range(3):
        rm.record_cache_lookup(hit=False)
    snap = rm.snapshot()
    assert snap["cache"]["lookups"] == 10
    assert snap["cache"]["hits"] == 7
    assert snap["cache"]["hit_rate"] == pytest.approx(0.7, abs=0.001)


def test_cache_zero_lookups_returns_none_rate():
    snap = rm.snapshot()
    assert snap["cache"]["lookups"] == 0
    assert snap["cache"]["hit_rate"] is None


@pytest.mark.asyncio
async def test_runtime_endpoint_returns_snapshot(client):
    rm.record_gzip(raw_bytes=500, compressed_bytes=100)
    rm.record_compaction(turns_dropped=2)
    rm.record_cache_lookup(hit=True)
    rm.record_cache_lookup(hit=False)

    r = await client.get("/api/v1/observability/runtime")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gzip"]["calls"] == 1
    assert body["gzip"]["compression_ratio"] == 5.0
    assert body["compaction"]["calls"] == 1
    assert body["compaction"]["turns_dropped"] == 2
    assert body["cache"]["lookups"] == 2
    assert body["cache"]["hit_rate"] == 0.5


@pytest.mark.asyncio
async def test_runtime_endpoint_zero_state_no_div_by_zero(client):
    r = await client.get("/api/v1/observability/runtime")
    assert r.status_code == 200
    body = r.json()
    assert body["gzip"]["compression_ratio"] is None
    assert body["compaction"]["avg_turns_dropped"] is None
    assert body["cache"]["hit_rate"] is None
