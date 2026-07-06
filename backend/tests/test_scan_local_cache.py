"""scan_local_models 30s TTL 缓存(性能 P1:原每次全走盘)。"""
import pytest

import src.services.model_metadata_service as mms


@pytest.fixture(autouse=True)
def _clear():
    mms.invalidate_local_scan_cache()
    yield
    mms.invalidate_local_scan_cache()


def test_cache_walks_disk_once_within_ttl(monkeypatch):
    calls = {"n": 0}
    def _fake_uncached():
        calls["n"] += 1
        return {"llm/x"}
    monkeypatch.setattr(mms, "_scan_local_models_uncached", _fake_uncached)
    r1 = mms.scan_local_models()
    r2 = mms.scan_local_models()
    r3 = mms.scan_local_models()
    assert r1 == r2 == r3 == {"llm/x"}
    assert calls["n"] == 1, f"应只走盘 1 次,实 {calls['n']}"


def test_invalidate_forces_rewalk(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(mms, "_scan_local_models_uncached",
                        lambda: (calls.__setitem__("n", calls["n"] + 1), {"a"})[1])
    mms.scan_local_models()
    mms.invalidate_local_scan_cache()
    mms.scan_local_models()
    assert calls["n"] == 2
