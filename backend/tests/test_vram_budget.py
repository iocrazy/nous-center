"""每模型显存预算(spec 2026-06-13):解析三模式 → utilization、推荐值口径、API wiring。
CI 安全:纯计算 + mock,不起真 vLLM。"""
import pytest

from src.config import recommend_vram_budget_gb, resolve_vram_utilization


# ---------------------------------------------------------------------------
# resolve_vram_utilization：三模式 → gpu_memory_utilization + 优先级
# ---------------------------------------------------------------------------

def test_resolve_auto_falls_through_to_auto_util():
    # mode=auto + 无 fallback → 走 adapter auto 公式(零回归)
    assert resolve_vram_utilization({"mode": "auto"}, 96.0, None, 0.55) == 0.55


def test_resolve_none_budget_uses_fallback_then_auto():
    assert resolve_vram_utilization(None, 96.0, 0.3, 0.55) == 0.3
    assert resolve_vram_utilization(None, 96.0, None, 0.55) == 0.55


def test_resolve_percent_direct():
    assert resolve_vram_utilization({"mode": "percent", "value": 0.42}, 96.0, 0.3, 0.55) == 0.42


def test_resolve_absolute_divides_by_card_total():
    # 11GB on a 96GB card ≈ 0.1146
    util = resolve_vram_utilization({"mode": "absolute", "value": 11}, 96.0, None, 0.55)
    assert util == pytest.approx(11 / 96, abs=1e-6)


def test_resolve_overlay_beats_yaml_fallback():
    # 显式 percent overlay 优先于 yaml gpu_memory_utilization(fallback)
    assert resolve_vram_utilization({"mode": "percent", "value": 0.5}, 96.0, 0.2, 0.55) == 0.5


def test_resolve_clamps_to_0_98():
    assert resolve_vram_utilization({"mode": "percent", "value": 1.5}, 96.0, None, 0.55) == 0.98
    assert resolve_vram_utilization({"mode": "absolute", "value": 200}, 96.0, None, 0.55) == 0.98


def test_resolve_invalid_value_falls_through():
    # 非法 value(<=0 / 缺省)不该崩,退回 fallback/auto
    assert resolve_vram_utilization({"mode": "percent", "value": 0}, 96.0, 0.3, 0.55) == 0.3
    assert resolve_vram_utilization({"mode": "absolute"}, 96.0, None, 0.55) == 0.55


def test_resolve_absolute_without_card_total_falls_through():
    assert resolve_vram_utilization({"mode": "absolute", "value": 11}, 0, 0.3, 0.55) == 0.3


# ---------------------------------------------------------------------------
# recommend_vram_budget_gb：分模态口径
# ---------------------------------------------------------------------------

def test_recommend_embedding_is_weights_times_1_25():
    assert recommend_vram_budget_gb("embedding", 8.0) == 10.0


def test_recommend_tts_same_as_embedding():
    assert recommend_vram_budget_gb("tts", 4.0) == 5.0


def test_recommend_llm_adds_kv_headroom():
    assert recommend_vram_budget_gb("llm", 20.0) == 26.0
    assert recommend_vram_budget_gb("vl", 20.0) == 26.0


def test_recommend_unknown_type_conservative():
    assert recommend_vram_budget_gb("image", 10.0) == 13.0


def test_recommend_floor_is_one():
    assert recommend_vram_budget_gb("embedding", 0) == 1.0


# ---------------------------------------------------------------------------
# 覆盖读写(数据加载统一 2026-06-16:DB typed 表 + 缓存,vram_budget 拆 mode/value 两列)
# ---------------------------------------------------------------------------

async def test_db_store_accepts_structured_vram_budget():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from src.models.database import Base
    from src.services import runtime_override_store as store

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    try:
        store.reset_cache()
        async with sf() as s:
            await store.set_override(s, "qwen3_embedding_4b", "vram_budget",
                                     {"mode": "absolute", "value": 11})
        data = store.get_overrides()
        assert data["qwen3_embedding_4b"]["vram_budget"] == {"mode": "absolute", "value": 11}
    finally:
        store.reset_cache()
        await engine.dispose()


async def test_db_store_rejects_unknown_key():
    from src.services import runtime_override_store as store
    with pytest.raises(ValueError):
        await store.set_override(None, "m", "bogus_key", 1)


# ---------------------------------------------------------------------------
# _card_total_gb_for_engine：真实落卡(已加载)优先于 detector 推断
# ---------------------------------------------------------------------------

def test_card_total_prefers_loaded_gpu(monkeypatch):
    import src.api.routes.engines as eng

    # gpu_summary/get_device_for_engine 在 _card_total_gb_for_engine 内部 import,patch 源模块。
    # 三卡:0=3090(24G) 1=Pro6000(96G) 2=3090(24G)。
    import src.gpu.detector as det
    monkeypatch.setattr(det, "gpu_summary", lambda: {"devices": [
        {"index": 0, "vram_gb": 24.0}, {"index": 1, "vram_gb": 96.0}, {"index": 2, "vram_gb": 24.0},
    ]})
    monkeypatch.setattr(det, "get_device_for_engine", lambda cfg: "cuda:0")  # detector 默认给 3090

    cfg = {"gpu": None, "type": "embedding"}
    # 未加载 → 走 detector → 24G(3090)
    assert eng._card_total_gb_for_engine(cfg, None) == 24.0
    # 已加载在 cuda:1 → 用真实落卡 96G,压过 detector
    assert eng._card_total_gb_for_engine(cfg, 1) == 96.0


# ---------------------------------------------------------------------------
# API wiring：GET / PATCH（admin gate 在测试里关闭）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_vram_budget_returns_recommended(client):
    resp = await client.get("/api/v1/engines/qwen3_embedding_4b/vram-budget")
    assert resp.status_code == 200
    body = resp.json()
    assert body["applicable"] is True
    assert body["current"] == {"mode": "auto"}  # 无 overlay 默认 auto
    assert body["recommended_gb"] > 0
    assert body["card_total_gb"] > 0


@pytest.mark.asyncio
async def test_get_vram_budget_unknown_404(client):
    resp = await client.get("/api/v1/engines/nonexistent/vram-budget")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_patch_vram_budget_rejects_bad_mode(client):
    resp = await client.patch("/api/v1/engines/qwen3_embedding_4b/vram-budget",
                              json={"mode": "bananas"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_patch_vram_budget_rejects_bad_value(client):
    resp = await client.patch("/api/v1/engines/qwen3_embedding_4b/vram-budget",
                              json={"mode": "percent", "value": 2})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_patch_then_get_roundtrip(db_client):
    """PATCH 写 DB(model_runtime_overrides)+ write-through 缓存 → GET 经
    load_runtime_overrides 缓存读回(数据加载统一 2026-06-16:db_client 提供 sqlite session)。"""
    resp = await db_client.patch("/api/v1/engines/qwen3_embedding_4b/vram-budget",
                                 json={"mode": "percent", "value": 0.25})
    assert resp.status_code == 200
    assert resp.json()["vram_budget"] == {"mode": "percent", "value": 0.25}
    assert "重新加载" in resp.json()["hint"]

    resp2 = await db_client.get("/api/v1/engines/qwen3_embedding_4b/vram-budget")
    assert resp2.json()["current"] == {"mode": "percent", "value": 0.25}
