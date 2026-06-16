"""运行时模型覆盖(resident / gpu / vram_budget)的 DB 持久化 + 进程内缓存。

数据加载统一(2026-06-16):覆盖从 runtime_overrides.json 文件迁到 Postgres typed 表
[[model_runtime_override]]。难点 = 配置读取是**同步**(load_model_configs/registry 到处同步调)
而 DB 是**异步**。解法:**DB 是真相源,启动时 hydrate 进 _CACHE,同步读走缓存**(零 blast
radius:load_runtime_overrides 仍同步返回同形状 dict);set_override 写 DB + 刷缓存(write-through)。

启动顺序保证(src/api/main.py lifespan):DB 连接/create_all → migrate_json_if_empty → hydrate
→ 才建 model_manager/registry(它们 _load 读缓存)→ 才预加载模型。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 允许的覆盖键(与旧 config._OVERRIDABLE_KEYS 一致)。
VALID_KEYS = ("resident", "gpu", "vram_budget")

# 进程内缓存:{model_id: {resident?, gpu?, vram_budget?}}。DB 的同步可读镜像。
_CACHE: dict[str, dict] = {}


def get_overrides() -> dict:
    """同步快照(load_runtime_overrides 用)。返回缓存深拷贝,调用方改不脏缓存。"""
    return {mid: dict(ov) for mid, ov in _CACHE.items()}


def reset_cache() -> None:
    """清空缓存(测试用 / 重新 hydrate 前)。"""
    _CACHE.clear()


def set_cache_for_test(overrides: dict) -> None:
    """直接灌缓存(同步单测用,免 DB)。{model_id: {resident?, gpu?, vram_budget?}}。"""
    _CACHE.clear()
    _CACHE.update({mid: dict(ov) for mid, ov in overrides.items()})


async def hydrate(session_factory) -> None:
    """从 DB 全量读进 _CACHE(启动时调,DB 连上后、建 registry 前)。"""
    from sqlalchemy import select  # noqa: PLC0415

    from src.models.model_runtime_override import ModelRuntimeOverride  # noqa: PLC0415
    async with session_factory() as session:
        rows = (await session.execute(select(ModelRuntimeOverride))).scalars().all()
    _CACHE.clear()
    for row in rows:
        ov = row.to_overrides()
        if ov:
            _CACHE[row.model_id] = ov
    logger.info("runtime overrides hydrated from DB: %d models", len(_CACHE))


async def set_override(session, model_id: str, key: str, value) -> None:
    """写一个覆盖键到 DB(upsert 对应列)+ 刷缓存。key ∈ VALID_KEYS。

    session: 调用方注入的 AsyncSession(API handler 经 Depends 拿,测试可注入 sqlite)。
    """
    if key not in VALID_KEYS:
        raise ValueError(f"non-overridable key: {key!r}(允许:{VALID_KEYS})")

    from src.models.model_runtime_override import ModelRuntimeOverride  # noqa: PLC0415
    row = await session.get(ModelRuntimeOverride, model_id)
    if row is None:
        row = ModelRuntimeOverride(model_id=model_id)
        session.add(row)
    if key == "resident":
        row.resident = bool(value)
    elif key == "gpu":
        row.gpu = int(value)
    elif key == "vram_budget":
        # value = {"mode": auto|percent|absolute, "value": float?}
        row.vram_budget_mode = (value or {}).get("mode")
        row.vram_budget_value = (value or {}).get("value")
    await session.commit()
    await session.refresh(row)
    ov = row.to_overrides()
    if ov:
        _CACHE[model_id] = ov
    else:
        _CACHE.pop(model_id, None)


async def migrate_json_if_empty(session_factory, json_path) -> int:
    """一次性迁移:表为空且 runtime_overrides.json 存在 → 导入,避免升级丢现有覆盖。
    返回导入的模型数(0 = 跳过)。幂等:表非空就不动。"""
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    from sqlalchemy import select  # noqa: PLC0415

    from src.models.model_runtime_override import ModelRuntimeOverride  # noqa: PLC0415
    if not os.path.exists(json_path):
        return 0
    async with session_factory() as session:
        existing = (await session.execute(select(ModelRuntimeOverride))).scalars().first()
        if existing is not None:
            return 0  # 表非空 → 已迁过,不覆盖
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("migrate_json_if_empty 读 %s 失败,跳过:%s", json_path, e)
            return 0
        n = 0
        for mid, ov in (data or {}).items():
            if not isinstance(ov, dict):
                continue
            row = ModelRuntimeOverride(model_id=mid)
            if "resident" in ov:
                row.resident = bool(ov["resident"])
            if "gpu" in ov:
                row.gpu = int(ov["gpu"])
            vb = ov.get("vram_budget")
            if isinstance(vb, dict):
                row.vram_budget_mode = vb.get("mode")
                row.vram_budget_value = vb.get("value")
            session.add(row)
            n += 1
        await session.commit()
    if n:
        logger.info("migrated %d runtime overrides from %s into DB", n, json_path)
    return n
