"""Alembic baseline 守门:fresh DB `upgrade head` 后 schema 必须与 models 零 diff。

这是引入 alembic 的安全基石 —— 证明 baseline 迁移 == Base.metadata(生产 schema)。
也顺带守未来:谁改了 model 却忘了生成对应迁移,`alembic check` 会检出 diff → 本测试红,
逼你补迁移。

用 subprocess 跑真实 alembic CLI(最忠实,就是运维会敲的命令),对一个**新建的空
PostgreSQL 库** —— 全局只用 PG,而且 alembic 对 PG 跑才是生产真正会走的路径
(sqlite 上 `check` 看不出 PG 专属类型/索引的漂移)。
"""

import asyncio
import os
import secrets
import subprocess
import sys
import urllib.parse
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


def _pg_fetch(dsn: str, sql: str) -> list:
    """裸 asyncpg 查一句 SQL(用来断言迁移真的把 DDL 落到了库上)。"""
    import asyncpg

    async def _run():
        conn = await asyncpg.connect(dsn)
        try:
            return await conn.fetch(sql)
        finally:
            await conn.close()

    return asyncio.run(_run())


def _pg_admin(admin_dsn: str, sql: str) -> None:
    """裸 asyncpg 执行一句管理 SQL(建库/删库不能在事务里,asyncpg 默认 autocommit)。"""
    import asyncpg

    async def _run():
        conn = await asyncpg.connect(admin_dsn)
        try:
            await conn.execute(sql)
        finally:
            await conn.close()

    asyncio.run(_run())


@pytest.fixture
def fresh_pg_db():
    """一个刚建好的**空** PG 库(alembic 要从零 upgrade,不能用已 create_all 的测试库)。
    基于当前 DATABASE_URL(conftest 已指向临时测试库)换个库名,用完 DROP。"""
    base = os.environ["DATABASE_URL"]
    parsed = urllib.parse.urlparse(base)
    name = f"nous_alembic_{secrets.token_hex(6)}"
    admin = urllib.parse.urlunparse(parsed._replace(path="/postgres")).replace(
        "postgresql+asyncpg://", "postgresql://")
    _pg_admin(admin, f'CREATE DATABASE "{name}"')
    try:
        yield urllib.parse.urlunparse(parsed._replace(path=f"/{name}"))
    finally:
        _pg_admin(admin, "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                         f"WHERE datname = '{name}' AND pid <> pg_backend_pid()")
        _pg_admin(admin, f'DROP DATABASE IF EXISTS "{name}"')


def _run_alembic(args: list[str], db_url: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "DATABASE_URL": db_url,
        "CUDA_VISIBLE_DEVICES": "",
    }
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=_BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_baseline_upgrade_then_zero_diff(fresh_pg_db):
    db = fresh_pg_db

    up = _run_alembic(["upgrade", "head"], db)
    assert up.returncode == 0, f"upgrade head failed:\n{up.stdout}\n{up.stderr}"

    check = _run_alembic(["check"], db)
    combined = check.stdout + check.stderr
    # check 返回 0 且明确"无新操作" = upgrade 后的库 schema 与 models 完全一致。
    assert check.returncode == 0, f"alembic check found drift (baseline != models):\n{combined}"
    assert "No new upgrade operations detected" in combined, combined


def test_upgrade_head_creates_memory_fts_gin_index(fresh_pg_db):
    """`upgrade head` 必须在全新库上产出 memory_entries 的表达式 GIN 索引。

    上面的 `alembic check` 只按**索引名**比对表达式索引(autogenerate 把它归到 `on '()'`),
    名字对了就算数 —— 一个同名的 btree 索引照样能骗过它。这里直接读 pg_indexes 的
    indexdef,把 `USING gin` 和 `to_tsvector('simple'` 钉死:表达式与
    pg_provider.py 查询侧差一个字,planner 就用不上索引。
    """
    db = fresh_pg_db

    up = _run_alembic(["upgrade", "head"], db)
    assert up.returncode == 0, f"upgrade head failed:\n{up.stdout}\n{up.stderr}"

    rows = _pg_fetch(
        db.replace("postgresql+asyncpg://", "postgresql://"),
        "SELECT indexdef FROM pg_indexes "
        "WHERE tablename = 'memory_entries' AND indexname = 'idx_mem_content_fts'",
    )
    assert rows, "upgrade head 后没有 idx_mem_content_fts"
    indexdef = rows[0]["indexdef"]
    assert "USING gin" in indexdef, indexdef
    assert "to_tsvector('simple'" in indexdef, indexdef


def test_upgrade_head_creates_model_runtime_overrides_gpus(fresh_pg_db):
    """`upgrade head` 必须在全新库上给 model_runtime_overrides 建出 gpus JSONB 列。

    模型级 GPU 组(张量并行)的落盘位置。`alembic check` 只保证「models 与库零 diff」——
    这里另外把**类型**钉死:退成 JSON(非 JSONB)虽然读写都通,却丢掉 GIN/包含查询能力,
    且与 main.py 那条微迁移(`ADD COLUMN ... JSONB`)不一致,生产/CI 会漂移。
    """
    db = fresh_pg_db

    up = _run_alembic(["upgrade", "head"], db)
    assert up.returncode == 0, f"upgrade head failed:\n{up.stdout}\n{up.stderr}"

    rows = _pg_fetch(
        db.replace("postgresql+asyncpg://", "postgresql://"),
        "SELECT data_type, is_nullable FROM information_schema.columns "
        "WHERE table_name = 'model_runtime_overrides' AND column_name = 'gpus'",
    )
    assert rows, "upgrade head 后 model_runtime_overrides 没有 gpus 列"
    assert rows[0]["data_type"] == "jsonb", rows[0]["data_type"]
    assert rows[0]["is_nullable"] == "YES"
