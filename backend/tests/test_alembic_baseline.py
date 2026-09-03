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
