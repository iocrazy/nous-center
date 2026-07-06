"""DB 连接池:pool_pre_ping + pool_recycle(防 PG 重启/空闲断连后陈旧连接抛错)。

单管理员长驻服务后台 PG,连接可能因 PG 重启/网络抖动/空闲超时被服务端关闭。默认无
pre_ping 时,下一个请求 checkout 到陈旧连接会抛 "server closed connection unexpectedly"。
pre_ping 在 checkout 前发轻量 SELECT 1,失败即丢弃取新的。pool_size/max_overflow 不在此
设置 —— 那需配合 PG max_connections 调,留默认。
"""

from src.models.database import _engine_kwargs


def test_pre_ping_enabled():
    kw = _engine_kwargs("postgresql+asyncpg://u:p@h/db")
    assert kw["pool_pre_ping"] is True


def test_pool_recycle_set():
    kw = _engine_kwargs("postgresql+asyncpg://u:p@h/db")
    assert kw["pool_recycle"] > 0


def test_postgres_gets_bounded_pool():
    # postgres 用 QueuePool,设保守常驻+溢出;峰值须远低于 PG 默认 max_connections=100。
    kw = _engine_kwargs("postgresql+asyncpg://u:p@h/db")
    assert kw["pool_size"] == 10
    assert kw["max_overflow"] == 20
    assert kw["pool_size"] + kw["max_overflow"] < 100


def test_sqlite_no_pool_size():
    # aiosqlite 池不接受 pool_size/max_overflow(设了 TypeError),只留 pre_ping/recycle。
    kw = _engine_kwargs("sqlite+aiosqlite:///./test.db")
    assert kw["pool_pre_ping"] is True
    assert "pool_size" not in kw
    assert "max_overflow" not in kw
