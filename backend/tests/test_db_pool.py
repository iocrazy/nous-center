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


def test_no_pool_size_left_to_default():
    # pool_size/max_overflow 需配合 PG max_connections 调,不在代码里硬设。
    kw = _engine_kwargs("postgresql+asyncpg://u:p@h/db")
    assert "pool_size" not in kw
    assert "max_overflow" not in kw


def test_sqlite_also_gets_pre_ping():
    # pool_pre_ping/pool_recycle 是 Pool 基类参数,sqlite 池也接受,不该分叉。
    kw = _engine_kwargs("sqlite+aiosqlite:///./test.db")
    assert kw["pool_pre_ping"] is True
