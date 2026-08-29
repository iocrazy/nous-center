"""回归护栏:测试套件绝不能绑到 .env 里的真实数据库。

2026-08-29 事故:conftest 只在 CI 的 `:memory:` 情况下把 DATABASE_URL 换成隔离
sqlite,注释还明写「local runs (postgres via .env) are left completely untouched」
—— 于是本地跑 pytest 直接读写生产 Postgres(nous_center)。实测一次全量跑在
comfy_templates / service_instances 各留下 16 行测试垃圾(bridge-db-test /
e2e-* / tpl-* / admin-e2e-*),并和生产行 `minimax-h3-r2v`(ComfyUI 桥 PR#672,
建于 2026-08-11)撞名 → test_create_template_creates_service 永久 409。

本文件锁死"测试库必须是隔离临时 sqlite"这条不变式。真要打真库调试,显式设
NOUS_TEST_USE_REAL_DB=1(此时本护栏自动跳过)。
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("NOUS_TEST_USE_REAL_DB") == "1",
    reason="显式选择了打真库(NOUS_TEST_USE_REAL_DB=1),护栏不适用",
)


def test_database_url_is_an_isolated_temp_sqlite():
    """conftest 必须已经把 DATABASE_URL 改写成 per-run 的临时 sqlite 文件。"""
    url = os.environ.get("DATABASE_URL", "")
    assert url.startswith("sqlite+aiosqlite:///"), (
        f"测试库不是 sqlite,疑似绑到了真实库:{url!r}"
    )
    assert "nous-test-db-" in url, (
        f"测试库不是 conftest 建的 per-run 临时目录:{url!r}"
    )


def test_session_factory_points_at_the_same_isolated_db():
    """应用侧的 engine(get_session_factory 走的那个)也必须落在隔离库上 ——
    只改环境变量不够,create_engine() 得真读到改后的值。"""
    from src.models.database import create_engine

    engine = create_engine()
    try:
        assert engine.url.get_backend_name() == "sqlite", (
            f"应用 engine 不是 sqlite,疑似绑到了真实库:{engine.url!r}"
        )
        assert "nous-test-db-" in str(engine.url.database or "")
    finally:
        # sync dispose 够了 —— 没有连接被打开过。
        pass
