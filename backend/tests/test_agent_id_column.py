"""Schema tests — agent_id column on response_sessions and llm_usage.

DB layer is async-only (`create_async_engine`), so these tests spin up an
SQLAlchemy 元数据直接读列(不起数据库)
within `run_sync`.
"""



from src.models.database import Base
from src.models import response_session, llm_usage  # noqa: F401 ensure registered


def _columns_for(table: str) -> list[str]:
    """直接读 SQLAlchemy 元数据。这个测试问的是「模型上有没有这列」,根本不需要
    起数据库 —— 以前起一个 in-memory sqlite 只是为了 create_all 再 inspect,绕远了。
    """
    return [c.name for c in Base.metadata.tables[table].columns]


def test_response_session_has_agent_id_column():
    cols = (_columns_for("response_sessions"))
    assert "agent_id" in cols


def test_llm_usage_has_agent_id_column():
    cols = (_columns_for("llm_usage"))
    assert "agent_id" in cols
