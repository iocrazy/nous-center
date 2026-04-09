# backend/tests/test_log_collector.py
import logging
import pytest
from src.services.log_db import init_log_db, query_logs
from src.services.log_collector import DbLogHandler


@pytest.fixture
def log_db(tmp_path):
    db_path = str(tmp_path / "test_logs.db")
    init_log_db(db_path)
    return db_path


def test_handler_captures_log_records(log_db):
    handler = DbLogHandler(db_path=log_db, flush_interval=0, flush_size=1)
    logger = logging.getLogger("test.collector")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("Test message from collector")
    handler.flush()

    result = query_logs(log_db, "app_logs", limit=10)
    assert result["total"] >= 1
    assert "Test message from collector" in result["items"][0]["message"]
    assert result["items"][0]["level"] == "INFO"

    logger.removeHandler(handler)


def test_handler_captures_error_with_location(log_db):
    handler = DbLogHandler(db_path=log_db, flush_interval=0, flush_size=1)
    logger = logging.getLogger("test.location")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.error("Something broke")
    handler.flush()

    result = query_logs(log_db, "app_logs", limit=10)
    item = result["items"][0]
    assert item["level"] == "ERROR"
    assert "test_log_collector.py" in item["location"]

    logger.removeHandler(handler)
