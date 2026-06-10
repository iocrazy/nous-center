# backend/tests/test_log_collector.py
import logging

from src.services import log_collector
from src.services.log_collector import DbLogHandler


def test_handler_forwards_record_to_store(monkeypatch):
    captured = []
    monkeypatch.setattr(log_collector, "enqueue", lambda kind, fields: captured.append((kind, fields)))

    handler = DbLogHandler()
    logger = logging.getLogger("test.collector")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("Test message from collector")
    finally:
        logger.removeHandler(handler)

    assert captured
    kind, fields = captured[-1]
    assert kind == "app"
    assert fields["level"] == "INFO"
    assert "Test message from collector" in fields["message"]


def test_handler_captures_error_with_location(monkeypatch):
    captured = []
    monkeypatch.setattr(log_collector, "enqueue", lambda kind, fields: captured.append((kind, fields)))

    handler = DbLogHandler()
    logger = logging.getLogger("test.location")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.error("Something broke")
    finally:
        logger.removeHandler(handler)

    _, fields = captured[-1]
    assert fields["level"] == "ERROR"
    assert "test_log_collector.py" in fields["location"]


def test_handler_never_raises(monkeypatch):
    def boom(kind, fields):
        raise RuntimeError("store down")
    monkeypatch.setattr(log_collector, "enqueue", boom)

    handler = DbLogHandler()
    logger = logging.getLogger("test.safe")
    logger.addHandler(handler)
    try:
        logger.info("should not propagate")  # must not raise
    finally:
        logger.removeHandler(handler)
