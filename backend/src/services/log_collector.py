"""Python logging handler that writes to logs.db with buffering."""
import logging
import threading
import time
from src.services.log_db import insert_app_log


class DbLogHandler(logging.Handler):
    """Buffered logging handler that writes to the log database."""

    def __init__(self, db_path: str | None = None, flush_interval: float = 1.0, flush_size: int = 50):
        super().__init__()
        self._db_path = db_path
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._flush_size = flush_size
        if flush_interval > 0:
            self._timer = threading.Thread(target=self._flush_loop, daemon=True)
            self._timer.start()

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "level": record.levelname,
            "module": record.name,
            "message": self.format(record),
            "location": f"{record.filename}:{record.lineno}",
        }
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._flush_size:
                self._do_flush()

    def flush(self) -> None:
        with self._lock:
            self._do_flush()

    def _do_flush(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        for entry in batch:
            try:
                insert_app_log(db_path=self._db_path, **entry)
            except Exception:
                pass  # Never let log writing crash the app

    def _flush_loop(self) -> None:
        while True:
            time.sleep(self._flush_interval)
            self.flush()
