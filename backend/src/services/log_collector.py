"""Python logging handler that forwards app logs to the structured-log store.

Buffering/batching now lives in ``log_store.LogWriter`` (one async consumer), so
this handler is a thin adapter: ``emit`` just enqueues. ``enqueue`` is
thread-safe and non-blocking (logging can fire from any thread), and drops
silently if the writer isn't running — it never raises into the logging path.
"""
import logging

from src.services.log_store import enqueue


class DbLogHandler(logging.Handler):
    """Forward log records into the PG-backed log store (non-blocking)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            enqueue("app", {
                "level": record.levelname,
                "module": record.name,
                "message": self.format(record),
                "location": f"{record.filename}:{record.lineno}",
            })
        except Exception:  # never let log writing crash the app
            pass
