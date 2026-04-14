"""Test isolation: prevent logs.db pollution during tests.

The global exception handler calls ``logger.exception(...)`` which, in production,
gets buffered by ``DbLogHandler`` and written to ``backend/data/logs.db``. Tests
that deliberately trigger 500s would otherwise pollute the real log DB and
appear as noise in the app's LogsOverlay.

``DbLogHandler`` is attached to the named loggers ``"src"`` and ``"nous"`` (not
the root logger) in ``main.py``'s lifespan. We detach from those specific
loggers and also monkeypatch ``log_db.insert_app_log`` as a belt-and-suspenders
guard against lifespan-startup handlers installed after the fixture runs.
"""

from __future__ import annotations

import logging
import pytest

_ATTACHED_LOGGERS = ("src", "nous")


@pytest.fixture(autouse=True)
def _silence_db_log_handler():
    saved: list[tuple[logging.Logger, logging.Handler]] = []
    for name in _ATTACHED_LOGGERS:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if h.__class__.__name__ == "DbLogHandler":
                lg.removeHandler(h)
                saved.append((lg, h))

    # Belt-and-suspenders: also stub the low-level write function so even
    # handlers installed later (e.g. inside a TestClient `with` block that
    # triggers lifespan) can't reach the DB.
    try:
        from src.services import log_db
        original = log_db.insert_app_log
        log_db.insert_app_log = lambda **kw: None
    except Exception:
        original = None

    try:
        yield
    finally:
        for lg, h in saved:
            lg.addHandler(h)
        if original is not None:
            from src.services import log_db
            log_db.insert_app_log = original
