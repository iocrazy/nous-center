"""Test isolation: detach the DB log handler during tests.

The global exception handler calls ``logger.exception(...)`` which, in
production, is forwarded by ``DbLogHandler`` to the structured-log store. Tests
that deliberately trigger 500s would otherwise enqueue noise. ``DbLogHandler``
is attached to the named loggers ``"src"`` and ``"nous"`` (not the root logger)
in ``main.py``'s lifespan; we detach it for the duration of each test.

(The store's ``enqueue`` is already a no-op when its writer isn't running, so
no low-level stub is needed anymore — the standalone SQLite ``log_db`` is gone.)
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
    try:
        yield
    finally:
        for lg, h in saved:
            lg.addHandler(h)
