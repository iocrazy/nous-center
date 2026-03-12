"""Snowflake ID generator.

Layout (64 bits total):
  - 1  bit: sign (always 0)
  - 41 bits: milliseconds since epoch (2024-01-01) → ~69 years
  - 10 bits: worker ID (0-1023)
  - 12 bits: sequence per millisecond (0-4095)
"""

import threading
import time

# Custom epoch: 2024-01-01 00:00:00 UTC
_EPOCH_MS = 1_704_067_200_000

_WORKER_BITS = 10
_SEQUENCE_BITS = 12

_MAX_WORKER_ID = (1 << _WORKER_BITS) - 1
_MAX_SEQUENCE = (1 << _SEQUENCE_BITS) - 1

_WORKER_SHIFT = _SEQUENCE_BITS
_TIMESTAMP_SHIFT = _WORKER_BITS + _SEQUENCE_BITS


class SnowflakeGenerator:
    def __init__(self, worker_id: int = 0):
        if not 0 <= worker_id <= _MAX_WORKER_ID:
            raise ValueError(f"worker_id must be 0-{_MAX_WORKER_ID}")
        self._worker_id = worker_id
        self._sequence = 0
        self._last_ts = -1
        self._lock = threading.Lock()

    def generate(self) -> int:
        with self._lock:
            ts = self._current_ms()
            if ts == self._last_ts:
                self._sequence = (self._sequence + 1) & _MAX_SEQUENCE
                if self._sequence == 0:
                    ts = self._wait_next_ms(ts)
            else:
                self._sequence = 0
            self._last_ts = ts
            return (
                ((ts - _EPOCH_MS) << _TIMESTAMP_SHIFT)
                | (self._worker_id << _WORKER_SHIFT)
                | self._sequence
            )

    @staticmethod
    def _current_ms() -> int:
        return int(time.time() * 1000)

    def _wait_next_ms(self, last_ts: int) -> int:
        ts = self._current_ms()
        while ts <= last_ts:
            ts = self._current_ms()
        return ts


# Default singleton
_default = SnowflakeGenerator(worker_id=1)


def snowflake_id() -> int:
    """Generate a unique Snowflake ID."""
    return _default.generate()
