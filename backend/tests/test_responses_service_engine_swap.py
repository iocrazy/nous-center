"""After migrating compact_messages -> GzipCompactContextEngine, behavior must match.

Subtask 3.3 regression guard: legacy shim `responses_service.compact_messages`
and new `GzipCompactContextEngine.compress` must produce identical output for
short input that doesn't trigger truncation. This protects the swap from silent
behavior drift.
"""

import pytest

from src.services.responses_service import compact_messages as legacy_compact


@pytest.mark.asyncio
async def test_engine_matches_legacy_for_short_input():
    from src.services.context.gzip_compact import GzipCompactContextEngine

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    legacy_out, legacy_trunc = legacy_compact(msgs, max_history_tokens=10_000)
    engine_out, engine_trunc = await GzipCompactContextEngine().compress(
        messages=msgs, max_tokens=10_000
    )
    assert legacy_out == engine_out
    assert legacy_trunc == engine_trunc
