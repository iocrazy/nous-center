"""Built-in ContextEngine implementation (migrated from responses_service.compact_messages).

Strategy: drop oldest non-system turns until estimated token count fits max_tokens.
Conservative over-estimation (len/2 + 4 per message) to avoid vLLM
`context_length_exceeded` mid-stream.
"""

from __future__ import annotations

import logging

from src.services.context.base import ContextEngine, ContextOverflowError

logger = logging.getLogger(__name__)


def _approx_tokens(messages: list[dict]) -> int:
    """Conservative OVER-estimate. Migrated from responses_service.approx_tokens."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c) // 2 + 4
        elif isinstance(c, list):
            for item in c:
                t = item.get("text", "")
                if isinstance(t, str):
                    total += len(t) // 2 + 4
                else:
                    total += 200  # image / other placeholder
    return total


class GzipCompactContextEngine(ContextEngine):
    name = "gzip-compact"

    async def initialize(self) -> None:
        pass  # stateless, nothing to do

    def should_compress(self, *, messages, max_tokens, current_tokens=None):
        est = current_tokens if current_tokens is not None else _approx_tokens(messages)
        return est > max_tokens

    async def compress(self, *, messages, max_tokens):
        """Drop oldest non-system turns until fit.

        Returns (compacted, was_truncated).
        Raises ContextOverflowError if final set (last turn + system) still exceeds.
        """
        if _approx_tokens(messages) <= max_tokens:
            return messages, False

        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        original_rest_len = len(rest)

        # Identify the last user message — this is the minimum we must preserve.
        last_user_idx = None
        for i in range(len(rest) - 1, -1, -1):
            if rest[i].get("role") == "user":
                last_user_idx = i
                break

        # Pop oldest until it fits, but never drop the last user message.
        while rest and _approx_tokens(system_msgs + rest) > max_tokens:
            if last_user_idx is not None and last_user_idx == 0:
                # About to drop the last user turn — stop and raise below.
                break
            rest.pop(0)
            if last_user_idx is not None:
                last_user_idx -= 1

        compacted = system_msgs + rest
        dropped = original_rest_len - len(rest)

        # 进程内累计 — 给 m04 dashboard 看 compaction 平均丢了多少 turn。
        try:
            from src.services.runtime_metrics import record_compaction
        except Exception:
            record_compaction = None

        if _approx_tokens(compacted) > max_tokens:
            if record_compaction:
                record_compaction(dropped, truncated=True)
            raise ContextOverflowError(
                f"context still exceeds max_tokens={max_tokens} "
                f"after compression (est={_approx_tokens(compacted)})"
            )

        if not rest:
            # 所有非 system 全被砍光了 — 说明最后一轮本身就超，raise
            if record_compaction:
                record_compaction(dropped, truncated=True)
            raise ContextOverflowError(
                "last turn alone exceeds max_tokens"
            )

        if record_compaction:
            record_compaction(dropped, truncated=False)
        return compacted, True
