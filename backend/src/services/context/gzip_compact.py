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
                # round4:① 裸 str / 非 dict item 不能 .get(否则 AttributeError 崩 500)。
                if not isinstance(item, dict):
                    total += 4
                    continue
                # ② image part 没有 "text" 键。早先 `get("text","")` 兜底成空串 → 命中
                #    str 分支只记 4,`else: 200` 成死代码 → 多模态请求严重低估、不触发
                #    压缩 → vLLM context_length_exceeded。用无默认 get 区分真有文本 vs 图。
                txt = item.get("text")
                if isinstance(txt, str):
                    total += len(txt) // 2 + 4
                else:
                    total += 200  # image / 其它非文本 part
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

        n = len(messages)
        # 最后一个 user 的位置 —— 最小必须保留,绝不丢。
        last_user_pos = None
        for i in range(n - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_pos = i
                break

        # round2 #9:**保持原始相对顺序**丢最旧的非 system 消息,而非把所有 system 提到最前。
        # 旧逻辑 `system_msgs + rest` 会把 responses 路由刻意放在历史之后的 instructions
        # (system role)拉到 agent 提示词前,丢失「instructions 覆盖历史」语义。system 不可丢;
        # 从最旧的非 system 往后丢(跳过最后一个 user),survivor 按原顺序保留。
        droppable = [
            i for i in range(n)
            if messages[i].get("role") != "system" and i != last_user_pos
        ]
        dropped_idxs: set[int] = set()

        def _surviving() -> list:
            return [m for i, m in enumerate(messages) if i not in dropped_idxs]

        di = 0
        while _approx_tokens(_surviving()) > max_tokens and di < len(droppable):
            dropped_idxs.add(droppable[di])
            di += 1

        compacted = _surviving()
        dropped = len(dropped_idxs)

        # 进程内累计 — 给 m04 dashboard 看 compaction 平均丢了多少 turn。
        try:
            from src.services.runtime_metrics import record_compaction
        except Exception:
            record_compaction = None

        if _approx_tokens(compacted) > max_tokens:
            # 能丢的都丢了仍超 → 最后一轮(+system)本身就超预算。
            if record_compaction:
                record_compaction(dropped, truncated=True)
            raise ContextOverflowError(
                f"context still exceeds max_tokens={max_tokens} "
                f"after compression (est={_approx_tokens(compacted)})"
            )

        if record_compaction:
            record_compaction(dropped, truncated=False)
        return compacted, True
