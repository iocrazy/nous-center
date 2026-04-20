"""ContextEngine ABC (stateless, critical-path component).

Contract (spec 决策 9): engine is stateless. Per-session token accounting
lives in ResponseSession.total_input_tokens / total_output_tokens.
"""

from __future__ import annotations

from abc import abstractmethod

from src.services.memory.base import PluginBase


class ContextOverflowError(Exception):
    """Compression failed: even after pruning, context exceeds model's max_tokens."""


class ContextEngine(PluginBase):
    """ABC for context compression strategies.

    Contract:
    - Stateless: do not hold per-session data in instance attributes.
    - compress() is critical: must succeed or raise ContextOverflowError.
    - should_compress() is synchronous + cheap (O(messages)).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name (e.g. 'gzip-compact'). Used in logs/config."""

    @abstractmethod
    def should_compress(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        current_tokens: int | None = None,
    ) -> bool:
        """Return True if messages would exceed max_tokens and need compression."""

    @abstractmethod
    async def compress(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
    ) -> tuple[list[dict], bool]:
        """Compress messages to fit max_tokens.

        Returns (compacted_messages, was_truncated).

        Raises:
            ContextOverflowError: if even the minimal retained set (e.g. last turn)
            exceeds max_tokens. Callers should return HTTP 400 input_too_long.
        """

    def update_from_response(self, usage: dict) -> None:
        """Optional: track running usage. Default no-op.

        Prefer writing to ResponseSession.total_input_tokens etc. for persistence;
        this method is for in-memory engines that need to react to usage.
        """
