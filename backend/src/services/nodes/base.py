"""Node Protocols (Wave 1 Task 4).

Each node class implements one or more of these Protocols, checked via
`isinstance(obj, Proto)` at dispatch time.

KEY CHANGE from Wave 0: StreamableNode accepts explicit on_token callback.
The global _on_progress_ref is eliminated (决策 7).
"""

from __future__ import annotations

from typing import AsyncIterator, Awaitable, Callable, Protocol, runtime_checkable


# Callback type for streaming nodes to push tokens back to caller.
OnTokenFn = Callable[[str], Awaitable[None]]


@runtime_checkable
class InvokableNode(Protocol):
    """Node that executes once, returns final dict."""

    async def invoke(self, data: dict, inputs: dict) -> dict: ...


@runtime_checkable
class StreamableNode(Protocol):
    """Node that streams tokens via on_token callback, returns final dict.

    on_token MUST be awaited for each chunk. Node is responsible for final
    usage aggregation in the returned dict.
    """

    async def stream(
        self,
        data: dict,
        inputs: dict,
        on_token: OnTokenFn,
    ) -> dict: ...


@runtime_checkable
class CollectableNode(Protocol):
    """Node that consumes an async stream of inputs and produces one final dict."""

    async def collect(
        self,
        data: dict,
        inputs_stream: AsyncIterator[dict],
    ) -> dict: ...


@runtime_checkable
class TransformableNode(Protocol):
    """Node that transforms a stream of inputs into a stream of outputs."""

    async def transform(
        self,
        data: dict,
        inputs_stream: AsyncIterator[dict],
    ) -> AsyncIterator[dict]: ...
