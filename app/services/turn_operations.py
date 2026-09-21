"""Owned physical work: cancellation is a request, not proof of completion."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from anyio import CancelScope

from app.errors import ServiceError

T = TypeVar('T')


async def bounded(
    factory: Callable[[], Awaitable[T]], deadline: float,
    operations: set[asyncio.Task], *, mutations: set[asyncio.Task] | None = None,
) -> T:
    # Each scope enters/exits in the same Task; never hold a timeout over yield.
    if asyncio.get_running_loop().time() >= deadline:
        raise TimeoutError

    async def invoke() -> T:
        return await factory()

    def finished(task: asyncio.Task) -> None:
        operations.discard(task)
        if mutations is not None:
            mutations.discard(task)
        if not task.cancelled():
            task.exception()  # Also retrieve failures after the consumer has left.

    task = asyncio.create_task(invoke())
    operations.add(task)
    if mutations is not None:
        mutations.add(task)
    task.add_done_callback(finished)
    try:
        async with asyncio.timeout_at(deadline):
            return await asyncio.shield(task)
    except (asyncio.CancelledError, TimeoutError):
        # A cancelled operation may await resource close in its finally block.
        # Let cleanup own that unwind instead of blocking this consumer.
        if not task.done() and not task.cancelling():
            task.cancel()
        raise


class TurnOperations:
    """Own all run() tasks and tracked iterators for exactly one turn.

    Consume tracked iterators through run(lambda: anext(iterator), deadline).
    Drain seals ownership, cancels each outstanding task at most once, waits
    for physical completion, then closes iterators. It has no grace timeout.
    Keep the session guard held until drain finishes, including when the
    waiting caller is cancelled repeatedly. Cancellation is re-raised only
    after completion. A stalled physical operation therefore retains the guard.
    A close error raises TURN_CLEANUP_FAILED; this is *not* a successful drain.
    """

    def __init__(self) -> None:
        self._operations: set[asyncio.Task] = set()
        self._mutations: set[asyncio.Task] = set()
        self._iterators: list = []
        self._drain_task: asyncio.Task | None = None

    async def run(
        self, factory: Callable[[], Awaitable[T]], deadline: float, *,
        mutation: bool = False,
    ) -> T:
        if self._drain_task is not None:
            raise RuntimeError('turn operations are draining or drained')
        return await bounded(factory, deadline, self._operations,
                             mutations=self._mutations if mutation else None)

    def track_iterator(self, iterator: T) -> T:
        if self._drain_task is not None:
            raise RuntimeError('turn operations are draining or drained')
        if not any(item is iterator for item in self._iterators):
            self._iterators.append(iterator)
        return iterator

    async def _drain(self) -> None:
        pending = tuple(self._operations)
        for task in pending:
            # Never interrupt a cancellation handler's delayed commit/close.
            if not task.done() and not task.cancelling():
                task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        failed = False
        for iterator in self._iterators:
            try:
                await iterator.aclose()
            except (Exception, asyncio.CancelledError):
                failed = True
        if failed:
            raise ServiceError('TURN_CLEANUP_FAILED', '本轮资源未能正常关闭', 503)
        self._iterators.clear()

    async def drain(self) -> None:
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._drain())
        task = self._drain_task
        cancelled = False
        # Match the existing ASGI cleanup shielding without a grace expiry.
        with CancelScope(shield=True):
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError
