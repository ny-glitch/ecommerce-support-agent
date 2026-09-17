"""Transport cancellation and deadlines for service event iterators."""
import asyncio
from collections.abc import AsyncIterator

from anyio import CancelScope
from fastapi import Request

from app.services.events import ChatEvent

# Retain tasks if third-party cleanup outlives the caller's bounded grace.
_closing_tasks: set[asyncio.Task] = set()


async def _close(upstream, pending) -> None:
    async def drain_and_close():
        if pending is not None:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await upstream.aclose()

    def finished(task):
        _closing_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    task = asyncio.create_task(drain_and_close())
    _closing_tasks.add(task)
    task.add_done_callback(finished)
    # The service has its own one-second cleanup grace. No cancel scope
    # crosses a yield, and we never wait indefinitely for cancellation unwind.
    with CancelScope(shield=True):
        try:
            async with asyncio.timeout(1.1):
                await asyncio.shield(task)
        except TimeoutError:
            task.cancel()


async def stream_events(
    request: Request, upstream: AsyncIterator[ChatEvent], *, deadline: float
) -> AsyncIterator[ChatEvent]:
    pending = None
    try:
        while not await request.is_disconnected():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError
            pending = asyncio.create_task(anext(upstream))
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                ready, _ = await asyncio.wait({pending}, timeout=min(0.05, remaining))
                if await request.is_disconnected():
                    return
                if ready:
                    break
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            yield event
            if event.name in {"done", "error"}:
                return
    except TimeoutError:
        yield ChatEvent("error", {
            "code": "UPSTREAM_TIMEOUT", "message": "模型服务响应超时，请重试"
        })
    finally:
        await _close(upstream, pending)
