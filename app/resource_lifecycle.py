from __future__ import annotations

import asyncio
from typing import Any, Protocol


class _Warmable(Protocol):
    def warmup(self) -> None: ...


async def warmup_local_models(local_models: _Warmable) -> None:
    """Own and drain physical model initialization before propagating cancellation."""
    task = asyncio.create_task(asyncio.to_thread(local_models.warmup))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await _drain_task(task)
        raise


async def _drain_task(task: asyncio.Task[Any]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            return
    if not task.cancelled():
        try:
            task.exception()
        except BaseException:
            pass


async def close_resources(
    resources_to_close: list[Any],
    *,
    cancellation: asyncio.CancelledError | None = None,
) -> None:
    """Close owned resources once in reverse order, despite repeated cancellation."""
    cleanup_task = asyncio.create_task(_close_resources_once(resources_to_close))
    close_error: Exception | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            close_error = exc
            break

    if close_error is None:
        try:
            cleanup_task.result()
        except Exception as exc:
            close_error = exc

    if cancellation is not None:
        if close_error is not None:
            raise cancellation from close_error
        raise cancellation
    if close_error is not None:
        raise close_error


async def _close_resources_once(resources_to_close: list[Any]) -> None:
    first_error: Exception | None = None
    for resource in reversed(resources_to_close):
        close = getattr(resource, "aclose", None)
        if close is None:
            continue
        try:
            await close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
