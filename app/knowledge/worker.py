from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable
from typing import TypeVar, cast


T = TypeVar("T")
_ABANDONED = object()


class InferenceQueueFullError(RuntimeError):
    pass


class InferenceDeadlineExceeded(TimeoutError):
    pass


class InferenceWorkerClosedError(RuntimeError):
    pass


class InferenceWorker:
    def __init__(self, queue_size: int) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._capacity = queue_size + 1
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="knowledge-inference",
        )
        self._state_lock = threading.Lock()
        self._inflight = 0
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._shutdown_task: asyncio.Task[None] | None = None

    async def run(self, fn: Callable[[], T], *, deadline: float) -> T:
        if time.monotonic() >= deadline:
            raise InferenceDeadlineExceeded("inference deadline exceeded")

        with self._state_lock:
            if self._closed:
                raise InferenceWorkerClosedError("inference worker is closed")
            if self._inflight >= self._capacity:
                raise InferenceQueueFullError("inference queue is full")
            self._inflight += 1

        abandoned = threading.Event()

        def run_before_deadline() -> T | object:
            if abandoned.is_set():
                return _ABANDONED
            if time.monotonic() >= deadline:
                raise InferenceDeadlineExceeded("inference deadline exceeded")
            return fn()

        try:
            future = self._executor.submit(run_before_deadline)
        except BaseException:
            self._release_capacity()
            raise
        future.add_done_callback(lambda _future: self._release_capacity())
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(_consume_future_result)

        try:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                abandoned.set()
                raise InferenceDeadlineExceeded("inference deadline exceeded")
            result = await asyncio.wait_for(asyncio.shield(wrapped), timeout=timeout)
            if result is _ABANDONED:
                raise asyncio.CancelledError
            return cast(T, result)
        except asyncio.CancelledError:
            abandoned.set()
            raise
        except TimeoutError as exc:
            if future.done() and not future.cancelled():
                underlying = future.exception()
                if isinstance(underlying, TimeoutError):
                    raise underlying
            abandoned.set()
            raise InferenceDeadlineExceeded("inference deadline exceeded") from exc

    async def aclose(self) -> None:
        with self._state_lock:
            self._closed = True
        async with self._close_lock:
            if self._shutdown_task is None:
                self._shutdown_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._executor.shutdown,
                        wait=True,
                        cancel_futures=False,
                    )
                )
            shutdown_task = self._shutdown_task
        await asyncio.shield(shutdown_task)

    def _release_capacity(self) -> None:
        with self._state_lock:
            self._inflight -= 1


def _consume_future_result(future: asyncio.Future[object]) -> None:
    if future.cancelled():
        return
    future.exception()
