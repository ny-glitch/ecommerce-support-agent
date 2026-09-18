from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


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

    async def run(self, fn: Callable[[], T], *, deadline: float) -> T:
        if time.monotonic() >= deadline:
            raise InferenceDeadlineExceeded("inference deadline exceeded")

        with self._state_lock:
            if self._closed:
                raise InferenceWorkerClosedError("inference worker is closed")
            if self._inflight >= self._capacity:
                raise InferenceQueueFullError("inference queue is full")
            self._inflight += 1

        def run_before_deadline() -> T:
            if time.monotonic() >= deadline:
                raise InferenceDeadlineExceeded("inference deadline exceeded")
            return fn()

        try:
            future = self._executor.submit(run_before_deadline)
        except BaseException:
            self._release_capacity()
            raise
        future.add_done_callback(lambda _future: self._release_capacity())

        try:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                future.cancel()
                raise InferenceDeadlineExceeded("inference deadline exceeded")
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
        except TimeoutError as exc:
            if future.done() and not future.cancelled():
                underlying = future.exception()
                if isinstance(underlying, TimeoutError):
                    raise underlying
            future.cancel()
            raise InferenceDeadlineExceeded("inference deadline exceeded") from exc

    async def aclose(self) -> None:
        async with self._close_lock:
            with self._state_lock:
                already_closed = self._closed
                self._closed = True
            if already_closed:
                return
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=False)

    def _release_capacity(self) -> None:
        with self._state_lock:
            self._inflight -= 1
