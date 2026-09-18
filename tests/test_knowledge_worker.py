from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.knowledge.worker import (
    InferenceDeadlineExceeded,
    InferenceQueueFullError,
    InferenceWorker,
    InferenceWorkerClosedError,
)


async def test_worker_runs_jobs_serially() -> None:
    worker = InferenceWorker(queue_size=2)
    first_entered = threading.Event()
    release_first = threading.Event()
    order: list[str] = []

    def first() -> str:
        first_entered.set()
        release_first.wait(2)
        order.append("first")
        return "one"

    def second() -> str:
        order.append("second")
        return "two"

    first_task = asyncio.create_task(worker.run(first, deadline=time.monotonic() + 5))
    try:
        assert await asyncio.to_thread(first_entered.wait, 1)
        second_task = asyncio.create_task(
            worker.run(second, deadline=time.monotonic() + 5)
        )
        await asyncio.sleep(0.02)
        assert order == []
        release_first.set()
        assert await first_task == "one"
        assert await second_task == "two"
        assert order == ["first", "second"]
    finally:
        release_first.set()
        await worker.aclose()


async def test_cancelled_running_job_keeps_slot_until_finished() -> None:
    worker = InferenceWorker(queue_size=4)
    entered = threading.Event()
    release = threading.Event()
    second = threading.Event()

    def first() -> None:
        entered.set()
        release.wait(2)

    task = asyncio.create_task(worker.run(first, deadline=time.monotonic() + 5))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        next_task = asyncio.create_task(
            worker.run(second.set, deadline=time.monotonic() + 5)
        )
        await asyncio.sleep(0.02)
        assert not second.is_set()
        release.set()
        await next_task
    finally:
        release.set()
        await worker.aclose()


async def test_cancelled_queued_job_keeps_churn_within_queue_bound() -> None:
    worker = InferenceWorker(queue_size=1)
    entered = threading.Event()
    release = threading.Event()
    queued_job_ran = threading.Event()

    def blocked() -> None:
        entered.set()
        release.wait(2)

    running = asyncio.create_task(worker.run(blocked, deadline=time.monotonic() + 5))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        admitted = 0
        rejected = 0
        for _ in range(8):
            candidate = asyncio.create_task(
                worker.run(queued_job_ran.set, deadline=time.monotonic() + 5)
            )
            await asyncio.sleep(0)
            if candidate.done():
                with pytest.raises(InferenceQueueFullError):
                    await candidate
                rejected += 1
            else:
                admitted += 1
                candidate.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await candidate

        assert admitted == 1
        assert rejected == 7
        assert not queued_job_ran.is_set()
    finally:
        release.set()
        await running
        await worker.aclose()


async def test_full_queue_rejects_new_job_without_running_it() -> None:
    worker = InferenceWorker(queue_size=1)
    entered = threading.Event()
    release = threading.Event()
    rejected = threading.Event()

    def blocked() -> None:
        entered.set()
        release.wait(2)

    running = asyncio.create_task(worker.run(blocked, deadline=time.monotonic() + 5))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        queued = asyncio.create_task(
            worker.run(lambda: None, deadline=time.monotonic() + 5)
        )
        await asyncio.sleep(0.02)
        with pytest.raises(InferenceQueueFullError):
            await worker.run(rejected.set, deadline=time.monotonic() + 5)
        assert not rejected.is_set()
        release.set()
        await running
        await queued
    finally:
        release.set()
        await worker.aclose()


async def test_expired_deadline_does_not_run_job() -> None:
    worker = InferenceWorker(queue_size=1)
    called = threading.Event()
    try:
        with pytest.raises(InferenceDeadlineExceeded):
            await worker.run(called.set, deadline=time.monotonic() - 0.01)
        assert not called.is_set()
    finally:
        await worker.aclose()


async def test_callable_timeout_error_is_not_relabelled_as_deadline() -> None:
    worker = InferenceWorker(queue_size=1)

    def fails() -> None:
        raise TimeoutError("model timeout")

    try:
        with pytest.raises(TimeoutError, match="model timeout"):
            await worker.run(fails, deadline=time.monotonic() + 2)
    finally:
        await worker.aclose()


async def test_close_waits_for_real_thread_and_rejects_new_jobs() -> None:
    worker = InferenceWorker(queue_size=1)
    entered = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        entered.set()
        release.wait(2)

    running = asyncio.create_task(worker.run(blocked, deadline=time.monotonic() + 5))
    assert await asyncio.to_thread(entered.wait, 1)
    close_task = asyncio.create_task(worker.aclose())
    await asyncio.sleep(0.02)
    assert not close_task.done()
    release.set()
    await running
    await close_task

    with pytest.raises(InferenceWorkerClosedError):
        await worker.run(lambda: None, deadline=time.monotonic() + 1)


async def test_cancelled_first_close_does_not_let_second_close_return_early() -> None:
    worker = InferenceWorker(queue_size=1)
    entered = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        entered.set()
        release.wait(2)

    running = asyncio.create_task(worker.run(blocked, deadline=time.monotonic() + 5))
    second_close: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        first_close = asyncio.create_task(worker.aclose())
        await asyncio.sleep(0.02)
        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close

        second_close = asyncio.create_task(worker.aclose())
        await asyncio.sleep(0.02)
        assert not second_close.done()
    finally:
        release.set()
        await running
        if second_close is not None:
            await second_close
        await worker.aclose()
