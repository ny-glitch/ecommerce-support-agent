from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import app.knowledge.milvus_store as milvus_module
from app.config import Settings
from app.knowledge.indexer import IndexerAlreadyRunningError, KnowledgeIndexer
from app.knowledge.milvus_store import (
    MilvusDeadlineExceeded,
    MilvusQueueFullError,
    MilvusStore,
)
from tests.ch04_helpers import make_chunk


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
        llm_api_key="test-key",
        knowledge_models_dir=tmp_path,
    )


async def _force_executor_cleanup(store: MilvusStore) -> None:
    await asyncio.to_thread(
        store._executor.shutdown,
        wait=True,
        cancel_futures=False,
    )


@pytest.mark.asyncio
async def test_cancelled_indexer_keeps_flock_until_physical_upsert_finishes(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class Client:
        def upsert(self, *_args: Any, **_kwargs: Any) -> None:
            entered.set()
            release.wait(2)
            finished.set()

        def close(self) -> None:
            return None

    inner = MilvusStore(_settings(tmp_path))
    inner._client = Client()  # type: ignore[assignment]

    class Repo:
        async def list_all(self):
            return [make_chunk()]

        async def mark_done_if_current(self, chunk_id: int, expected_hash: str):
            raise AssertionError("cancelled indexing must remain pending")

    class Store:
        async def ensure_schema(self) -> None:
            return None

        async def upsert(self, chunks, vectors, *, deadline: float) -> None:
            await inner._call("upsert", chunks, vectors, deadline=deadline)

    class Models:
        async def embed(self, texts, *, deadline: float):
            return [[1.0] + [0.0] * 1023 for _text in texts]

    lock_path = tmp_path / "index.lock"
    first = asyncio.create_task(
        KnowledgeIndexer(
            Repo(), Store(), Models(), lock_path=lock_path
        ).run()
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        first.cancel()
        await asyncio.sleep(0.02)
        assert not first.done()

        class EmptyRepo:
            async def list_all(self):
                return []

        with pytest.raises(IndexerAlreadyRunningError):
            await KnowledgeIndexer(
                EmptyRepo(), Store(), Models(), lock_path=lock_path
            ).run()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert finished.is_set()
        await inner.aclose()


@pytest.mark.asyncio
async def test_rpc_admission_stays_bounded_until_cancelled_job_dequeues(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    invoked: list[str] = []

    class Client:
        def execute(self, label: str, *, timeout: float) -> str:
            invoked.append(label)
            if label == "running":
                entered.set()
                release.wait(2)
            return label

        def close(self) -> None:
            return None

    store = MilvusStore(_settings(tmp_path))
    store._client = Client()  # type: ignore[assignment]
    running = asyncio.create_task(
        store._call("execute", "running", deadline=time.monotonic() + 5)
    )
    queued: asyncio.Task[Any] | None = None
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        queued = asyncio.create_task(
            store._call("execute", "cancelled", deadline=time.monotonic() + 5)
        )
        await asyncio.sleep(0.02)
        with pytest.raises(MilvusQueueFullError):
            await store._call(
                "execute", "rejected", deadline=time.monotonic() + 1
            )

        queued.cancel()
        await asyncio.sleep(0.02)
        assert not queued.done()
        with pytest.raises(MilvusQueueFullError):
            await store._call(
                "execute", "still-rejected", deadline=time.monotonic() + 1
            )
    finally:
        release.set()
        await running
        if queued is not None:
            with pytest.raises(asyncio.CancelledError):
                await queued
        await store.aclose()

    assert invoked == ["running"]


@pytest.mark.asyncio
async def test_queued_rpc_recomputes_timeout_and_rejects_expired_deadline(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()
    second_release = threading.Event()
    calls: list[tuple[str, float]] = []

    class Client:
        def execute(self, label: str, *, timeout: float) -> str:
            calls.append((label, timeout))
            if label == "running":
                entered.set()
                release.wait(2)
            if label == "second-running":
                second_entered.set()
                second_release.wait(2)
            return label

        def close(self) -> None:
            return None

    store = MilvusStore(_settings(tmp_path))
    store._client = Client()  # type: ignore[assignment]
    running = asyncio.create_task(
        store._call("execute", "running", deadline=time.monotonic() + 5)
    )
    expired: asyncio.Task[Any] | None = None
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        expired = asyncio.create_task(
            store._call("execute", "expired", deadline=time.monotonic() + 0.05)
        )
        await asyncio.sleep(0.08)
        release.set()
        await running
        with pytest.raises(MilvusDeadlineExceeded):
            await expired

        second_running = asyncio.create_task(
            store._call(
                "execute", "second-running", deadline=time.monotonic() + 5
            )
        )
        assert await asyncio.to_thread(second_entered.wait, 1)
        deadline = time.monotonic() + 0.5
        fresh = asyncio.create_task(
            store._call("execute", "fresh", deadline=deadline)
        )
        await asyncio.sleep(0.1)
        second_release.set()
        await second_running
        assert await fresh == "fresh"
        fresh_timeout = dict(calls)["fresh"]
        assert 0 < fresh_timeout < 0.45
        assert fresh_timeout <= deadline - time.monotonic() + 0.05
    finally:
        release.set()
        second_release.set()
        if not running.done():
            await running
        if expired is not None and not expired.done():
            with pytest.raises(MilvusDeadlineExceeded):
                await expired
        await store.aclose()

    assert [label for label, _timeout in calls] == [
        "running",
        "second-running",
        "fresh",
    ]


@pytest.mark.asyncio
async def test_cancelled_close_is_shared_and_waits_for_active_rpc(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class Client:
        def __init__(self) -> None:
            self.close_calls = 0

        def execute(self, *, timeout: float) -> None:
            entered.set()
            release.wait(2)

        def close(self) -> None:
            self.close_calls += 1

    client = Client()
    store = MilvusStore(_settings(tmp_path))
    store._client = client  # type: ignore[assignment]
    rpc = asyncio.create_task(
        store._call("execute", deadline=time.monotonic() + 5)
    )
    second_close: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        first_close = asyncio.create_task(store.aclose())
        await asyncio.sleep(0.02)
        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close

        second_close = asyncio.create_task(store.aclose())
        await asyncio.sleep(0.02)
        assert not second_close.done()
    finally:
        release.set()
        await rpc
        if second_close is not None:
            await second_close
        else:
            await store.aclose()
        await _force_executor_cleanup(store)

    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_close_does_not_block_loop_and_closes_client_created_inflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    instances: list[Any] = []

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            self.close_calls = 0
            instances.append(self)
            constructor_entered.set()
            release_constructor.wait(2)

        def execute(self, *, timeout: float) -> str:
            return "done"

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(milvus_module, "MilvusClient", Client)
    store = MilvusStore(_settings(tmp_path))
    rpc = asyncio.create_task(
        store._call("execute", deadline=time.monotonic() + 5)
    )
    close_task: asyncio.Task[None] | None = None
    timer = threading.Timer(0.3, release_constructor.set)
    timer.start()
    try:
        assert await asyncio.to_thread(constructor_entered.wait, 1)
        started = time.monotonic()
        close_task = asyncio.create_task(store.aclose())
        await asyncio.sleep(0.02)
        assert time.monotonic() - started < 0.15
        assert not close_task.done()
    finally:
        release_constructor.set()
        timer.cancel()
        assert await rpc == "done"
        if close_task is not None:
            await close_task
        else:
            await store.aclose()
        await _force_executor_cleanup(store)

    assert len(instances) == 1
    assert instances[0].close_calls == 1
