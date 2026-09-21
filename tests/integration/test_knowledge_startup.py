from __future__ import annotations

import asyncio
import threading

import pytest

import app.knowledge.bootstrap as bootstrap_module
from app.knowledge.bootstrap import build_knowledge_components
from app.resource_lifecycle import close_resources
from tests.ch04_helpers import make_chunk
from tests.helpers import RecordingGateway, settings


class KnowledgeCapableGateway(RecordingGateway):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def create_knowledge_gateway(self, **_kwargs):
        self.events.append("knowledge_gateway")
        return object()


def install_runtime_fakes(
    monkeypatch,
    events: list[str],
    *,
    chunks=None,
    manifest_error: Exception | None = None,
    warmup=None,
    close_models=None,
):
    selected_chunks = [make_chunk(vectorize_status="done")] if chunks is None else chunks

    class Repository:
        def __init__(self, _sessions):
            events.append("repository_created")

        async def list_all(self):
            events.append("corpus_checked")
            return selected_chunks

    class Store:
        def __init__(self, _settings):
            events.append("store_created")

        async def prepare_existing_collection(self):
            events.append("store_checked")

        async def aclose(self):
            events.append("store_closed")

    class Models:
        def __init__(self, _settings):
            events.append("models_created")

        def warmup(self):
            events.append("warmup_started")
            if warmup is not None:
                warmup(self)
            events.append("warmup_finished")

        async def aclose(self):
            events.append("models_close_started")
            if close_models is not None:
                await close_models(self)
            events.append("models_closed")

    async def check_tables(_database):
        events.append("tables_checked")

    def manifest(_settings):
        events.append("manifest_checked")
        if manifest_error is not None:
            raise manifest_error
        return "a" * 64

    monkeypatch.setattr(bootstrap_module, "KnowledgeRepository", Repository)
    monkeypatch.setattr(bootstrap_module, "MilvusStore", Store)
    monkeypatch.setattr(bootstrap_module, "LocalModels", Models)
    monkeypatch.setattr(bootstrap_module, "check_knowledge_tables", check_tables)
    monkeypatch.setattr(bootstrap_module, "model_manifest_fingerprint", manifest)
    monkeypatch.setattr(
        bootstrap_module, "LowConfidenceRepository", lambda sessions: object()
    )


def fake_database():
    return type("Database", (), {"sessions": object()})()


async def test_components_validate_existing_resources_then_warm_models(monkeypatch) -> None:
    events: list[str] = []
    install_runtime_fakes(monkeypatch, events)
    owned = []
    gateway = KnowledgeCapableGateway(events)

    components = await build_knowledge_components(
        settings(), fake_database(), gateway, owned
    )

    assert components.repository is not None
    assert components.retriever is not None
    assert components.knowledge_gateway_factory == gateway.create_knowledge_gateway
    assert events == [
        "store_created", "models_created", "tables_checked", "store_checked",
        "repository_created", "corpus_checked", "manifest_checked",
        "warmup_started", "warmup_finished",
    ]
    assert owned == [components.store, components.local_models]
    await close_resources(owned)


@pytest.mark.parametrize(
    ("chunks", "manifest_error", "message"),
    [
        ([], None, "knowledge corpus is empty"),
        (None, RuntimeError("local model manifest revision mismatch"), "manifest revision"),
    ],
)
async def test_missing_corpus_or_bad_model_manifest_fails_before_warmup_and_closes(
    monkeypatch, chunks, manifest_error, message
) -> None:
    events: list[str] = []
    install_runtime_fakes(
        monkeypatch, events, chunks=chunks, manifest_error=manifest_error
    )
    owned = []
    with pytest.raises(RuntimeError, match=message):
        try:
            await build_knowledge_components(
                settings(), fake_database(), KnowledgeCapableGateway(events), owned
            )
        finally:
            await close_resources(owned)

    assert "warmup_started" not in events
    assert events[-2:] == ["models_closed", "store_closed"]


async def test_startup_cancellation_drains_physical_warmup_before_cleanup(monkeypatch) -> None:
    events: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def blocked_warmup(_models):
        entered.set()
        release.wait(2)

    install_runtime_fakes(monkeypatch, events, warmup=blocked_warmup)
    owned = []

    async def start():
        cancellation = None
        try:
            await build_knowledge_components(
                settings(), fake_database(), KnowledgeCapableGateway(events), owned
            )
        except asyncio.CancelledError as exc:
            cancellation = exc
            raise
        finally:
            await close_resources(owned, cancellation=cancellation)

    task = asyncio.create_task(start())
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert "models_closed" not in events
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events.index("warmup_finished") < events.index("models_closed")
    assert events[-2:] == ["models_closed", "store_closed"]


async def test_repeated_cancellation_drains_cleanup_once(monkeypatch) -> None:
    events: list[str] = []
    entered = threading.Event()
    release_warmup = threading.Event()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()

    def blocked_warmup(_models):
        entered.set()
        release_warmup.wait(2)

    async def blocked_close(_models):
        close_entered.set()
        await release_close.wait()

    install_runtime_fakes(
        monkeypatch, events, warmup=blocked_warmup, close_models=blocked_close
    )
    owned = []

    async def start():
        cancellation = None
        try:
            await build_knowledge_components(
                settings(), fake_database(), KnowledgeCapableGateway(events), owned
            )
        except asyncio.CancelledError as exc:
            cancellation = exc
            raise
        finally:
            await close_resources(owned, cancellation=cancellation)

    task = asyncio.create_task(start())
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    release_warmup.set()
    await close_entered.wait()
    task.cancel()
    assert not task.done()
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events.count("models_close_started") == 1
    assert events.count("models_closed") == 1
    assert events.count("store_closed") == 1
