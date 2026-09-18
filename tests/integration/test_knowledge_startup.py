from __future__ import annotations

import asyncio
import threading

import pytest

import app.main as main_module
from app.main import create_app
from tests.ch04_helpers import make_chunk
from tests.helpers import RecordingGateway, settings


class KnowledgeCapableGateway(RecordingGateway):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.knowledge_gateway = object()

    def create_knowledge_gateway(self):
        self.events.append("knowledge_gateway")
        return self.knowledge_gateway

    async def aclose(self):
        self.events.append("gateway_closed")
        await super().aclose()


def install_runtime_fakes(
    monkeypatch,
    events: list[str],
    *,
    warmup=None,
    close_models=None,
):
    class Database:
        def __init__(self, _url):
            self.sessions = object()
            self.engine = object()
            events.append("database_created")

        async def check(self):
            events.append("database_checked")

        async def aclose(self):
            events.append("database_closed")

    class Repository:
        def __init__(self, _sessions):
            events.append("repository_created")

        async def list_all(self):
            return [make_chunk(vectorize_status="done")]

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

    def calibration(_settings, *, corpus_fingerprint):
        events.append("calibration_checked")
        assert len(corpus_fingerprint) == 64
        return type("Artifact", (), {"threshold": 0.42})()

    monkeypatch.setattr(main_module, "Database", Database)
    monkeypatch.setattr(main_module, "KnowledgeRepository", Repository)
    monkeypatch.setattr(main_module, "MilvusStore", Store)
    monkeypatch.setattr(main_module, "LocalModels", Models)
    monkeypatch.setattr(main_module, "_check_knowledge_tables", check_tables)
    monkeypatch.setattr(main_module, "load_runtime_calibration", calibration)


async def test_production_lifespan_checks_then_assembles_and_closes_once(monkeypatch) -> None:
    events: list[str] = []
    install_runtime_fakes(monkeypatch, events)
    gateway = KnowledgeCapableGateway(events)
    app = create_app(
        settings(database_url="mysql+asyncmy://local/test"), gateway
    )

    async with app.router.lifespan_context(app):
        assert app.state.knowledge_repository is not None
        assert app.state.chat_service.knowledge_runner is not None
        assert events.index("tables_checked") < events.index("store_checked")
        assert events.index("calibration_checked") < events.index("warmup_started")
        assert events.index("warmup_finished") < events.index("knowledge_gateway")

    assert events.count("models_closed") == 1
    assert events.count("store_closed") == 1
    assert events.count("database_closed") == 1
    assert events.count("gateway_closed") == 1


async def test_startup_cancellation_drains_physical_warmup_before_cleanup(monkeypatch) -> None:
    events: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def blocked_warmup(_models):
        entered.set()
        release.wait(2)

    install_runtime_fakes(monkeypatch, events, warmup=blocked_warmup)
    gateway = KnowledgeCapableGateway(events)
    app = create_app(
        settings(database_url="mysql+asyncmy://local/test"), gateway
    )

    async def start() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("cancelled startup must not become ready")

    task = asyncio.create_task(start())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        assert "models_closed" not in events
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert events.index("warmup_finished") < events.index("models_closed")
    assert events[-4:] == [
        "models_closed",
        "store_closed",
        "database_closed",
        "gateway_closed",
    ]


async def test_repeated_cancellation_drains_all_cleanup_once(monkeypatch) -> None:
    events: list[str] = []
    warmup_entered = threading.Event()
    release_warmup = threading.Event()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()

    def blocked_warmup(_models):
        warmup_entered.set()
        release_warmup.wait(2)

    async def blocked_close(_models):
        close_entered.set()
        await release_close.wait()

    install_runtime_fakes(
        monkeypatch,
        events,
        warmup=blocked_warmup,
        close_models=blocked_close,
    )
    gateway = KnowledgeCapableGateway(events)
    app = create_app(
        settings(database_url="mysql+asyncmy://local/test"), gateway
    )

    async def start() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("cancelled startup must not become ready")

    task = asyncio.create_task(start())
    try:
        assert await asyncio.to_thread(warmup_entered.wait, 1)
        task.cancel()
        release_warmup.set()
        await asyncio.wait_for(close_entered.wait(), 1)

        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert "store_closed" not in events
    finally:
        release_warmup.set()
        release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert events.count("models_close_started") == 1
    assert events.count("models_closed") == 1
    assert events.count("store_closed") == 1
    assert events.count("database_closed") == 1
    assert events.count("gateway_closed") == 1
    assert events[-4:] == [
        "models_closed",
        "store_closed",
        "database_closed",
        "gateway_closed",
    ]


async def test_cancelled_cleanup_keeps_close_error_observable(monkeypatch) -> None:
    events: list[str] = []
    warmup_entered = threading.Event()
    release_warmup = threading.Event()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()

    def blocked_warmup(_models):
        warmup_entered.set()
        release_warmup.wait(2)

    async def failing_close(_models):
        close_entered.set()
        await release_close.wait()
        raise RuntimeError("model close failed")

    install_runtime_fakes(
        monkeypatch,
        events,
        warmup=blocked_warmup,
        close_models=failing_close,
    )
    gateway = KnowledgeCapableGateway(events)
    app = create_app(
        settings(database_url="mysql+asyncmy://local/test"), gateway
    )

    async def start() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("cancelled startup must not become ready")

    task = asyncio.create_task(start())
    try:
        assert await asyncio.to_thread(warmup_entered.wait, 1)
        task.cancel()
        release_warmup.set()
        await asyncio.wait_for(close_entered.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release_warmup.set()
        release_close.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "model close failed"
    assert events.count("models_close_started") == 1
    assert events.count("store_closed") == 1
    assert events.count("database_closed") == 1
    assert events.count("gateway_closed") == 1


async def test_missing_dependency_still_closes_created_resources(monkeypatch) -> None:
    events: list[str] = []
    install_runtime_fakes(monkeypatch, events)

    async def missing_tables(_database):
        raise RuntimeError(
            "knowledge tables are missing: qa_extraction_staging; run schema setup"
        )

    monkeypatch.setattr(main_module, "_check_knowledge_tables", missing_tables)
    gateway = KnowledgeCapableGateway(events)
    app = create_app(
        settings(database_url="mysql+asyncmy://local/test"), gateway
    )

    with pytest.raises(RuntimeError, match="qa_extraction_staging"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")

    assert "warmup_started" not in events
    assert events[-4:] == [
        "models_closed",
        "store_closed",
        "database_closed",
        "gateway_closed",
    ]
