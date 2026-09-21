from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from pydantic import SecretStr

import app.main as main_module
from app.main import create_app
from app.workflow.checkpoints import CheckpointStore as RealCheckpointStore
from tests.helpers import RecordingGateway, settings


class Resource:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    async def aclose(self) -> None:
        self.events.append(f"{self.name}_closed")


class FakeDatabase(Resource):
    def __init__(self, _url: str, events: list[str]) -> None:
        super().__init__("database", events)
        self.sessions = object()

    async def check(self) -> None:
        self.events.append("database_checked")


class FakeCheckpointStore(Resource):
    def __init__(self, _settings, events: list[str], *, fail_check=False) -> None:
        super().__init__("checkpoint", events)
        self.saver = object()
        self.fail_check = fail_check

    async def open(self):
        self.events.append("checkpoint_opened")
        return self.saver

    async def check(self) -> None:
        self.events.append("checkpoint_checked")
        if self.fail_check:
            raise RuntimeError("CHECKPOINT_SCHEMA_UNAVAILABLE")


class FakeService(Resource):
    pass


@dataclass
class FakeComponents:
    repository: object
    store: object
    local_models: object
    retriever: object = None
    low_confidence: object = None
    knowledge_gateway_factory: object = None


@pytest.fixture
def production_fakes(monkeypatch):
    events: list[str] = []
    components = FakeComponents(
        repository=object(),
        store=Resource("store", events),
        local_models=Resource("models", events),
    )
    dependencies = type("Dependencies", (), {
        "agent_dependencies": type("Agent", (), {
            "faq": object(), "tickets": object(), "executor": object(),
        })(),
        "actions": object(),
        "conversations": object(),
    })()

    monkeypatch.setattr(
        main_module, "Database", lambda url: FakeDatabase(url, events)
    )
    monkeypatch.setattr(
        main_module, "CheckpointStore", lambda cfg: FakeCheckpointStore(cfg, events)
    )

    async def migrate(_database, *, check_only):
        assert check_only is True
        events.append("mysql_schema_checked")
        return {"ready": 1, "backfilled": 0}

    async def knowledge(_settings, _database, _gateway, owned_resources):
        events.append("knowledge_checked")
        owned_resources.extend((components.store, components.local_models))
        return components

    def workflow(_settings, _database, _gateway, selected_components):
        assert selected_components is components
        events.append("workflow_dependencies_built")
        return dependencies

    def graph(selected_dependencies, saver):
        assert selected_dependencies is dependencies
        assert saver is not None
        events.append("workflow_built")
        return object()

    def service(_settings, _graph, conversations, _guard):
        assert conversations is dependencies.conversations
        events.append("service_built")
        return FakeService("service", events)

    monkeypatch.setattr(main_module, "migrate_workflow", migrate)
    monkeypatch.setattr(main_module, "build_knowledge_components", knowledge)
    monkeypatch.setattr(main_module, "build_workflow_dependencies", workflow)
    monkeypatch.setattr(main_module, "build_workflow", graph)
    monkeypatch.setattr(main_module, "WorkflowChatService", service)
    monkeypatch.setattr(main_module, "ActionService", lambda *args: object())
    return events, components


async def test_normal_startup_assembles_workflow_and_closes_service_first(
    production_fakes,
) -> None:
    events, components = production_fakes
    gateway = RecordingGateway()
    app = create_app(
        settings(
            database_url="mysql+asyncmy://local/test",
            checkpoint_database_url="postgresql://local/test",
        ),
        gateway,
    )

    async with app.router.lifespan_context(app):
        assert isinstance(app.state.chat_service, FakeService)
        assert app.state.knowledge_repository is components.repository
        assert app.state.action_service is not None

    assert events[:8] == [
        "database_checked",
        "mysql_schema_checked",
        "knowledge_checked",
        "checkpoint_opened",
        "checkpoint_checked",
        "workflow_dependencies_built",
        "workflow_built",
        "service_built",
    ]
    assert events[-5:] == [
        "service_closed",
        "checkpoint_closed",
        "models_closed",
        "store_closed",
        "database_closed",
    ]
    assert gateway.closed


async def test_missing_checkpoint_configuration_fails_without_building_graph(
    monkeypatch, production_fakes
) -> None:
    events, _ = production_fakes
    monkeypatch.setattr(main_module, "CheckpointStore", RealCheckpointStore)
    gateway = RecordingGateway()
    app = create_app(settings(database_url="mysql+asyncmy://local/test"), gateway)

    with pytest.raises(RuntimeError, match="CHECKPOINT_CONFIG_MISSING"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")

    assert "workflow_built" not in events
    assert events[-3:] == ["models_closed", "store_closed", "database_closed"]
    assert gateway.closed


async def test_missing_checkpoint_tables_fails_and_closes_partial_resources(
    monkeypatch, production_fakes
) -> None:
    events, _ = production_fakes
    monkeypatch.setattr(
        main_module,
        "CheckpointStore",
        lambda cfg: FakeCheckpointStore(cfg, events, fail_check=True),
    )
    app = create_app(
        settings(
            database_url="mysql+asyncmy://local/test",
            checkpoint_database_url="postgresql://local/test",
        ),
        RecordingGateway(),
    )

    with pytest.raises(RuntimeError, match="CHECKPOINT_SCHEMA_UNAVAILABLE"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")

    assert "workflow_built" not in events
    assert events[-5:] == [
        "checkpoint_checked",
        "checkpoint_closed",
        "models_closed",
        "store_closed",
        "database_closed",
    ]


async def test_real_checkpoint_missing_table_schema_fails_startup(
    monkeypatch, checkpoint_settings
) -> None:
    events: list[str] = []
    raw_url = checkpoint_settings.checkpoint_database_url.get_secret_value()
    separator = "&" if "?" in raw_url else "?"
    isolated_url = raw_url + separator + "options=-csearch_path%3Dch05_missing_startup"
    selected = checkpoint_settings.model_copy(
        update={
            "database_url": SecretStr("mysql+asyncmy://local/test"),
            "checkpoint_database_url": SecretStr(isolated_url),
        }
    )
    dependencies = type("Dependencies", (), {"conversations": object()})()

    monkeypatch.setattr(
        main_module, "Database", lambda url: FakeDatabase(url, events)
    )

    async def ready(_database, *, check_only):
        assert check_only is True
        return {"ready": 1, "backfilled": 0}

    monkeypatch.setattr(main_module, "migrate_workflow", ready)
    app = create_app(
        selected,
        RecordingGateway(),
        workflow_dependencies=dependencies,
    )

    with pytest.raises(RuntimeError, match="CHECKPOINT_SCHEMA_UNAVAILABLE"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail before graph compilation")

    assert events == ["database_checked", "database_closed"]


async def test_missing_mysql_migration_fails_before_knowledge_resources(
    monkeypatch, production_fakes
) -> None:
    events, _ = production_fakes

    async def missing(_database, *, check_only):
        assert check_only is True
        return {"ready": 0, "backfilled": 0}

    monkeypatch.setattr(main_module, "migrate_workflow", missing)
    app = create_app(
        settings(
            database_url="mysql+asyncmy://local/test",
            checkpoint_database_url="postgresql://local/test",
        ),
        RecordingGateway(),
    )
    with pytest.raises(RuntimeError, match="workflow migration"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")
    assert "knowledge_checked" not in events
    assert events[-1] == "database_closed"


async def test_explicit_service_does_not_create_production_resources(monkeypatch) -> None:
    service = FakeService("service", [])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("production resource was created")

    monkeypatch.setattr(main_module, "Database", forbidden)
    monkeypatch.setattr(main_module, "CheckpointStore", forbidden)
    app = create_app(settings(), RecordingGateway(), chat_service=service)
    async with app.router.lifespan_context(app):
        assert app.state.chat_service is service


async def test_explicit_service_without_gateway_does_not_create_model_owner(
    monkeypatch,
) -> None:
    service = FakeService("service", [])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("production model owner was created")

    monkeypatch.setattr(main_module, "OpenAIModelGateway", forbidden)
    app = create_app(settings(), chat_service=service)
    async with app.router.lifespan_context(app):
        assert app.state.chat_service is service
        assert not hasattr(app.state, "gateway")


async def test_repeated_startup_cancellation_drains_cleanup_once(
    monkeypatch, production_fakes
) -> None:
    events, _ = production_fakes
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_knowledge(*args):
        owned_resources = args[-1]
        resource = Resource("blocked", events)
        owned_resources.append(resource)
        entered.set()
        await release.wait()
        raise asyncio.CancelledError

    monkeypatch.setattr(main_module, "build_knowledge_components", blocked_knowledge)
    app = create_app(
        settings(
            database_url="mysql+asyncmy://local/test",
            checkpoint_database_url="postgresql://local/test",
        ),
        RecordingGateway(),
    )

    async def start():
        async with app.router.lifespan_context(app):
            pytest.fail("cancelled startup must not become ready")

    task = asyncio.create_task(start())
    await entered.wait()
    task.cancel()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events.count("blocked_closed") == 1
    assert events.count("database_closed") == 1
