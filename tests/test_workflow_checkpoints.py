from __future__ import annotations

import traceback
from dataclasses import dataclass

import pytest

from app.config import Settings
from app.workflow.checkpoints import CheckpointStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from scripts import init_workflow_checkpoints


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "llm_base_url": "https://api.example.com/v1",
        "llm_model": "example-chat-model",
        "llm_api_key": "test-key",
    }
    values.update(overrides)
    return Settings(**values)


@dataclass
class ControlledSaver:
    setup_calls: int = 0
    setup_error: Exception | None = None
    check_error: Exception | None = None

    async def setup(self) -> None:
        self.setup_calls += 1
        if self.setup_error is not None:
            raise self.setup_error

    async def aget_tuple(self, config: dict[str, object]) -> None:
        assert config == {"configurable": {"thread_id": "healthcheck-only"}}
        if self.check_error is not None:
            raise self.check_error


@dataclass
class ControlledManager:
    saver: ControlledSaver
    enter_calls: int = 0
    exit_calls: int = 0
    enter_error: Exception | None = None

    async def __aenter__(self) -> ControlledSaver:
        self.enter_calls += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self.saver

    async def __aexit__(self, *_exc_info: object) -> None:
        self.exit_calls += 1


def install_manager(
    monkeypatch: pytest.MonkeyPatch,
    manager: ControlledManager,
) -> list[str]:
    urls: list[str] = []

    def from_conn_string(url: str) -> ControlledManager:
        urls.append(url)
        return manager

    monkeypatch.setattr(
        AsyncPostgresSaver,
        "from_conn_string",
        staticmethod(from_conn_string),
    )
    return urls


async def test_missing_url_fails_without_opening_a_connection() -> None:
    store = CheckpointStore(make_settings())

    with pytest.raises(RuntimeError, match="CHECKPOINT_DATABASE_URL") as exc_info:
        await store.open()

    assert "postgresql://" not in str(exc_info.value)


async def test_setup_is_explicit_and_open_only_enters_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = ControlledSaver()
    manager = ControlledManager(saver)
    urls = install_manager(monkeypatch, manager)
    secret_url = "postgresql://support_graph:secret@127.0.0.1:5433/support_graph"
    store = CheckpointStore(make_settings(checkpoint_database_url=secret_url))

    opened = await store.open()

    assert opened is saver
    assert urls == [secret_url]
    assert saver.setup_calls == 0

    await store.setup()

    assert saver.setup_calls == 1
    await store.aclose()


async def test_check_wraps_schema_error_without_exposing_connection_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_url = "postgresql://support_graph:secret@127.0.0.1:5433/support_graph"
    saver = ControlledSaver(
        check_error=RuntimeError(f'relation "checkpoints" missing at {secret_url}')
    )
    manager = ControlledManager(saver)
    install_manager(monkeypatch, manager)
    store = CheckpointStore(make_settings(checkpoint_database_url=secret_url))

    with pytest.raises(RuntimeError, match="checkpoint schema") as exc_info:
        await store.check()

    assert secret_url not in str(exc_info.value)
    await store.aclose()


async def test_connection_error_traceback_has_safe_code_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_url = "postgresql://support_graph:obvious-fake-secret@db.invalid/graph"
    manager = ControlledManager(
        ControlledSaver(),
        enter_error=RuntimeError(f"could not connect using {secret_url}"),
    )
    install_manager(monkeypatch, manager)
    store = CheckpointStore(make_settings(checkpoint_database_url=secret_url))

    with pytest.raises(RuntimeError, match="CHECKPOINT_CONNECTION_FAILED") as exc_info:
        await store.open()

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "obvious-fake-secret" not in formatted
    assert secret_url not in formatted


async def test_init_cli_error_traceback_has_safe_code_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_url = "postgresql://support_graph:obvious-fake-secret@db.invalid/graph"
    saver = ControlledSaver(
        setup_error=PermissionError(f"permission denied for {secret_url}")
    )
    manager = ControlledManager(saver)
    install_manager(monkeypatch, manager)
    settings = make_settings(checkpoint_database_url=secret_url)
    monkeypatch.setattr(init_workflow_checkpoints, "load_settings", lambda: settings)

    with pytest.raises(RuntimeError, match="CHECKPOINT_SETUP_FAILED") as exc_info:
        await init_workflow_checkpoints.initialize_workflow_checkpoints()

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "obvious-fake-secret" not in formatted
    assert secret_url not in formatted
    assert manager.exit_calls == 1


async def test_aclose_exits_owned_manager_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControlledManager(ControlledSaver())
    install_manager(monkeypatch, manager)
    store = CheckpointStore(
        make_settings(
            checkpoint_database_url=(
                "postgresql://support_graph:secret@127.0.0.1:5433/support_graph"
            )
        )
    )
    await store.open()

    await store.aclose()
    await store.aclose()

    assert manager.enter_calls == 1
    assert manager.exit_calls == 1
