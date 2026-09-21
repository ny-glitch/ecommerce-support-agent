from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from psycopg import ProgrammingError
from psycopg.conninfo import conninfo_to_dict
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from app.db.contracts import TurnRef


class MySQLTestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.test",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    test_database_url: SecretStr | None = None


class PostgreSQLTestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.test",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    test_checkpoint_database_url: SecretStr | None = None


def load_test_checkpoint_database_url(require_postgres: bool) -> str:
    secret = PostgreSQLTestSettings().test_checkpoint_database_url
    if secret is None:
        if require_postgres:
            pytest.fail(
                "--require-postgres requires TEST_CHECKPOINT_DATABASE_URL"
            )
        pytest.skip("TEST_CHECKPOINT_DATABASE_URL is not configured")

    raw_url = secret.get_secret_value()
    try:
        connection = conninfo_to_dict(raw_url)
    except ProgrammingError:
        pytest.fail("TEST_CHECKPOINT_DATABASE_URL is not a valid PostgreSQL URL")
    if (
        connection.get("host") != "127.0.0.1"
        or connection.get("port") != "15433"
        or connection.get("dbname") != "support_graph_test"
        or connection.get("user") != "support_graph_test"
    ):
        pytest.fail(
            "TEST_CHECKPOINT_DATABASE_URL must use PostgreSQL, the isolated "
            "support_graph_test account and database, and 127.0.0.1:15433"
        )
    return raw_url


async def clear_checkpoint_test_threads(store) -> None:
    saver = await store.open()
    thread_ids = {
        checkpoint.config["configurable"]["thread_id"]
        async for checkpoint in saver.alist(None)
        if checkpoint.config["configurable"]["thread_id"].startswith(
            "ch05_test_"
        )
    }
    for thread_id in thread_ids:
        if not thread_id.startswith("ch05_test_"):
            raise RuntimeError("refusing to clean a non-test checkpoint thread")
        await saver.adelete_thread(thread_id)


@pytest_asyncio.fixture
async def checkpoint_settings(request: pytest.FixtureRequest):
    from app.config import Settings
    from app.workflow.checkpoints import CheckpointStore

    raw_url = load_test_checkpoint_database_url(
        require_postgres=request.config.getoption("--require-postgres")
    )
    settings = Settings(
        _env_file=None,
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
        llm_api_key="test-key",
        checkpoint_database_url=raw_url,
    )
    store = CheckpointStore(settings, test_mode=True)
    ready = False
    try:
        try:
            await store.setup()
            await store.check()
            ready = True
        except Exception as exc:
            if request.config.getoption("--require-postgres"):
                pytest.fail(f"--require-postgres requires PostgreSQL 17.11: {exc}")
            pytest.skip(f"PostgreSQL 17.11 is not available: {exc}")
        await clear_checkpoint_test_threads(store)
        yield settings
    finally:
        if ready:
            await clear_checkpoint_test_threads(store)
        await store.aclose()


def load_test_database_url(require_mysql: bool) -> str:
    secret = MySQLTestSettings().test_database_url
    if secret is None:
        if require_mysql:
            pytest.fail("--require-mysql requires TEST_DATABASE_URL")
        pytest.skip("TEST_DATABASE_URL is not configured")

    raw_url = secret.get_secret_value()
    try:
        url = make_url(raw_url)
    except ArgumentError:
        pytest.fail("TEST_DATABASE_URL is not a valid SQLAlchemy URL")
    if (
        url.drivername != "mysql+asyncmy"
        or url.host != "127.0.0.1"
        or url.port != 13307
        or url.database != "support_test"
        or url.username != "support_test"
    ):
        pytest.fail(
            "TEST_DATABASE_URL must use mysql+asyncmy, the isolated support_test "
            "account, database, and 127.0.0.1:13307"
        )
    return raw_url


async def clear_business_rows(db) -> None:
    async with db.engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE knowledge_chunks "
                "SET prev_chunk_id = NULL, next_chunk_id = NULL"
            )
        )
        for table_name in (
            "low_confidence_questions",
            "qa_extraction_staging",
            "knowledge_chunks",
        ):
            await connection.execute(text(f"DELETE FROM {table_name}"))
        for table_name in ("tickets", "messages", "conversations", "faq"):
            await connection.execute(text(f"DELETE FROM {table_name}"))


@pytest_asyncio.fixture
async def mysql_db(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    from app.db.database import Database

    database_url = load_test_database_url(
        require_mysql=request.config.getoption("--require-mysql")
    )
    db = Database(database_url, test_mode=True)
    schema_ready = False
    try:
        await db.check()
        await db.create_schema()
        schema_ready = True
        await clear_business_rows(db)
        yield db
    finally:
        if schema_ready:
            await clear_business_rows(db)
        await db.aclose()


@pytest_asyncio.fixture
async def repos(mysql_db):
    from app.db.conversations import ConversationRepository
    from app.db.faq import FaqRepository
    from app.db.seed import seed_database
    from app.db.tickets import TicketRepository

    await seed_database(mysql_db)
    return (
        ConversationRepository(mysql_db.sessions),
        FaqRepository(mysql_db.sessions),
        TicketRepository(mysql_db.sessions),
    )


@pytest_asyncio.fixture
async def new_turn(repos) -> TurnRef:
    conversations, _, _ = repos
    ref = TurnRef(conversation_id=str(uuid4()), turn_id=str(uuid4()))
    await conversations.create(ref.conversation_id, "demo")
    return ref


@pytest_asyncio.fixture
async def milvus_store(request: pytest.FixtureRequest):
    from pymilvus import MilvusClient

    from app.config import Settings
    from app.knowledge.milvus_store import MilvusStore, validate_test_collection

    collection_name = f"ch04_test_{uuid4().hex}"
    settings = Settings(
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
        llm_api_key="test-key",
    )
    store = MilvusStore(settings, collection_name=collection_name)
    available = False
    try:
        try:
            await store.check()
            await store.ensure_schema()
            available = True
        except Exception as exc:
            if request.config.getoption("--require-milvus"):
                pytest.fail(f"--require-milvus requires Milvus 2.6.23: {exc}")
            pytest.skip(f"Milvus 2.6.23 is not available: {exc}")
        yield store, settings, collection_name
    finally:
        await store.aclose()
        if available:
            validate_test_collection(collection_name)

            def cleanup() -> None:
                client = MilvusClient(uri=settings.milvus_uri)
                try:
                    if client.has_collection(collection_name):
                        client.drop_collection(collection_name)
                finally:
                    client.close()

            try:
                await asyncio.to_thread(cleanup)
            except Exception:
                if request.config.getoption("--require-milvus"):
                    raise
