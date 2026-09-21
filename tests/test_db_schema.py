from __future__ import annotations

from importlib import import_module, util

from pydantic import SecretStr
from sqlalchemy import CheckConstraint
from sqlalchemy.pool import NullPool

from app.config import Settings


EXPECTED_COLUMNS = {
    "faq": {"id", "question", "answer", "category"},
    "conversations": {"id", "user_id", "status", "created_at"},
    "messages": {
        "id",
        "conversation_id",
        "turn_id",
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "event_key",
        "event_data",
        "turn_status",
        "created_at",
    },
    "tickets": {
        "ticket_no",
        "conversation_id",
        "issue_description",
        "ticket_type",
        "status",
        "created_at",
    },
    "conversation_actions": {
        "action_id", "conversation_id", "turn_id", "action_type",
        "issue_description", "ticket_type", "status", "ticket_no",
        "created_at", "updated_at",
    },
    "knowledge_chunks": {
        "id",
        "category",
        "questions",
        "answer",
        "section_path",
        "content_type",
        "is_key_clause",
        "prev_chunk_id",
        "next_chunk_id",
        "vector_id",
        "vectorize_status",
        "created_at",
        "updated_at",
    },
    "qa_extraction_staging": {
        "id",
        "batch_no",
        "source_ref",
        "question",
        "answer",
        "status",
        "created_at",
    },
    "low_confidence_questions": {
        "id",
        "original_question",
        "conversation_id",
        "turn_id",
        "entry_point",
        "reason_code",
        "reason",
        "created_at",
    },
}


def load_db_module(name: str):
    package = util.find_spec("app.db")
    assert package is not None, "app.db package must provide the database layer"
    return import_module(f"app.db.{name}")


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "llm_base_url": "https://api.example.com/v1",
        "llm_model": "example-chat-model",
        "llm_api_key": "test-key",
    }
    values.update(overrides)
    return Settings(**values)


def test_exact_business_tables_and_columns() -> None:
    Base = load_db_module("models").Base

    assert set(Base.metadata.tables) == set(EXPECTED_COLUMNS)
    assert {
        name: set(table.columns.keys())
        for name, table in Base.metadata.tables.items()
    } == EXPECTED_COLUMNS


def test_required_fields_constraints_foreign_keys_and_indexes() -> None:
    Base = load_db_module("models").Base
    tables = Base.metadata.tables

    assert all(not column.nullable for column in tables["faq"].columns)
    assert all(not column.nullable for column in tables["conversations"].columns)
    assert {
        column.name for column in tables["messages"].columns if column.nullable
    } == {"tool_calls", "tool_call_id", "event_data"}
    assert tables["messages"].c.event_key.nullable is False
    assert all(not column.nullable for column in tables["conversation_actions"].columns)
    assert all(not column.nullable for column in tables["tickets"].columns)

    assert {
        table_name: {
            (foreign_key.parent.name, foreign_key.target_fullname)
            for foreign_key in tables[table_name].foreign_keys
        }
        for table_name in ("messages", "tickets", "conversation_actions")
    } == {
        "messages": {("conversation_id", "conversations.id")},
        "tickets": {("conversation_id", "conversations.id")},
        "conversation_actions": {("conversation_id", "conversations.id")},
    }

    check_names = {
        constraint.name
        for table in tables.values()
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert check_names == {
        "ck_conversations_status",
        "ck_messages_role",
        "ck_messages_turn_status",
        "ck_tickets_status",
        "ck_actions_type",
        "ck_actions_status",
    }

    message_indexes = {
        tuple(column.name for column in index.columns)
        for index in tables["messages"].indexes
    }
    assert message_indexes == {
        ("conversation_id", "id"),
        ("conversation_id", "turn_id"),
        ("conversation_id", "turn_id", "event_key"),
    }
    assert {
        index.name: tuple(column.name for column in index.columns)
        for index in tables["messages"].indexes if index.unique
    } == {"uq_messages_event": ("conversation_id", "turn_id", "event_key")}
    assert {
        index.name: (tuple(column.name for column in index.columns), index.unique)
        for index in tables["conversation_actions"].indexes
    } == {
        "uq_actions_turn_type": (("conversation_id", "turn_id", "action_type"), True),
        "uq_actions_ticket_no": (("ticket_no",), True),
    }


def test_knowledge_metadata_preserves_source_ddl_contract() -> None:
    Base = load_db_module("models").Base
    table = Base.metadata.tables["knowledge_chunks"]

    assert table.comment == "知识库 chunk 原文权威源"
    assert table.c.id.comment == "chunk 主键,与 Milvus 集合主键对齐"
    assert table.c.questions.comment == (
        "问法或本节标题,多个问法换行分隔,进向量化文本"
    )
    assert table.c.updated_at.server_default is not None
    assert "ON UPDATE CURRENT_TIMESTAMP" in str(table.c.updated_at.server_default.arg)
    assert {index.name for index in table.indexes} == {
        "idx_category",
        "idx_vectorize_status",
    }
    assert {
        (foreign_key.parent.name, foreign_key.target_fullname, foreign_key.ondelete)
        for foreign_key in table.foreign_keys
    } == {
        ("prev_chunk_id", "knowledge_chunks.id", "SET NULL"),
        ("next_chunk_id", "knowledge_chunks.id", "SET NULL"),
    }


def test_database_test_mode_uses_null_pool() -> None:
    Database = load_db_module("database").Database

    db = Database(
        "mysql+asyncmy://support_test:unused@127.0.0.1:13307/support_test",
        test_mode=True,
    )

    assert isinstance(db.engine.pool, NullPool)


def test_database_url_is_optional_and_secret() -> None:
    assert make_settings().database_url is None

    settings = make_settings(
        database_url="mysql+asyncmy://support:password@127.0.0.1:3307/support"
    )

    assert isinstance(settings.database_url, SecretStr)
    assert "password" not in repr(settings)
