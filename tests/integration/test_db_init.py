from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import DBAPIError


TABLE_NAMES = (
    "faq",
    "conversations",
    "messages",
    "tickets",
    "knowledge_chunks",
    "qa_extraction_staging",
    "low_confidence_questions",
)


@pytest.mark.asyncio
async def test_seed_is_idempotent_in_real_mysql(mysql_db) -> None:
    from app.db.seed import seed_database

    await seed_database(mysql_db)
    async with mysql_db.engine.connect() as connection:
        before = tuple(
            [
                (await connection.execute(text(f"SELECT COUNT(*) FROM {name}")))
                .scalar_one()
                for name in TABLE_NAMES
            ]
        )

    await seed_database(mysql_db)
    async with mysql_db.engine.connect() as connection:
        tables = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).get_table_names()
        )
        after = tuple(
            [
                (await connection.execute(text(f"SELECT COUNT(*) FROM {name}")))
                .scalar_one()
                for name in TABLE_NAMES
            ]
        )

    assert set(tables) == set(TABLE_NAMES)
    assert all(count > 0 for count in before[:4])
    assert before[4:] == (0, 0, 0)
    assert after == before


@pytest.mark.asyncio
async def test_mysql_knowledge_ddl_matches_authoritative_contract(mysql_db) -> None:
    async with mysql_db.engine.connect() as connection:
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"]: column
                for column in inspect(sync_connection).get_columns(
                    "knowledge_chunks"
                )
            }
        )
        foreign_keys = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).get_foreign_keys(
                "knowledge_chunks"
            )
        )
        table_comment = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).get_table_comment(
                "knowledge_chunks"
            )["text"]
        )
        create_sql = (
            await connection.execute(text("SHOW CREATE TABLE knowledge_chunks"))
        ).one()[1]
        staging_sql = (
            await connection.execute(
                text("SHOW CREATE TABLE qa_extraction_staging")
            )
        ).one()[1]
        low_confidence_indexes = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).get_indexes(
                "low_confidence_questions"
            )
        )
        low_confidence_unique = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).get_unique_constraints(
                "low_confidence_questions"
            )
        )

    assert table_comment == "知识库 chunk 原文权威源"
    assert columns["id"]["comment"] == "chunk 主键,与 Milvus 集合主键对齐"
    assert columns["questions"]["comment"] == (
        "问法或本节标题,多个问法换行分隔,进向量化文本"
    )
    assert "bigint unsigned" in create_sql.casefold()
    assert "enum('pending','done')" in create_sql.casefold()
    assert "on update current_timestamp" in create_sql.casefold()
    assert "charset=utf8mb4" in create_sql.casefold()
    assert "enum('extracted','kept','discarded')" in staging_sql.casefold()
    assert "charset=utf8mb4" in staging_sql.casefold()
    assert {
        tuple(constraint["column_names"])
        for constraint in low_confidence_unique
    } == {("conversation_id", "turn_id")}
    assert {
        tuple(index["column_names"])
        for index in low_confidence_indexes
        if not index["unique"]
    } == {("created_at",), ("reason_code",)}
    assert {
        (
            foreign_key["constrained_columns"][0],
            foreign_key["referred_table"],
            foreign_key["options"].get("ondelete"),
        )
        for foreign_key in foreign_keys
    } == {
        ("prev_chunk_id", "knowledge_chunks", "SET NULL"),
        ("next_chunk_id", "knowledge_chunks", "SET NULL"),
    }


@pytest.mark.asyncio
async def test_mysql_rejects_invalid_role_and_unknown_conversation(mysql_db) -> None:
    from app.db.models import Conversation, Message

    conversation_id = str(uuid4())
    async with mysql_db.sessions.begin() as session:
        session.add(
            Conversation(
                id=conversation_id,
                user_id="demo-user",
                status="open",
            )
        )

    with pytest.raises(DBAPIError):
        async with mysql_db.sessions.begin() as session:
            session.add(
                Message(
                    conversation_id=conversation_id,
                    turn_id="bad-role-turn",
                    role="invalid",
                    content="invalid",
                    turn_status="failed",
                )
            )

    with pytest.raises(DBAPIError):
        async with mysql_db.sessions.begin() as session:
            session.add(
                Message(
                    conversation_id=str(uuid4()),
                    turn_id="missing-conversation-turn",
                    role="user",
                    content="找不到会话",
                    turn_status="completed",
                )
            )

    async with mysql_db.sessions() as session:
        invalid_count = (
            await session.execute(
                select(func.count(Message.id)).where(
                    Message.turn_id.in_(
                        ("bad-role-turn", "missing-conversation-turn")
                    )
                )
            )
        ).scalar_one()

    assert invalid_count == 0


@pytest.mark.asyncio
async def test_json_tool_calls_and_chinese_round_trip(mysql_db) -> None:
    from app.db.models import Conversation, Message

    conversation_id = str(uuid4())
    tool_calls = [
        {
            "id": "call_demo",
            "type": "function",
            "function": {
                "name": "create_ticket",
                "arguments": {"问题": "商品破损", "数量": 1},
            },
        }
    ]
    async with mysql_db.sessions.begin() as session:
        session.add(
            Conversation(
                id=conversation_id,
                user_id="演示用户",
                status="open",
            )
        )
        await session.flush()
        session.add(
            Message(
                conversation_id=conversation_id,
                turn_id="中文工具轮次",
                role="assistant",
                content="我来为您创建售后工单。",
                tool_calls=tool_calls,
                tool_call_id="call_demo",
                turn_status="completed",
            )
        )

    async with mysql_db.sessions() as session:
        message = (
            await session.execute(
                select(Message).where(Message.conversation_id == conversation_id)
            )
        ).scalar_one()

    assert message.content == "我来为您创建售后工单。"
    assert message.tool_calls == tool_calls


@pytest.mark.asyncio
async def test_create_and_seed_do_not_overwrite_existing_records(mysql_db) -> None:
    from app.db.models import FAQ
    from app.db.seed import FAQ_RETURN_ID, seed_database

    await seed_database(mysql_db)
    async with mysql_db.sessions.begin() as session:
        faq = await session.get(FAQ, FAQ_RETURN_ID)
        assert faq is not None
        faq.answer = "客服人工补充后的答案"

    await mysql_db.create_schema()
    await seed_database(mysql_db)

    async with mysql_db.sessions() as session:
        faq = await session.get(FAQ, FAQ_RETURN_ID)

    assert faq is not None
    assert faq.answer == "客服人工补充后的答案"
