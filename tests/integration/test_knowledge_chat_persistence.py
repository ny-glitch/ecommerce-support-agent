from __future__ import annotations

import json

from sqlalchemy import func, select

from app.db.conversations import ConversationRepository
from app.db.knowledge_models import LowConfidenceQuestion
from app.db.low_confidence import LowConfidenceRepository
from app.sessions import SessionGuard
from app.services.chat import ChatService
from app.tools.executor import ToolExecutor
from tests.ch02_helpers import ChatGateway, selection
from tests.ch04_helpers import FakeKnowledgePipeline, make_decision
from tests.helpers import settings


async def test_refusal_pool_and_tool_audit_persist_atomically_in_order(repos, mysql_db) -> None:
    conversations, faq, tickets = repos
    gateway = ChatGateway()
    gateway.selection = selection("query_faq", {})
    pipeline = FakeKnowledgePipeline(make_decision(status="not_found"))
    service = ChatService(
        settings(), gateway, conversations, faq, tickets, SessionGuard(10), ToolExecutor(),
        knowledge_pipeline=pipeline,
        low_confidence=LowConfidenceRepository(mysql_db.sessions),
    )

    async with service.prepare("C65-Pro支持什么协议？", None, category="数码配件") as prepared:
        events = [event async for event in service.stream(prepared)]

    assert events[-1].name == "done"
    audit = await conversations.audit(prepared.ref.conversation_id, "demo")
    assert [row["role"] for row in audit] == ["user", "assistant", "tool", "assistant"]
    assert {row["turn_status"] for row in audit} == {"completed"}
    assert json.loads(audit[2]["content"])["status"] == "not_found"
    async with mysql_db.sessions() as session:
        pooled = (await session.execute(select(LowConfidenceQuestion))).scalar_one()
    assert pooled.original_question == "C65-Pro支持什么协议？"
    assert pooled.conversation_id == prepared.ref.conversation_id
    assert pooled.turn_id == prepared.ref.turn_id


async def test_large_knowledge_result_survives_database_round_trip(repos, mysql_db) -> None:
    conversations, faq, tickets = repos
    gateway = ChatGateway()
    gateway.selection = selection("query_faq", {})
    gateway.fragments = ["结论[1]"]
    decision = make_decision(answer_size=3000)
    service = ChatService(
        settings(context_window_tokens=30_000), gateway, conversations, faq, tickets,
        SessionGuard(10), ToolExecutor(),
        knowledge_pipeline=FakeKnowledgePipeline(decision),
        low_confidence=LowConfidenceRepository(mysql_db.sessions),
    )
    async with service.prepare("C65-Pro支持什么协议？", None) as prepared:
        events = [event async for event in service.stream(prepared)]
    assert events[-1].name == "done"
    audit = await conversations.audit(prepared.ref.conversation_id, "demo")
    assert len(audit[2]["content"].encode("utf-8")) > 4096
    assert json.loads(audit[2]["content"])["sources"][0]["answer"] == "支" * 3000


async def test_pool_disconnect_before_write_never_appends_tool_result(repos, mysql_db) -> None:
    conversations, faq, tickets = repos

    class DisconnectedPool:
        async def record_once(self, *args, **kwargs):
            from app.errors import ServiceError

            raise ServiceError("DATABASE_ERROR", "数据库操作失败", 503)

    gateway = ChatGateway()
    gateway.selection = selection("query_faq", {})
    service = ChatService(
        settings(), gateway, conversations, faq, tickets, SessionGuard(10), ToolExecutor(),
        knowledge_pipeline=FakeKnowledgePipeline(make_decision(status="not_found")),
        low_confidence=DisconnectedPool(),
    )
    async with service.prepare("C65-Pro支持什么协议？", None) as prepared:
        events = [event async for event in service.stream(prepared)]

    assert events[-1].data["code"] == "DATABASE_ERROR"
    audit = await conversations.audit(prepared.ref.conversation_id, "demo")
    assert [row["role"] for row in audit] == ["user", "assistant", "tool"]
    assert {row["turn_status"] for row in audit} == {"failed"}
    assert json.loads(audit[2]["content"])["code"] == "DATABASE_ERROR"
    async with mysql_db.sessions() as session:
        count = await session.scalar(select(func.count(LowConfidenceQuestion.id)))
    assert count == 0


async def test_lost_ack_after_tool_audit_never_exposes_refusal(repos, mysql_db) -> None:
    real_conversations, faq, tickets = repos

    class LostToolAcknowledgment:
        def __getattr__(self, name):
            return getattr(real_conversations, name)

        async def append_result(self, ref, message):
            await real_conversations.append_result(ref, message)
            raise RuntimeError("connection lost after commit")

    gateway = ChatGateway()
    gateway.selection = selection("query_faq", {})
    service = ChatService(
        settings(), gateway, LostToolAcknowledgment(), faq, tickets,
        SessionGuard(10), ToolExecutor(),
        knowledge_pipeline=FakeKnowledgePipeline(make_decision(status="not_found")),
        low_confidence=LowConfidenceRepository(mysql_db.sessions),
    )
    async with service.prepare("C65-Pro支持什么协议？", None) as prepared:
        events = [event async for event in service.stream(prepared)]

    assert events[-1].data["code"] == "DB_ERROR"
    assert not any(event.name in {"refusal", "done"} for event in events)
    audit = await real_conversations.audit(prepared.ref.conversation_id, "demo")
    assert [row["role"] for row in audit] == ["user", "assistant", "tool"]
    assert {row["turn_status"] for row in audit} == {"failed"}
    async with mysql_db.sessions() as session:
        pooled = (await session.execute(select(LowConfidenceQuestion))).scalar_one()
    assert pooled.original_question == "C65-Pro支持什么协议？"
