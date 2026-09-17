import asyncio
import json

import pytest
from sqlalchemy import select

from app.db.conversations import ConversationRepository
from app.db.database import Database
from app.db.faq import FaqRepository
from app.db.models import Ticket
from app.db.tickets import TicketRepository
from app.errors import ServiceError
from tests.ch02_helpers import ChatHarness, selection
from tests.integration.conftest import load_test_database_url


async def test_tool_turn_survives_service_and_database_recreation(repos):
    h = ChatHarness(repos=repos)
    h.gateway.selection = selection()
    events = await h.collect("订单1001物流")
    assert events[-1].name == "done"
    sid = events[0].data["session_id"]
    audit = await h.conversations.audit(sid, "demo")
    assert [row["role"] for row in audit] == ["user", "assistant", "tool", "assistant"]
    assert {row["turn_status"] for row in audit} == {"completed"}
    assert audit[1]["tool_call_id"] == audit[2]["tool_call_id"] == "call-1"
    assert {row["turn_id"] for row in audit} == {events[0].data["turn_id"]}

    db = Database(load_test_database_url(True), test_mode=True)
    try:
        restored = ChatHarness(repos=(
            ConversationRepository(db.sessions), FaqRepository(db.sessions),
            TicketRepository(db.sessions),
        ))
        await restored.collect("继续说明", sid)
        messages = restored.gateway.select_calls[0]
        assert [m.type for m in messages] == ["system", "human", "ai", "tool", "ai", "human"]
        assert messages[1].content == "订单1001物流"
        assert messages[3].tool_call_id == "call-1"
        assert messages[4].content == "你好，这是结果"
        assert len(await restored.conversations.history(sid, "demo", 12)) == 2
    finally:
        await db.aclose()


async def test_failed_partial_turn_is_audited_but_excluded_from_next_context(repos):
    h = ChatHarness(repos=repos)
    h.gateway.error = ServiceError("UPSTREAM_ERROR", "模型服务暂时不可用", 502)
    events = await h.collect("第一条")
    sid = events[0].data["session_id"]
    assert events[-1].name == "error"
    audit = await h.conversations.audit(sid, "demo")
    assert [row["role"] for row in audit] == ["user", "assistant"]
    assert {row["turn_status"] for row in audit} == {"failed"}
    assert audit[-1]["content"] == "你好，这是结果"
    assert await h.conversations.history(sid, "demo", 12) == []
    h.gateway.error = None
    await h.collect("新问题", sid)
    assert [m.type for m in h.gateway.select_calls[-1]] == ["system", "human"]
    assert h.gateway.select_calls[-1][-1].content == "新问题"


async def test_committed_ticket_survives_cancellation(repos, mysql_db):
    h = ChatHarness(repos=repos)
    h.gateway.selection = selection("create_ticket", {
        "issue_description": "收到的商品损坏", "ticket_type": "repair",
    })
    h.gateway.stream_gate = asyncio.Event()
    async with h.service.prepare("请帮我创建维修工单", None) as p:
        async def consume():
            return [e async for e in h.service.stream(p)]
        task = asyncio.create_task(consume())
        await asyncio.wait_for(h.gateway.waiting.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    async with mysql_db.sessions() as session:
        tickets = (await session.execute(select(Ticket).where(Ticket.conversation_id == p.ref.conversation_id))).scalars().all()
        assert len(tickets) == 1
        assert tickets[0].conversation_id == p.ref.conversation_id
        assert tickets[0].issue_description == "收到的商品损坏"
        assert tickets[0].ticket_no.startswith("TK-")
    audit = await h.conversations.audit(p.ref.conversation_id, "demo")
    assert [row["role"] for row in audit] == ["user", "assistant", "tool", "assistant"]
    assert {row["turn_status"] for row in audit} == {"cancelled"}
    assert json.loads(audit[2]["content"])["status"] == "ok"
    assert await h.conversations.history(p.ref.conversation_id, "demo", 12) == []
    assert (await h.conversations.get(p.ref.conversation_id, "demo"))["status"] == "human_pending"
    h.guard.acquire(p.ref.conversation_id)
    h.guard.release(p.ref.conversation_id)


async def test_uncertain_completion_commit_preserves_durable_truth(repos, caplog):
    real, faq, tickets = repos

    class LostAcknowledgment:
        def __getattr__(self, name):
            return getattr(real, name)

        async def finish_turn(self, ref, content, status):
            await real.finish_turn(ref, content, status)
            raise RuntimeError("private connection details")

    h = ChatHarness(repos=(LostAcknowledgment(), faq, tickets))
    events = await h.collect("你好")
    assert events[-1].data["code"] == "DB_ERROR"
    assert not any(e.name == "done" for e in events)
    sid = events[0].data["session_id"]
    audit = await real.audit(sid, "demo")
    assert len(audit) == 2
    assert {row["turn_status"] for row in audit} == {"completed"}
    assert "private connection details" not in str(events) + caplog.text
