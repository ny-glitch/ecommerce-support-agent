from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError

from app.db.contracts import TurnRef
from app.db.models import Conversation, Message, Ticket
from app.errors import ServiceError


def injected_db_error() -> DBAPIError:
    return DBAPIError(
        statement=None,
        params=None,
        orig=RuntimeError("simulated database commit uncertainty"),
    )


class RollbackThenRaise:
    def __init__(self, transaction) -> None:
        self._transaction = transaction

    async def __aenter__(self):
        return await self._transaction.__aenter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            return await self._transaction.__aexit__(
                exc_type, exc_value, traceback
            )
        error = injected_db_error()
        await self._transaction.__aexit__(type(error), error, error.__traceback__)
        raise error


class RaiseOnEnter:
    async def __aenter__(self):
        raise injected_db_error()

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class FailureInjectingSessions:
    def __init__(
        self,
        sessions,
        *,
        fail_on_exit: set[int] | None = None,
        fail_on_enter: set[int] | None = None,
    ) -> None:
        self._sessions = sessions
        self._fail_on_exit = fail_on_exit or set()
        self._fail_on_enter = fail_on_enter or set()
        self._begin_count = 0

    def begin(self):
        self._begin_count += 1
        if self._begin_count in self._fail_on_enter:
            return RaiseOnEnter()
        transaction = self._sessions.begin()
        if self._begin_count in self._fail_on_exit:
            return RollbackThenRaise(transaction)
        return transaction


def tool_call(call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


@pytest.mark.asyncio
async def test_completed_turn_preserves_tool_pair(repos, new_turn) -> None:
    conversations, _, _ = repos
    await conversations.start_turn(new_turn, "demo", "订单1001的物流")
    await conversations.append_call(new_turn, tool_call())
    await conversations.append_result(
        new_turn,
        ToolMessage(content='{"status":"ok"}', tool_call_id="call-1"),
    )

    assert await conversations.history(new_turn.conversation_id, "demo", 12) == []

    await conversations.finish_turn(new_turn, "演示物流已发出", "completed")
    turns = await conversations.history(new_turn.conversation_id, "demo", 12)

    assert [message.type for message in turns[0].messages] == [
        "human",
        "ai",
        "tool",
        "ai",
    ]
    assert turns[0].messages[1].tool_calls[0]["id"] == "call-1"
    assert turns[0].messages[2].tool_call_id == "call-1"
    assert turns[0].messages[-1].content == "演示物流已发出"


@pytest.mark.asyncio
async def test_literal_faq_search_escapes_wildcards(repos) -> None:
    _, faq, _ = repos

    matches = await faq.search("退货")

    assert matches
    assert set(matches[0]) == {"id", "question", "answer", "category"}
    assert await faq.search("邮费") == []
    assert await faq.search("%_") == []


@pytest.mark.asyncio
async def test_ticket_retry_returns_one_ticket_and_marks_conversation(repos, new_turn, mysql_db) -> None:
    _, _, tickets = repos

    first = await tickets.create_once(
        "TICKET-001",
        new_turn.conversation_id,
        "demo",
        "包装破损",
        "complaint",
    )
    second = await tickets.create_once(
        "TICKET-001",
        new_turn.conversation_id,
        "demo",
        "包装破损",
        "complaint",
    )

    assert first == second == {
        "ticket_no": "TICKET-001",
        "conversation_id": new_turn.conversation_id,
        "status": "pending",
    }
    async with mysql_db.sessions() as session:
        ticket_count = (
            await session.execute(
                select(func.count(Ticket.ticket_no)).where(
                    Ticket.ticket_no == "TICKET-001"
                )
            )
        ).scalar_one()
        conversation = await session.get(Conversation, new_turn.conversation_id)

    assert ticket_count == 1
    assert conversation is not None
    assert conversation.status == "human_pending"


@pytest.mark.asyncio
async def test_ticket_retry_rejects_conflicting_fields(repos, new_turn) -> None:
    _, _, tickets = repos
    await tickets.create_once(
        "TICKET-CONFLICT",
        new_turn.conversation_id,
        "demo",
        "包装破损",
        "complaint",
    )

    with pytest.raises(ServiceError) as exc_info:
        await tickets.create_once(
            "TICKET-CONFLICT",
            new_turn.conversation_id,
            "demo",
            "不同的问题",
            "complaint",
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_noncompleted_turns_stay_in_audit_but_not_history(repos, new_turn) -> None:
    conversations, _, _ = repos
    completed = new_turn
    failed = TurnRef(completed.conversation_id, str(uuid4()))
    cancelled = TurnRef(completed.conversation_id, str(uuid4()))
    pending = TurnRef(completed.conversation_id, str(uuid4()))

    await conversations.start_turn(completed, "demo", "已完成")
    await conversations.finish_turn(completed, "完成答复", "completed")
    await conversations.start_turn(failed, "demo", "失败")
    await conversations.finish_turn(failed, "部分答复", "failed")
    await conversations.start_turn(cancelled, "demo", "取消")
    await conversations.finish_turn(cancelled, "", "cancelled")
    await conversations.start_turn(pending, "demo", "仍在处理")

    turns = await conversations.history(completed.conversation_id, "demo", 12)
    audit = await conversations.audit(completed.conversation_id, "demo")

    assert [turn.turn_id for turn in turns] == [completed.turn_id]
    assert {row["turn_status"] for row in audit} == {
        "completed",
        "failed",
        "cancelled",
        "pending",
    }
    assert [row["id"] for row in audit] == sorted(row["id"] for row in audit)


@pytest.mark.asyncio
async def test_history_limit_counts_complete_turns_and_survives_repository_restart(
    repos, new_turn, mysql_db
) -> None:
    from app.db.conversations import ConversationRepository

    conversations, _, _ = repos
    refs = [new_turn]
    refs.extend(
        TurnRef(new_turn.conversation_id, str(uuid4())) for _ in range(2)
    )
    for index, ref in enumerate(refs):
        await conversations.start_turn(ref, "demo", f"问题{index}")
        if index == 1:
            await conversations.append_call(ref, tool_call("call-middle"))
            await conversations.append_result(
                ref,
                ToolMessage(content="中间结果", tool_call_id="call-middle"),
            )
        await conversations.finish_turn(ref, f"答复{index}", "completed")

    restarted = ConversationRepository(mysql_db.sessions)
    turns = await restarted.history(new_turn.conversation_id, "demo", 2)

    assert [turn.turn_id for turn in turns] == [refs[1].turn_id, refs[2].turn_id]
    assert [len(turn.messages) for turn in turns] == [4, 2]


@pytest.mark.asyncio
async def test_other_user_cannot_read_or_write_conversation(repos, new_turn) -> None:
    conversations, _, tickets = repos
    await conversations.start_turn(new_turn, "demo", "仅本人可见")

    assert await conversations.get(new_turn.conversation_id, "other") is None
    assert await conversations.history(new_turn.conversation_id, "other", 12) == []
    assert await conversations.audit(new_turn.conversation_id, "other") == []
    with pytest.raises(ServiceError):
        await tickets.create_once(
            "TICKET-OTHER",
            new_turn.conversation_id,
            "other",
            "越权工单",
            "other",
        )


@pytest.mark.asyncio
async def test_concurrent_ticket_retry_relies_on_unique_key(repos, new_turn, mysql_db) -> None:
    _, _, tickets = repos

    results = await asyncio.gather(
        tickets.create_once(
            "TICKET-RACE",
            new_turn.conversation_id,
            "demo",
            "并发重试",
            "other",
        ),
        tickets.create_once(
            "TICKET-RACE",
            new_turn.conversation_id,
            "demo",
            "并发重试",
            "other",
        ),
    )

    assert results[0] == results[1]
    async with mysql_db.sessions() as session:
        count = (
            await session.execute(
                select(func.count(Ticket.ticket_no)).where(
                    Ticket.ticket_no == "TICKET-RACE"
                )
            )
        ).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_ticket_write_failure_rolls_back_conversation_status(repos, new_turn, mysql_db) -> None:
    _, _, tickets = repos

    with pytest.raises(DBAPIError):
        await tickets.create_once(
            "TICKET-ROLLBACK",
            new_turn.conversation_id,
            "demo",
            "数据库拒绝这个超长类型",
            "x" * 100,
        )

    async with mysql_db.sessions() as session:
        conversation = await session.get(Conversation, new_turn.conversation_id)
        ticket = await session.get(Ticket, "TICKET-ROLLBACK")

    assert conversation is not None
    assert conversation.status == "open"
    assert ticket is None


@pytest.mark.asyncio
async def test_ticket_recovery_restores_conversation_status_before_success(
    new_turn, mysql_db
) -> None:
    from app.db.tickets import TicketRepository

    async with mysql_db.sessions.begin() as session:
        session.add(
            Ticket(
                ticket_no="TICKET-RECOVER",
                conversation_id=new_turn.conversation_id,
                issue_description="恢复幂等状态",
                ticket_type="other",
                status="pending",
            )
        )
    tickets = TicketRepository(
        FailureInjectingSessions(mysql_db.sessions, fail_on_exit={1})
    )

    result = await tickets.create_once(
        "TICKET-RECOVER",
        new_turn.conversation_id,
        "demo",
        "恢复幂等状态",
        "other",
    )

    async with mysql_db.sessions.begin() as session:
        conversation = await session.get(Conversation, new_turn.conversation_id)
    assert result["ticket_no"] == "TICKET-RECOVER"
    assert conversation is not None
    assert conversation.status == "human_pending"


@pytest.mark.asyncio
async def test_ticket_recovery_normalizes_cross_user_primary_key_conflict(
    new_turn, mysql_db
) -> None:
    from app.db.tickets import TicketRepository

    other_conversation_id = str(uuid4())
    async with mysql_db.sessions.begin() as session:
        session.add(
            Conversation(
                id=other_conversation_id,
                user_id="other",
                status="open",
            )
        )
        await session.flush()
        session.add(
            Ticket(
                ticket_no="TICKET-CROSS-USER",
                conversation_id=other_conversation_id,
                issue_description="其他用户的问题",
                ticket_type="other",
                status="pending",
            )
        )
    tickets = TicketRepository(
        FailureInjectingSessions(mysql_db.sessions, fail_on_enter={1})
    )

    with pytest.raises(ServiceError) as exc_info:
        await tickets.create_once(
            "TICKET-CROSS-USER",
            new_turn.conversation_id,
            "demo",
            "当前用户的问题",
            "other",
        )

    assert exc_info.value.code == "TICKET_CONFLICT"
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_ticket_recovery_commit_failure_does_not_report_success(
    new_turn, mysql_db
) -> None:
    from app.db.tickets import TicketRepository

    async with mysql_db.sessions.begin() as session:
        session.add(
            Ticket(
                ticket_no="TICKET-RECOVERY-FAILS",
                conversation_id=new_turn.conversation_id,
                issue_description="恢复提交仍然失败",
                ticket_type="other",
                status="pending",
            )
        )
    tickets = TicketRepository(
        FailureInjectingSessions(mysql_db.sessions, fail_on_exit={1, 2})
    )

    with pytest.raises(DBAPIError):
        await tickets.create_once(
            "TICKET-RECOVERY-FAILS",
            new_turn.conversation_id,
            "demo",
            "恢复提交仍然失败",
            "other",
        )

    async with mysql_db.sessions.begin() as session:
        conversation = await session.get(Conversation, new_turn.conversation_id)
    assert conversation is not None
    assert conversation.status == "open"


@pytest.mark.asyncio
async def test_malformed_completed_tool_group_is_audited_but_not_returned(
    repos, new_turn, mysql_db
) -> None:
    conversations, _, _ = repos
    invalid_schema_turn_id = str(uuid4())
    async with mysql_db.sessions.begin() as session:
        session.add_all(
            [
                Message(
                    conversation_id=new_turn.conversation_id,
                    turn_id=new_turn.turn_id,
                    role="user",
                    content="损坏轮次",
                    turn_status="completed",
                ),
                Message(
                    conversation_id=new_turn.conversation_id,
                    turn_id=new_turn.turn_id,
                    role="assistant",
                    content="",
                    tool_calls=tool_call().tool_calls,
                    turn_status="completed",
                ),
                Message(
                    conversation_id=new_turn.conversation_id,
                    turn_id=new_turn.turn_id,
                    role="assistant",
                    content="缺少工具结果",
                    turn_status="completed",
                ),
                Message(
                    conversation_id=new_turn.conversation_id,
                    turn_id=invalid_schema_turn_id,
                    role="user",
                    content="工具结构损坏",
                    turn_status="completed",
                ),
                Message(
                    conversation_id=new_turn.conversation_id,
                    turn_id=invalid_schema_turn_id,
                    role="assistant",
                    content="",
                    tool_calls=[{"id": "bad-call"}],
                    tool_call_id="bad-call",
                    turn_status="completed",
                ),
                Message(
                    conversation_id=new_turn.conversation_id,
                    turn_id=invalid_schema_turn_id,
                    role="tool",
                    content="损坏结果",
                    tool_call_id="bad-call",
                    turn_status="completed",
                ),
                Message(
                    conversation_id=new_turn.conversation_id,
                    turn_id=invalid_schema_turn_id,
                    role="assistant",
                    content="不应进入历史",
                    turn_status="completed",
                ),
            ]
        )

    assert await conversations.history(new_turn.conversation_id, "demo", 12) == []
    audit = await conversations.audit(new_turn.conversation_id, "demo")
    assert len(audit) == 7
    assert {row["turn_id"] for row in audit} == {
        new_turn.turn_id,
        invalid_schema_turn_id,
    }
