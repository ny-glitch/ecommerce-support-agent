from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import func, select, update

from app.db.actions import ActionRepository
from app.db.contracts import TurnRef
from app.db.models import Conversation, Ticket
from app.db.workflow_models import ConversationAction
from app.errors import ServiceError
from app.services.actions import ActionService
from app.tools.executor import ToolExecutor
from app.tools.schemas import TicketInput

pytestmark = pytest.mark.asyncio


def build_service(actions, faq, tickets) -> ActionService:
    return ActionService(
        actions,
        faq,
        tickets,
        ToolExecutor(timeout_seconds=1, max_attempts=2),
        SimpleNamespace(request_timeout_seconds=3),
    )


async def completed_offer(repos, mysql_db, ref: TurnRef, *, user_id: str = "demo"):
    conversations, faq, tickets = repos
    actions = ActionRepository(mysql_db.sessions)
    await conversations.start_turn(ref, user_id, "收到的商品外壳破损")
    offer = await actions.offer_once(
        ref,
        user_id,
        TicketInput(issue_description="收到的商品外壳破损", ticket_type="complaint"),
    )
    await conversations.finish_turn(ref, "建议创建工单", "completed")
    return build_service(actions, faq, tickets), actions, offer


async def ticket_count(mysql_db, conversation_id: str) -> int:
    async with mysql_db.sessions() as session:
        return await session.scalar(
            select(func.count()).select_from(Ticket).where(
                Ticket.conversation_id == conversation_id
            )
        )


async def test_real_mysql_confirmation_is_concurrent_and_idempotent(
    repos, new_turn, mysql_db
) -> None:
    service, _, offer = await completed_offer(repos, mysql_db, new_turn)
    assert await ticket_count(mysql_db, new_turn.conversation_id) == 0

    first, second = await asyncio.gather(
        service.confirm(offer.conversation_id, offer.action_id),
        service.confirm(offer.conversation_id, offer.action_id),
    )

    assert first == second
    assert first["ticket_no"] == offer.ticket_no
    assert first["status"] == "completed"
    assert await ticket_count(mysql_db, new_turn.conversation_id) == 1
    assert (await repos[0].get(new_turn.conversation_id, "demo"))["status"] == "open"


class FailFirstMark:
    def __init__(self, delegate: ActionRepository) -> None:
        self.delegate = delegate
        self.first = True

    async def get_confirmable(self, *args):
        return await self.delegate.get_confirmable(*args)

    async def mark_completed(self, *args):
        if self.first:
            self.first = False
            raise ServiceError("ACTION_UPDATE_FAILED", "安全的状态更新失败", 503)
        return await self.delegate.mark_completed(*args)


async def test_ticket_commit_survives_action_update_failure_and_retry(
    repos, new_turn, mysql_db
) -> None:
    _, actions, offer = await completed_offer(repos, mysql_db, new_turn)
    flaky = FailFirstMark(actions)
    service = build_service(flaky, repos[1], repos[2])

    with pytest.raises(ServiceError) as first:
        await service.confirm(offer.conversation_id, offer.action_id)
    assert first.value.code == "ACTION_UPDATE_FAILED"
    assert await ticket_count(mysql_db, new_turn.conversation_id) == 1
    assert (await repos[0].get(new_turn.conversation_id, "demo"))["status"] == "open"

    retried = await service.confirm(offer.conversation_id, offer.action_id)

    assert retried["ticket_no"] == offer.ticket_no
    assert retried["status"] == "completed"
    assert await ticket_count(mysql_db, new_turn.conversation_id) == 1
    assert (await repos[0].get(new_turn.conversation_id, "demo"))["status"] == "open"


async def test_wrong_conversation_user_and_forged_action_are_rejected(
    repos, new_turn, mysql_db
) -> None:
    service, _, offer = await completed_offer(repos, mysql_db, new_turn)
    cases = (
        (str(uuid4()), offer.action_id, "demo"),
        (offer.conversation_id, offer.action_id, "other"),
        (offer.conversation_id, str(uuid4()), "demo"),
    )
    for conversation_id, action_id, user_id in cases:
        with pytest.raises(ServiceError):
            await service.confirm(conversation_id, action_id, user_id)
    assert await ticket_count(mysql_db, new_turn.conversation_id) == 0


@pytest.mark.parametrize("turn_status", ["pending", "failed"])
async def test_pending_and_failed_turn_offers_are_rejected(
    repos, new_turn, mysql_db, turn_status: str
) -> None:
    conversations, faq, tickets = repos
    actions = ActionRepository(mysql_db.sessions)
    await conversations.start_turn(new_turn, "demo", "问题")
    offer = await actions.offer_once(
        new_turn,
        "demo",
        TicketInput(issue_description="问题", ticket_type="other"),
    )
    if turn_status == "failed":
        await conversations.finish_turn(new_turn, "", "failed")

    with pytest.raises(ServiceError) as caught:
        await build_service(actions, faq, tickets).confirm(
            offer.conversation_id, offer.action_id
        )

    assert caught.value.code == "TURN_INVALID"
    assert await ticket_count(mysql_db, new_turn.conversation_id) == 0


async def test_closed_conversation_rejects_new_confirmation_but_allows_completed_replay(
    repos, new_turn, mysql_db
) -> None:
    service, _, offer = await completed_offer(repos, mysql_db, new_turn)
    async with mysql_db.sessions.begin() as session:
        await session.execute(
            update(Conversation)
            .where(Conversation.id == new_turn.conversation_id)
            .values(status="closed")
        )
    with pytest.raises(ServiceError) as caught:
        await service.confirm(offer.conversation_id, offer.action_id)
    assert caught.value.code == "CONVERSATION_CLOSED"
    assert await ticket_count(mysql_db, new_turn.conversation_id) == 0

    async with mysql_db.sessions.begin() as session:
        await session.execute(
            update(Conversation)
            .where(Conversation.id == new_turn.conversation_id)
            .values(status="open")
        )
    first = await service.confirm(offer.conversation_id, offer.action_id)
    async with mysql_db.sessions.begin() as session:
        await session.execute(
            update(Conversation)
            .where(Conversation.id == new_turn.conversation_id)
            .values(status="closed")
        )

    replay = await service.confirm(offer.conversation_id, offer.action_id)

    assert replay == first
    assert await ticket_count(mysql_db, new_turn.conversation_id) == 1


async def test_corrupt_persisted_draft_is_safely_rejected(
    repos, new_turn, mysql_db
) -> None:
    service, _, offer = await completed_offer(repos, mysql_db, new_turn)
    async with mysql_db.sessions.begin() as session:
        await session.execute(
            update(ConversationAction)
            .where(ConversationAction.action_id == offer.action_id)
            .values(ticket_type="not-a-ticket-type")
        )

    with pytest.raises(ServiceError) as caught:
        await service.confirm(offer.conversation_id, offer.action_id)

    assert caught.value.code == "ACTION_CONFLICT"
    assert caught.value.status_code == 409
    assert await ticket_count(mysql_db, new_turn.conversation_id) == 0
