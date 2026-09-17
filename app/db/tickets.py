from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Conversation, Ticket
from app.errors import ServiceError


def _conversation_not_found() -> ServiceError:
    return ServiceError("CONVERSATION_NOT_FOUND", "会话不存在", 404)


def _ticket_conflict() -> ServiceError:
    return ServiceError("TICKET_CONFLICT", "工单号已用于其他请求", 409)


def _same_ticket(
    ticket: Ticket,
    conversation_id: str,
    issue_description: str,
    ticket_type: str,
) -> bool:
    return (
        ticket.conversation_id == conversation_id
        and ticket.issue_description == issue_description
        and ticket.ticket_type == ticket_type
    )


def _ticket_dto(ticket: Ticket) -> dict:
    return {
        "ticket_no": ticket.ticket_no,
        "conversation_id": ticket.conversation_id,
        "status": ticket.status,
    }


async def _load_idempotency_state(
    session: AsyncSession,
    ticket_no: str,
    conversation_id: str,
    user_id: str,
    issue_description: str,
    ticket_type: str,
) -> tuple[Conversation, Ticket | None]:
    conversation = (
        await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if conversation is None:
        raise _conversation_not_found()

    ticket = await session.get(Ticket, ticket_no)
    if ticket is not None and not _same_ticket(
        ticket, conversation_id, issue_description, ticket_type
    ):
        raise _ticket_conflict()
    return conversation, ticket


class TicketRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create_once(
        self,
        ticket_no: str,
        conversation_id: str,
        user_id: str,
        issue_description: str,
        ticket_type: str,
    ) -> dict:
        try:
            async with self._sessions.begin() as session:
                conversation, ticket = await _load_idempotency_state(
                    session,
                    ticket_no,
                    conversation_id,
                    user_id,
                    issue_description,
                    ticket_type,
                )
                if ticket is not None:
                    conversation.status = "human_pending"
                    return _ticket_dto(ticket)

                ticket = Ticket(
                    ticket_no=ticket_no,
                    conversation_id=conversation_id,
                    issue_description=issue_description,
                    ticket_type=ticket_type,
                    status="pending",
                )
                session.add(ticket)
                conversation.status = "human_pending"
                return _ticket_dto(ticket)
        except DBAPIError:
            # A unique-key race or uncertain commit leaves the failed Session
            # unusable. Verify the durable outcome in a new short transaction.
            async with self._sessions.begin() as session:
                conversation, ticket = await _load_idempotency_state(
                    session,
                    ticket_no,
                    conversation_id,
                    user_id,
                    issue_description,
                    ticket_type,
                )
                if ticket is None:
                    raise
                conversation.status = "human_pending"
                return _ticket_dto(ticket)
