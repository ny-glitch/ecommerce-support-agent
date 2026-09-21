"""Short transactions around offers; business tool execution belongs to the service."""
import json
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from app.db.contracts import TurnRef
from app.db.conversations import _restore_complete_turn
from app.db.models import Conversation, Message, Ticket
from app.db.workflow_models import ConversationAction
from app.errors import ServiceError
from app.tools.schemas import TicketInput
from app.workflow.contracts import ActionOffer


def _error(code='ACTION_CONFLICT'):
    return ServiceError(code, '操作建议不存在、不可执行或与已保存内容冲突', 409)


def _dto(row):
    return ActionOffer(action_id=row.action_id, conversation_id=row.conversation_id,
        turn_id=row.turn_id, ticket_no=row.ticket_no, status=row.status,
        draft=TicketInput(issue_description=row.issue_description, ticket_type=row.ticket_type))


class ActionRepository:
    def __init__(self, sessions):
        self._sessions = sessions

    async def _owner(self, session, cid, uid, *, lock=False):
        query = select(Conversation).where(Conversation.id == cid)
        row = await session.scalar(query.with_for_update() if lock else query)
        if row is None or row.id != cid or row.user_id != uid:
            raise _error('CONVERSATION_NOT_FOUND')
        return row

    async def offer_once(self, ref: TurnRef, user_id: str, draft: TicketInput) -> ActionOffer:
        draft = TicketInput.model_validate(draft.model_dump())
        identity = uuid5(NAMESPACE_URL, json.dumps([ref.conversation_id,ref.turn_id,'create_ticket'], separators=(',',':')))
        action_id, ticket_no = str(identity), 'TK-' + identity.hex
        def verify(row):
            if (row.action_id != action_id or row.conversation_id != ref.conversation_id
                    or row.turn_id != ref.turn_id or row.action_type != 'create_ticket'
                    or row.ticket_no != ticket_no or row.issue_description != draft.issue_description
                    or row.ticket_type != draft.ticket_type):
                raise _error()
            return _dto(row)
        try:
            async with self._sessions.begin() as session:
                owner = await self._owner(session, ref.conversation_id, user_id, lock=True)
                row = await session.get(ConversationAction, action_id)
                if row:
                    return verify(row)
                if owner.status == 'closed':
                    raise _error('CONVERSATION_CLOSED')
                rows = list((await session.scalars(select(Message).where(
                    Message.conversation_id == ref.conversation_id, Message.turn_id == ref.turn_id))).all())
                if not rows or any(r.turn_id != ref.turn_id or r.turn_status not in ('pending','completed') for r in rows):
                    raise _error('TURN_INVALID')
                row = ConversationAction(action_id=action_id, conversation_id=ref.conversation_id,
                    turn_id=ref.turn_id, action_type='create_ticket', issue_description=draft.issue_description,
                    ticket_type=draft.ticket_type, status='offered', ticket_no=ticket_no)
                session.add(row)
                return _dto(row)
        except DBAPIError:
            async with self._sessions.begin() as session:
                await self._owner(session, ref.conversation_id, user_id)
                row = await session.get(ConversationAction, action_id)
                if row is None:
                    raise
                return verify(row)

    async def get_confirmable(self, conversation_id: str, action_id: str, user_id: str) -> ActionOffer:
        async with self._sessions.begin() as session:
            owner = await self._owner(session, conversation_id, user_id)
            row = await session.get(ConversationAction, action_id)
            if row is None or row.action_id != action_id or row.conversation_id != conversation_id:
                raise _error('ACTION_NOT_FOUND')
            rows = list((await session.scalars(select(Message).where(
                Message.conversation_id == conversation_id, Message.turn_id == row.turn_id).order_by(Message.id))).all())
            if _restore_complete_turn(rows) is None:
                raise _error('TURN_INVALID')
            if owner.status == 'closed' and row.status != 'completed':
                raise _error('CONVERSATION_CLOSED')
            return _dto(row)

    async def mark_completed(self, action_id: str, ticket_no: str) -> ActionOffer:
        async def verify(session, *, recovery=False):
            row = await session.scalar(select(ConversationAction).where(
                ConversationAction.action_id == action_id).with_for_update())
            if row is None or row.action_id != action_id or row.ticket_no != ticket_no:
                raise _error()
            ticket = await session.get(Ticket, ticket_no)
            if (ticket is None or ticket.ticket_no != ticket_no or ticket.conversation_id != row.conversation_id
                    or ticket.issue_description != row.issue_description or ticket.ticket_type != row.ticket_type):
                raise _error()
            if recovery and row.status != 'completed':
                return None
            if not recovery:
                row.status = 'completed'
            return _dto(row)
        try:
            async with self._sessions.begin() as session:
                return await verify(session)
        except DBAPIError:
            async with self._sessions.begin() as session:
                result = await verify(session, recovery=True)
                if result is None:
                    raise
                return result
