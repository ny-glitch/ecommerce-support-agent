from __future__ import annotations

import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from app.db.contracts import StoredTurn, TurnRef, TurnSnapshot
from app.db.models import Conversation, Message
from app.errors import ServiceError
from app.workflow.state import validate_turn_messages


def _conversation_not_found():
    return ServiceError('CONVERSATION_NOT_FOUND', '会话不存在', 404)


def _invalid_turn(message):
    return ServiceError('TURN_INVALID', message, 409)


def _conflict():
    return ServiceError('EVENT_CONFLICT', '该事件与已保存内容不一致', 409)


def same_event(existing: Message, candidate: dict) -> bool:
    return all(getattr(existing, key) == candidate.get(key) for key in
               ('role', 'content', 'tool_calls', 'tool_call_id', 'event_data'))


def _stored_tool_calls(value):
    """Validate persisted JSON before AIMessage can normalize or discard fields.

    This checks raw shape only. Turn ordering and call/result pairing continue
    to belong to validate_turn_messages.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError('stored tool calls must be a list')
    for call in value:
        if (
            not isinstance(call, dict)
            or set(call) != {'name', 'args', 'id', 'type'}
            or not isinstance(call['name'], str) or not call['name']
            or not isinstance(call['args'], dict)
            or not isinstance(call['id'], str) or not call['id']
            or call['type'] != 'tool_call'
        ):
            raise ValueError('invalid stored tool call')
    return value


def _messages(rows):
    messages = []
    for row in rows:
        calls = _stored_tool_calls(row.tool_calls)
        if row.role == 'user':
            if calls or row.tool_call_id:
                raise ValueError('user tool fields')
            messages.append(HumanMessage(content=row.content))
        elif row.role == 'assistant':
            if calls:
                if len(calls) != 1 or calls[0]['id'] != row.tool_call_id:
                    raise ValueError('call identity')
                messages.append(AIMessage(content=row.content, tool_calls=calls))
            else:
                if row.tool_call_id:
                    raise ValueError('orphan call identity')
                messages.append(AIMessage(content=row.content))
        elif row.role == 'tool':
            if calls or not row.tool_call_id:
                raise ValueError('invalid tool fields')
            messages.append(ToolMessage(content=row.content, tool_call_id=row.tool_call_id))
        else:
            raise ValueError('unknown role')
    return tuple(messages)


def _restore_complete_turn(rows):
    if not rows or any(row.turn_status != 'completed' for row in rows):
        return None
    try:
        messages = _messages(rows)
        validate_turn_messages(messages)
    except (ValueError, TypeError, KeyError):
        return None
    return StoredTurn(turn_id=rows[0].turn_id, messages=messages)


def _metadata(data):
    if data is None:
        return None
    try:
        if not isinstance(data, dict):
            raise ValueError()
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode('utf-8')) > 65536:
            raise ValueError()
        decoded = json.loads(encoded)
        if decoded != data:
            raise ValueError()
        return decoded
    except (TypeError, ValueError, RecursionError):
        raise _invalid_turn('事件元数据必须为有限大小的 JSON 对象') from None


class ConversationRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, conversation_id: str, user_id: str) -> dict | None:
        async with self._sessions.begin() as session:
            conversation = (
                await session.execute(
                    select(Conversation).where(
                        Conversation.id == conversation_id,
                        Conversation.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()
            if conversation is None or conversation.id != conversation_id or conversation.user_id != user_id:
                return None
            return {
                "id": conversation.id,
                "user_id": conversation.user_id,
                "status": conversation.status,
            }

    async def create(self, conversation_id: str, user_id: str) -> None:
        def verify(row):
            if row.id != conversation_id or row.user_id != user_id:
                raise _conflict()
        try:
            async with self._sessions.begin() as session:
                row = await session.get(Conversation, conversation_id)
                if row:
                    verify(row)
                else:
                    session.add(Conversation(id=conversation_id, user_id=user_id, status='open'))
        except DBAPIError:
            async with self._sessions.begin() as session:
                row = await session.get(Conversation, conversation_id)
                if row is None:
                    raise
                verify(row)

    async def _rows(self, session, ref):
        return list((await session.scalars(select(Message).where(
            Message.conversation_id == ref.conversation_id,
            Message.turn_id == ref.turn_id).order_by(Message.id))).all())

    async def _write(self, ref, key, candidate, validate, *, user_id=None, status=None):
        def check(existing):
            if (existing.conversation_id != ref.conversation_id or existing.turn_id != ref.turn_id
                    or existing.event_key != key or not same_event(existing, candidate)
                    or (status is not None and existing.turn_status != status)):
                raise _conflict()
        try:
            async with self._sessions.begin() as session:
                owner = await session.scalar(select(Conversation).where(
                    Conversation.id == ref.conversation_id).with_for_update())
                if owner is None or owner.id != ref.conversation_id or (user_id is not None and owner.user_id != user_id):
                    raise _conversation_not_found()
                rows = await self._rows(session, ref)
                if any(row.conversation_id != ref.conversation_id or row.turn_id != ref.turn_id for row in rows):
                    raise _conflict()
                existing = next((r for r in rows if r.event_key == key), None)
                if existing:
                    check(existing)
                    return
                if owner.status == 'closed':
                    raise _invalid_turn('会话已关闭')
                validate(rows)
                session.add(Message(conversation_id=ref.conversation_id, turn_id=ref.turn_id,
                    event_key=key, turn_status=status or 'pending', **candidate))
                if status:
                    for row in rows:
                        row.turn_status = status
        except DBAPIError:
            # A failed/uncertain transaction must never be reused.
            async with self._sessions.begin() as session:
                owner = await session.get(Conversation, ref.conversation_id)
                if owner is None or owner.id != ref.conversation_id or (user_id is not None and owner.user_id != user_id):
                    raise _conversation_not_found()
                existing = await session.scalar(select(Message).where(
                    Message.conversation_id == ref.conversation_id, Message.turn_id == ref.turn_id,
                    Message.event_key == key))
                if existing is None:
                    raise
                check(existing)

    async def start_turn(self, ref: TurnRef, user_id: str, content: str) -> None:
        def validate(rows):
            if rows:
                raise _invalid_turn('轮次已存在')
        await self._write(ref, 'user', dict(role='user',content=content), validate, user_id=user_id)

    async def append_call(self, ref: TurnRef, message: AIMessage, *, step: int = 0) -> None:
        if type(step) is not int or step < 0 or len(message.tool_calls) != 1:
            raise _invalid_turn('工具调用必须包含一个有效调用标识')
        def validate(rows):
            if len(rows) != 1 + 2 * step or any(r.turn_status != 'pending' for r in rows):
                raise _invalid_turn('当前轮次不能追加工具调用')
            try:
                validate_turn_messages((*_messages(rows), message,
                    ToolMessage(content='',tool_call_id=message.tool_calls[0]['id']), AIMessage(content='validate')))
            except (ValueError, TypeError, KeyError):
                raise _invalid_turn('工具调用序列无效') from None
        await self._write(ref, f'call:{step}', dict(role='assistant', content=message.content,
            tool_calls=message.tool_calls, tool_call_id=message.tool_calls[0].get('id')), validate)

    async def append_result(self, ref: TurnRef, message: ToolMessage, *, step: int = 0) -> None:
        if type(step) is not int or step < 0:
            raise _invalid_turn('步骤无效')
        def validate(rows):
            if len(rows) != 2 + 2 * step or any(r.turn_status != 'pending' for r in rows):
                raise _invalid_turn('工具结果与当前轮次的调用不匹配')
            try:
                validate_turn_messages((*_messages(rows), message, AIMessage(content='validate')))
            except (ValueError, TypeError, KeyError):
                raise _invalid_turn('工具结果与当前轮次的调用不匹配') from None
        await self._write(ref, f'result:{step}', dict(role='tool',content=message.content,
            tool_call_id=message.tool_call_id), validate)

    async def finish_turn(self, ref: TurnRef, content: str, status: str, *, event_data: dict | None = None) -> None:
        if status not in ('completed', 'failed', 'cancelled'):
            raise _invalid_turn('结束状态无效')
        data = _metadata(event_data)
        def validate(rows):
            if not rows or any(r.turn_status != 'pending' for r in rows):
                raise _invalid_turn('当前轮次不能结束')
            if status == 'completed':
                try:
                    validate_turn_messages((*_messages(rows), AIMessage(content=content)))
                except (ValueError, TypeError, KeyError):
                    raise _invalid_turn('完成的轮次包含不完整的工具调用或回答') from None
        await self._write(ref, 'final', dict(role='assistant',content=content,event_data=data), validate, status=status)

    async def get_turn(self, ref: TurnRef, user_id: str) -> TurnSnapshot | None:
        async with self._sessions.begin() as session:
            owner = await session.get(Conversation, ref.conversation_id)
            if owner is None or owner.id != ref.conversation_id or owner.user_id != user_id:
                return None
            rows = await self._rows(session, ref)
            if not rows:
                return None
            if any(r.turn_id != ref.turn_id for r in rows) or len({r.turn_status for r in rows}) != 1:
                raise _invalid_turn('审计轮次状态冲突')
            final = rows[-1] if rows[-1].role == 'assistant' and not rows[-1].tool_calls else None
            try:
                messages = _messages(rows)
            except (ValueError, TypeError, KeyError):
                raise _invalid_turn('审计轮次结构无效') from None
            if rows[0].turn_status == 'completed' and _restore_complete_turn(rows) is None:
                raise _invalid_turn('审计轮次结构无效')
            return TurnSnapshot(ref, rows[0].content, rows[0].turn_status,
                final.content if final else None, final.event_data if final else None, messages)

    async def unfinished_turns(self, conversation_id: str, user_id: str) -> list[TurnSnapshot]:
        audit = await self.audit(conversation_id, user_id)
        ids = dict.fromkeys(row['turn_id'] for row in audit if row['turn_status'] == 'pending')
        snapshots = [await self.get_turn(TurnRef(conversation_id, tid), user_id) for tid in ids]
        return [s for s in snapshots if s is not None]

    async def history(
        self, conversation_id: str, user_id: str, limit: int
    ) -> list[StoredTurn]:
        async with self._sessions.begin() as session:
            if limit <= 0:
                return []
            owner = (
                await session.execute(
                    select(Conversation).where(
                        Conversation.id == conversation_id,
                        Conversation.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()
            if owner is None or owner.id != conversation_id or owner.user_id != user_id:
                return []
            rows = (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.id)
                )
            ).scalars().all()

            grouped: dict[str, list[Message]] = {}
            for row in rows:
                grouped.setdefault(row.turn_id, []).append(row)
            complete = [
                stored
                for turn_rows in grouped.values()
                if (stored := _restore_complete_turn(turn_rows)) is not None
            ]
            return complete[-limit:]

    async def audit(self, conversation_id: str, user_id: str) -> list[dict]:
        async with self._sessions.begin() as session:
            owner = (
                await session.execute(
                    select(Conversation).where(
                        Conversation.id == conversation_id,
                        Conversation.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()
            if owner is None or owner.id != conversation_id or owner.user_id != user_id:
                return []
            rows = (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.id)
                )
            ).scalars().all()
            return [
                {
                    "id": row.id,
                    "turn_id": row.turn_id,
                    "role": row.role,
                    "content": row.content,
                    "tool_calls": row.tool_calls,
                    "tool_call_id": row.tool_call_id,
                    "turn_status": row.turn_status,
                }
                for row in rows
            ]
