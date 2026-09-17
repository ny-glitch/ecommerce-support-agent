from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.contracts import StoredTurn, TurnRef
from app.db.models import Conversation, Message
from app.errors import ServiceError


def _conversation_not_found() -> ServiceError:
    return ServiceError("CONVERSATION_NOT_FOUND", "会话不存在", 404)


def _invalid_turn(message: str) -> ServiceError:
    return ServiceError("TURN_INVALID", message, 409)


def _restore_complete_turn(rows: list[Message]) -> StoredTurn | None:
    if not rows or any(row.turn_status != "completed" for row in rows):
        return None
    if rows[0].tool_calls or rows[0].tool_call_id:
        return None
    roles = [row.role for row in rows]
    if roles == ["user", "assistant"]:
        if rows[1].tool_calls or rows[1].tool_call_id or not rows[1].content.strip():
            return None
        messages = (
            HumanMessage(content=rows[0].content),
            AIMessage(content=rows[1].content),
        )
    elif roles == ["user", "assistant", "tool", "assistant"]:
        calls = rows[1].tool_calls
        call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("name"), str)
            or not call["name"]
            or not isinstance(call.get("args"), dict)
            or not isinstance(call.get("id"), str)
            or not call["id"]
            or call.get("type") != "tool_call"
            or call["id"] != rows[1].tool_call_id
            or rows[1].tool_call_id != rows[2].tool_call_id
            or rows[2].tool_calls
            or rows[3].tool_calls
            or rows[3].tool_call_id
            or not rows[3].content.strip()
        ):
            return None
        messages = (
            HumanMessage(content=rows[0].content),
            AIMessage(content=rows[1].content, tool_calls=calls),
            ToolMessage(content=rows[2].content, tool_call_id=rows[2].tool_call_id),
            AIMessage(content=rows[3].content),
        )
    else:
        return None
    return StoredTurn(turn_id=rows[0].turn_id, messages=messages)


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
            if conversation is None:
                return None
            return {
                "id": conversation.id,
                "user_id": conversation.user_id,
                "status": conversation.status,
            }

    async def create(self, conversation_id: str, user_id: str) -> None:
        async with self._sessions.begin() as session:
            session.add(
                Conversation(id=conversation_id, user_id=user_id, status="open")
            )

    async def start_turn(self, ref: TurnRef, user_id: str, content: str) -> None:
        async with self._sessions.begin() as session:
            conversation = (
                await session.execute(
                    select(Conversation).where(
                        Conversation.id == ref.conversation_id,
                        Conversation.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()
            if conversation is None:
                raise _conversation_not_found()
            session.add(
                Message(
                    conversation_id=ref.conversation_id,
                    turn_id=ref.turn_id,
                    role="user",
                    content=content,
                    turn_status="pending",
                )
            )

    async def append_call(self, ref: TurnRef, message: AIMessage) -> None:
        if len(message.tool_calls) != 1 or not message.tool_calls[0].get("id"):
            raise _invalid_turn("工具调用必须包含一个有效调用标识")
        async with self._sessions.begin() as session:
            rows = (
                await session.execute(
                    select(Message)
                    .where(
                        Message.conversation_id == ref.conversation_id,
                        Message.turn_id == ref.turn_id,
                    )
                    .order_by(Message.id)
                )
            ).scalars().all()
            if (
                len(rows) != 1
                or rows[0].role != "user"
                or rows[0].turn_status != "pending"
            ):
                raise _invalid_turn("当前轮次不能追加工具调用")
            session.add(
                Message(
                    conversation_id=ref.conversation_id,
                    turn_id=ref.turn_id,
                    role="assistant",
                    content=message.content,
                    tool_calls=message.tool_calls,
                    tool_call_id=message.tool_calls[0]["id"],
                    turn_status="pending",
                )
            )

    async def append_result(self, ref: TurnRef, message: ToolMessage) -> None:
        async with self._sessions.begin() as session:
            rows = (
                await session.execute(
                    select(Message)
                    .where(
                        Message.conversation_id == ref.conversation_id,
                        Message.turn_id == ref.turn_id,
                    )
                    .order_by(Message.id)
                )
            ).scalars().all()
            if (
                len(rows) != 2
                or [row.role for row in rows] != ["user", "assistant"]
                or any(row.turn_status != "pending" for row in rows)
                or rows[1].tool_call_id != message.tool_call_id
            ):
                raise _invalid_turn("工具结果与当前轮次的调用不匹配")
            session.add(
                Message(
                    conversation_id=ref.conversation_id,
                    turn_id=ref.turn_id,
                    role="tool",
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                    turn_status="pending",
                )
            )

    async def finish_turn(
        self,
        ref: TurnRef,
        content: str,
        status: Literal["completed", "failed", "cancelled"],
    ) -> None:
        if status == "completed" and not content.strip():
            raise _invalid_turn("完成的轮次必须包含最终回答")
        async with self._sessions.begin() as session:
            rows = (
                await session.execute(
                    select(Message)
                    .where(
                        Message.conversation_id == ref.conversation_id,
                        Message.turn_id == ref.turn_id,
                    )
                    .order_by(Message.id)
                )
            ).scalars().all()
            if not rows or any(row.turn_status != "pending" for row in rows):
                raise _invalid_turn("当前轮次不能结束")
            roles = [row.role for row in rows]
            valid_prefix = roles == ["user"]
            if roles == ["user", "assistant", "tool"]:
                calls = rows[1].tool_calls
                valid_prefix = bool(
                    isinstance(calls, list)
                    and len(calls) == 1
                    and isinstance(calls[0], dict)
                    and calls[0].get("id")
                    and calls[0]["id"] == rows[1].tool_call_id
                    and rows[1].tool_call_id == rows[2].tool_call_id
                )
            if status == "completed" and not valid_prefix:
                raise _invalid_turn("完成的轮次包含不完整的工具调用")
            if content.strip():
                session.add(
                    Message(
                        conversation_id=ref.conversation_id,
                        turn_id=ref.turn_id,
                        role="assistant",
                        content=content,
                        turn_status=status,
                    )
                )
            await session.execute(
                update(Message)
                .where(
                    Message.conversation_id == ref.conversation_id,
                    Message.turn_id == ref.turn_id,
                )
                .values(turn_status=status)
            )

    async def history(
        self, conversation_id: str, user_id: str, limit: int
    ) -> list[StoredTurn]:
        async with self._sessions.begin() as session:
            if limit <= 0:
                return []
            owner = (
                await session.execute(
                    select(Conversation.id).where(
                        Conversation.id == conversation_id,
                        Conversation.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()
            if owner is None:
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
                    select(Conversation.id).where(
                        Conversation.id == conversation_id,
                        Conversation.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()
            if owner is None:
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
