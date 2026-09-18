from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.contracts import TurnRef
from app.db.knowledge_models import LowConfidenceQuestion
from app.db.models import Message
from app.errors import ServiceError


def _turn_mismatch() -> ServiceError:
    return ServiceError(
        "LOW_CONFIDENCE_TURN_MISMATCH",
        "低置信度问题必须匹配当前轮次的用户原话",
        409,
    )


def _database_error() -> ServiceError:
    return ServiceError("DATABASE_ERROR", "数据库操作失败", 503)


async def _validate_turn(session: AsyncSession, ref: TurnRef, question: str) -> None:
    original_questions = (
        await session.execute(
            select(Message.content).where(
                Message.conversation_id == ref.conversation_id,
                Message.turn_id == ref.turn_id,
                Message.role == "user",
            )
        )
    ).scalars().all()
    if question not in original_questions:
        raise _turn_mismatch()


async def _existing(
    session: AsyncSession,
    ref: TurnRef,
) -> LowConfidenceQuestion | None:
    return (
        await session.execute(
            select(LowConfidenceQuestion).where(
                LowConfidenceQuestion.conversation_id == ref.conversation_id,
                LowConfidenceQuestion.turn_id == ref.turn_id,
            )
        )
    ).scalar_one_or_none()


class LowConfidenceRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record_once(
        self,
        ref: TurnRef,
        question: str,
        reason_code: str,
        reason: str,
        entry_point: str = "chat",
    ) -> int:
        try:
            async with self._sessions.begin() as session:
                await _validate_turn(session, ref, question)
                existing = await _existing(session, ref)
                if existing is not None:
                    return existing.id
                record = LowConfidenceQuestion(
                    original_question=question,
                    conversation_id=ref.conversation_id,
                    turn_id=ref.turn_id,
                    entry_point=entry_point,
                    reason_code=reason_code,
                    reason=reason,
                )
                session.add(record)
                await session.flush()
                return record.id
        except ServiceError:
            raise
        except DBAPIError as original_error:
            return await self._recover_unique_race(ref, question, original_error)

    async def _recover_unique_race(
        self,
        ref: TurnRef,
        question: str,
        original_error: DBAPIError,
    ) -> int:
        try:
            async with self._sessions.begin() as session:
                await _validate_turn(session, ref, question)
                existing = await _existing(session, ref)
                if existing is not None:
                    return existing.id
        except ServiceError:
            raise
        except DBAPIError as recovery_error:
            raise _database_error() from recovery_error
        raise _database_error() from original_error
