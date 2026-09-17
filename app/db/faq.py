from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import FAQ


class FaqRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def search(self, keyword: str) -> list[dict]:
        stmt = (
            select(FAQ)
            .where(
                or_(
                    FAQ.question.contains(keyword, autoescape=True),
                    FAQ.answer.contains(keyword, autoescape=True),
                    FAQ.category.contains(keyword, autoescape=True),
                )
            )
            .order_by(FAQ.id)
            .limit(5)
        )
        async with self._sessions.begin() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": row.id,
                    "question": row.question,
                    "answer": row.answer,
                    "category": row.category,
                }
                for row in rows
            ]
