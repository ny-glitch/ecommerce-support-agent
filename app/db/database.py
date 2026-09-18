from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.db.models import Base
import app.db.knowledge_models  # noqa: F401


class Database:
    def __init__(self, url: str, *, test_mode: bool = False) -> None:
        engine_options: dict[str, object] = {
            "pool_pre_ping": True,
            "echo": False,
        }
        if test_mode:
            engine_options["poolclass"] = NullPool

        self.engine: AsyncEngine = create_async_engine(url, **engine_options)
        self.sessions: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

    async def check(self) -> None:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def aclose(self) -> None:
        await self.engine.dispose()
