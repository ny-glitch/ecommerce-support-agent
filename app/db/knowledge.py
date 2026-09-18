from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.knowledge_models import KnowledgeChunkRecord
from app.errors import ServiceError
from app.knowledge.contracts import KnowledgeChunk
from app.knowledge.text import source_hash


def _chunk_conflict() -> ServiceError:
    return ServiceError(
        "KNOWLEDGE_CHUNK_CONFLICT",
        "知识条目标识已用于其他原文",
        409,
    )


def _database_error() -> ServiceError:
    return ServiceError("DATABASE_ERROR", "数据库操作失败", 503)


def _to_chunk(record: KnowledgeChunkRecord) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=record.id,
        category=record.category,
        questions=record.questions,
        answer=record.answer,
        section_path=record.section_path,
        content_type=record.content_type,
        is_key_clause=bool(record.is_key_clause),
        prev_chunk_id=record.prev_chunk_id,
        next_chunk_id=record.next_chunk_id,
        vector_id=record.vector_id,
        vectorize_status=record.vectorize_status,
    )


def _new_record(chunk: KnowledgeChunk) -> KnowledgeChunkRecord:
    return KnowledgeChunkRecord(
        id=chunk.id,
        category=chunk.category,
        questions=chunk.questions,
        answer=chunk.answer,
        section_path=chunk.section_path,
        content_type=chunk.content_type,
        is_key_clause=chunk.is_key_clause,
        prev_chunk_id=None,
        next_chunk_id=None,
        vector_id=chunk.vector_id,
        vectorize_status=chunk.vectorize_status,
    )


class KnowledgeRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, chunk_id: int) -> KnowledgeChunk | None:
        try:
            async with self._sessions.begin() as session:
                record = await session.get(KnowledgeChunkRecord, chunk_id)
                return None if record is None else _to_chunk(record)
        except DBAPIError as exc:
            raise _database_error() from exc

    async def get_many(self, ids: Iterable[int]) -> dict[int, KnowledgeChunk]:
        chunk_ids = tuple(dict.fromkeys(ids))
        if not chunk_ids:
            return {}
        try:
            async with self._sessions.begin() as session:
                records = (
                    await session.execute(
                        select(KnowledgeChunkRecord).where(
                            KnowledgeChunkRecord.id.in_(chunk_ids)
                        )
                    )
                ).scalars().all()
                return {record.id: _to_chunk(record) for record in records}
        except DBAPIError as exc:
            raise _database_error() from exc

    async def list_all(self) -> list[KnowledgeChunk]:
        try:
            async with self._sessions.begin() as session:
                records = (
                    await session.execute(
                        select(KnowledgeChunkRecord).order_by(
                            KnowledgeChunkRecord.id
                        )
                    )
                ).scalars().all()
                return [_to_chunk(record) for record in records]
        except DBAPIError as exc:
            raise _database_error() from exc

    async def categories(self) -> list[str]:
        try:
            async with self._sessions.begin() as session:
                return list(
                    (
                        await session.execute(
                            select(KnowledgeChunkRecord.category)
                            .distinct()
                            .order_by(KnowledgeChunkRecord.category)
                        )
                    ).scalars()
                )
        except DBAPIError as exc:
            raise _database_error() from exc

    async def insert_seed(self, chunks: Iterable[KnowledgeChunk]) -> None:
        by_id: dict[int, KnowledgeChunk] = {}
        for chunk in chunks:
            previous = by_id.get(chunk.id)
            if previous is not None and source_hash(previous) != source_hash(chunk):
                raise _chunk_conflict()
            by_id[chunk.id] = chunk
        if not by_id:
            return

        try:
            async with self._sessions.begin() as session:
                existing = {
                    record.id: record
                    for record in (
                        await session.execute(
                            select(KnowledgeChunkRecord)
                            .where(KnowledgeChunkRecord.id.in_(by_id))
                            .with_for_update()
                        )
                    ).scalars()
                }
                if any(
                    source_hash(_to_chunk(record)) != source_hash(by_id[chunk_id])
                    for chunk_id, record in existing.items()
                ):
                    raise _chunk_conflict()

                inserted: dict[int, KnowledgeChunkRecord] = {}
                for chunk_id, chunk in by_id.items():
                    if chunk_id not in existing:
                        record = _new_record(chunk)
                        session.add(record)
                        inserted[chunk_id] = record
                await session.flush()
                for chunk_id, record in inserted.items():
                    chunk = by_id[chunk_id]
                    record.prev_chunk_id = chunk.prev_chunk_id
                    record.next_chunk_id = chunk.next_chunk_id
        except ServiceError:
            raise
        except DBAPIError as exc:
            await self._recover_seed_race(by_id, exc)

    async def _recover_seed_race(
        self,
        by_id: dict[int, KnowledgeChunk],
        original_error: DBAPIError,
    ) -> None:
        try:
            async with self._sessions.begin() as session:
                existing = {
                    record.id: record
                    for record in (
                        await session.execute(
                            select(KnowledgeChunkRecord).where(
                                KnowledgeChunkRecord.id.in_(by_id)
                            )
                        )
                    ).scalars()
                }
                if len(existing) == len(by_id):
                    if all(
                        source_hash(_to_chunk(existing[chunk_id]))
                        == source_hash(chunk)
                        for chunk_id, chunk in by_id.items()
                    ):
                        return
                    raise _chunk_conflict()
        except ServiceError:
            raise
        except DBAPIError as recovery_error:
            raise _database_error() from recovery_error
        raise _database_error() from original_error

    async def mark_pending(self, ids: Iterable[int]) -> None:
        chunk_ids = tuple(dict.fromkeys(ids))
        if not chunk_ids:
            return
        try:
            async with self._sessions.begin() as session:
                await session.execute(
                    update(KnowledgeChunkRecord)
                    .where(KnowledgeChunkRecord.id.in_(chunk_ids))
                    .values(vector_id=None, vectorize_status="pending")
                )
        except DBAPIError as exc:
            raise _database_error() from exc

    async def mark_done_if_current(
        self,
        chunk_id: int,
        expected_hash: str,
    ) -> bool:
        try:
            async with self._sessions.begin() as session:
                record = (
                    await session.execute(
                        select(KnowledgeChunkRecord)
                        .where(KnowledgeChunkRecord.id == chunk_id)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if record is None or source_hash(_to_chunk(record)) != expected_hash:
                    return False
                record.vector_id = str(record.id)
                record.vectorize_status = "done"
                return True
        except DBAPIError as exc:
            raise _database_error() from exc
