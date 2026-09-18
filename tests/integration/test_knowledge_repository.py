from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select

from app.db.contracts import TurnRef
from app.errors import ServiceError
from app.knowledge.text import source_hash
from tests.ch04_helpers import make_chunk


@pytest.mark.asyncio
async def test_seed_and_conditional_done(mysql_db) -> None:
    from app.db.knowledge import KnowledgeRepository

    repo = KnowledgeRepository(mysql_db.sessions)
    chunk = make_chunk()
    await repo.insert_seed([chunk])
    await repo.insert_seed([chunk])

    assert len(await repo.list_all()) == 1
    assert await repo.get_many([chunk.id, chunk.id + 1]) == {chunk.id: chunk}
    assert await repo.categories() == ["数码配件/充电器"]
    assert not await repo.mark_done_if_current(chunk.id, "outdated")
    assert (await repo.get(chunk.id)).vectorize_status == "pending"
    assert await repo.mark_done_if_current(chunk.id, source_hash(chunk))
    assert (await repo.get(chunk.id)).vector_id == str(chunk.id)


@pytest.mark.asyncio
async def test_seed_sets_in_batch_pointers_and_delete_nulls_neighbor(mysql_db) -> None:
    from app.db.knowledge import KnowledgeRepository
    from app.db.knowledge_models import KnowledgeChunkRecord

    repo = KnowledgeRepository(mysql_db.sessions)
    first = make_chunk(id=910010, next_chunk_id=910011)
    second = make_chunk(id=910011, prev_chunk_id=910010)
    await repo.insert_seed([first, second])

    assert (await repo.get(first.id)).next_chunk_id == second.id
    assert (await repo.get(second.id)).prev_chunk_id == first.id

    async with mysql_db.sessions.begin() as session:
        await session.execute(
            delete(KnowledgeChunkRecord).where(KnowledgeChunkRecord.id == first.id)
        )

    assert (await repo.get(second.id)).prev_chunk_id is None


@pytest.mark.asyncio
async def test_seed_conflict_rolls_back_entire_batch(mysql_db) -> None:
    from app.db.knowledge import KnowledgeRepository

    repo = KnowledgeRepository(mysql_db.sessions)
    original = make_chunk(id=910020)
    await repo.insert_seed([original])

    with pytest.raises(ServiceError) as exc_info:
        await repo.insert_seed(
            [
                make_chunk(id=910021),
                replace(original, answer="冲突内容"),
            ]
        )

    assert exc_info.value.code == "KNOWLEDGE_CHUNK_CONFLICT"
    assert await repo.get(910021) is None
    assert await repo.get(original.id) == original


@pytest.mark.asyncio
async def test_changed_source_cannot_be_marked_done_with_stale_hash(mysql_db) -> None:
    from app.db.knowledge import KnowledgeRepository
    from app.db.knowledge_models import KnowledgeChunkRecord

    repo = KnowledgeRepository(mysql_db.sessions)
    chunk = make_chunk(id=910030)
    await repo.insert_seed([chunk])

    async with mysql_db.sessions.begin() as session:
        record = await session.get(KnowledgeChunkRecord, chunk.id)
        assert record is not None
        record.answer = "原文已更新"

    assert not await repo.mark_done_if_current(chunk.id, source_hash(chunk))
    changed = await repo.get(chunk.id)
    assert changed is not None
    assert changed.vector_id is None
    assert changed.vectorize_status == "pending"


@pytest.mark.asyncio
async def test_mark_pending_clears_vector_identity(mysql_db) -> None:
    from app.db.knowledge import KnowledgeRepository

    repo = KnowledgeRepository(mysql_db.sessions)
    chunk = make_chunk(id=910040)
    await repo.insert_seed([chunk])
    assert await repo.mark_done_if_current(chunk.id, source_hash(chunk))

    await repo.mark_pending([chunk.id])

    pending = await repo.get(chunk.id)
    assert pending is not None
    assert pending.vector_id is None
    assert pending.vectorize_status == "pending"


@pytest.mark.asyncio
async def test_low_confidence_record_once_is_stable_under_unique_race(mysql_db) -> None:
    from app.db.conversations import ConversationRepository
    from app.db.low_confidence import LowConfidenceRepository
    from app.db.knowledge_models import LowConfidenceQuestion

    conversation_id = str(uuid4())
    ref = TurnRef(conversation_id, "turn-low-confidence")
    conversations = ConversationRepository(mysql_db.sessions)
    await conversations.create(conversation_id, "demo")
    await conversations.start_turn(ref, "demo", "这个型号支持量子充电吗？")
    first_repo = LowConfidenceRepository(mysql_db.sessions)
    second_repo = LowConfidenceRepository(mysql_db.sessions)

    ids = await asyncio.gather(
        first_repo.record_once(
            ref,
            "这个型号支持量子充电吗？",
            "no_hits",
            "知识库中没有该型号资料",
        ),
        second_repo.record_once(
            ref,
            "这个型号支持量子充电吗？",
            "no_hits",
            "知识库中没有该型号资料",
        ),
    )

    assert ids[0] == ids[1]
    async with mysql_db.sessions() as session:
        rows = (
            await session.execute(
                select(LowConfidenceQuestion).where(
                    LowConfidenceQuestion.conversation_id == conversation_id,
                    LowConfidenceQuestion.turn_id == ref.turn_id,
                )
            )
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].original_question == "这个型号支持量子充电吗？"
    assert rows[0].entry_point == "chat"
    assert rows[0].reason_code == "no_hits"


@pytest.mark.asyncio
async def test_low_confidence_requires_matching_user_turn_and_original_text(
    mysql_db,
) -> None:
    from app.db.conversations import ConversationRepository
    from app.db.low_confidence import LowConfidenceRepository
    from app.db.knowledge_models import LowConfidenceQuestion

    conversation_id = str(uuid4())
    real_ref = TurnRef(conversation_id, "turn-real")
    conversations = ConversationRepository(mysql_db.sessions)
    await conversations.create(conversation_id, "demo")
    await conversations.start_turn(real_ref, "demo", "Model ABC")
    repo = LowConfidenceRepository(mysql_db.sessions)

    for ref, question in (
        (real_ref, "model abc"),
        (real_ref, "改写后的问题"),
        (TurnRef(conversation_id, "turn-missing"), "Model ABC"),
    ):
        with pytest.raises(ServiceError) as exc_info:
            await repo.record_once(
                ref,
                question,
                "insufficient_evidence",
                "证据不足",
            )
        assert exc_info.value.code == "LOW_CONFIDENCE_TURN_MISMATCH"

    async with mysql_db.sessions() as session:
        count = (
            await session.execute(select(func.count(LowConfidenceQuestion.id)))
        ).scalar_one()
    assert count == 0
