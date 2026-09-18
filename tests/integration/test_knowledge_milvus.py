from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import pytest
from pymilvus import DataType, MilvusClient
from sqlalchemy import update

from app.db.knowledge import KnowledgeRepository
from app.db.knowledge_models import KnowledgeChunkRecord
from app.knowledge.indexer import KnowledgeIndexer
from app.knowledge.milvus_store import (
    IncompatibleMilvusSchemaError,
    MilvusStore,
    validate_test_collection,
)
from app.knowledge.text import source_hash
from tests.ch04_helpers import make_chunk


@pytest.mark.asyncio
async def test_native_bm25_dense_and_hybrid_honor_same_category_filter(
    milvus_store,
) -> None:
    store, _settings, _collection_name = milvus_store
    target = make_chunk(
        id=910101,
        category='配件"\\\n控制',
        answer="C65-Pro 支持 PD 3.0",
    )
    other = make_chunk(id=910102, category="其他", answer="C65-Pro 不相关条目")
    target_vector = [1.0] + [0.0] * 1023
    other_vector = [0.9, 0.1] + [0.0] * 1022
    deadline = time.monotonic() + 30

    await store.upsert(
        [target, other],
        [target_vector, other_vector],
        deadline=deadline,
    )

    dense = await store.search_dense(
        target_vector, target.category, deadline=deadline
    )
    bm25 = await store.search_bm25("C65-Pro", target.category, deadline=deadline)
    hybrid = await store.search_hybrid(
        target_vector,
        "C65-Pro",
        target.category,
        deadline=deadline,
    )

    assert [hit.id for hit in dense] == [target.id]
    assert [hit.id for hit in bm25] == [target.id]
    assert [hit.id for hit in hybrid] == [target.id]
    assert dense[0].source_hash == source_hash(target)


@pytest.mark.asyncio
async def test_chinese_analyzer_handles_chinese_english_number_and_model(
    milvus_store,
) -> None:
    _store, settings, collection_name = milvus_store

    def analyze():
        client = MilvusClient(uri=settings.milvus_uri)
        try:
            return client.run_analyzer(
                "充电器 USB PD 3.0 C65-Pro",
                collection_name=collection_name,
                field_name="text",
                with_detail=True,
            )
        finally:
            client.close()

    result = await asyncio.to_thread(analyze)
    tokens = [item["token"] for item in result.tokens]
    assert "充" in tokens or "充电器" in tokens
    assert "usb" in [token.casefold() for token in tokens]
    assert any("65" in token for token in tokens)


@pytest.mark.asyncio
async def test_incompatible_existing_collection_is_never_dropped(milvus_store) -> None:
    _store, settings, _collection_name = milvus_store
    incompatible_name = f"ch04_test_{uuid4().hex}"
    validate_test_collection(incompatible_name)

    def create_incompatible() -> None:
        client = MilvusClient(uri=settings.milvus_uri)
        try:
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=False,
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=8,
            )
            client.create_collection(incompatible_name, schema=schema)
        finally:
            client.close()

    await asyncio.to_thread(create_incompatible)
    incompatible = MilvusStore(settings, collection_name=incompatible_name)
    try:
        with pytest.raises(IncompatibleMilvusSchemaError):
            await incompatible.ensure_schema()

        def still_exists() -> bool:
            client = MilvusClient(uri=settings.milvus_uri)
            try:
                return client.has_collection(incompatible_name)
            finally:
                client.close()

        assert await asyncio.to_thread(still_exists)
    finally:
        await incompatible.aclose()

        def cleanup() -> None:
            client = MilvusClient(uri=settings.milvus_uri)
            try:
                if client.has_collection(incompatible_name):
                    client.drop_collection(incompatible_name)
            finally:
                client.close()

        await asyncio.to_thread(cleanup)


@pytest.mark.asyncio
async def test_repair_recovers_stale_done_sql_row_with_stable_id(
    milvus_store,
    mysql_db,
    tmp_path,
) -> None:
    store, _settings, _collection_name = milvus_store
    repo = KnowledgeRepository(mysql_db.sessions)
    original = make_chunk(id=910201, answer="C65-Pro 原始答案")
    await repo.insert_seed([original])

    class Models:
        async def embed(self, texts, *, deadline: float):
            return [[1.0] + [0.0] * 1023 for _text in texts]

    indexer = KnowledgeIndexer(
        repo,
        store,
        Models(),
        lock_path=tmp_path / "index.lock",
    )
    assert await indexer.run() == {"indexed": 1, "skipped": 0, "failed": 0}
    assert await indexer.run() == {"indexed": 0, "skipped": 1, "failed": 0}

    async with mysql_db.sessions.begin() as session:
        await session.execute(
            update(KnowledgeChunkRecord)
            .where(KnowledgeChunkRecord.id == original.id)
            .values(answer="C65-Pro 更新答案")
        )

    stale = await repo.get(original.id)
    assert stale is not None
    assert stale.vectorize_status == "done"
    assert (await store.fingerprints([original.id]))[original.id] != source_hash(stale)

    assert await indexer.run(repair=True) == {
        "indexed": 1,
        "skipped": 0,
        "failed": 0,
    }
    repaired = await repo.get(original.id)
    assert repaired is not None
    assert repaired.vectorize_status == "done"
    assert repaired.vector_id == str(original.id)
    assert await store.fingerprints([original.id]) == {
        original.id: source_hash(repaired)
    }
