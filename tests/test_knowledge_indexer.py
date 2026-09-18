from __future__ import annotations

from dataclasses import replace
import fcntl

import pytest

from app.knowledge.indexer import IndexerAlreadyRunningError, KnowledgeIndexer
from app.knowledge.milvus_store import validate_test_collection
from tests.ch04_helpers import make_chunk


@pytest.mark.parametrize("name", ["knowledge", "support", "ch04_test_"])
def test_cleanup_rejects_non_test_collection(name: str) -> None:
    with pytest.raises(ValueError):
        validate_test_collection(name)


def test_cleanup_accepts_uuid_test_collection() -> None:
    validate_test_collection("ch04_test_0123456789abcdef0123456789abcdef")


@pytest.mark.asyncio
async def test_repeat_upsert_uses_same_id_after_unconfirmed_sql(tmp_path) -> None:
    chunk = make_chunk()

    class Repo:
        def __init__(self) -> None:
            self.value = chunk
            self.attempts = 0

        async def list_all(self):
            return [self.value]

        async def mark_done_if_current(self, chunk_id: int, expected_hash: str):
            self.attempts += 1
            if self.attempts == 1:
                return False
            self.value = replace(
                self.value,
                vectorize_status="done",
                vector_id=str(chunk_id),
            )
            return True

    class Store:
        def __init__(self) -> None:
            self.ids: list[int] = []

        async def ensure_schema(self):
            return None

        async def upsert(self, chunks, vectors, *, deadline: float):
            self.ids.extend(row.id for row in chunks)

    class Models:
        async def embed(self, texts, *, deadline: float):
            return [[1.0] + [0.0] * 1023 for _text in texts]

    repo, store = Repo(), Store()
    indexer = KnowledgeIndexer(
        repo,
        store,
        Models(),
        lock_path=tmp_path / "index.lock",
    )

    first = await indexer.run()
    assert first == {"indexed": 0, "skipped": 0, "failed": 1}
    assert repo.value.vectorize_status == "pending"

    second = await indexer.run()
    assert second == {"indexed": 1, "skipped": 0, "failed": 0}
    assert store.ids == [chunk.id, chunk.id]


@pytest.mark.asyncio
async def test_repeat_upsert_uses_same_id_after_sql_confirmation_error(
    tmp_path,
) -> None:
    chunk = make_chunk(id=910002)

    class Repo:
        def __init__(self) -> None:
            self.value = chunk
            self.attempts = 0

        async def list_all(self):
            return [self.value]

        async def mark_done_if_current(self, chunk_id: int, expected_hash: str):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("simulated commit failure")
            self.value = replace(
                self.value,
                vectorize_status="done",
                vector_id=str(chunk_id),
            )
            return True

    class Store:
        def __init__(self) -> None:
            self.ids: list[int] = []

        async def ensure_schema(self):
            return None

        async def upsert(self, chunks, vectors, *, deadline: float):
            self.ids.extend(row.id for row in chunks)

    class Models:
        async def embed(self, texts, *, deadline: float):
            return [[1.0] + [0.0] * 1023 for _text in texts]

    repo, store = Repo(), Store()
    indexer = KnowledgeIndexer(
        repo, store, Models(), lock_path=tmp_path / "index.lock"
    )

    assert (await indexer.run())["failed"] == 1
    assert repo.value.vectorize_status == "pending"
    assert (await indexer.run())["indexed"] == 1
    assert store.ids == [chunk.id, chunk.id]


@pytest.mark.asyncio
async def test_invalid_int64_id_remains_pending_without_model_call(tmp_path) -> None:
    chunk = make_chunk(id=2**63)

    class Repo:
        async def list_all(self):
            return [chunk]

        async def mark_done_if_current(self, chunk_id: int, expected_hash: str):
            raise AssertionError("invalid ID must not be marked done")

    class Store:
        async def ensure_schema(self):
            return None

        async def upsert(self, chunks, vectors, *, deadline: float):
            raise AssertionError("invalid ID must not be upserted")

    class Models:
        async def embed(self, texts, *, deadline: float):
            raise AssertionError("invalid ID must not be embedded")

    result = await KnowledgeIndexer(
        Repo(), Store(), Models(), lock_path=tmp_path / "index.lock"
    ).run()

    assert result == {"indexed": 0, "skipped": 0, "failed": 1}


@pytest.mark.asyncio
async def test_oversized_utf8_metadata_remains_pending_without_model_call(
    tmp_path,
) -> None:
    chunk = make_chunk(category="😀" * 255 + "x")

    class Repo:
        async def list_all(self):
            return [chunk]

        async def mark_done_if_current(self, chunk_id: int, expected_hash: str):
            raise AssertionError("oversized metadata must not be marked done")

    class Store:
        async def ensure_schema(self):
            return None

        async def upsert(self, chunks, vectors, *, deadline: float):
            raise AssertionError("oversized metadata must not be upserted")

    class Models:
        def __init__(self) -> None:
            self.called = False

        async def embed(self, texts, *, deadline: float):
            self.called = True
            return [[1.0] + [0.0] * 1023 for _text in texts]

    models = Models()
    result = await KnowledgeIndexer(
        Repo(), Store(), models, lock_path=tmp_path / "index.lock"
    ).run()

    assert result == {"indexed": 0, "skipped": 0, "failed": 1}
    assert not models.called


@pytest.mark.asyncio
async def test_repair_reindexes_done_row_with_missing_vector(tmp_path) -> None:
    chunk = make_chunk(vectorize_status="done", vector_id="910001")

    class Repo:
        def __init__(self) -> None:
            self.value = chunk
            self.pending_ids: list[int] = []

        async def list_all(self):
            return [self.value]

        async def mark_pending(self, ids):
            self.pending_ids.extend(ids)
            self.value = replace(
                self.value,
                vectorize_status="pending",
                vector_id=None,
            )

        async def mark_done_if_current(self, chunk_id: int, expected_hash: str):
            self.value = replace(
                self.value,
                vectorize_status="done",
                vector_id=str(chunk_id),
            )
            return True

    class Store:
        def __init__(self) -> None:
            self.ids: list[int] = []

        async def ensure_schema(self):
            return None

        async def fingerprints(self, ids):
            return {}

        async def upsert(self, chunks, vectors, *, deadline: float):
            self.ids.extend(row.id for row in chunks)

    class Models:
        async def embed(self, texts, *, deadline: float):
            return [[1.0] + [0.0] * 1023 for _text in texts]

    repo, store = Repo(), Store()
    result = await KnowledgeIndexer(
        repo, store, Models(), lock_path=tmp_path / "index.lock"
    ).run(repair=True)

    assert result == {"indexed": 1, "skipped": 0, "failed": 0}
    assert repo.pending_ids == [chunk.id]
    assert store.ids == [chunk.id]


@pytest.mark.asyncio
async def test_nonblocking_lock_rejects_concurrent_indexer(tmp_path) -> None:
    lock_path = tmp_path / "index.lock"

    class Unused:
        async def ensure_schema(self):
            raise AssertionError("contended indexer must not start")

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(IndexerAlreadyRunningError):
            await KnowledgeIndexer(
                Unused(), Unused(), Unused(), lock_path=lock_path
            ).run()


def test_lock_path_uses_database_identity_without_credentials() -> None:
    from scripts.index_knowledge import index_lock_path

    first = index_lock_path(
        "mysql+asyncmy://reader:first@db.example:3307/support",
        "knowledge",
    )
    second = index_lock_path(
        "mysql+asyncmy://reader:second@db.example:3307/support",
        "knowledge",
    )
    other_database = index_lock_path(
        "mysql+asyncmy://reader:first@db.example:3307/other",
        "knowledge",
    )

    assert first == second
    assert first != other_database
    assert "first" not in first.name
    assert "second" not in first.name


@pytest.mark.asyncio
async def test_pending_chunks_are_embedded_and_upserted_as_one_batch(tmp_path) -> None:
    chunks = [make_chunk(id=910011), make_chunk(id=910012)]

    class Repo:
        async def list_all(self):
            return chunks

        async def mark_done_if_current(self, chunk_id: int, expected_hash: str):
            return True

    class Store:
        def __init__(self) -> None:
            self.batches: list[list[int]] = []

        async def ensure_schema(self):
            return None

        async def upsert(self, batch, vectors, *, deadline: float):
            self.batches.append([chunk.id for chunk in batch])

    class Models:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        async def embed(self, texts, *, deadline: float):
            self.batch_sizes.append(len(texts))
            return [[1.0] + [0.0] * 1023 for _text in texts]

    store, models = Store(), Models()
    result = await KnowledgeIndexer(
        Repo(), store, models, lock_path=tmp_path / "index.lock"
    ).run()

    assert result == {"indexed": 2, "skipped": 0, "failed": 0}
    assert models.batch_sizes == [2]
    assert store.batches == [[910011, 910012]]
