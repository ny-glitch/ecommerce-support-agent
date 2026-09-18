from __future__ import annotations

import time

import pytest

from app.knowledge.contracts import KnowledgeChunk, QueryPlan, SearchHit
from app.knowledge.retrieval import KnowledgeRetriever
from app.knowledge.text import embedding_text, source_hash


def chunk(
    chunk_id: int,
    *,
    status: str = "done",
    answer: str | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id,
        category="数码配件/充电器",
        questions=f"问题 {chunk_id}",
        answer=answer or f"答案 {chunk_id}",
        vector_id=str(chunk_id) if status == "done" else None,
        vectorize_status=status,  # type: ignore[arg-type]
    )


class Repo:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = {item.id: item for item in chunks}
        self.calls: list[list[int]] = []

    async def get_many(self, ids: list[int]) -> dict[int, KnowledgeChunk]:
        self.calls.append(list(ids))
        return {item_id: self.chunks[item_id] for item_id in ids if item_id in self.chunks}


class Store:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[object, ...]] = []

    async def search_dense(
        self, vector: list[float], category: str | None, *, deadline: float
    ) -> list[SearchHit]:
        self.calls.append(("dense", vector, category, deadline))
        return self.hits

    async def search_bm25(
        self, text: str, category: str | None, *, deadline: float
    ) -> list[SearchHit]:
        self.calls.append(("bm25", text, category, deadline))
        return self.hits

    async def search_hybrid(
        self,
        vector: list[float],
        text: str,
        category: str | None,
        *,
        deadline: float,
    ) -> list[SearchHit]:
        self.calls.append(("hybrid", vector, text, category, deadline))
        return self.hits


class Models:
    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = scores or []
        self.embed_calls: list[tuple[list[str], float]] = []
        self.score_calls: list[tuple[str, list[str], float]] = []

    async def embed(
        self, texts: list[str], *, deadline: float
    ) -> list[list[float]]:
        self.embed_calls.append((texts, deadline))
        return [[0.1, 0.2]]

    async def score(
        self, query: str, texts: list[str], *, deadline: float
    ) -> list[float]:
        self.score_calls.append((query, texts, deadline))
        return self.scores


def plan() -> QueryPlan:
    return QueryPlan(
        original="充电头温升正常吗？",
        normalized="充电器发热是否正常？",
        synonyms=("温升", "发热", "温升"),
        category="数码配件/充电器",
    )


def hits_for(chunks: list[KnowledgeChunk]) -> list[SearchHit]:
    return [
        SearchHit(id=item.id, score=0.9 - index / 10, source_hash=source_hash(item))
        for index, item in enumerate(chunks)
    ]


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
async def test_baseline_strategy_calls_only_its_required_search_path(
    strategy: str,
) -> None:
    original = chunk(1)
    repo = Repo([original])
    store = Store(hits_for([original]))
    models = Models()
    events: list[str] = []
    deadline = time.monotonic() + 5

    async def emit(stage: str) -> None:
        events.append(stage)

    result = await KnowledgeRetriever(repo, store, models).retrieve(
        plan(),  # type: ignore[arg-type]
        strategy,  # type: ignore[arg-type]
        deadline=deadline,
        emit=emit,
    )

    assert result.strategy == strategy
    assert [item.chunk.id for item in result.ranked] == [1]
    assert events == ["retrieving"]
    assert models.score_calls == []
    assert [call[0] for call in store.calls] == [strategy]
    if strategy == "bm25":
        assert models.embed_calls == []
        assert store.calls[0][1] == "充电头温升正常吗？\n充电器发热是否正常？\n温升\n发热"
    else:
        assert models.embed_calls == [(["充电器发热是否正常？"], deadline)]


async def test_hybrid_rerank_scores_all_valid_candidates_and_emits_progress() -> None:
    first = chunk(7)
    second = chunk(3)
    repo = Repo([first, second])
    store = Store(hits_for([first, second]))
    models = Models(scores=[0.4, 0.9])
    events: list[str] = []
    deadline = time.monotonic() + 5

    async def emit(stage: str) -> None:
        events.append(stage)

    result = await KnowledgeRetriever(repo, store, models).retrieve(
        plan(),
        "hybrid_rerank",
        deadline=deadline,
        emit=emit,
    )

    assert events == ["retrieving", "reranking"]
    assert models.score_calls == [
        (
            "充电器发热是否正常？",
            [embedding_text(first), embedding_text(second)],
            deadline,
        )
    ]
    assert [(item.chunk.id, item.score) for item in result.ranked] == [
        (3, 0.9),
        (7, 0.4),
    ]


async def test_results_use_current_mysql_originals_and_drop_stale_candidates() -> None:
    valid = chunk(2)
    pending = chunk(3, status="pending")
    changed = chunk(4, answer="current answer")
    duplicate = SearchHit(id=valid.id, score=0.4, source_hash=source_hash(valid))
    store = Store(
        [
            SearchHit(id=valid.id, score=0.7, source_hash=source_hash(valid)),
            duplicate,
            SearchHit(id=pending.id, score=0.8, source_hash=source_hash(pending)),
            SearchHit(id=changed.id, score=0.9, source_hash="old-hash"),
            SearchHit(id=99, score=1.0, source_hash="missing"),
        ]
    )
    repo = Repo([valid, pending, changed])

    result = await KnowledgeRetriever(repo, store, Models()).retrieve(
        plan(),
        "bm25",
        deadline=time.monotonic() + 5,
    )

    assert result.raw_count == 5
    assert result.stale_count == 3
    assert [(item.chunk, item.score) for item in result.ranked] == [(valid, 0.7)]
    assert repo.calls == [[2, 3, 4, 99]]


async def test_equal_scores_are_ordered_by_chunk_id() -> None:
    high_id = chunk(9)
    low_id = chunk(4)
    hits = [
        SearchHit(id=high_id.id, score=0.5, source_hash=source_hash(high_id)),
        SearchHit(id=low_id.id, score=0.5, source_hash=source_hash(low_id)),
    ]

    result = await KnowledgeRetriever(
        Repo([high_id, low_id]), Store(hits), Models()
    ).retrieve(plan(), "dense", deadline=time.monotonic() + 5)

    assert [item.chunk.id for item in result.ranked] == [4, 9]


async def test_bm25_query_is_bounded_without_extra_search_rounds() -> None:
    long_plan = QueryPlan(
        original="原" * 3_000,
        normalized="标准问法",
        synonyms=("同义词",),
        category=None,
    )
    store = Store([])

    await KnowledgeRetriever(Repo([]), store, Models()).retrieve(
        long_plan,
        "bm25",
        deadline=time.monotonic() + 5,
    )

    assert len(store.calls) == 1
    assert len(store.calls[0][1]) <= 2_048
