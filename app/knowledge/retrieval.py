from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Literal, Protocol

from app.knowledge.contracts import (
    KnowledgeChunk,
    QueryPlan,
    RankedChunk,
    RetrievalResult,
    SearchHit,
)
from app.knowledge.text import embedding_text, source_hash


RetrievalStrategy = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
_BM25_QUERY_MAX_CHARS = 2_048
_BM25_ORIGINAL_MAX_CHARS = 1_024


class _Repository(Protocol):
    async def get_many(self, ids: Iterable[int]) -> dict[int, KnowledgeChunk]: ...


class _Store(Protocol):
    async def search_dense(
        self, vector: list[float], category: str | None, *, deadline: float
    ) -> list[SearchHit]: ...

    async def search_bm25(
        self, text: str, category: str | None, *, deadline: float
    ) -> list[SearchHit]: ...

    async def search_hybrid(
        self,
        vector: list[float],
        text: str,
        category: str | None,
        *,
        deadline: float,
    ) -> list[SearchHit]: ...


class _Models(Protocol):
    async def embed(
        self, texts: list[str], *, deadline: float
    ) -> list[list[float]]: ...

    async def score(
        self, query: str, texts: list[str], *, deadline: float
    ) -> list[float]: ...


def _bm25_query(plan: QueryPlan) -> str:
    values = [plan.original[:_BM25_ORIGINAL_MAX_CHARS], plan.normalized, *plan.synonyms]
    parts: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = raw.strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            parts.append(value)
    return "\n".join(parts)[:_BM25_QUERY_MAX_CHARS]


def _deduplicate_hits(hits: list[SearchHit]) -> list[SearchHit]:
    order: list[int] = []
    selected: dict[int, SearchHit] = {}
    for hit in hits:
        previous = selected.get(hit.id)
        if previous is None:
            order.append(hit.id)
            selected[hit.id] = hit
        elif hit.score > previous.score:
            selected[hit.id] = hit
    return [selected[chunk_id] for chunk_id in order]


class KnowledgeRetriever:
    def __init__(
        self,
        repo: _Repository,
        store: _Store,
        models: _Models,
    ) -> None:
        self._repo = repo
        self._store = store
        self._models = models

    async def retrieve(
        self,
        plan: QueryPlan,
        strategy: RetrievalStrategy,
        *,
        deadline: float,
        emit: Callable[[str], Awaitable[None]] | None = None,
    ) -> RetrievalResult:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("knowledge retrieval deadline exceeded")
        async with asyncio.timeout(remaining):
            return await self._retrieve(
                plan,
                strategy,
                deadline=deadline,
                emit=emit,
            )

    async def _retrieve(
        self,
        plan: QueryPlan,
        strategy: RetrievalStrategy,
        *,
        deadline: float,
        emit: Callable[[str], Awaitable[None]] | None,
    ) -> RetrievalResult:
        if emit is not None:
            await emit("retrieving")

        text_query = _bm25_query(plan)
        if strategy == "bm25":
            hits = await self._store.search_bm25(
                text_query,
                plan.category,
                deadline=deadline,
            )
        elif strategy in {"dense", "hybrid", "hybrid_rerank"}:
            vectors = await self._models.embed([plan.normalized], deadline=deadline)
            if len(vectors) != 1:
                raise ValueError("query embedding output count must be one")
            if strategy == "dense":
                hits = await self._store.search_dense(
                    vectors[0],
                    plan.category,
                    deadline=deadline,
                )
            else:
                hits = await self._store.search_hybrid(
                    vectors[0],
                    text_query,
                    plan.category,
                    deadline=deadline,
                )
        else:
            raise ValueError(f"unknown retrieval strategy: {strategy}")

        unique_hits = _deduplicate_hits(hits)
        originals = await self._repo.get_many([hit.id for hit in unique_hits])
        ranked: list[RankedChunk] = []
        stale_count = 0
        for hit in unique_hits:
            original = originals.get(hit.id)
            if (
                original is None
                or original.vectorize_status != "done"
                or source_hash(original) != hit.source_hash
            ):
                stale_count += 1
                continue
            ranked.append(RankedChunk(original, hit.score))

        if strategy == "hybrid_rerank" and ranked:
            if emit is not None:
                await emit("reranking")
            scores = await self._models.score(
                plan.normalized,
                [embedding_text(item.chunk) for item in ranked],
                deadline=deadline,
            )
            ranked = [
                RankedChunk(item.chunk, score)
                for item, score in zip(ranked, scores, strict=True)
            ]

        ranked.sort(key=lambda item: (-item.score, item.chunk.id))
        return RetrievalResult(
            query=plan,
            strategy=strategy,
            ranked=tuple(ranked),
            raw_count=len(hits),
            stale_count=stale_count,
        )
