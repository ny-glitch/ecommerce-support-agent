"""Fixed single-pass workflow retrieval and independent evidence routing."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict
import math
import time
from typing import Protocol, cast

from app.config import Settings
from app.db.contracts import StoredTurn
from app.knowledge.contracts import (
    Citation,
    EvidenceAssessment,
    KnowledgeChunk,
    KnowledgeDecision,
    QueryPlan,
    RankedChunk,
    RetrievalResult,
)
from app.knowledge.evidence import WorkflowEvidenceBudget
from app.knowledge.pipeline import decide_evidence
from app.workflow.contracts import (
    IntentResult,
    KnowledgeBand,
    KnowledgeTarget,
    WorkflowKnowledgeResult,
)
from app.workflow.routing import knowledge_band
from app.workflow.state import TurnRuntime


_RETRIEVAL_STRATEGIES = frozenset({"dense", "bm25", "hybrid", "hybrid_rerank"})
_QUERY_KEYS = frozenset({"original", "normalized", "synonyms", "category", "fallback"})
_CHUNK_KEYS = frozenset(
    {
        "id", "category", "questions", "answer", "section_path", "content_type",
        "is_key_clause", "prev_chunk_id", "next_chunk_id", "vector_id",
        "vectorize_status",
    }
)
_RETRIEVAL_KEYS = frozenset(
    {"query", "strategy", "ranked", "raw_count", "stale_count"}
)


class _Normalizer(Protocol):
    async def prepare(
        self, question: str, category: str | None, *, deadline: float
    ) -> QueryPlan: ...


class _Retriever(Protocol):
    async def retrieve(
        self,
        plan: QueryPlan,
        strategy: str,
        *,
        deadline: float,
        emit: Callable[[str], Awaitable[None]] | None = None,
    ) -> RetrievalResult: ...


class _WorkflowGateway(Protocol):
    async def assess(
        self,
        question: str,
        sources: Sequence[Citation],
        *,
        normalized_question: str,
        intent: IntentResult,
    ) -> EvidenceAssessment: ...


NormalizerFactory = Callable[[TurnRuntime], _Normalizer]
GatewayFactory = Callable[[TurnRuntime], _WorkflowGateway]


def knowledge_target(
    intent: IntentResult,
    band: KnowledgeBand | None,
    sufficient: bool,
) -> KnowledgeTarget:
    if not sufficient:
        return "fallback"
    if intent.needs_business_data or band == "middle":
        return "agent_tools"
    return "workflow_answer" if band == "high" else "agent_generate"


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"invalid {name}")
    return value


def _text(value: object, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"invalid {name}")
    return value


def _score(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("invalid reranker score")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("invalid reranker score")
    return score


def serialize_retrieval(result: RetrievalResult) -> dict:
    payload = {
        "query": {
            **asdict(result.query),
            "synonyms": list(result.query.synonyms),
        },
        "strategy": result.strategy,
        "ranked": [
            {"chunk": asdict(item.chunk), "score": item.score}
            for item in result.ranked[:50]
        ],
        "raw_count": result.raw_count,
        "stale_count": result.stale_count,
    }
    deserialize_retrieval(payload)
    return payload


def deserialize_retrieval(value: dict) -> RetrievalResult:
    if not isinstance(value, dict) or set(value) != _RETRIEVAL_KEYS:
        raise ValueError("invalid retrieval snapshot")
    query_value = value["query"]
    if not isinstance(query_value, dict) or set(query_value) != _QUERY_KEYS:
        raise ValueError("invalid retrieval query")
    synonyms = query_value["synonyms"]
    if not isinstance(synonyms, list) or not all(isinstance(item, str) for item in synonyms):
        raise ValueError("invalid retrieval synonyms")
    category = _text(query_value["category"], "query category", optional=True)
    if type(query_value["fallback"]) is not bool:
        raise ValueError("invalid retrieval fallback")
    query = QueryPlan(
        original=cast(str, _text(query_value["original"], "original query")),
        normalized=cast(str, _text(query_value["normalized"], "normalized query")),
        synonyms=tuple(synonyms),
        category=category,
        fallback=query_value["fallback"],
    )
    strategy = value["strategy"]
    if not isinstance(strategy, str) or strategy not in _RETRIEVAL_STRATEGIES:
        raise ValueError("invalid retrieval strategy")
    raw_count = _integer(value["raw_count"], "raw count", minimum=0)
    stale_count = _integer(value["stale_count"], "stale count", minimum=0)
    ranked_value = value["ranked"]
    if not isinstance(ranked_value, list) or len(ranked_value) > 50:
        raise ValueError("invalid ranked snapshot")
    ranked: list[RankedChunk] = []
    seen: set[int] = set()
    for item in ranked_value:
        if not isinstance(item, dict) or set(item) != {"chunk", "score"}:
            raise ValueError("invalid ranked item")
        chunk_value = item["chunk"]
        if not isinstance(chunk_value, dict) or set(chunk_value) != _CHUNK_KEYS:
            raise ValueError("invalid chunk snapshot")
        chunk_id = _integer(chunk_value["id"], "chunk id")
        if chunk_id in seen:
            raise ValueError("duplicate chunk id")
        seen.add(chunk_id)
        if type(chunk_value["is_key_clause"]) is not bool:
            raise ValueError("invalid key-clause flag")
        vectorize_status = chunk_value["vectorize_status"]
        if not isinstance(vectorize_status, str) or vectorize_status not in {
            "pending",
            "done",
        }:
            raise ValueError("invalid vectorize status")
        chunk = KnowledgeChunk(
            id=chunk_id,
            category=cast(str, _text(chunk_value["category"], "chunk category")),
            questions=cast(str, _text(chunk_value["questions"], "chunk questions")),
            answer=cast(str, _text(chunk_value["answer"], "chunk answer")),
            section_path=_text(chunk_value["section_path"], "section path", optional=True),
            content_type=_text(chunk_value["content_type"], "content type", optional=True),
            is_key_clause=chunk_value["is_key_clause"],
            prev_chunk_id=(
                None if chunk_value["prev_chunk_id"] is None
                else _integer(chunk_value["prev_chunk_id"], "previous chunk id")
            ),
            next_chunk_id=(
                None if chunk_value["next_chunk_id"] is None
                else _integer(chunk_value["next_chunk_id"], "next chunk id")
            ),
            vector_id=_text(chunk_value["vector_id"], "vector id", optional=True),
            vectorize_status=vectorize_status,
        )
        ranked.append(RankedChunk(chunk, _score(item["score"])))
    if stale_count > raw_count or len(ranked) > raw_count:
        raise ValueError("retrieval counts are inconsistent")
    return RetrievalResult(query, strategy, tuple(ranked), raw_count, stale_count)


class _IntentBoundGateway:
    def __init__(self, gateway: _WorkflowGateway, intent: IntentResult) -> None:
        self._gateway = gateway
        self._intent = intent

    async def assess(self, question, sources, *, normalized_question):
        return await self._gateway.assess(
            question,
            sources,
            normalized_question=normalized_question,
            intent=self._intent,
        )


class KnowledgeStage:
    """Factories bind one runtime without mutating a shared gateway owner."""

    def __init__(
        self,
        normalizer_factory: NormalizerFactory,
        retriever: _Retriever,
        gateway_factory: GatewayFactory,
        settings: Settings,
    ) -> None:
        self._normalizer_factory = normalizer_factory
        self._retriever = retriever
        self._gateway_factory = gateway_factory
        self._settings = settings

    async def retrieve(
        self,
        question: str,
        category: str | None,
        *,
        runtime: TurnRuntime,
        emit: Callable[[str], Awaitable[None]],
    ) -> RetrievalResult:
        remaining = runtime.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("knowledge stage deadline exceeded")
        async with asyncio.timeout(remaining):
            await emit("normalizing")
            query = await self._normalizer_factory(runtime).prepare(
                question, category, deadline=runtime.deadline
            )
            return await self._retriever.retrieve(
                query,
                "hybrid_rerank",
                deadline=runtime.deadline,
                emit=emit,
            )

    async def assess(
        self,
        retrieval: RetrievalResult,
        intent: IntentResult,
        history: Sequence[StoredTurn],
        tool_schemas: Sequence[dict],
        *,
        runtime: TurnRuntime,
        emit: Callable[[str], Awaitable[None]],
    ) -> WorkflowKnowledgeResult:
        score = max((_score(item.score) for item in retrieval.ranked), default=None)
        await emit("checking_evidence")
        decision = await decide_evidence(
            retrieval.query,
            retrieval,
            WorkflowEvidenceBudget(
                self._settings,
                history,
                retrieval.query.original,
                intent,
                tool_schemas,
            ),
            gateway=_IntentBoundGateway(self._gateway_factory(runtime), intent),
            threshold=None,
            deadline=runtime.deadline,
        )
        band = (
            None
            if score is None
            else cast(
                KnowledgeBand,
                knowledge_band(
                    score,
                    lower=self._settings.workflow_knowledge_lower_threshold,
                    upper=self._settings.workflow_knowledge_upper_threshold,
                ),
            )
        )
        sufficient = (
            decision.status == "ok"
            and decision.assessment is not None
            and decision.assessment.sufficient
        )
        return WorkflowKnowledgeResult(
            decision=decision,
            score=score,
            band=band,
            target=knowledge_target(intent, band, sufficient),
        )

    async def run(
        self,
        question: str,
        category: str | None,
        intent: IntentResult,
        history: Sequence[StoredTurn],
        tool_schemas: Sequence[dict],
        *,
        runtime: TurnRuntime,
        emit: Callable[[str], Awaitable[None]],
    ) -> WorkflowKnowledgeResult:
        result = await self.retrieve(
            question, category, runtime=runtime, emit=emit
        )
        return await self.assess(
            result,
            intent,
            history,
            tool_schemas,
            runtime=runtime,
            emit=emit,
        )


def render_workflow_answer(result: WorkflowKnowledgeResult) -> str:
    if result.target != "workflow_answer":
        raise ValueError("workflow answer template requires workflow_answer target")
    assessment = result.decision.assessment
    if assessment is None or not assessment.sufficient:
        raise ValueError("workflow answer template requires sufficient evidence")
    sources = {source.chunk_id: source for source in result.decision.sources}
    blocks: list[str] = []
    for chunk_id in assessment.supporting_chunk_ids:
        source = sources.get(chunk_id)
        if source is None:
            raise ValueError("supporting source is unavailable")
        path = source.section_path or source.category
        blocks.append(f"{path}\n{source.answer}[{source.number}]")
    if not blocks:
        raise ValueError("workflow answer template requires supporting sources")
    return "\n\n".join(blocks)
