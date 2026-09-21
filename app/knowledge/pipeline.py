from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from app.errors import ServiceError
from app.knowledge.contracts import (
    EvidenceAssessment,
    KnowledgeDecision,
    QueryPlan,
    RetrievalResult,
)
from app.knowledge.evidence import EvidenceBudget
from app.knowledge.gateway import KnowledgeGateway


REFUSALS = {
    "no_hits": "现有知识不足以确认这个问题，请补充具体商品或政策信息，也可以联系人工客服。",
    "low_relevance": "现有知识不足以确认这个问题，相关证据不足，请补充具体信息。",
    "insufficient_evidence": "现有知识不足以确认您询问的事项，建议补充信息或联系人工客服核实。",
    "ambiguous_question": "现有知识不足以确认您指的是哪件商品或哪项政策，请补充具体型号和问题。",
    "stale_evidence": "现有知识不足以确认答案，相关内容正在同步，请稍后重试。",
    "context_budget": "现有知识不足以确认答案，本轮暂时无法完整放入所需证据，请缩小问题范围。",
}


class _Normalizer(Protocol):
    async def prepare(
        self,
        question: str,
        category: str | None,
        *,
        deadline: float,
    ) -> QueryPlan: ...


class _Retriever(Protocol):
    async def retrieve(
        self,
        plan: QueryPlan,
        strategy: str,
        *,
        deadline: float,
        emit: Callable[[str], Awaitable[None]],
    ) -> RetrievalResult: ...


class _EvidenceBudget(Protocol):
    def select(self, ranked, query): ...


class _EvidenceGateway(Protocol):
    async def assess(
        self, question, sources, *, normalized_question
    ) -> EvidenceAssessment: ...


def _refusal(
    query: QueryPlan,
    reason_code: str,
    *,
    assessment: EvidenceAssessment | None = None,
    sources=(),
) -> KnowledgeDecision:
    return KnowledgeDecision(
        query=query,
        status="not_found",
        sources=tuple(sources),
        assessment=assessment,
        reason_code=reason_code,
        refusal=REFUSALS[reason_code],
    )


def _assessment_error() -> ServiceError:
    return ServiceError(
        "EVIDENCE_ASSESSMENT_ERROR",
        "证据充分性校验失败，请重试",
        502,
    )


async def decide_evidence(
    query: QueryPlan,
    retrieval: RetrievalResult,
    budget: _EvidenceBudget,
    *,
    gateway: _EvidenceGateway,
    threshold: float | None,
    deadline: float,
) -> KnowledgeDecision:
    if retrieval.raw_count == 0:
        return _refusal(query, "no_hits")
    if not retrieval.ranked:
        reason = "stale_evidence" if retrieval.stale_count else "no_hits"
        return _refusal(query, reason)
    if threshold is not None and retrieval.ranked[0].score < threshold:
        return _refusal(query, "low_relevance")

    selected = budget.select(retrieval.ranked, query)
    if not selected.sources:
        return _refusal(query, "context_budget")

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ServiceError(
            "KNOWLEDGE_UNAVAILABLE",
            "知识服务暂时不可用，请稍后重试",
            502,
        )
    try:
        async with asyncio.timeout(remaining):
            assessment = await gateway.assess(
                query.original,
                selected.sources,
                normalized_question=query.normalized,
            )
    except TimeoutError as exc:
        raise ServiceError(
            "KNOWLEDGE_UNAVAILABLE",
            "知识服务暂时不可用，请稍后重试",
            502,
        ) from exc

    available_ids = {source.chunk_id for source in selected.sources}
    supporting_ids = set(assessment.supporting_chunk_ids)
    valid_supported = (
        assessment.sufficient
        and assessment.reason_code == "supported"
        and bool(supporting_ids)
        and supporting_ids.issubset(available_ids)
    )
    valid_refusal = (
        not assessment.sufficient
        and assessment.reason_code
        in {"insufficient_evidence", "ambiguous_question"}
        and not supporting_ids
    )
    if not (valid_supported or valid_refusal):
        raise _assessment_error()

    if valid_supported:
        return KnowledgeDecision(
            query=query,
            status="ok",
            sources=selected.sources,
            assessment=assessment,
            reason_code=None,
            refusal=None,
        )
    return _refusal(
        query,
        assessment.reason_code,
        assessment=assessment,
        sources=selected.sources,
    )


class KnowledgePipeline:
    def __init__(
        self,
        normalizer: _Normalizer,
        retriever: _Retriever,
        gateway: KnowledgeGateway,
        threshold: float,
    ) -> None:
        self._normalizer = normalizer
        self._retriever = retriever
        self._gateway = gateway
        self._threshold = threshold

    async def run(
        self,
        question: str,
        category: str | None,
        budget: EvidenceBudget,
        *,
        deadline: float,
        emit: Callable[[str], Awaitable[None]],
    ) -> KnowledgeDecision:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("knowledge pipeline deadline exceeded")
        async with asyncio.timeout(remaining):
            return await self._run(
                question,
                category,
                budget,
                deadline=deadline,
                emit=emit,
            )

    async def _run(
        self,
        question: str,
        category: str | None,
        budget: EvidenceBudget,
        *,
        deadline: float,
        emit: Callable[[str], Awaitable[None]],
    ) -> KnowledgeDecision:
        await emit("normalizing")
        query = await self._normalizer.prepare(
            question,
            category,
            deadline=deadline,
        )
        retrieval = await self._retriever.retrieve(
            query,
            "hybrid_rerank",
            deadline=deadline,
            emit=emit,
        )
        await emit("checking_evidence")
        return await decide_evidence(
            query,
            retrieval,
            budget,
            gateway=self._gateway,
            threshold=self._threshold,
            deadline=deadline,
        )
