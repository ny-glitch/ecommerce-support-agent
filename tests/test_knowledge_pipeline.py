from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.messages import AIMessage

from app.config import Settings
from app.errors import ServiceError
from app.knowledge.contracts import (
    EvidenceAssessment,
    QueryPlan,
    RankedChunk,
    RetrievalResult,
)
from app.knowledge.evidence import EvidenceBudget
from app.knowledge.pipeline import KnowledgePipeline, REFUSALS, decide_evidence
from ch04_helpers import make_chunk


def settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="https://upstream.example/v1",
        llm_model="test-chat-model",
        llm_api_key="test-key",
        context_window_tokens=8_192,
        max_output_tokens=128,
        token_safety_margin=128,
    )


def plan() -> QueryPlan:
    return QueryPlan(
        original="C65-Pro 支持什么协议？",
        normalized="C65-Pro 支持哪些协议？",
        synonyms=(),
        category=None,
    )


def ranked(score: float = 0.9) -> tuple[RankedChunk, ...]:
    return (RankedChunk(make_chunk(vectorize_status="done"), score),)


def retrieval(
    *,
    score: float = 0.9,
    raw_count: int = 1,
    stale_count: int = 0,
    include_ranked: bool = True,
) -> RetrievalResult:
    return RetrievalResult(
        query=plan(),
        strategy="hybrid_rerank",
        ranked=ranked(score) if include_ranked else (),
        raw_count=raw_count,
        stale_count=stale_count,
    )


def call() -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {
                "id": "call-knowledge",
                "name": "query_knowledge",
                "args": {"question": plan().original},
                "type": "tool_call",
            }
        ],
    )


def budget(question: str | None = None) -> EvidenceBudget:
    return EvidenceBudget(settings(), [], question or plan().original, call())


class Gateway:
    def __init__(self, assessment: EvidenceAssessment | Exception) -> None:
        self.assessment = assessment
        self.calls: list[tuple[str, tuple[int, ...], str]] = []

    async def assess(self, question, sources, *, normalized_question):
        self.calls.append(
            (
                question,
                tuple(source.chunk_id for source in sources),
                normalized_question,
            )
        )
        if isinstance(self.assessment, Exception):
            raise self.assessment
        return self.assessment


@pytest.mark.parametrize(
    ("result", "reason_code"),
    [
        (retrieval(raw_count=0, include_ranked=False), "no_hits"),
        (
            retrieval(raw_count=2, stale_count=2, include_ranked=False),
            "stale_evidence",
        ),
        (retrieval(score=0.2), "low_relevance"),
    ],
)
async def test_early_refusals_do_not_call_assessment(
    result: RetrievalResult, reason_code: str
) -> None:
    gateway = Gateway(AssertionError("assessment must not run"))

    decision = await decide_evidence(
        plan(),
        result,
        budget(),
        gateway=gateway,  # type: ignore[arg-type]
        threshold=0.5,
        deadline=time.monotonic() + 1,
    )

    assert decision.status == "not_found"
    assert decision.reason_code == reason_code
    assert decision.refusal == REFUSALS[reason_code]
    assert gateway.calls == []


async def test_budget_failure_refuses_without_assessment() -> None:
    long_question = "超长问题" * 300
    long_plan = QueryPlan(long_question, long_question, (), None)
    result = RetrievalResult(long_plan, "hybrid_rerank", ranked(), 1, 0)
    constrained = settings().model_copy(
        update={
            "context_window_tokens": 1_500,
            "max_output_tokens": 300,
            "token_safety_margin": 200,
        }
    )
    gateway = Gateway(AssertionError("assessment must not run"))

    decision = await decide_evidence(
        long_plan,
        result,
        EvidenceBudget(constrained, [], long_question, call()),
        gateway=gateway,  # type: ignore[arg-type]
        threshold=0.5,
        deadline=time.monotonic() + 1,
    )

    assert decision.reason_code == "context_budget"
    assert gateway.calls == []


async def test_assessment_runs_once_on_the_actual_cropped_sources() -> None:
    gateway = Gateway(
        EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="原文直接支持",
            supporting_chunk_ids=[910001],
        )
    )

    decision = await decide_evidence(
        plan(),
        retrieval(),
        budget(),
        gateway=gateway,  # type: ignore[arg-type]
        threshold=0.5,
        deadline=time.monotonic() + 1,
    )

    assert decision.status == "ok"
    assert [source.chunk_id for source in decision.sources] == [910001]
    assert gateway.calls == [(plan().original, (910001,), plan().normalized)]


async def test_unknown_supporting_chunk_id_is_a_controlled_technical_error() -> None:
    gateway = Gateway(
        EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="声称有证据",
            supporting_chunk_ids=[999999],
        )
    )

    with pytest.raises(ServiceError) as exc_info:
        await decide_evidence(
            plan(),
            retrieval(),
            budget(),
            gateway=gateway,  # type: ignore[arg-type]
            threshold=0.5,
            deadline=time.monotonic() + 1,
        )

    assert exc_info.value.code == "EVIDENCE_ASSESSMENT_ERROR"


@pytest.mark.parametrize(
    "assessment",
    [
        EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="未列支持证据",
            supporting_chunk_ids=[],
        ),
        EvidenceAssessment(
            sufficient=False,
            reason_code="supported",
            reason="状态矛盾",
            supporting_chunk_ids=[910001],
        ),
    ],
)
async def test_internally_inconsistent_assessment_never_defaults_to_answer(
    assessment: EvidenceAssessment,
) -> None:
    with pytest.raises(ServiceError) as exc_info:
        await decide_evidence(
            plan(),
            retrieval(),
            budget(),
            gateway=Gateway(assessment),  # type: ignore[arg-type]
            threshold=0.5,
            deadline=time.monotonic() + 1,
        )

    assert exc_info.value.code == "EVIDENCE_ASSESSMENT_ERROR"


async def test_insufficient_and_ambiguous_assessments_use_fixed_refusals() -> None:
    for reason_code in ("insufficient_evidence", "ambiguous_question"):
        gateway = Gateway(
            EvidenceAssessment(
                sufficient=False,
                reason_code=reason_code,  # type: ignore[arg-type]
                reason="现有证据不足",
                supporting_chunk_ids=[],
            )
        )
        decision = await decide_evidence(
            plan(),
            retrieval(),
            budget(),
            gateway=gateway,  # type: ignore[arg-type]
            threshold=None,
            deadline=time.monotonic() + 1,
        )
        assert decision.status == "not_found"
        assert decision.refusal == REFUSALS[reason_code]


class Normalizer:
    def __init__(self) -> None:
        self.calls = 0

    async def prepare(self, question, category, *, deadline):
        self.calls += 1
        return QueryPlan(question, "标准问法", (), category)


class Retriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[tuple[str, float]] = []

    async def retrieve(self, query, strategy, *, deadline, emit):
        self.calls.append((strategy, deadline))
        await emit("retrieving")
        await emit("reranking")
        return RetrievalResult(
            query=query,
            strategy=strategy,
            ranked=self.result.ranked,
            raw_count=self.result.raw_count,
            stale_count=self.result.stale_count,
        )


async def test_pipeline_has_fixed_single_pass_stages_and_default_strategy() -> None:
    normalizer = Normalizer()
    retriever = Retriever(retrieval())
    gateway = Gateway(
        EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="有依据",
            supporting_chunk_ids=[910001],
        )
    )
    events: list[str] = []

    async def emit(stage: str) -> None:
        events.append(stage)

    decision = await KnowledgePipeline(
        normalizer,  # type: ignore[arg-type]
        retriever,  # type: ignore[arg-type]
        gateway,  # type: ignore[arg-type]
        0.5,
    ).run(
        plan().original,
        None,
        budget(),
        deadline=time.monotonic() + 1,
        emit=emit,
    )

    assert decision.status == "ok"
    assert normalizer.calls == 1
    assert [strategy for strategy, _ in retriever.calls] == ["hybrid_rerank"]
    assert len(gateway.calls) == 1
    assert events == ["normalizing", "retrieving", "reranking", "checking_evidence"]


async def test_pipeline_deadline_includes_progress_callbacks() -> None:
    pipeline = KnowledgePipeline(
        Normalizer(),  # type: ignore[arg-type]
        Retriever(retrieval()),  # type: ignore[arg-type]
        Gateway(  # type: ignore[arg-type]
            EvidenceAssessment(
                sufficient=True,
                reason_code="supported",
                reason="有依据",
                supporting_chunk_ids=[910001],
            )
        ),
        0.5,
    )

    async def slow_emit(_stage: str) -> None:
        await asyncio.sleep(0.1)

    with pytest.raises(TimeoutError):
        await pipeline.run(
            plan().original,
            None,
            budget(),
            deadline=time.monotonic() + 0.01,
            emit=slow_emit,
        )
