from __future__ import annotations

from app.evaluation_io import strict_json_dumps

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
import json
import logging
import math
import time
from typing import Any

from langchain_core.messages import AIMessage

from app.config import Settings
from app.context import build_tool_context
from app.errors import ServiceError
from app.knowledge.contracts import QueryPlan, RetrievalResult
from app.knowledge.evidence import (
    EvidenceBudget,
    build_knowledge_tool_message,
    validate_citation_numbers,
)
from app.knowledge.gateway import (
    FaithfulnessRequestError,
    FaithfulnessResponseError,
    NormalizationOutput,
    NormalizationRequestError,
    NormalizationResponseError,
)
from app.knowledge.pipeline import decide_evidence
from app.knowledge.query import QueryNormalizer
from app.prompts import knowledge_answer_system_prompt


_QUERY_LOGGER_NAME = "app.knowledge.query"


def retrieval_metrics(
    ranked_ids: list[int], relevant_ids: set[int]
) -> dict[str, float | None]:
    if not relevant_ids:
        return {
            "recall_at_5": None,
            "recall_at_10": None,
            "recall_at_50": None,
            "mrr_at_50": None,
        }
    metrics: dict[str, float | None] = {}
    for limit in (5, 10, 50):
        matches = relevant_ids.intersection(ranked_ids[:limit])
        metrics[f"recall_at_{limit}"] = len(matches) / len(relevant_ids)
    metrics["mrr_at_50"] = next(
        (
            1.0 / rank
            for rank, chunk_id in enumerate(ranked_ids[:50], start=1)
            if chunk_id in relevant_ids
        ),
        0.0,
    )
    return metrics


def faithfulness_score(claims: list[dict[str, Any]]) -> float | None:
    if not claims:
        return None
    return sum(claim.get("supported") is True for claim in claims) / len(claims)


def calibrate_threshold(
    samples: list[tuple[float | None, bool]], *, max_false_accept: float = 0.1
) -> float:
    if not math.isfinite(max_false_accept) or not 0 <= max_false_accept <= 1:
        raise ValueError("max_false_accept must be a finite value between 0 and 1")
    scores = [score for score, _ in samples if score is not None]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("calibration scores must be finite")
    if not scores:
        return 0.0
    all_refuse = math.nextafter(max(scores), math.inf)
    if not math.isfinite(all_refuse):
        raise ValueError("cannot construct a finite all-refuse threshold")
    candidates = sorted(set(scores) | {all_refuse})
    unknown_count = sum(not answerable for _, answerable in samples)
    valid: list[tuple[int, float]] = []
    for threshold in candidates:
        false_accepts = sum(
            score is not None and score >= threshold
            for score, answerable in samples
            if not answerable
        )
        rate = false_accepts / unknown_count if unknown_count else 0.0
        if rate <= max_false_accept:
            accepted_answers = sum(
                score is not None and score >= threshold
                for score, answerable in samples
                if answerable
            )
            valid.append((accepted_answers, threshold))
    return max(valid, key=lambda item: (item[0], item[1]))[1] if valid else all_refuse


class _NormalizationGatewayObserver:
    def __init__(self, gateway: Any) -> None:
        self._gateway = gateway
        self.request_success = False
        self.gateway_error_code: str | None = None
        self.output: NormalizationOutput | None = None

    async def normalize(self, question: str) -> NormalizationOutput:
        try:
            output = await self._gateway.normalize(question)
        except Exception as exc:
            self.gateway_error_code = _normalization_error_code(exc)
            raise
        self.request_success = True
        self.output = output
        return output


def _normalization_error_code(exc: Exception) -> str:
    if isinstance(exc, NormalizationResponseError):
        return "invalid_response"
    if isinstance(exc, NormalizationRequestError):
        return "request_error"
    if isinstance(exc, ServiceError) and exc.code == "INPUT_TOO_LONG":
        return "budget"
    if isinstance(exc, TimeoutError):
        return "deadline"
    return "gateway_error"


class _FallbackHandler(logging.Handler):
    def __init__(self, task: asyncio.Task[Any] | None) -> None:
        super().__init__(logging.INFO)
        self._task = task
        self.reason: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            current = asyncio.current_task()
        except RuntimeError:
            return
        if (
            current is self._task
            and record.name == _QUERY_LOGGER_NAME
            and record.getMessage() == "query normalization fallback"
        ):
            reason = getattr(record, "fallback_reason", None)
            if isinstance(reason, str):
                self.reason = reason


async def observe_normalization(
    gateway: Any,
    question: str,
    category: str | None,
    *,
    deadline: float,
) -> tuple[QueryPlan, dict[str, Any]]:
    """Run the production normalizer once and retain its existing safe event."""
    observer = _NormalizationGatewayObserver(gateway)
    logger = logging.getLogger(_QUERY_LOGGER_NAME)
    previous_level = logger.level
    handler = _FallbackHandler(asyncio.current_task())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        plan = await QueryNormalizer(observer).prepare(
            question, category, deadline=deadline
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    fallback_reason = handler.reason
    if plan.fallback and fallback_reason is None:
        fallback_reason = "unobserved_fallback"
    return plan, {
        "request_success": observer.request_success,
        "gateway_error_code": observer.gateway_error_code,
        "gateway_output": (
            None if observer.output is None else observer.output.model_dump(mode="json")
        ),
        "accepted": not plan.fallback,
        "fallback_reason": fallback_reason,
    }


@dataclass(frozen=True)
class EvaluationDependencies:
    settings: Settings
    knowledge_gateway: Any
    retriever: Any
    model_gateway: Any


async def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    strategies: tuple[str, ...],
    calibration_threshold: float | None,
    dependencies: EvaluationDependencies,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalizations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for case in cases:
        deadline = time.monotonic() + dependencies.settings.knowledge_request_timeout_seconds
        plan, observation = await observe_normalization(
            dependencies.knowledge_gateway,
            case["query"],
            case.get("category"),
            deadline=deadline,
        )
        normalizations.append(
            {
                "query_id": case["query_id"],
                "original": plan.original,
                "normalized": plan.normalized,
                "synonyms": list(plan.synonyms),
                "category": plan.category,
                "fallback": plan.fallback,
                **observation,
            }
        )
        normalization_error = None
        if not observation["request_success"]:
            normalization_error = {
                "stage": "normalization",
                "error_code": observation["gateway_error_code"] or "gateway_error",
            }
        for strategy in strategies:
            results.append(
                await _evaluate_strategy(
                    case,
                    plan,
                    strategy,
                    calibration_threshold=(
                        calibration_threshold if strategy == "hybrid_rerank" else None
                    ),
                    dependencies=dependencies,
                    initial_error=normalization_error,
                )
            )
    return normalizations, results


def _base_result(
    case: dict[str, Any], strategy: str, *, initial_error: dict[str, str] | None
) -> dict[str, Any]:
    return {
        "query_id": case["query_id"],
        "question": case["query"],
        "category": case.get("category"),
        "query_type": case["query_type"],
        "difficulty": case["difficulty"],
        "answerable": case["answerable"],
        "relevant_chunk_ids": list(case["relevant_chunk_ids"]),
        "reference_answer": case["reference_answer"],
        "rationale": case["rationale"],
        "strategy": strategy,
        "ranked_ids": [],
        "ranked_scores": [],
        "raw_count": 0,
        "stale_count": 0,
        "retrievable_count": 0,
        "retrieval_metrics": _failed_metrics(set(case["relevant_chunk_ids"])),
        "top_score": None,
        "threshold": None,
        "sources": [],
        "context_relevant_coverage": (
            0.0 if case["relevant_chunk_ids"] else None
        ),
        "assessment": None,
        "answered": False,
        "refused": False,
        "reason_code": None,
        "answer": None,
        "generation_diagnostic_output": None,
        "claims": [],
        "judge_raw_response": None,
        "faithfulness": None,
        "error": initial_error,
        "timings_ms": {
            "retrieval": 0.0,
            "rerank": 0.0,
            "evidence": 0.0,
            "generation": 0.0,
            "judge": 0.0,
            "total": 0.0,
        },
    }


async def _evaluate_strategy(
    case: dict[str, Any],
    plan: QueryPlan,
    strategy: str,
    *,
    calibration_threshold: float | None,
    dependencies: EvaluationDependencies,
    initial_error: dict[str, str] | None,
) -> dict[str, Any]:
    result = _base_result(case, strategy, initial_error=initial_error)
    started = time.perf_counter()
    deadline = time.monotonic() + dependencies.settings.knowledge_request_timeout_seconds
    retrieval_timing = _RetrievalTiming(time.perf_counter())
    try:
        retrieval: RetrievalResult = await dependencies.retriever.retrieve(
            plan, strategy, deadline=deadline, emit=retrieval_timing.emit
        )
    except Exception as exc:
        result["error"] = _safe_error("retrieval", exc)
        retrieval_timing.finish(result["timings_ms"])
        result["timings_ms"]["total"] = _elapsed_ms(started)
        return result
    ranked_ids = [item.chunk.id for item in retrieval.ranked]
    relevant_ids = set(case["relevant_chunk_ids"])
    result.update(
        {
            "ranked_ids": ranked_ids,
            "ranked_scores": [item.score for item in retrieval.ranked],
            "raw_count": retrieval.raw_count,
            "stale_count": retrieval.stale_count,
            "retrievable_count": len(retrieval.ranked),
            "retrieval_metrics": retrieval_metrics(ranked_ids, relevant_ids),
            "top_score": retrieval.ranked[0].score if retrieval.ranked else None,
            "threshold": calibration_threshold,
        }
    )
    retrieval_timing.finish(result["timings_ms"])

    call = _evaluation_call(case["query_id"], strategy)
    budget = EvidenceBudget(dependencies.settings, [], case["query"], call)
    evidence_started = time.perf_counter()
    try:
        decision = await decide_evidence(
            plan,
            retrieval,
            budget,
            gateway=dependencies.knowledge_gateway,
            threshold=calibration_threshold,
            deadline=deadline,
        )
    except Exception as exc:
        result["error"] = _safe_error("evidence", exc)
        result["timings_ms"]["evidence"] = _elapsed_ms(evidence_started)
        result["timings_ms"]["total"] = _elapsed_ms(started)
        return result
    result["timings_ms"]["evidence"] = _elapsed_ms(evidence_started)
    result["sources"] = [source.model_dump(mode="json") for source in decision.sources]
    result["assessment"] = (
        None if decision.assessment is None else decision.assessment.model_dump(mode="json")
    )
    result["reason_code"] = decision.reason_code
    if relevant_ids:
        retained = relevant_ids.intersection(source.chunk_id for source in decision.sources)
        result["context_relevant_coverage"] = len(retained) / len(relevant_ids)
    if decision.status == "not_found":
        result["refused"] = True
        result["answer"] = decision.refusal
        result["timings_ms"]["total"] = _elapsed_ms(started)
        return result

    generation_started = time.perf_counter()
    try:
        tool_message = build_knowledge_tool_message(call, decision)
        window = build_tool_context(
            knowledge_answer_system_prompt(),
            [],
            case["query"],
            dependencies.settings,
            tool_schemas=[],
            current_tool_messages=[call, tool_message],
        )
        parts: list[str] = []
        upstream = dependencies.model_gateway.stream(window.messages)
        async with asyncio.timeout_at(deadline):
            async with aclosing(upstream) as tokens:
                async for part in tokens:
                    parts.append(part)
                    result["generation_diagnostic_output"] = "".join(parts)
        answer = "".join(parts)
        if not answer.strip():
            raise ValueError("generated answer is empty")
        validate_citation_numbers(answer, {source.number for source in decision.sources})
    except Exception as exc:
        result["error"] = _safe_error("generation", exc)
        result["timings_ms"]["generation"] = _elapsed_ms(generation_started)
        result["timings_ms"]["total"] = _elapsed_ms(started)
        return result
    result["timings_ms"]["generation"] = _elapsed_ms(generation_started)
    result["answered"] = True
    result["answer"] = answer

    judge_started = time.perf_counter()
    try:
        async with asyncio.timeout_at(deadline):
            judgement = await dependencies.knowledge_gateway.judge(
                case["query"], answer, decision.sources
            )
        claims = [claim.model_dump(mode="json") for claim in judgement.claims]
        result["claims"] = claims
        result["judge_raw_response"] = judgement.raw_response
        result["faithfulness"] = faithfulness_score(claims)
    except Exception as exc:
        if isinstance(exc, FaithfulnessResponseError):
            result["judge_raw_response"] = exc.diagnostic_raw_response
        result["error"] = _safe_error("judge", exc)
    result["timings_ms"]["judge"] = _elapsed_ms(judge_started)
    result["timings_ms"]["total"] = _elapsed_ms(started)
    return result


class _RetrievalTiming:
    def __init__(self, started: float) -> None:
        self.started = started
        self.rerank_started: float | None = None

    async def emit(self, stage: str) -> None:
        if stage == "reranking" and self.rerank_started is None:
            self.rerank_started = time.perf_counter()

    def finish(self, target: dict[str, float]) -> None:
        finished = time.perf_counter()
        if self.rerank_started is None:
            target["retrieval"] = round((finished - self.started) * 1_000, 3)
            target["rerank"] = 0.0
        else:
            target["retrieval"] = round((self.rerank_started - self.started) * 1_000, 3)
            target["rerank"] = round((finished - self.rerank_started) * 1_000, 3)


def _evaluation_call(query_id: str, strategy: str) -> AIMessage:
    import hashlib

    suffix = hashlib.sha256(f"{query_id}\0{strategy}".encode()).hexdigest()[:24]
    return AIMessage(
        "",
        tool_calls=[
            {
                "id": f"eval-{suffix}",
                "name": "query_faq",
                "args": {},
                "type": "tool_call",
            }
        ],
    )


def _failed_metrics(relevant_ids: set[int]) -> dict[str, float | None]:
    if not relevant_ids:
        return retrieval_metrics([], set())
    return {
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "recall_at_50": 0.0,
        "mrr_at_50": 0.0,
    }


def _safe_error(stage: str, exc: Exception) -> dict[str, str]:
    if isinstance(exc, ServiceError):
        code = exc.code.lower()
    elif isinstance(exc, FaithfulnessRequestError):
        code = "request_error"
    elif isinstance(exc, FaithfulnessResponseError):
        code = "invalid_response"
    elif isinstance(exc, TimeoutError):
        code = "timeout"
    elif isinstance(exc, (ValueError, TypeError)):
        code = "invalid_result"
    else:
        code = "unexpected_error"
    return {"stage": stage, "error_code": code}


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1_000, 3)
