from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from typing import Any

import httpx
import pytest
from langchain_openai import ChatOpenAI
import scripts.evaluate_knowledge as evaluation_cli

from app.config import Settings
from app.knowledge.contracts import Citation
from app.knowledge.evaluation import (
    EvaluationDependencies,
    calibrate_threshold,
    evaluate_cases,
    faithfulness_score,
    observe_normalization,
    retrieval_metrics,
    strict_json_dumps,
)
from app.knowledge.evaluation_artifacts import (
    EvaluationDataError,
    build_calibration_artifact,
    execute_run,
    invalidate_run,
    safe_run_configuration,
    summarize_results,
)
from app.knowledge.calibration import RuntimeProvenance
from app.knowledge.gateway import (
    FaithfulnessClaim,
    FaithfulnessJudgement,
    FaithfulnessRequestError,
    FaithfulnessResponseError,
    KnowledgeGateway,
    NormalizationOutput,
    NormalizationResponseError,
)
from app.knowledge.contracts import (
    EvidenceAssessment,
    QueryPlan,
    RankedChunk,
    RetrievalResult,
)
from ch04_helpers import make_chunk
from app.resource_lifecycle import close_resources, warmup_local_models
from scripts.evaluate_knowledge import build_parser, validate_case_set


def test_retrieval_metrics_use_only_answerable_ground_truth() -> None:
    scored = retrieval_metrics([99, 10, 11, 12, 13], {10, 12})

    assert scored == {
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "recall_at_50": 1.0,
        "mrr_at_50": 0.5,
    }
    assert retrieval_metrics([99], set()) == {
        "recall_at_5": None,
        "recall_at_10": None,
        "recall_at_50": None,
        "mrr_at_50": None,
    }


def test_retrieval_miss_scores_zero_and_duplicate_ids_do_not_raise_recall() -> None:
    scored = retrieval_metrics([99, 99, 10], {10, 11})

    assert scored["recall_at_5"] == 0.5
    assert scored["mrr_at_50"] == pytest.approx(1 / 3)
    assert retrieval_metrics([99], {10})["recall_at_50"] == 0.0


def test_faithfulness_rejection_is_na_and_claims_are_hand_scored() -> None:
    assert faithfulness_score([]) is None
    assert faithfulness_score(
        [
            {"statement": "a", "supported": True, "source_ids": [10]},
            {"statement": "b", "supported": False, "source_ids": []},
        ]
    ) == 0.5


def test_calibration_maximizes_answerable_accepts_under_unknown_budget() -> None:
    samples = [
        (0.91, True),
        (0.80, True),
        (0.71, True),
        (0.79, False),
        (None, False),
    ]

    assert calibrate_threshold(samples, max_false_accept=0.1) == 0.8


def test_calibration_uses_conservative_boundary_and_finite_all_refuse() -> None:
    assert calibrate_threshold(
        [(0.8, True), (0.8, False)], max_false_accept=0.0
    ) > 0.8
    threshold = calibrate_threshold([(None, True), (None, False)])
    assert math.isfinite(threshold)
    assert all(score is None or score < threshold for score, _ in [(None, True)])


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_calibration_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        calibrate_threshold([(score, True)])


def test_strict_json_never_emits_non_standard_nan() -> None:
    with pytest.raises(ValueError):
        strict_json_dumps({"score": math.nan})
    assert json.loads(strict_json_dumps({"score": None})) == {"score": None}


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="https://upstream.example/v1",
        llm_model="test-chat-model",
        llm_api_key="test-key",
        context_window_tokens=8_192,
        max_output_tokens=128,
        token_safety_margin=128,
    )


def _completion(content: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-faithfulness-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "test-chat-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


def _source(chunk_id: int = 910001) -> Citation:
    return Citation(
        number=1,
        chunk_id=chunk_id,
        category="数码配件/充电器",
        section_path="商品手册/C65-Pro/协议",
        questions="C65-Pro 支持什么协议？",
        answer="USB-C 口支持 PD 3.0 和 PPS。",
        content_hash="a" * 64,
        url=f"/api/knowledge/chunks/{chunk_id}?expected_hash=" + "a" * 64,
        score=0.91,
    )


def _judge_gateway(
    handler,
) -> tuple[KnowledgeGateway, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = ChatOpenAI(
        model="test-chat-model",
        api_key="test-key",
        base_url="https://upstream.example/v1",
        max_retries=0,
        http_async_client=client,
    )
    return (
        KnowledgeGateway(
            model,
            chat_extra_body={
                "thinking": {"type": "disabled"},
                "max_completion_tokens": 128,
            },
            settings=_settings(),
        ),
        client,
    )


async def test_judge_preserves_raw_response_and_validates_supported_sources() -> None:
    requests: list[dict[str, Any]] = []
    raw = json.dumps(
        {
            "claims": [
                {
                    "statement": "C65-Pro 支持 PD 3.0。",
                    "supported": True,
                    "source_ids": [910001],
                    "reason": "证据原文明确列出该协议。",
                },
                {
                    "statement": "包装内附送充电线。",
                    "supported": False,
                    "source_ids": [],
                    "reason": "提供的证据未说明包装内容。",
                },
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(raw))

    gateway, client = _judge_gateway(handler)
    try:
        judgement = await gateway.judge(
            "C65-Pro 支持哪些协议？",
            "它支持 PD 3.0，但包装内也附送充电线。[1]",
            (_source(),),
        )
    finally:
        await client.aclose()

    assert judgement.raw_response == raw
    assert [claim.supported for claim in judgement.claims] == [True, False]
    assert judgement.claims[0].source_ids == [910001]
    assert faithfulness_score(
        [claim.model_dump(mode="json") for claim in judgement.claims]
    ) == 0.5
    assert len(requests) == 1
    body = requests[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_completion_tokens"] == 128
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    judge_input = json.loads(body["messages"][1]["content"])
    assert set(judge_input) == {"question", "answer", "sources"}
    assert judge_input["sources"][0]["chunk_id"] == 910001


@pytest.mark.parametrize(
    "raw",
    [
        '{"claims":[{"statement":"x","supported":true,'
        '"source_ids":[999999],"reason":"bad id"}]}',
        '{"claims":[{"statement":"x","supported":false,'
        '"source_ids":[910001],"reason":"not supporting"}]}',
        "{}",
    ],
)
async def test_judge_rejects_invalid_schema_or_support_ids(raw: str) -> None:
    gateway, client = _judge_gateway(
        lambda _request: httpx.Response(200, json=_completion(raw))
    )
    try:
        with pytest.raises(FaithfulnessResponseError):
            await gateway.judge("问题", "答案", (_source(),))
    finally:
        await client.aclose()


async def test_judge_validation_error_retains_raw_structured_diagnostic() -> None:
    raw = (
        '{"claims":[{"statement":"x","supported":true,'
        '"source_ids":[999999],"reason":"bad id"}]}'
    )
    gateway, client = _judge_gateway(
        lambda _request: httpx.Response(200, json=_completion(raw))
    )
    try:
        with pytest.raises(FaithfulnessResponseError) as exc_info:
            await gateway.judge("问题", "答案", (_source(),))
    finally:
        await client.aclose()

    assert exc_info.value.diagnostic_raw_response == raw
    assert raw not in str(exc_info.value)


async def test_judge_separates_transport_failure_from_invalid_response() -> None:
    gateway, client = _judge_gateway(
        lambda _request: httpx.Response(503, json={"private": "do not expose"})
    )
    try:
        with pytest.raises(FaithfulnessRequestError, match="request failed") as exc_info:
            await gateway.judge("问题", "答案", (_source(),))
    finally:
        await client.aclose()

    assert "private" not in str(exc_info.value)
    assert getattr(exc_info.value, "diagnostic_raw_response", None) is None


class _NormalizationGateway:
    def __init__(self, result: NormalizationOutput | Exception) -> None:
        self.result = result
        self.calls: list[str] = []

    async def normalize(self, question: str) -> NormalizationOutput:
        self.calls.append(question)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_observed_normalization_records_success_without_leaking_logger_state() -> None:
    logger = logging.getLogger("app.knowledge.query")
    previous_level = logger.level
    previous_handlers = tuple(logger.handlers)
    logger.setLevel(logging.WARNING)
    gateway = _NormalizationGateway(
        NormalizationOutput(normalized="C65-Pro 支持哪些协议？", synonyms=("PD",))
    )

    try:
        plan, observation = await observe_normalization(
            gateway,
            "C65-Pro 支持什么协议？",
            "数码配件/充电器",
            deadline=time.monotonic() + 1,
        )

        assert plan.fallback is False
        assert observation == {
            "request_success": True,
            "gateway_error_code": None,
            "gateway_output": {
                "normalized": "C65-Pro 支持哪些协议？",
                "synonyms": ["PD"],
            },
            "accepted": True,
            "fallback_reason": None,
        }
        assert gateway.calls == ["C65-Pro 支持什么协议？"]
        assert logger.level == logging.WARNING
        assert tuple(logger.handlers) == previous_handlers
    finally:
        logger.setLevel(previous_level)


async def test_observed_normalization_captures_info_fallback_at_default_warning_level() -> None:
    logger = logging.getLogger("app.knowledge.query")
    previous_level = logger.level
    previous_handlers = tuple(logger.handlers)
    logger.setLevel(logging.WARNING)
    gateway = _NormalizationGateway(
        NormalizationOutput(normalized="C65 支持哪些协议？", synonyms=())
    )

    try:
        plan, observation = await observe_normalization(
            gateway,
            "C65-Pro 支持什么协议？",
            None,
            deadline=time.monotonic() + 1,
        )

        assert plan.fallback is True
        assert observation["request_success"] is True
        assert observation["gateway_output"]["normalized"] == "C65 支持哪些协议？"
        assert observation["accepted"] is False
        assert observation["fallback_reason"] == "protected_terms"
        assert logger.level == logging.WARNING
        assert tuple(logger.handlers) == previous_handlers
    finally:
        logger.setLevel(previous_level)


async def test_observed_normalization_distinguishes_invalid_response() -> None:
    gateway = _NormalizationGateway(NormalizationResponseError("private body"))

    plan, observation = await observe_normalization(
        gateway,
        "充电器保修多久？",
        None,
        deadline=time.monotonic() + 1,
    )

    assert plan.fallback is True
    assert observation == {
        "request_success": False,
        "gateway_error_code": "invalid_response",
        "gateway_output": None,
        "accepted": False,
        "fallback_reason": "invalid_response",
    }
    assert "private body" not in json.dumps(observation, ensure_ascii=False)


class _EvaluationGateway(_NormalizationGateway):
    def __init__(self) -> None:
        super().__init__(
            NormalizationOutput(
                normalized="C65-Pro 支持哪些协议？",
                synonyms=("PD 协议",),
            )
        )
        self.assess_calls: list[tuple[str, tuple[int, ...], str]] = []
        self.judge_calls: list[tuple[str, str, tuple[int, ...]]] = []

    async def assess(self, question, sources, *, normalized_question):
        self.assess_calls.append(
            (question, tuple(source.chunk_id for source in sources), normalized_question)
        )
        return EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="证据直接支持",
            supporting_chunk_ids=[sources[0].chunk_id],
        )

    async def judge(self, question, answer, sources):
        self.judge_calls.append(
            (question, answer, tuple(source.chunk_id for source in sources))
        )
        judgement = FaithfulnessJudgement(
            claims=[
                FaithfulnessClaim(
                    statement="C65-Pro 支持 PD 3.0。",
                    supported=True,
                    source_ids=[sources[0].chunk_id],
                    reason="证据直接支持",
                )
            ]
        )
        judgement._raw_response = '{"claims":[]}'
        return judgement


class _EvaluationRetriever:
    def __init__(self, *, failure_strategy: str | None = None) -> None:
        self.failure_strategy = failure_strategy
        self.calls: list[tuple[QueryPlan, str]] = []

    async def retrieve(self, plan, strategy, *, deadline, emit=None):
        self.calls.append((plan, strategy))
        if strategy == self.failure_strategy:
            raise RuntimeError("private retrieval detail")
        if emit is not None:
            await emit("retrieving")
        score = 0.4 if strategy == "hybrid_rerank" else 0.9
        return RetrievalResult(
            query=plan,
            strategy=strategy,
            ranked=(RankedChunk(make_chunk(vectorize_status="done"), score),),
            raw_count=1,
            stale_count=0,
        )


class _GenerationGateway:
    def __init__(self) -> None:
        self.messages: list[list[Any]] = []

    async def stream(self, messages):
        self.messages.append(messages)
        yield "C65-Pro 支持 PD 3.0。[1]"


def _answerable_case() -> dict[str, Any]:
    return {
        "query_id": "case-1",
        "query": "C65-Pro 支持什么协议？",
        "category": "数码配件/充电器",
        "relevant_chunk_ids": [910001],
        "reference_answer": "支持 PD 3.0。",
        "answerable": True,
        "query_type": "literal",
        "difficulty": "easy",
        "rationale": "原文直接支持。",
    }


async def test_evaluate_cases_normalizes_once_and_applies_threshold_only_to_rerank() -> None:
    knowledge = _EvaluationGateway()
    retriever = _EvaluationRetriever()
    generation = _GenerationGateway()

    normalizations, results = await evaluate_cases(
        [_answerable_case()],
        strategies=("dense", "bm25", "hybrid", "hybrid_rerank"),
        calibration_threshold=0.5,
        dependencies=EvaluationDependencies(
            settings=_settings(),
            knowledge_gateway=knowledge,
            retriever=retriever,
            model_gateway=generation,
        ),
    )

    assert knowledge.calls == ["C65-Pro 支持什么协议？"]
    assert len(normalizations) == 1
    assert [strategy for _, strategy in retriever.calls] == [
        "dense",
        "bm25",
        "hybrid",
        "hybrid_rerank",
    ]
    assert len({id(plan) for plan, _ in retriever.calls}) == 1
    by_strategy = {item["strategy"]: item for item in results}
    assert by_strategy["hybrid_rerank"]["retrieval_metrics"]["recall_at_5"] == 1.0
    assert by_strategy["hybrid_rerank"]["top_score"] == 0.4
    assert by_strategy["hybrid_rerank"]["refused"] is True
    assert by_strategy["hybrid_rerank"]["reason_code"] == "low_relevance"
    assert by_strategy["hybrid_rerank"]["faithfulness"] is None
    assert all(by_strategy[name]["answered"] for name in ("dense", "bm25", "hybrid"))
    assert len(generation.messages) == 3
    assert len(knowledge.judge_calls) == 3
    serialized_messages = json.dumps(
        [[message.content for message in group] for group in generation.messages],
        ensure_ascii=False,
    )
    assert "原文直接支持" not in serialized_messages
    assert "reference_answer" not in serialized_messages


async def test_retrieval_failure_counts_zero_and_keeps_other_strategy_results() -> None:
    knowledge = _EvaluationGateway()
    retriever = _EvaluationRetriever(failure_strategy="dense")

    _, results = await evaluate_cases(
        [_answerable_case()],
        strategies=("dense", "bm25"),
        calibration_threshold=0.5,
        dependencies=EvaluationDependencies(
            settings=_settings(),
            knowledge_gateway=knowledge,
            retriever=retriever,
            model_gateway=_GenerationGateway(),
        ),
    )

    dense, bm25 = results
    assert dense["error"] == {"stage": "retrieval", "error_code": "unexpected_error"}
    assert dense["retrieval_metrics"] == {
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "recall_at_50": 0.0,
        "mrr_at_50": 0.0,
    }
    assert "private retrieval detail" not in json.dumps(dense, ensure_ascii=False)
    assert bm25["error"] is None
    assert bm25["answered"] is True


async def test_invalid_generation_keeps_diagnostic_output_without_counting_answer() -> None:
    class InvalidCitationGateway:
        async def stream(self, messages):
            yield "C65-Pro 支持 PD 3.0。[99]"

    knowledge = _EvaluationGateway()
    _, results = await evaluate_cases(
        [_answerable_case()],
        strategies=("bm25",),
        calibration_threshold=0.5,
        dependencies=EvaluationDependencies(
            settings=_settings(),
            knowledge_gateway=knowledge,
            retriever=_EvaluationRetriever(),
            model_gateway=InvalidCitationGateway(),
        ),
    )

    result = results[0]
    assert result["error"] == {
        "stage": "generation",
        "error_code": "invalid_result",
    }
    assert result["generation_diagnostic_output"] == "C65-Pro 支持 PD 3.0。[99]"
    assert result["answer"] is None
    assert result["answered"] is False
    assert knowledge.judge_calls == []


async def test_invalid_judge_keeps_valid_answer_and_raw_diagnostic_without_score() -> None:
    raw = '{"claims":[{"statement":"x","supported":true,"source_ids":[9]}]}'

    class InvalidJudgeGateway(_EvaluationGateway):
        async def judge(self, question, answer, sources):
            raise FaithfulnessResponseError(
                "invalid structured faithfulness response",
                diagnostic_raw_response=raw,
            )

    _, results = await evaluate_cases(
        [_answerable_case()],
        strategies=("bm25",),
        calibration_threshold=0.5,
        dependencies=EvaluationDependencies(
            settings=_settings(),
            knowledge_gateway=InvalidJudgeGateway(),
            retriever=_EvaluationRetriever(),
            model_gateway=_GenerationGateway(),
        ),
    )

    result = results[0]
    assert result["error"] == {
        "stage": "judge",
        "error_code": "invalid_response",
    }
    assert result["answered"] is True
    assert result["answer"] == "C65-Pro 支持 PD 3.0。[1]"
    assert result["judge_raw_response"] == raw
    assert result["claims"] == []
    assert result["faithfulness"] is None


async def test_generation_deadline_closes_stream_and_keeps_partial_output() -> None:
    class SlowGenerationGateway:
        def __init__(self) -> None:
            self.closed = False

        async def stream(self, messages):
            try:
                yield "C65-Pro 支持"
                await asyncio.sleep(0.15)
                yield " PD 3.0。[1]"
            finally:
                self.closed = True

    generation = SlowGenerationGateway()
    knowledge = _EvaluationGateway()
    settings = _settings().model_copy(
        update={"knowledge_request_timeout_seconds": 0.05}
    )
    _, results = await evaluate_cases(
        [_answerable_case()],
        strategies=("bm25",),
        calibration_threshold=0.5,
        dependencies=EvaluationDependencies(
            settings=settings,
            knowledge_gateway=knowledge,
            retriever=_EvaluationRetriever(),
            model_gateway=generation,
        ),
    )

    result = results[0]
    assert result["error"] == {"stage": "generation", "error_code": "timeout"}
    assert result["generation_diagnostic_output"] == "C65-Pro 支持"
    assert result["answered"] is False
    assert result["answer"] is None
    assert generation.closed is True
    assert knowledge.judge_calls == []


async def test_judge_uses_remaining_deadline_and_keeps_valid_generation() -> None:
    class SlowJudgeGateway(_EvaluationGateway):
        def __init__(self) -> None:
            super().__init__()
            self.judge_started = 0

        async def judge(self, question, answer, sources):
            self.judge_started += 1
            await asyncio.sleep(0.15)
            return await super().judge(question, answer, sources)

    knowledge = SlowJudgeGateway()
    settings = _settings().model_copy(
        update={"knowledge_request_timeout_seconds": 0.05}
    )
    _, results = await evaluate_cases(
        [_answerable_case()],
        strategies=("bm25",),
        calibration_threshold=0.5,
        dependencies=EvaluationDependencies(
            settings=settings,
            knowledge_gateway=knowledge,
            retriever=_EvaluationRetriever(),
            model_gateway=_GenerationGateway(),
        ),
    )

    result = results[0]
    assert knowledge.judge_started == 1
    assert result["error"] == {"stage": "judge", "error_code": "timeout"}
    assert result["answered"] is True
    assert result["answer"] == "C65-Pro 支持 PD 3.0。[1]"
    assert result["faithfulness"] is None


async def test_execute_run_writes_atomic_resume_artifacts_and_reuses_cache(tmp_path) -> None:
    output_dir = tmp_path / "run"
    configuration = {
        "schema_version": 1,
        "mode": "compare",
        "strategies": ["bm25"],
        "corpus_fingerprint": "a" * 64,
        "cases_sha256": "b" * 64,
    }
    knowledge = _EvaluationGateway()
    retriever = _EvaluationRetriever()
    generation = _GenerationGateway()
    dependencies = EvaluationDependencies(
        settings=_settings(),
        knowledge_gateway=knowledge,
        retriever=retriever,
        model_gateway=generation,
    )

    first = await execute_run(
        [_answerable_case()],
        strategies=("bm25",),
        calibration_threshold=0.5,
        dependencies=dependencies,
        output_dir=output_dir,
        configuration=configuration,
    )

    assert first["status"] == "complete"
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "normalizations.json").is_file()
    assert (output_dir / "results.jsonl").is_file()
    assert (output_dir / "report.md").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_results"] == 1
    assert "llm_api_key" not in json.dumps(manifest)

    class _MustNotRun:
        def __getattr__(self, name):
            raise AssertionError(f"cached run unexpectedly accessed {name}")

    resumed = await execute_run(
        [_answerable_case()],
        strategies=("bm25",),
        calibration_threshold=0.5,
        dependencies=EvaluationDependencies(
            settings=_settings(),
            knowledge_gateway=_MustNotRun(),
            retriever=_MustNotRun(),
            model_gateway=_MustNotRun(),
        ),
        output_dir=output_dir,
        configuration=configuration,
    )
    assert resumed == first

    with pytest.raises(EvaluationDataError, match="configuration"):
        await execute_run(
            [_answerable_case()],
            strategies=("bm25",),
            calibration_threshold=0.6,
            dependencies=dependencies,
            output_dir=output_dir,
            configuration={**configuration, "cases_sha256": "c" * 64},
        )


async def test_invalidated_run_is_terminal_and_preserves_measurement_evidence(
    tmp_path,
) -> None:
    output_dir = tmp_path / "run"
    configuration = {
        "schema_version": 1,
        "mode": "compare",
        "strategies": ["bm25"],
        "corpus_fingerprint": "a" * 64,
        "cases_sha256": "b" * 64,
    }
    knowledge = _EvaluationGateway()
    dependencies = EvaluationDependencies(
        settings=_settings(),
        knowledge_gateway=knowledge,
        retriever=_EvaluationRetriever(),
        model_gateway=_GenerationGateway(),
    )
    await execute_run(
        [_answerable_case()],
        strategies=("bm25",),
        calibration_threshold=0.5,
        dependencies=dependencies,
        output_dir=output_dir,
        configuration=configuration,
    )
    original_results = (output_dir / "results.jsonl").read_text(encoding="utf-8")
    invalidate_run(output_dir, "corpus_changed_during_run")

    with pytest.raises(
        EvaluationDataError,
        match="invalid runs require a fresh output directory",
    ):
        await execute_run(
            [_answerable_case()],
            strategies=("bm25",),
            calibration_threshold=0.5,
            dependencies=dependencies,
            output_dir=output_dir,
            configuration=configuration,
        )

    assert len(knowledge.calls) == 1
    assert (output_dir / "results.jsonl").read_text(encoding="utf-8") == original_results
    assert json.loads((output_dir / "manifest.json").read_text())["status"] == "invalid"
    assert "INVALID: corpus_changed_during_run" in (
        output_dir / "report.md"
    ).read_text(encoding="utf-8")


def test_summary_uses_full_denominators_and_keeps_failures_separate() -> None:
    successful = {
        "strategy": "hybrid",
        "query_type": "literal",
        "difficulty": "easy",
        "answerable": True,
        "answered": True,
        "refused": False,
        "retrieval_metrics": {
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "recall_at_50": 1.0,
            "mrr_at_50": 1.0,
        },
        "faithfulness": 1.0,
        "retrievable_count": 24,
        "context_relevant_coverage": 0.5,
        "error": None,
        "timings_ms": {"retrieval": 10.0, "rerank": 0.0, "total": 20.0},
    }
    failed = {
        **successful,
        "query_id": "failed",
        "difficulty": "hard",
        "answered": False,
        "retrieval_metrics": {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_50": 0.0,
            "mrr_at_50": 0.0,
        },
        "faithfulness": None,
        "retrievable_count": 0,
        "context_relevant_coverage": 0.0,
        "error": {"stage": "retrieval", "error_code": "timeout"},
        "timings_ms": {"retrieval": 30.0, "rerank": 0.0, "total": 40.0},
    }
    unknown = {
        **successful,
        "query_id": "unknown",
        "query_type": "unanswerable",
        "answerable": False,
        "answered": False,
        "refused": True,
        "retrieval_metrics": {
            "recall_at_5": None,
            "recall_at_10": None,
            "recall_at_50": None,
            "mrr_at_50": None,
        },
        "faithfulness": None,
    }

    summary = summarize_results([successful, failed, unknown])
    overall = summary["strategies"]["hybrid"]["overall"]

    assert overall["sample_count"] == 3
    assert overall["retrieval_denominator"] == 2
    assert overall["recall_at_5"] == 0.5
    assert overall["faithfulness"] == 1.0
    assert overall["faithfulness_count"] == 1
    assert overall["mean_retrievable_count"] == 16.0
    assert overall["context_relevant_coverage"] == 0.25
    assert overall["answer_rate"] == 0.5
    assert overall["unknown_refusal_rate"] == 1.0
    assert overall["unknown_wrong_answer_rate"] == 0.0
    assert overall["technical_failure_count"] == 1
    assert set(summary["strategies"]["hybrid"]["by_difficulty"]) == {"easy", "hard"}


def test_build_calibration_artifact_reuses_runtime_contract_and_exact_scores() -> None:
    provenance = RuntimeProvenance(
        corpus_fingerprint="a" * 64,
        embedding_model="BAAI/bge-m3",
        embedding_revision="b" * 40,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_revision="c" * 40,
        model_manifest_sha256="d" * 64,
        prompt_bundle_sha256="e" * 64,
    )
    cases = [
        {"query_id": f"answer-{index}", "answerable": True}
        for index in range(20)
    ] + [
        {"query_id": f"unknown-{index}", "answerable": False}
        for index in range(10)
    ]
    results = [
        {
            "query_id": case["query_id"],
            "strategy": "hybrid_rerank",
            "top_score": 0.8 if case["answerable"] else (0.85 if index == 20 else 0.7),
            "error": None,
        }
        for index, case in enumerate(cases)
    ]

    artifact = build_calibration_artifact(
        provenance,
        calibration_cases_sha256="f" * 64,
        cases=cases,
        results=results,
        max_false_accept=0.1,
    )

    assert artifact.readiness == "production"
    assert artifact.threshold == 0.8
    assert artifact.sample_count == 30
    assert artifact.answerable_count == 20
    assert artifact.unknown_count == 10
    assert artifact.technical_failure_count == 0
    assert artifact.unknown_false_accept_count == 1


async def test_warmup_cancellation_drains_physical_thread() -> None:
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class Models:
        def warmup(self) -> None:
            started.set()
            release.wait(timeout=5)
            completed.set()

    task = asyncio.create_task(warmup_local_models(Models()))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed.is_set()


async def test_close_resources_reverses_attempts_and_preserves_first_error() -> None:
    calls: list[str] = []

    class Resource:
        def __init__(self, name: str, error: Exception | None = None) -> None:
            self.name = name
            self.error = error

        async def aclose(self) -> None:
            calls.append(self.name)
            if self.error is not None:
                raise self.error

    first = RuntimeError("first close failure")
    resources = [Resource("database"), Resource("models", first), Resource("store")]

    with pytest.raises(RuntimeError, match="first close failure"):
        await close_resources(resources)

    assert calls == ["store", "models", "database"]


def test_safe_run_configuration_is_an_explicit_secret_free_allowlist() -> None:
    provenance = RuntimeProvenance(
        corpus_fingerprint="a" * 64,
        embedding_model="BAAI/bge-m3",
        embedding_revision="b" * 40,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_revision="c" * 40,
        model_manifest_sha256="d" * 64,
        prompt_bundle_sha256="e" * 64,
    )

    configuration = safe_run_configuration(
        mode="compare",
        strategies=("dense", "hybrid_rerank"),
        cases_sha256="f" * 64,
        provenance=provenance,
        settings=_settings(),
        judge_prompt_sha256="1" * 64,
        calibration_sha256="2" * 64,
        threshold=0.812345678901,
        limit=None,
        code_commit="8420256",
        dependency_versions={"langchain-openai": "1.6.2"},
    )

    serialized = json.dumps(configuration, ensure_ascii=False)
    assert configuration["threshold"] == 0.812345678901
    assert configuration["llm_model"] == "test-chat-model"
    assert configuration["llm_revision"] == "provider-configured-unpinned"
    assert "api_key" not in serialized
    assert "database" not in serialized
    assert "base_url" not in serialized


async def test_resume_rejects_changed_effective_llm_request_controls(tmp_path) -> None:
    provenance = RuntimeProvenance(
        corpus_fingerprint="a" * 64,
        embedding_model="BAAI/bge-m3",
        embedding_revision="b" * 40,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_revision="c" * 40,
        model_manifest_sha256="d" * 64,
        prompt_bundle_sha256="e" * 64,
    )
    disabled = _settings().model_copy(
        update={"llm_chat_extra_body": {"thinking": {"type": "disabled"}}}
    )
    enabled = _settings().model_copy(
        update={"llm_chat_extra_body": {"thinking": {"type": "enabled"}}}
    )
    different_endpoint = _settings().model_copy(
        update={"llm_base_url": "https://alternate.example/v1"}
    )

    def configuration(settings: Settings) -> dict[str, Any]:
        return safe_run_configuration(
            mode="compare",
            strategies=("bm25",),
            cases_sha256="f" * 64,
            provenance=provenance,
            settings=settings,
            judge_prompt_sha256="1" * 64,
            calibration_sha256="2" * 64,
            threshold=0.8,
            limit=None,
            code_commit="a1b6956",
            dependency_versions={"langchain-openai": "1.6.2"},
        )

    first_configuration = configuration(disabled)
    changed_configuration = configuration(enabled)
    assert first_configuration["llm_chat_extra_body_sha256"] != (
        changed_configuration["llm_chat_extra_body_sha256"]
    )
    assert first_configuration["llm_endpoint_sha256"] == (
        changed_configuration["llm_endpoint_sha256"]
    )
    assert first_configuration["llm_endpoint_sha256"] != (
        configuration(different_endpoint)["llm_endpoint_sha256"]
    )
    assert first_configuration["llm_token_limit_param"] == "max_completion_tokens"
    assert first_configuration["llm_request_timeout_seconds"] == 60
    serialized = json.dumps(first_configuration)
    assert "upstream.example" not in serialized
    assert "thinking" not in serialized

    output_dir = tmp_path / "run"
    dependencies = EvaluationDependencies(
        settings=disabled,
        knowledge_gateway=_EvaluationGateway(),
        retriever=_EvaluationRetriever(),
        model_gateway=_GenerationGateway(),
    )
    await execute_run(
        [_answerable_case()],
        strategies=("bm25",),
        calibration_threshold=0.8,
        dependencies=dependencies,
        output_dir=output_dir,
        configuration=first_configuration,
    )
    with pytest.raises(EvaluationDataError, match="configuration"):
        await execute_run(
            [_answerable_case()],
            strategies=("bm25",),
            calibration_threshold=0.8,
            dependencies=dependencies,
            output_dir=output_dir,
            configuration=changed_configuration,
        )


def test_cli_parser_has_full_compare_defaults_and_rejects_bad_limits() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--cases",
            "evals/ch04/test.jsonl",
            "--calibration",
            "calibration.json",
            "--output-dir",
            "reports/run",
        ]
    )

    assert args.strategies == ("dense", "bm25", "hybrid", "hybrid_rerank")
    assert args.limit is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "compare",
                "--cases",
                "evals/ch04/test.jsonl",
                "--calibration",
                "calibration.json",
                "--output-dir",
                "reports/run",
                "--limit",
                "0",
            ]
        )


def test_cli_validates_formal_case_set_shapes_without_model_calls() -> None:
    calibration = [
        {"answerable": True, "query_type": kind}
        for kind in ("literal", "model", "colloquial", "synonym", "category_filter")
        for _ in range(4)
    ] + [
        {"answerable": False, "query_type": "unanswerable"}
        for _ in range(10)
    ]
    validate_case_set("calibrate", calibration, limit=None)

    with pytest.raises(EvaluationDataError, match="20 answerable"):
        validate_case_set("calibrate", calibration[:-1], limit=None)

    comparison = [
        {"answerable": kind != "unanswerable", "query_type": kind}
        for kind in ("literal", "model", "colloquial", "synonym", "category_filter", "unanswerable")
        for _ in range(10)
    ]
    validate_case_set("compare", comparison, limit=None)
    validate_case_set("compare", comparison, limit=3)


def test_invalidating_a_run_marks_manifest_and_report(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "status": "complete"}),
        encoding="utf-8",
    )
    (output / "report.md").write_text("# report\n", encoding="utf-8")

    invalidate_run(output, "corpus_changed_during_run")

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "invalid"
    assert manifest["invalid_reason"] == "corpus_changed_during_run"
    assert "INVALID: corpus_changed_during_run" in (
        output / "report.md"
    ).read_text(encoding="utf-8")


async def test_cli_compare_assembles_read_only_runtime_and_closes_in_reverse(
    monkeypatch, tmp_path
) -> None:
    events: list[str] = []
    chunk = make_chunk()
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("{}\n", encoding="utf-8")
    calibration_path = tmp_path / "input-calibration.json"
    calibration_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    cases = [
        {"answerable": kind != "unanswerable", "query_type": kind}
        for kind in (
            "literal",
            "model",
            "colloquial",
            "synonym",
            "category_filter",
            "unanswerable",
        )
        for _ in range(10)
    ]
    configuration = Settings(
        _env_file=None,
        llm_base_url="https://upstream.example/v1",
        llm_model="test-chat-model",
        llm_api_key="test-key",
        database_url="mysql+asyncmy://local/test",
    )
    provenance = RuntimeProvenance(
        corpus_fingerprint="a" * 64,
        embedding_model="BAAI/bge-m3",
        embedding_revision="b" * 40,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_revision="c" * 40,
        model_manifest_sha256="d" * 64,
        prompt_bundle_sha256="e" * 64,
    )
    artifact = build_calibration_artifact(
        provenance,
        calibration_cases_sha256="f" * 64,
        cases=[
            {"query_id": str(index), "answerable": index < 20}
            for index in range(30)
        ],
        results=[
            {
                "query_id": str(index),
                "strategy": "hybrid_rerank",
                "top_score": 0.9 if index < 20 else 0.1,
                "error": None,
            }
            for index in range(30)
        ],
    )

    class Resource:
        def __init__(self, name: str, *args, **kwargs) -> None:
            self.name = name
            self.sessions = object()

        async def check(self) -> None:
            events.append(f"check:{self.name}")

        async def prepare_existing_collection(self) -> None:
            events.append("prepare:store")

        async def aclose(self) -> None:
            events.append(f"close:{self.name}")

        def warmup(self) -> None:
            events.append("warmup:models")

    class Gateway(Resource):
        def create_knowledge_gateway(self):
            return object()

    class Repository:
        def __init__(self, sessions) -> None:
            pass

        async def list_all(self):
            return [chunk]

    async def fake_execute_run(*args, **kwargs):
        events.append("execute")
        return {"status": "complete"}

    monkeypatch.setattr(evaluation_cli, "load_corpus", lambda path: [chunk])
    monkeypatch.setattr(evaluation_cli, "load_cases", lambda path: cases)
    monkeypatch.setattr(evaluation_cli, "validate_cases", lambda values, chunks: [])
    monkeypatch.setattr(evaluation_cli, "load_settings", lambda: configuration)
    monkeypatch.setattr(evaluation_cli, "OpenAIModelGateway", lambda settings: Gateway("gateway"))
    monkeypatch.setattr(evaluation_cli, "Database", lambda url: Resource("database"))
    monkeypatch.setattr(evaluation_cli, "MilvusStore", lambda settings: Resource("store"))
    monkeypatch.setattr(evaluation_cli, "LocalModels", lambda settings: Resource("models"))
    monkeypatch.setattr(evaluation_cli, "KnowledgeRepository", Repository)
    monkeypatch.setattr(evaluation_cli, "KnowledgeRetriever", lambda *args: object())
    monkeypatch.setattr(
        evaluation_cli,
        "corpus_fingerprint",
        lambda chunks: provenance.corpus_fingerprint,
    )
    monkeypatch.setattr(evaluation_cli, "build_runtime_provenance", lambda *a, **k: provenance)
    monkeypatch.setattr(evaluation_cli, "load_calibration", lambda *a, **k: artifact)
    monkeypatch.setattr(evaluation_cli, "execute_run", fake_execute_run)
    monkeypatch.setattr(evaluation_cli, "_code_commit", lambda: "8420256")
    monkeypatch.setattr(
        evaluation_cli,
        "_dependency_versions",
        lambda: {"langchain-openai": "1.6.2"},
    )

    args = build_parser().parse_args(
        [
            "compare",
            "--cases",
            str(cases_path),
            "--calibration",
            str(calibration_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    status = await evaluation_cli.run_command(args)

    assert status == 0
    assert json.loads((output_dir / "calibration.json").read_text())["threshold"] == 0.9
    assert events[-4:] == [
        "close:models",
        "close:store",
        "close:database",
        "close:gateway",
    ]
