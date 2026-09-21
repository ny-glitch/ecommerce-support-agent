from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from app.config import Settings
from app.db.contracts import TurnRef
from app.knowledge.contracts import (
    Citation,
    EvidenceAssessment,
    KnowledgeChunk,
    QueryPlan,
    RankedChunk,
    RetrievalResult,
)
from app.knowledge.evidence import EvidenceSelector, WorkflowEvidenceBudget
from app.services.turn_operations import TurnOperations
from app.workflow.budget import RequestBudget
from app.workflow.contracts import IntentResult, WorkflowKnowledgeResult
from app.workflow.knowledge import (
    KnowledgeStage,
    deserialize_retrieval,
    knowledge_target,
    render_workflow_answer,
    serialize_retrieval,
)
from app.workflow.routing import knowledge_band
from app.workflow.state import TurnRuntime, load_turns
from ch04_helpers import make_chunk


def settings(**changes: object) -> Settings:
    value = Settings(
        _env_file=None,
        llm_base_url="https://upstream.example/v1",
        llm_model="test-chat-model",
        llm_api_key="test-key",
        context_window_tokens=16_000,
        max_output_tokens=128,
        token_safety_margin=128,
    )
    return value.model_copy(update=changes)


def plan(*, category: str | None = None) -> QueryPlan:
    return QueryPlan(
        original="C65-Pro 支持什么协议？",
        normalized="C65-Pro 支持哪些协议？",
        synonyms=("充电协议",),
        category=category,
    )


def ranked(
    score: float = .9,
    *,
    chunk_id: int = 910001,
    answer: str = "支持 PD 3.0。",
    section_path: str | None = "商品手册/C65-Pro/协议",
) -> RankedChunk:
    return RankedChunk(
        replace(
            make_chunk(vectorize_status="done"),
            id=chunk_id,
            answer=answer,
            section_path=section_path,
        ),
        score,
    )


def retrieval(
    score: float = .9,
    *,
    raw_count: int = 1,
    stale_count: int = 0,
    include_ranked: bool = True,
    query: QueryPlan | None = None,
) -> RetrievalResult:
    current = query or plan()
    return RetrievalResult(
        query=current,
        strategy="hybrid_rerank",
        ranked=(ranked(score),) if include_ranked else (),
        raw_count=raw_count,
        stale_count=stale_count,
    )


def runtime() -> TurnRuntime:
    started = time.monotonic()
    return TurnRuntime(
        ref=TurnRef("conversation-1", "turn-1"),
        user_id="user-1",
        started_at=started,
        deadline=started + 30,
        budget=RequestBudget(49_152),
        operations=TurnOperations(),
    )


@pytest.mark.parametrize(
    ("score", "sufficient", "target"),
    [
        (.65, True, "agent_generate"),
        (.95, False, "fallback"),
        (.9, True, "workflow_answer"),
        (.75, True, "agent_tools"),
        (.7, True, "agent_tools"),
        (.8, True, "agent_tools"),
    ],
)
def test_knowledge_target_is_not_score_only(
    score: float, sufficient: bool, target: str
) -> None:
    intent = IntentResult(intent="product", needs_business_data=False)
    assert knowledge_target(intent, knowledge_band(score), sufficient) == target


def test_business_data_only_routes_to_tools_after_policy_evidence_passes() -> None:
    intent = IntentResult(intent="return_refund", needs_business_data=True)

    assert knowledge_target(intent, "high", False) == "fallback"
    assert knowledge_target(intent, "high", True) == "agent_tools"


def test_evidence_selector_keeps_stable_numbers_and_drops_whole_weakest_chunks() -> None:
    candidates = tuple(ranked(1 - i / 10, chunk_id=910000 + i) for i in range(1, 5))
    attempts: list[tuple[int, ...]] = []

    def fits(sources, _query):
        attempts.append(tuple(source.chunk_id for source in sources))
        return len(sources) <= 2

    result = EvidenceSelector(fits).select(candidates, plan())

    assert [set(value) for value in attempts] == [
        {910001, 910002, 910003, 910004},
        {910001, 910002, 910003},
        {910001, 910002},
    ]
    assert [(source.chunk_id, source.number) for source in result.sources] == [
        (910001, 1),
        (910002, 2),
    ]
    assert result.dropped_ids == (910003, 910004)


def test_workflow_budget_counts_real_evidence_agent_answer_and_tool_schema_requests() -> None:
    source = (ranked(.9, answer="完整答案" * 80),)
    small_schema = ({"type": "function", "function": {"name": "query_order"}},)
    huge_schema = (
        {
            "type": "function",
            "function": {"name": "query_order", "description": "参" * 10_000},
        },
    )
    intent = IntentResult(intent="product", needs_business_data=True)

    accepted = WorkflowEvidenceBudget(
        settings(), [], plan().original, intent, small_schema
    ).select(source, plan())
    rejected = WorkflowEvidenceBudget(
        settings(), [], plan().original, intent, huge_schema
    ).select(source, plan())

    assert accepted.sources[0].answer == source[0].chunk.answer
    assert rejected.sources == ()
    assert rejected.dropped_ids == (910001,)


def test_retrieval_serialization_round_trips_full_chunks_and_caps_snapshot_at_50() -> None:
    candidates = tuple(
        ranked(1 - index / 100, chunk_id=910000 + index)
        for index in range(1, 61)
    )
    result = RetrievalResult(plan(category="数码配件"), "hybrid_rerank", candidates, 60, 0)

    payload = serialize_retrieval(result)
    restored = deserialize_retrieval(payload)

    assert len(payload["ranked"]) == 50
    assert restored == replace(result, ranked=candidates[:50])
    assert payload["ranked"][0]["chunk"] == {
        "id": 910001,
        "category": "数码配件/充电器",
        "questions": "C65-Pro 支持什么协议？",
        "answer": "支持 PD 3.0。",
        "section_path": "商品手册/C65-Pro/协议",
        "content_type": "manual",
        "is_key_clause": False,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "vector_id": None,
        "vectorize_status": "done",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value["query"].update(category=7),
        lambda value: value["ranked"][0].update(score=float("nan")),
        lambda value: value["ranked"][0].update(score=1.01),
        lambda value: value["ranked"][0]["chunk"].update(vectorize_status=[]),
        lambda value: value.update(raw_count=-1),
        lambda value: value.update(stale_count=2),
        lambda value: value["ranked"].extend(value["ranked"] * 50),
    ],
)
def test_retrieval_deserialization_rejects_invalid_fields_counts_and_scores(mutate) -> None:
    value = serialize_retrieval(retrieval())
    mutate(value)

    with pytest.raises(ValueError):
        deserialize_retrieval(value)


class Normalizer:
    def __init__(self) -> None:
        self.categories: list[str | None] = []

    async def prepare(self, question, category, *, deadline):
        self.categories.append(category)
        return QueryPlan(question, question, (), category)


class Retriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[tuple[QueryPlan, str]] = []

    async def retrieve(self, query, strategy, *, deadline, emit=None):
        self.calls.append((query, strategy))
        return replace(self.result, query=query, strategy=strategy)


class Gateway:
    def __init__(self, assessment: EvidenceAssessment) -> None:
        self.assessment = assessment
        self.calls: list[tuple[str, tuple[int, ...], IntentResult]] = []

    async def assess(self, question, sources, *, normalized_question, intent):
        self.calls.append((question, tuple(source.chunk_id for source in sources), intent))
        return self.assessment


def stage(
    result: RetrievalResult,
    assessment: EvidenceAssessment,
    *,
    stage_settings: Settings | None = None,
):
    normalizers: list[tuple[TurnRuntime, Normalizer]] = []
    gateways: list[tuple[TurnRuntime, Gateway]] = []
    retriever = Retriever(result)

    def normalizer_factory(bound_runtime: TurnRuntime) -> Normalizer:
        value = Normalizer()
        normalizers.append((bound_runtime, value))
        return value

    def gateway_factory(bound_runtime: TurnRuntime) -> Gateway:
        value = Gateway(assessment)
        gateways.append((bound_runtime, value))
        return value

    return (
        KnowledgeStage(
            normalizer_factory,
            retriever,
            gateway_factory,
            stage_settings or settings(),
        ),
        normalizers,
        gateways,
        retriever,
    )


async def silent_emit(_stage: str) -> None:
    return None


async def test_retrieve_binds_factory_to_runtime_once_and_category_never_inherits() -> None:
    current_runtime = runtime()
    knowledge, normalizers, _gateways, retriever = stage(
        retrieval(),
        EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="完整支持",
            supporting_chunk_ids=[910001],
        ),
    )

    await knowledge.retrieve(
        "第一问", "数码配件", runtime=current_runtime, emit=silent_emit
    )
    await knowledge.retrieve("第二问", None, runtime=current_runtime, emit=silent_emit)

    assert [bound for bound, _ in normalizers] == [current_runtime, current_runtime]
    assert [value.categories for _, value in normalizers] == [["数码配件"], [None]]
    assert [strategy for _, strategy in retriever.calls] == [
        "hybrid_rerank",
        "hybrid_rerank",
    ]


async def test_low_score_complete_evidence_is_assessed_and_not_cut_off_by_old_threshold() -> None:
    assessment = EvidenceAssessment(
        sufficient=True,
        reason_code="supported",
        reason="低相关分但原文完整",
        supporting_chunk_ids=[910001],
    )
    knowledge, _normalizers, gateways, _retriever = stage(retrieval(.65), assessment)
    current_runtime = runtime()
    intent = IntentResult(intent="product", needs_business_data=False)

    result = await knowledge.run(
        plan().original,
        None,
        intent,
        [],
        (),
        runtime=current_runtime,
        emit=silent_emit,
    )

    assert result.score == .65
    assert result.band == "low"
    assert result.target == "agent_generate"
    assert result.decision.status == "ok"
    assert gateways[0][0] is current_runtime
    assert len(gateways[0][1].calls) == 1


async def test_invalid_reranker_score_fails_before_gateway_is_created() -> None:
    knowledge, _normalizers, gateways, _retriever = stage(
        retrieval(float("nan")),
        EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="不应调用",
            supporting_chunk_ids=[910001],
        ),
    )

    with pytest.raises(ValueError, match="reranker score"):
        await knowledge.run(
            plan().original,
            None,
            IntentResult(intent="product", needs_business_data=False),
            [],
            (),
            runtime=runtime(),
            emit=silent_emit,
        )

    assert gateways == []


async def test_workflow_budget_drops_whole_chunks_before_single_assessment() -> None:
    candidates = tuple(
        ranked(
            1 - number / 10,
            chunk_id=910000 + number,
            answer=f"完整答案 {number} " + "甲" * 1_200,
        )
        for number in range(1, 4)
    )
    result = RetrievalResult(plan(), "hybrid_rerank", candidates, 3, 0)
    assessment = EvidenceAssessment(
        sufficient=True,
        reason_code="supported",
        reason="保留来源完整支持",
        supporting_chunk_ids=[910001],
    )
    constrained = settings(
        context_window_tokens=12_000,
        max_output_tokens=300,
        token_safety_margin=200,
    )
    knowledge, _normalizers, gateways, _retriever = stage(
        result, assessment, stage_settings=constrained
    )

    routed = await knowledge.run(
        plan().original,
        None,
        IntentResult(intent="product", needs_business_data=False),
        [],
        (),
        runtime=runtime(),
        emit=silent_emit,
    )

    assessed_ids = gateways[0][1].calls[0][1]
    assert 0 < len(assessed_ids) < len(candidates)
    assert set(assessed_ids) == {
        source.chunk_id for source in routed.decision.sources
    }
    assert len(gateways[0][1].calls) == 1


async def test_high_score_insufficient_evidence_falls_back_independently() -> None:
    assessment = EvidenceAssessment(
        sufficient=False,
        reason_code="insufficient_evidence",
        reason="型号不匹配",
        supporting_chunk_ids=[],
    )
    knowledge, *_ = stage(retrieval(.95), assessment)

    result = await knowledge.run(
        plan().original,
        None,
        IntentResult(intent="product", needs_business_data=False),
        [],
        (),
        runtime=runtime(),
        emit=silent_emit,
    )

    assert result.score == .95
    assert result.band == "high"
    assert result.target == "fallback"
    assert result.decision.status == "not_found"


@pytest.mark.parametrize(
    ("empty", "reason"),
    [
        (retrieval(raw_count=0, include_ranked=False), "no_hits"),
        (retrieval(raw_count=2, stale_count=2, include_ranked=False), "stale_evidence"),
    ],
)
async def test_zero_hits_and_stale_only_have_no_fake_score(
    empty: RetrievalResult, reason: str
) -> None:
    knowledge, *_ = stage(
        empty,
        EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="不应调用",
            supporting_chunk_ids=[910001],
        ),
    )

    result = await knowledge.run(
        plan().original,
        None,
        IntentResult(intent="product", needs_business_data=False),
        [],
        (),
        runtime=runtime(),
        emit=silent_emit,
    )

    assert result.score is None
    assert result.band is None
    assert result.target == "fallback"
    assert result.decision.reason_code == reason


async def test_stage_consumes_configured_threshold_pair() -> None:
    assessment = EvidenceAssessment(
        sufficient=True,
        reason_code="supported",
        reason="完整支持",
        supporting_chunk_ids=[910001],
    )
    configured = settings(
        workflow_knowledge_lower_threshold=.6,
        workflow_knowledge_upper_threshold=.7,
    )
    knowledge, *_ = stage(retrieval(.65), assessment, stage_settings=configured)

    result = await knowledge.run(
        plan().original,
        None,
        IntentResult(intent="product", needs_business_data=False),
        [],
        (),
        runtime=runtime(),
        emit=silent_emit,
    )

    assert result.band == "middle"
    assert result.target == "agent_tools"


async def test_compound_refund_policy_can_query_order_only_after_policy_gate() -> None:
    policy = ranked(
        .91,
        answer="未拆封且签收后七日内可以申请退货。",
        chunk_id=920001,
        section_path="退换货政策/七日无理由",
    )
    result = RetrievalResult(plan(), "hybrid_rerank", (policy,), 1, 0)
    knowledge, *_ = stage(
        result,
        EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="政策条件完整；订单 1001 状态需后续工具查询",
            supporting_chunk_ids=[920001],
        ),
    )

    routed = await knowledge.run(
        "订单 1001 未拆封，可以退吗？",
        None,
        IntentResult(intent="return_refund", needs_business_data=True),
        [],
        (),
        runtime=runtime(),
        emit=silent_emit,
    )

    assert routed.target == "agent_tools"
    assert routed.decision.assessment is not None
    assert routed.decision.assessment.supporting_chunk_ids == [920001]


def test_high_band_template_uses_only_supported_full_sources() -> None:
    selected = ranked(
        .91,
        chunk_id=920001,
        answer="未拆封且签收后七日内可以申请退货。",
        section_path="退换货政策/七日无理由",
    )
    unused = ranked(.89, chunk_id=920002, answer="这一块不能展示。")
    budget = EvidenceSelector(lambda _sources, _query: True)
    sources = budget.select((selected, unused), plan()).sources
    assessment = EvidenceAssessment(
        sufficient=True,
        reason_code="supported",
        reason="第一块完整支持",
        supporting_chunk_ids=[920001],
    )
    from app.knowledge.contracts import KnowledgeDecision

    result = WorkflowKnowledgeResult(
        decision=KnowledgeDecision(plan(), "ok", sources, assessment, None, None),
        score=.91,
        band="high",
        target="workflow_answer",
    )

    assert render_workflow_answer(result) == (
        "退换货政策/七日无理由\n"
        "未拆封且签收后七日内可以申请退货。[1]"
    )


def _jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def _corpus_chunks() -> dict[int, dict]:
    corpus = json.loads(
        Path("data/knowledge/ch04/chunks.json").read_text(encoding="utf-8")
    )
    return {
        chunk["id"]: chunk
        for document in corpus["documents"]
        for chunk in document["chunks"]
    }


def test_formal_evidence_labels_are_grounded_in_the_real_ch04_corpus() -> None:
    cases = _jsonl("evals/ch05/evidence.jsonl")
    chunks = _corpus_chunks()
    corpus_text = json.dumps(list(chunks.values()), ensure_ascii=False)
    expected_facts = {
        "evidence-001": {910001: ("PD 3.0", "PPS", "QC 3.0")},
        "evidence-005": {
            910071: ("签收后 7 天内", "刷头密封完好"),
            910110: ("寄回运费由买家承担",),
        },
        "evidence-006": {910030: ("机身启动键", "全屋清扫", "回充")},
        "evidence-007": {910015: ("签收后 7 天内", "仅验货", "无使用痕迹")},
        "evidence-008": {910082: ("不可以", "微波炉")},
        "evidence-009": {910088: ("密封圈正确安装", "可能渗漏")},
        "evidence-010": {
            910007: ("不超过 65W", "可能充电缓慢或无法充电"),
            910006: ("不含充电线",),
        },
        "evidence-011": {910053: ("IPX7", "不得长时间浸泡", "水下充电")},
        "evidence-012": {910070: ("密封包装拆封后不支持无理由退货",)},
    }
    absent_terms = {
        "evidence-002": "C65-Air",
        "evidence-003": "Z99-Pro",
        "evidence-004": "刻字",
    }

    assert len(chunks) == 120
    assert set(chunks) == set(range(910001, 910121))
    assert len(cases) == 12
    assert len({case["id"] for case in cases}) == 12
    assert {
        "model_mismatch",
        "no_answer",
        "policy_conditions_missing",
        "cross_chunk_support",
        "low_relevance_complete",
    }.issubset({case["case_type"] for case in cases})
    assert {case["id"] for case in cases if case["expected_sufficient"]} == set(
        expected_facts
    )
    for case in cases:
        assert set(case) == {
            "id",
            "case_type",
            "question",
            "history",
            "category",
            "expected_intent",
            "expected_sufficient",
            "supporting_chunk_ids",
            "needs_business_data",
            "reason",
        }
        assert "sources" not in case
        assert "score" not in json.dumps(case, ensure_ascii=False).lower()
        assert isinstance(case["question"], str) and case["question"].strip()
        assert case["category"] is None or isinstance(case["category"], str)
        load_turns(case["history"])
        IntentResult(
            intent=case["expected_intent"],
            needs_business_data=case["needs_business_data"],
        )
        support = case["supporting_chunk_ids"]
        assert bool(support) == case["expected_sufficient"]
        assert set(support).issubset(chunks)
        if case["category"] is not None:
            assert all(chunks[chunk_id]["category"] == case["category"] for chunk_id in support)
        for chunk_id, facts in expected_facts.get(case["id"], {}).items():
            assert chunk_id in support
            for fact in facts:
                assert fact in chunks[chunk_id]["answer"]
                assert fact in case["reason"]
        if case["id"] in absent_terms:
            term = absent_terms[case["id"]]
            assert term in case["question"]
            assert term in case["reason"]
            assert term not in corpus_text


def test_adversarial_evidence_cases_are_isolated_assessor_inputs() -> None:
    cases = _jsonl("evals/ch05/evidence_adversarial.jsonl")

    assert len(cases) == 2
    assert {case["case_type"] for case in cases} == {
        "source_prompt_injection",
        "conflicting_sources",
    }
    for case in cases:
        assert set(case) == {
            "id",
            "case_type",
            "evaluation_scope",
            "input",
            "expected",
            "reason",
        }
        assert case["evaluation_scope"] == "assessor_fixture"
        assert set(case["input"]) == {"question", "intent", "sources"}
        assert set(case["expected"]) == {
            "sufficient",
            "reason_code",
            "supporting_chunk_ids",
        }
        IntentResult.model_validate(case["input"]["intent"], strict=True)
        sources = [Citation.model_validate(source, strict=True) for source in case["input"]["sources"]]
        assert sources
        assert case["expected"] == {
            "sufficient": False,
            "reason_code": "insufficient_evidence",
            "supporting_chunk_ids": [],
        }
        assert "expected" not in json.dumps(case["input"], ensure_ascii=False)
