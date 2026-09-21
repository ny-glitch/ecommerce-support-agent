from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.config import Settings
from app.db.contracts import StoredTurn
from app.errors import ServiceError
from app.knowledge.contracts import (
    Citation,
    EvidenceAssessment,
    KnowledgeDecision,
    QueryPlan,
    RankedChunk,
)
from app.knowledge.evidence import (
    EvidenceBudget,
    edge_order,
    validate_citation_numbers,
)
from app.knowledge.gateway import KnowledgeGateway
from ch04_helpers import make_chunk


def settings(**changes: object) -> Settings:
    base = Settings(
        _env_file=None,
        llm_base_url="https://upstream.example/v1",
        llm_model="test-chat-model",
        llm_api_key="test-key",
        context_window_tokens=8_192,
        max_output_tokens=128,
        token_safety_margin=128,
        max_history_turns=12,
    )
    return base.model_copy(update=changes)


def plan(question: str = "C65-Pro 支持什么协议？") -> QueryPlan:
    return QueryPlan(question, "C65-Pro 支持哪些协议？", (), None)


def call() -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {
                "id": "call-knowledge",
                "name": "query_faq",
                "args": {"question": "C65-Pro 支持什么协议？"},
                "type": "tool_call",
            }
        ],
    )


def ranked(count: int, *, answer_size: int = 8) -> tuple[RankedChunk, ...]:
    return tuple(
        RankedChunk(
            replace(
                make_chunk(),
                id=910_000 + number,
                questions=f"问题 {number}",
                answer=f"答案 {number} " + "甲" * answer_size,
            ),
            1.0 - number / 100,
        )
        for number in range(1, count + 1)
    )


def test_best_evidence_is_at_both_edges() -> None:
    assert edge_order(list(range(1, 11))) == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    assert edge_order([1, 2, 3]) == [1, 3, 2]


def test_unknown_or_missing_citation_is_generation_error() -> None:
    with pytest.raises(ValueError, match="unknown citation"):
        validate_citation_numbers("支持该协议[99]。", {1, 2})
    with pytest.raises(ValueError, match="citation required"):
        validate_citation_numbers("支持该协议。", {1, 2})
    assert validate_citation_numbers("分别支持[1][2]。", {1, 2}) == {1, 2}


def test_contracts_reject_invalid_bounds_and_oversized_payload() -> None:
    with pytest.raises(ValidationError):
        Citation(
            number=11,
            chunk_id=1,
            category="分类",
            section_path=None,
            questions="问题",
            answer="答案",
            content_hash="0" * 64,
            url="/api/knowledge/chunks/1?expected_hash=" + "0" * 64,
            score=0.5,
        )
    with pytest.raises(ValidationError):
        EvidenceAssessment(
            sufficient=False,
            reason_code="insufficient_evidence",
            reason="证" * 601,
            supporting_chunk_ids=[],
        )

    source = Citation(
        number=1,
        chunk_id=1,
        category="分类",
        section_path=None,
        questions="问题",
        answer="答" * 48_000,
        content_hash="0" * 64,
        url="/api/knowledge/chunks/1?expected_hash=" + "0" * 64,
        score=0.5,
    )
    decision = KnowledgeDecision(
        query=plan(),
        status="ok",
        sources=(source,),
        assessment=EvidenceAssessment(
            sufficient=True,
            reason_code="supported",
            reason="有直接证据",
            supporting_chunk_ids=[1],
        ),
        reason_code=None,
        refusal=None,
    )
    with pytest.raises(ValueError, match="48000"):
        decision.to_payload()


def test_budget_numbers_by_relevance_then_places_second_best_at_final_edge() -> None:
    result = EvidenceBudget(
        settings(context_window_tokens=12_000), [], plan().original, call()
    ).select(
        ranked(10), plan()
    )

    assert [source.number for source in result.sources] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    assert [source.chunk_id for source in result.sources] == [
        910001,
        910003,
        910005,
        910007,
        910009,
        910010,
        910008,
        910006,
        910004,
        910002,
    ]
    assert result.dropped_ids == ()
    assert result.sources[0].content_hash != result.sources[-1].content_hash
    assert result.sources[0].url.endswith(result.sources[0].content_hash)


def test_budget_drops_old_complete_turns_before_evidence() -> None:
    old = StoredTurn(
        "old",
        (HumanMessage("旧问题" * 700), AIMessage("旧回答" * 700)),
    )
    compact = settings(
        context_window_tokens=8_000,
        max_output_tokens=300,
        token_safety_margin=200,
    )

    result = EvidenceBudget(compact, [old], plan().original, call()).select(
        ranked(3), plan()
    )

    assert {source.chunk_id for source in result.sources} == {910001, 910002, 910003}
    assert result.dropped_ids == ()


def test_budget_removes_only_whole_lowest_ranked_chunks() -> None:
    compact = settings(
        context_window_tokens=12_000,
        max_output_tokens=300,
        token_safety_margin=200,
    )
    candidates = ranked(3, answer_size=650)

    result = EvidenceBudget(compact, [], plan().original, call()).select(
        candidates, plan()
    )

    assert 0 < len(result.sources) < 3
    kept = {source.chunk_id for source in result.sources}
    assert kept == {item.chunk.id for item in candidates[: len(kept)]}
    assert result.dropped_ids == tuple(
        item.chunk.id for item in candidates[len(kept) :]
    )
    for source in result.sources:
        original = next(item.chunk for item in candidates if item.chunk.id == source.chunk_id)
        assert source.answer == original.answer


def test_budget_returns_empty_plan_when_question_leaves_no_room_for_one_chunk() -> None:
    compact = settings(
        context_window_tokens=1_500,
        max_output_tokens=300,
        token_safety_margin=200,
    )
    question = "超长问题" * 300

    result = EvidenceBudget(compact, [], question, call()).select(
        ranked(1), QueryPlan(question, question, (), None)
    )

    assert result.sources == ()
    assert result.dropped_ids == (910001,)


def _completion(content: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-assessment-test",
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


async def test_assess_binds_controls_and_sends_original_normalized_and_sources_once() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_completion(
                '{"sufficient":true,"reason_code":"supported",'
                '"reason":"原文直接支持","supporting_chunk_ids":[910001]}'
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = ChatOpenAI(
        model="test-chat-model",
        api_key="test-key",
        base_url="https://upstream.example/v1",
        max_retries=0,
        http_async_client=client,
    )
    gateway = KnowledgeGateway(
        model,
        chat_extra_body={
            "thinking": {"type": "disabled"},
            "max_completion_tokens": 128,
        },
        settings=settings(),
    )
    sources = EvidenceBudget(settings(), [], plan().original, call()).select(
        ranked(1), plan()
    ).sources
    try:
        result = await gateway.assess(
            plan().original,
            sources,
            normalized_question=plan().normalized,
        )
    finally:
        await client.aclose()

    assert result.supporting_chunk_ids == [910001]
    assert len(requests) == 1
    body = requests[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_completion_tokens"] == 128
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assessment_input = json.loads(body["messages"][1]["content"])
    assert assessment_input["original_question"] == plan().original
    assert assessment_input["normalized_question"] == plan().normalized
    assert assessment_input["sources"][0]["chunk_id"] == 910001


async def test_assess_invalid_structured_result_is_a_controlled_service_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("{}"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = ChatOpenAI(
        model="test-chat-model",
        api_key="test-key",
        base_url="https://upstream.example/v1",
        max_retries=0,
        http_async_client=client,
    )
    gateway = KnowledgeGateway(
        model,
        chat_extra_body={"max_completion_tokens": 128},
        settings=settings(),
    )
    sources = EvidenceBudget(settings(), [], plan().original, call()).select(
        ranked(1), plan()
    ).sources
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.assess(
                plan().original,
                sources,
                normalized_question=plan().normalized,
            )
    finally:
        await client.aclose()

    assert exc_info.value.code == "EVIDENCE_ASSESSMENT_ERROR"


@pytest.mark.parametrize("supporting_ids", [[999999], [910001, 910001]])
async def test_existing_assess_reuses_support_id_validation(
    supporting_ids: list[int],
) -> None:
    raw = json.dumps(
        {
            "sufficient": True,
            "reason_code": "supported",
            "reason": "无效引用",
            "supporting_chunk_ids": supporting_ids,
        }
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=_completion(raw))
        )
    )
    gateway = KnowledgeGateway(
        ChatOpenAI(
            model="test-chat-model",
            api_key="test-key",
            base_url="https://upstream.example/v1",
            max_retries=0,
            http_async_client=client,
        ),
        chat_extra_body={"max_completion_tokens": 128},
        settings=settings(),
    )
    sources = EvidenceBudget(settings(), [], plan().original, call()).select(
        ranked(1), plan()
    ).sources
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.assess(
                plan().original,
                sources,
                normalized_question=plan().normalized,
            )
    finally:
        await client.aclose()

    assert exc_info.value.code == "EVIDENCE_ASSESSMENT_ERROR"
