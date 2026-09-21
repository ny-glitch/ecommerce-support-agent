from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.config import Settings
from app.db.contracts import StoredTurn
from app.errors import ServiceError
from app.knowledge.contracts import Citation
from app.tools.executor import ToolExecutor, ToolOutcome, ToolProgress
from app.tools.registry import ToolRegistry
from app.tools.schemas import OrderInput
from app.workflow.contracts import FinalControl, IntentResult
from app.workflow.gateway import WorkflowGateway
from app.workflow.prompts import build_workflow_messages


def _settings(**changes: object) -> Settings:
    base = Settings(
        _env_file=None,
        llm_base_url="https://upstream.example/v1",
        llm_model="test-chat-model",
        llm_api_key="test-key",
        context_window_tokens=16_384,
        max_output_tokens=128,
        token_safety_margin=128,
        max_history_turns=12,
    )
    return base.model_copy(update=changes)


def _completion(
    content: str | None,
    *,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-workflow-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "test-chat-model",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": usage
        or {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


def _tool_call(
    name: str = "query_order",
    *,
    arguments: str = '{"order_id":"1001"}',
    call_id: str = "call-1",
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _sse_chunk(
    content: str | None = None,
    *,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> bytes:
    payload: dict[str, Any] = {
        "id": "chatcmpl-workflow-stream-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "test-chat-model",
        "choices": [
            {
                "index": 0,
                "delta": {} if content is None else {"content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n".encode()


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> str:
    """查询指定订单的演示状态。"""
    return order_id


@tool(args_schema=OrderInput)
async def query_logistics(order_id: str) -> str:
    """查询指定订单的演示物流。"""
    return order_id


def _gateway(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    before_request: Callable[[str, list, list[dict]], None] | None = None,
    record_usage: Callable[[str, dict[str, int] | None], None] | None = None,
    settings: Settings | None = None,
) -> tuple[WorkflowGateway, httpx.AsyncClient]:
    selected = settings or _settings()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = ChatOpenAI(
        model="test-chat-model",
        api_key="test-key",
        base_url="https://upstream.example/v1",
        max_retries=0,
        stream_usage=False,
        use_responses_api=False,
        http_async_client=client,
    )
    return (
        WorkflowGateway(
            model,
            selected,
            chat_extra_body={
                "thinking": {"type": "disabled"},
                "max_completion_tokens": selected.max_output_tokens,
            },
            before_request=before_request,
            record_usage=record_usage,
        ),
        client,
    )


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


def test_control_cannot_request_ticket_without_draft() -> None:
    with pytest.raises(ValidationError):
        FinalControl(kind="respond", actions=["create_ticket"], ticket=None)


def test_control_actions_are_not_executable_tools() -> None:
    value = FinalControl(kind="clarify", actions=["handoff"], ticket=None)
    assert value.actions == ["handoff"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"intent":"order"}',
        '{"intent":"order","needs_business_data":false,"route":"business"}',
        '{"intent":"other","needs_business_data":false}',
        '{"intent":"order","needs_business_data":"false"}',
    ],
)
async def test_classify_rejects_missing_extra_unknown_and_coerced_fields(
    raw: str,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_completion(raw))

    gateway, client = _gateway(handler)
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.classify([HumanMessage("订单 1001 在哪？")])
    finally:
        await client.aclose()

    assert exc_info.value.code == "INTENT_PROTOCOL_ERROR"
    assert calls == 1


async def test_classify_sends_one_json_mode_request_and_records_safe_usage() -> None:
    bodies: list[dict[str, Any]] = []
    hooks: list[tuple[str, int, int]] = []
    usages: list[tuple[str, dict[str, int] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_completion(
                '{"intent":"logistics","needs_business_data":true}',
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                    "provider_private": "secret",
                },
            ),
        )

    gateway, client = _gateway(
        handler,
        before_request=lambda stage, messages, schemas: hooks.append(
            (stage, len(messages), len(schemas))
        ),
        record_usage=lambda stage, usage: usages.append((stage, usage)),
    )
    try:
        result = await gateway.classify([HumanMessage("你好，订单1001物流到哪")])
    finally:
        await client.aclose()

    assert result == IntentResult(intent="logistics", needs_business_data=True)
    assert hooks == [("intent", 1, 0)]
    assert usages == [
        (
            "intent",
            {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        )
    ]
    assert len(bodies) == 1
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert bodies[0]["thinking"] == {"type": "disabled"}
    assert bodies[0]["max_completion_tokens"] == 128
    assert "tools" not in bodies[0]


async def test_before_request_failure_prevents_classify_transport() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    def reject(_stage: str, _messages: list, _schemas: list[dict]) -> None:
        raise ServiceError("TURN_BUDGET_EXHAUSTED", "本轮模型预算已用尽", 429)

    gateway, client = _gateway(handler, before_request=reject)
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.classify([HumanMessage("订单 1001 在哪？")])
    finally:
        await client.aclose()

    assert exc_info.value.code == "TURN_BUDGET_EXHAUSTED"
    assert requests == []


async def test_decide_returns_one_whitelisted_native_tool_call() -> None:
    bodies: list[dict[str, Any]] = []
    hook_schemas: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_completion(
                None,
                finish_reason="tool_calls",
                tool_calls=[_tool_call()],
            ),
        )

    gateway, client = _gateway(
        handler,
        before_request=lambda _stage, _messages, schemas: hook_schemas.append(schemas),
    )
    try:
        result = await gateway.decide(
            [HumanMessage("查询订单 1001")], [query_order, query_logistics]
        )
    finally:
        await client.aclose()

    assert isinstance(result, AIMessage)
    assert result.tool_calls == [
        {
            "name": "query_order",
            "args": {"order_id": "1001"},
            "id": "call-1",
            "type": "tool_call",
        }
    ]
    assert bodies[0]["tool_choice"] == "auto"
    assert bodies[0]["parallel_tool_calls"] is False
    assert {item["function"]["name"] for item in bodies[0]["tools"]} == {
        "query_order",
        "query_logistics",
    }
    assert hook_schemas == [bodies[0]["tools"]]


@pytest.mark.parametrize(
    "arguments",
    [
        "{}",
        '{"order_id":"1001","unexpected":"x"}',
        '{"order_id":["1001"]}',
    ],
    ids=["missing", "extra", "wrong-type"],
)
async def test_parseable_business_argument_errors_reach_original_executor_once(
    arguments: str,
) -> None:
    requests: list[httpx.Request] = []
    business_invocations = 0

    @tool("query_order", args_schema=OrderInput)
    async def observed_query_order(order_id: str) -> str:
        """查询指定订单的演示状态。"""
        nonlocal business_invocations
        business_invocations += 1
        return order_id

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_completion(
                None,
                finish_reason="tool_calls",
                tool_calls=[
                    _tool_call(
                        arguments=arguments,
                        call_id="call-invalid-business-args",
                    )
                ],
            ),
        )

    gateway, client = _gateway(handler)
    try:
        decision = await gateway.decide(
            [HumanMessage("查询订单")], [observed_query_order]
        )
        assert isinstance(decision, AIMessage)
        events = [
            event
            async for event in ToolExecutor(max_attempts=2).run(
                decision.tool_calls[0],
                ToolRegistry([observed_query_order]),
                deadline=1e30,
            )
        ]
    finally:
        await client.aclose()

    assert len(requests) == 1
    assert business_invocations == 0
    assert len(events) == 2
    assert isinstance(events[0], ToolProgress)
    assert events[0].attempt == 1
    assert isinstance(events[1], ToolOutcome)
    assert events[1].attempt == 1
    assert events[1].terminal_status == "failed"
    assert events[1].message.tool_call_id == "call-invalid-business-args"
    assert json.loads(str(events[1].message.content)) == {
        "status": "error",
        "code": "INVALID_TOOL_ARGUMENTS",
    }


async def test_decide_treats_tool_name_in_control_json_as_non_executable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion(
                '{"kind":"clarify","actions":["handoff"],"ticket":null}'
            ),
        )

    gateway, client = _gateway(handler)
    try:
        result = await gateway.decide([HumanMessage("我要人工")], [query_order])
    finally:
        await client.aclose()

    assert result == FinalControl(kind="clarify", actions=["handoff"], ticket=None)


@pytest.mark.parametrize(
    "response",
    [
        _completion(
            None,
            finish_reason="tool_calls",
            tool_calls=[_tool_call("unknown_tool")],
        ),
        _completion(
            None,
            finish_reason="tool_calls",
            tool_calls=[
                _tool_call(call_id="call-1"),
                _tool_call("query_logistics", call_id="call-2"),
            ],
        ),
        _completion(
            None,
            finish_reason="tool_calls",
            tool_calls=[_tool_call(arguments="{broken-json")],
        ),
        _completion(
            '{"kind":"respond","actions":[],"ticket":null}',
            finish_reason="length",
        ),
    ],
)
async def test_decide_rejects_unknown_parallel_invalid_and_unfinished_calls(
    response: dict[str, Any],
) -> None:
    gateway, client = _gateway(
        lambda _request: httpx.Response(200, json=response)
    )
    try:
        with pytest.raises(ServiceError):
            await gateway.decide([HumanMessage("测试")], [query_order, query_logistics])
    finally:
        await client.aclose()


async def test_decide_rejects_ticket_action_without_valid_draft() -> None:
    gateway, client = _gateway(
        lambda _request: httpx.Response(
            200,
            json=_completion(
                '{"kind":"respond","actions":["create_ticket"],"ticket":null}'
            ),
        )
    )
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.decide([HumanMessage("帮我建工单")], [query_order])
    finally:
        await client.aclose()

    assert exc_info.value.code == "AGENT_CONTROL_ERROR"


async def test_assess_uses_workflow_evidence_prompt_and_validates_source_ids() -> None:
    bodies: list[dict[str, Any]] = []
    hooks: list[tuple[str, int]] = []
    usages: list[tuple[str, dict[str, int] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_completion(
                '{"sufficient":true,"reason_code":"supported",'
                '"reason":"政策原文支持","supporting_chunk_ids":[910001]}'
            ),
        )

    gateway, client = _gateway(
        handler,
        before_request=lambda stage, messages, _schemas: hooks.append(
            (stage, len(messages))
        ),
        record_usage=lambda stage, usage: usages.append((stage, usage)),
    )
    try:
        result = await gateway.assess(
            "订单1001未拆封能否退货",
            (_source(),),
            normalized_question="未拆封商品是否可退货",
            intent=IntentResult(intent="return_refund", needs_business_data=True),
        )
    finally:
        await client.aclose()

    assert result.supporting_chunk_ids == [910001]
    body = bodies[0]
    assert body["response_format"] == {"type": "json_object"}
    payload = json.loads(body["messages"][-1]["content"])
    assert payload["intent"] == {
        "intent": "return_refund",
        "needs_business_data": True,
    }
    assert payload["normalized_question"] == "未拆封商品是否可退货"
    assert hooks == [("evidence", 2)]
    assert usages == [
        (
            "evidence",
            {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        )
    ]


async def test_assess_rejects_unknown_or_duplicate_support_ids() -> None:
    for ids in ([999999], [910001, 910001]):
        raw = json.dumps(
            {
                "sufficient": True,
                "reason_code": "supported",
                "reason": "invalid",
                "supporting_chunk_ids": ids,
            }
        )
        gateway, client = _gateway(
            lambda _request, raw=raw: httpx.Response(200, json=_completion(raw))
        )
        try:
            with pytest.raises(ServiceError) as exc_info:
                await gateway.assess(
                    "问题",
                    (_source(),),
                    normalized_question="问题",
                    intent=IntentResult(
                        intent="product", needs_business_data=False
                    ),
                )
        finally:
            await client.aclose()
        assert exc_info.value.code == "EVIDENCE_ASSESSMENT_ERROR"


async def test_stream_final_emits_text_only_and_requires_terminal_stop() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=b"".join(
                [
                    _sse_chunk("演示"),
                    _sse_chunk("答案"),
                    _sse_chunk(finish_reason="stop"),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    gateway, client = _gateway(handler)
    try:
        parts = [
            part
            async for part in gateway.stream_final([HumanMessage("给出最终答复")])
        ]
    finally:
        await client.aclose()

    assert parts == ["演示", "答案"]
    assert "tools" not in bodies[0]
    assert bodies[0]["thinking"] == {"type": "disabled"}


async def test_stream_final_rejects_incomplete_stream() -> None:
    gateway, client = _gateway(
        lambda _request: httpx.Response(
            200,
            content=b"".join(
                [
                    _sse_chunk("半截"),
                    _sse_chunk(finish_reason="length"),
                    b"data: [DONE]\n\n",
                ]
            ),
        )
    )
    try:
        with pytest.raises(ServiceError) as exc_info:
            _ = [part async for part in gateway.stream_final([HumanMessage("回答")])]
    finally:
        await client.aclose()

    assert exc_info.value.code == "UPSTREAM_INCOMPLETE"


def test_build_workflow_messages_excludes_history_from_evidence() -> None:
    history = (
        StoredTurn(
            "old-turn",
            (HumanMessage("旧问题"), AIMessage("旧答案")),
        ),
    )
    window = build_workflow_messages(
        "evidence",
        settings=_settings(),
        question="当前问题",
        history=history,
        sources=(_source(),),
        tool_messages=(
            ToolMessage(
                "private business result",
                tool_call_id="call-old",
                name="query_order",
            ),
        ),
        intent=IntentResult(intent="product", needs_business_data=False),
        normalized_question="当前规范问题",
    )

    wire = json.dumps(
        [message.content for message in window.messages], ensure_ascii=False
    )
    assert "当前问题" in wire
    assert "旧问题" not in wire
    assert "private business result" not in wire
    assert window.retained_turns == []


def test_build_workflow_messages_keeps_tool_pair_and_counts_schemas() -> None:
    selected = AIMessage(
        "",
        tool_calls=[
            {
                "id": "call-1",
                "name": "query_order",
                "args": {"order_id": "1001"},
                "type": "tool_call",
            }
        ],
    )
    tool_messages = (
        selected,
        ToolMessage(
            '{"status":"ok"}',
            tool_call_id="call-1",
            name="query_order",
        ),
    )
    schema = {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询订单",
            "parameters": {"type": "object"},
        },
    }
    with_schema = build_workflow_messages(
        "agent",
        settings=_settings(),
        question="订单1001怎么样",
        tool_messages=tool_messages,
        intent=IntentResult(intent="order", needs_business_data=True),
        tool_schemas=(schema,),
    )
    without_schema = build_workflow_messages(
        "agent",
        settings=_settings(),
        question="订单1001怎么样",
        tool_messages=tool_messages,
        intent=IntentResult(intent="order", needs_business_data=True),
    )

    assert [message.type for message in with_schema.messages[-2:]] == ["ai", "tool"]
    assert with_schema.estimated_input_tokens > without_schema.estimated_input_tokens
    assert '"kind"' in str(with_schema.messages[0].content)


def test_intent_fixture_has_five_human_checked_cases_per_label() -> None:
    path = Path("evals/ch05/intents.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 35
    assert {tuple(sorted(row)) for row in rows} == {
        ("expected_intent", "history", "id", "needs_business_data", "question")
    }
    counts = {label: 0 for label in IntentResult.model_json_schema()["properties"]["intent"]["enum"]}
    for row in rows:
        IntentResult(
            intent=row["expected_intent"],
            needs_business_data=row["needs_business_data"],
        )
        counts[row["expected_intent"]] += 1
    assert set(counts.values()) == {5}
    questions = {row["question"] for row in rows}
    assert {
        "你好，订单1001物流到哪",
        "我要投诉",
        "订单1001未拆封能否退货",
        "退货政策是什么",
        "你叫什么",
    }.issubset(questions)
