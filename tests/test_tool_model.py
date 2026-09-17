from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.model as model_module
from app.config import Settings
from app.db.contracts import TurnRef
from app.db.faq import FaqRepository
from app.db.tickets import TicketRepository
from app.errors import ServiceError
from app.model import OpenAIModelGateway
from app.tools.business import ToolContext, build_registry


def completion(
    *,
    content: str | None = None,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-tool-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "test-chat-model",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


def tool_call(*, arguments: str = '{"order_id":"1001"}') -> dict[str, Any]:
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": "query_logistics", "arguments": arguments},
    }


def sse_chunk(
    content: str | None = None, *, finish_reason: str | None = None
) -> bytes:
    payload = {
        "id": "chatcmpl-stream-tool-test",
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
    return f"data: {json.dumps(payload)}\n\n".encode()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="https://upstream.example/v1",
        llm_model="test-chat-model",
        llm_api_key="test-key",
        context_window_tokens=8_192,
        max_output_tokens=256,
        token_safety_margin=128,
        request_timeout_seconds=9,
        llm_chat_extra_body={"thinking": {"type": "disabled"}},
    )


@pytest.fixture
def tools() -> list[Any]:
    context = ToolContext(
        ref=TurnRef(conversation_id="conversation-1", turn_id="turn-1"),
        user_id="demo-user",
        user_message="订单 1001 的物流到哪了",
        ticket_no="TICKET-1",
    )
    registry = build_registry(
        context,
        AsyncMock(spec=FaqRepository),
        AsyncMock(spec=TicketRepository),
    )
    return registry.tools


@pytest.fixture
def two_phase_messages() -> list[Any]:
    selected = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "type": "tool_call",
            }
        ],
    )
    return [
        HumanMessage("订单 1001 的物流到哪了"),
        selected,
        ToolMessage(
            '{"status":"ok","simulated":true}',
            tool_call_id="call-1",
            name="query_logistics",
        ),
    ]


def make_gateway(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    handler: Callable[[httpx.Request], Any],
) -> tuple[OpenAIModelGateway, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(model_module, "AsyncClient", lambda **_kwargs: client)
    return OpenAIModelGateway(settings), client


@pytest.fixture
def recorded_bodies() -> list[dict[str, Any]]:
    return []


@pytest.fixture
async def tool_gateway(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    recorded_bodies: list[dict[str, Any]],
) -> AsyncIterator[OpenAIModelGateway]:
    async def handler(request: httpx.Request) -> httpx.Response:
        recorded_bodies.append(json.loads(request.content))
        if len(recorded_bodies) == 1:
            return httpx.Response(
                200,
                json=completion(finish_reason="tool_calls", tool_calls=[tool_call()]),
            )
        chunks = [
            sse_chunk("演示"),
            sse_chunk("物流"),
            sse_chunk(finish_reason="stop"),
            b"data: [DONE]\n\n",
        ]
        return httpx.Response(200, content=b"".join(chunks))

    gateway, _client = make_gateway(monkeypatch, settings, handler)
    try:
        yield gateway
    finally:
        await gateway.aclose()


async def test_select_then_final_stream_uses_disabled_thinking(
    tool_gateway: OpenAIModelGateway,
    recorded_bodies: list[dict[str, Any]],
    tools: list[Any],
    two_phase_messages: list[Any],
) -> None:
    selected = await tool_gateway.select(two_phase_messages[:1], tools)
    streamed = [part async for part in tool_gateway.stream(two_phase_messages)]

    assert selected.tool_calls[0]["id"] == "call-1"
    assert streamed == ["演示", "物流"]
    assert len(recorded_bodies) == 2
    assert recorded_bodies[0]["tools"]
    assert recorded_bodies[0]["tool_choice"] == "auto"
    assert recorded_bodies[0]["parallel_tool_calls"] is False
    assert "tools" not in recorded_bodies[1]
    assert all(
        body["thinking"] == {"type": "disabled"} for body in recorded_bodies
    )
    assert all(body["max_completion_tokens"] == 256 for body in recorded_bodies)
    assert recorded_bodies[1]["messages"][-1]["tool_call_id"] == "call-1"


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (completion(content="半截", finish_reason="length"), "UPSTREAM_INCOMPLETE"),
        (
            completion(
                finish_reason="tool_calls",
                tool_calls=[tool_call(arguments="{not-json")],
            ),
            "UPSTREAM_INCOMPLETE",
        ),
        (
            {"error": {"message": "private upstream detail", "type": "test"}},
            "UPSTREAM_ERROR",
        ),
    ],
)
async def test_select_maps_incomplete_malformed_and_upstream_failures(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    tools: list[Any],
    response: dict[str, Any],
    expected_code: str,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        status = 500 if "error" in response else 200
        return httpx.Response(status, json=response)

    gateway, _client = make_gateway(monkeypatch, settings, handler)
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.select([HumanMessage("测试")], tools)
    finally:
        await gateway.aclose()

    assert exc_info.value.code == expected_code
    assert exc_info.value.status_code == 502
    assert "private upstream detail" not in exc_info.value.message


@pytest.mark.parametrize(
    "protected_key",
    [
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "stream_options",
        "model",
        "max_tokens",
        "max_completion_tokens",
        "response_format",
    ],
)
def test_chat_extra_body_cannot_override_protocol_controls(
    protected_key: str,
) -> None:
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            llm_base_url="https://upstream.example/v1",
            llm_model="test-chat-model",
            llm_api_key="test-key",
            llm_chat_extra_body={protected_key: "override"},
        )


def test_chat_extra_body_reads_json_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://upstream.example/v1")
    monkeypatch.setenv("LLM_MODEL", "test-chat-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv(
        "LLM_CHAT_EXTRA_BODY", '{"thinking":{"type":"disabled"}}'
    )

    settings = Settings(_env_file=None)

    assert settings.llm_chat_extra_body == {"thinking": {"type": "disabled"}}
