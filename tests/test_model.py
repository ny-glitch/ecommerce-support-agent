from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

import app.model as model_module
from app.config import Settings
from app.errors import ServiceError
from app.model import OpenAIModelGateway
from app.prompts import customer_system_prompt, extraction_system_prompt
from app.schemas import AfterSalesResult, ChatRequest, ExtractRequest


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name, raising=False)
        monkeypatch.delenv(field_name.upper(), raising=False)

    return Settings(
        _env_file=None,
        llm_base_url="https://upstream.example/v1",
        llm_model="test-chat-model",
        llm_api_key="test-key",
        context_window_tokens=8_192,
        max_output_tokens=512,
        token_safety_margin=128,
        request_timeout_seconds=9,
    )


def completion(content: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
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
        "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
    }


def sse_chunk(
    content: str | None = None, *, finish_reason: str | None = None
) -> bytes:
    payload = {
        "id": "chatcmpl-stream-test",
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


class ControlledStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], pause_after: int | None = None) -> None:
        self.chunks = chunks
        self.pause_after = pause_after
        self.resume = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            for index, chunk in enumerate(self.chunks):
                yield chunk
                if self.pause_after == index:
                    await self.resume.wait()
        finally:
            self.closed.set()

    async def aclose(self) -> None:
        self.closed.set()


def make_gateway(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    handler: Callable[[httpx.Request], Any],
) -> tuple[OpenAIModelGateway, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(model_module, "AsyncClient", lambda **_kwargs: client)
    return OpenAIModelGateway(settings), client


def test_request_models_trim_text_and_accept_session_uuid() -> None:
    session_id = UUID("12345678-1234-5678-1234-567812345678")

    chat = ChatRequest(message="  我的订单在哪？  ", session_id=session_id)
    extraction = ExtractRequest(description="  包裹破损  ")

    assert chat.message == "我的订单在哪？"
    assert chat.session_id == session_id
    assert extraction.description == "包裹破损"


@pytest.mark.parametrize("model_type, field", [(ChatRequest, "message"), (ExtractRequest, "description")])
@pytest.mark.parametrize("value", ["", " \n\t ", "字" * 32_001, 123])
def test_request_models_reject_invalid_text(
    model_type: type[ChatRequest] | type[ExtractRequest], field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        model_type.model_validate({field: value})


def test_after_sales_result_requires_exact_strict_fields() -> None:
    result = AfterSalesResult(
        order_id="A123",
        request_type="exchange",
        expected_resolution="换一个新的",
    )

    assert result.model_dump() == {
        "order_id": "A123",
        "request_type": "exchange",
        "expected_resolution": "换一个新的",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"order_id": "A123", "request_type": "exchange"},
        {
            "order_id": "A123",
            "request_type": "exchange",
            "expected_resolution": None,
            "unexpected": "value",
        },
        {"order_id": 123, "request_type": "exchange", "expected_resolution": None},
        {"order_id": "A123", "request_type": "replace", "expected_resolution": None},
        {"order_id": " \t ", "request_type": "unknown", "expected_resolution": None},
        {"order_id": None, "request_type": "unknown", "expected_resolution": " \n "},
    ],
)
def test_after_sales_result_rejects_missing_extra_coerced_or_blank_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AfterSalesResult.model_validate(payload)


def test_prompt_resources_render_outside_project_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    customer_prompt = customer_system_prompt()
    extraction_prompt = extraction_system_prompt()

    assert customer_prompt.strip()
    assert extraction_prompt.strip()
    assert "{schema_json}" not in extraction_prompt
    assert '"request_type"' in extraction_prompt


async def test_gateway_disables_responses_usage_and_retries(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            500,
            json={"error": {"message": "private upstream detail", "type": "test"}},
        )

    gateway, _client = make_gateway(monkeypatch, settings, handler)
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.extract("订单有售后问题")
    finally:
        await gateway.aclose()

    assert attempts == 1
    assert gateway._model.use_responses_api is False
    assert gateway._model.stream_usage is False
    assert exc_info.value.code == "UPSTREAM_ERROR"
    assert "private upstream detail" not in exc_info.value.message


@pytest.mark.parametrize("token_field", ["max_tokens", "max_completion_tokens"])
async def test_extract_uses_chat_completions_json_mode_and_selected_token_field(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, token_field: str
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=completion(
                json.dumps(
                    {
                        "order_id": "A123",
                        "request_type": "exchange",
                        "expected_resolution": "换一个新的",
                    },
                    ensure_ascii=False,
                )
            ),
        )

    selected = settings.model_copy(update={"llm_token_limit_param": token_field})
    gateway, client = make_gateway(monkeypatch, selected, handler)
    try:
        result = await gateway.extract("订单 A123 到货破损，希望换一个新的")
    finally:
        await gateway.aclose()

    assert result.model_dump() == {
        "order_id": "A123",
        "request_type": "exchange",
        "expected_resolution": "换一个新的",
    }
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body[token_field] == 512
    competing_field = (
        "max_completion_tokens" if token_field == "max_tokens" else "max_tokens"
    )
    assert competing_field not in body
    assert body["response_format"] == {"type": "json_object"}
    assert "tools" not in body
    assert "tool_choice" not in body
    assert body["messages"][-1] == {
        "role": "user",
        "content": "订单 A123 到货破损，希望换一个新的",
    }
    assert client.is_closed


async def test_extract_keeps_user_braces_in_a_human_message(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    body: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json=completion(
                '{"order_id":null,"request_type":"unknown",'
                '"expected_resolution":null}'
            ),
        )

    gateway, _client = make_gateway(monkeypatch, settings, handler)
    try:
        await gateway.extract("保留这些字符：{schema_json}")
    finally:
        await gateway.aclose()

    assert body["messages"][-1] == {
        "role": "user",
        "content": "保留这些字符：{schema_json}",
    }


@pytest.mark.parametrize(
    "raw_content, finish_reason",
    [
        (
            '{"order_id":"A123","request_type":"exchange",'
            '"expected_resolution":"换货"',
            "stop",
        ),
        ('{"order_id":"A123","request_type":"exchange"}', "stop"),
        (
            '{"order_id":"A123","request_type":"replace",'
            '"expected_resolution":null}',
            "stop",
        ),
        (
            '{"order_id":"A123","request_type":"exchange",'
            '"expected_resolution":null,"extra":true}',
            "stop",
        ),
        (
            '{"order_id":"A123","request_type":"exchange",'
            '"expected_resolution":null}',
            "length",
        ),
        (
            '{"order_id":"A123","request_type":"exchange",'
            '"expected_resolution":null}',
            "content_filter",
        ),
        ("", "stop"),
    ],
)
async def test_extract_rejects_malformed_schema_or_incomplete_completion(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    raw_content: str,
    finish_reason: str,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=completion(raw_content, finish_reason=finish_reason)
        )

    gateway, _client = make_gateway(monkeypatch, settings, handler)
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.extract("订单有售后问题")
    finally:
        await gateway.aclose()

    assert exc_info.value.code == "STRUCTURED_OUTPUT_ERROR"
    assert exc_info.value.status_code == 502
    if raw_content:
        assert raw_content not in exc_info.value.message


async def test_extract_checks_input_budget_before_calling_upstream(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    called = False

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    constrained = settings.model_copy(
        update={
            "context_window_tokens": 700,
            "max_output_tokens": 500,
            "token_safety_margin": 100,
        }
    )
    gateway, _client = make_gateway(monkeypatch, constrained, handler)
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.extract("太长" * 100)
    finally:
        await gateway.aclose()

    assert exc_info.value.code == "INPUT_TOO_LONG"
    assert called is False


async def test_stream_yields_first_text_before_upstream_finishes_and_skips_empty(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    requests: list[httpx.Request] = []
    stream = ControlledStream(
        [
            sse_chunk("你"),
            sse_chunk(""),
            sse_chunk("好"),
            sse_chunk(finish_reason="stop"),
            b"data: [DONE]\n\n",
        ],
        pause_after=0,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, stream=stream)

    gateway, _client = make_gateway(monkeypatch, settings, handler)
    output = gateway.stream([SystemMessage("客服"), HumanMessage("你好")])
    try:
        first = await asyncio.wait_for(anext(output), timeout=1)
        assert first == "你"
        assert stream.closed.is_set() is False

        stream.resume.set()
        rest = [chunk async for chunk in output]
    finally:
        await output.aclose()
        await gateway.aclose()

    assert rest == ["好"]
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert body["stream"] is True
    assert body["max_completion_tokens"] == 512
    assert "stream_options" not in body
    assert "tools" not in body
    assert "tool_choice" not in body
    assert stream.closed.is_set()


@pytest.mark.parametrize(
    "chunks",
    [
        [sse_chunk(finish_reason="stop"), b"data: [DONE]\n\n"],
        [sse_chunk("半段"), sse_chunk(finish_reason="length"), b"data: [DONE]\n\n"],
    ],
)
async def test_stream_rejects_empty_output_or_non_stop_finish(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, chunks: list[bytes]
) -> None:
    stream = ControlledStream(chunks)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    gateway, _client = make_gateway(monkeypatch, settings, handler)
    output = gateway.stream([HumanMessage("测试")])
    try:
        with pytest.raises(ServiceError) as exc_info:
            _received = [chunk async for chunk in output]
    finally:
        await output.aclose()
        await gateway.aclose()

    assert exc_info.value.code == "UPSTREAM_INCOMPLETE"
    assert stream.closed.is_set()


async def test_closing_stream_closes_paused_upstream_response(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    stream = ControlledStream(
        [sse_chunk("首段"), sse_chunk("不会读取")], pause_after=0
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    gateway, _client = make_gateway(monkeypatch, settings, handler)
    output = gateway.stream([HumanMessage("测试取消")])
    assert await anext(output) == "首段"

    await output.aclose()
    await gateway.aclose()

    await asyncio.wait_for(stream.closed.wait(), timeout=1)
