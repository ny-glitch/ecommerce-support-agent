from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.errors import ServiceError
from app.knowledge.gateway import (
    KnowledgeGateway,
    NormalizationOutput,
    NormalizationResponseError,
)
from app.knowledge.query import QueryNormalizer, protected_terms_preserved


class StubGateway:
    def __init__(
        self,
        result: NormalizationOutput | Exception,
    ) -> None:
        self.result = result
        self.questions: list[str] = []

    async def normalize(self, question: str) -> NormalizationOutput:
        self.questions.append(question)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _completion(
    content: str,
    *,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-normalize-test",
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


def _gateway(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    settings: Settings | None = None,
) -> tuple[KnowledgeGateway, httpx.AsyncClient]:
    if settings is None:
        settings = Settings(
            _env_file=None,
            llm_base_url="https://upstream.example/v1",
            llm_model="test-chat-model",
            llm_api_key="test-key",
            context_window_tokens=8_192,
            max_output_tokens=128,
            token_safety_margin=128,
        )
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
            settings=settings,
        ),
        client,
    )


def test_rewrite_must_keep_model_and_negation() -> None:
    assert not protected_terms_preserved(
        "C65-Pro 不支持哪些协议？", "C65 支持哪些协议？"
    )
    assert protected_terms_preserved(
        "C65-Pro 不支持啥协议？", "C65-Pro 不支持哪些协议？"
    )
    assert not protected_terms_preserved(
        "C65-Pro 支持哪些协议？", "C65-Pro 不支持哪些协议？"
    )


@pytest.mark.parametrize(
    ("original", "normalized"),
    [
        ("C2 口最高 30W 吗？", "C2 口最高功率是多少？"),
        ("R8 普通版不能抬拖布吗？", "R8 普通版支持抬拖布吗？"),
        ("订单 A-2048 未发货吗？", "订单 A-2048 发货了吗？"),
    ],
)
def test_rewrite_must_keep_alphanumeric_numbers_and_negation(
    original: str,
    normalized: str,
) -> None:
    assert not protected_terms_preserved(original, normalized)


async def test_prepare_reads_only_current_question_and_deduplicates_synonyms() -> None:
    gateway = StubGateway(
        NormalizationOutput(
            normalized="充电器高负载发热是否正常？",
            synonyms=("温升", " 温升 ", "发热"),
        )
    )

    plan = await QueryNormalizer(gateway).prepare(
        "充电头高负载时有点温升，是不是故障？",
        "数码配件/充电器",
        deadline=time.monotonic() + 1,
    )

    assert gateway.questions == ["充电头高负载时有点温升，是不是故障？"]
    assert plan.original == "充电头高负载时有点温升，是不是故障？"
    assert plan.normalized == "充电器高负载发热是否正常？"
    assert plan.synonyms == ("温升", "发热")
    assert plan.category == "数码配件/充电器"
    assert plan.fallback is False


@pytest.mark.parametrize(
    "output",
    [
        NormalizationOutput(
            normalized="C65 支持哪些协议？",
            synonyms=("协议兼容",),
        ),
        NormalizationOutput(
            normalized="C65-Pro 支持哪些协议？",
            synonyms=("协议兼容",),
        ),
        RuntimeError("upstream unavailable"),
    ],
)
async def test_prepare_falls_back_once_when_rewrite_is_unsafe_or_fails(
    output: NormalizationOutput | Exception,
) -> None:
    gateway = StubGateway(output)
    original = "C65-Pro 不支持哪些协议？"

    plan = await QueryNormalizer(gateway).prepare(
        original,
        None,
        deadline=time.monotonic() + 1,
    )

    assert gateway.questions == [original]
    assert plan.normalized == original
    assert plan.synonyms == ()
    assert plan.fallback is True


async def test_prepare_rejects_synonym_that_adds_model_or_number(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway = StubGateway(
        NormalizationOutput(
            normalized="C65-Pro 支持哪些协议？",
            synonyms=("R8 的 2.4GHz 协议",),
        )
    )

    with caplog.at_level("INFO", logger="app.knowledge.query"):
        plan = await QueryNormalizer(gateway).prepare(
            "C65-Pro 支持哪些协议？",
            None,
            deadline=time.monotonic() + 1,
        )

    assert plan.normalized == "C65-Pro 支持哪些协议？"
    assert plan.synonyms == ()
    assert plan.fallback is True
    assert caplog.records[-1].fallback_reason == "protected_terms"


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        (NormalizationResponseError("invalid response"), "invalid_response"),
        (
            ServiceError("INPUT_TOO_LONG", "输入内容超过可处理的上下文长度", 413),
            "budget",
        ),
        (RuntimeError("transport failed"), "gateway_error"),
    ],
)
async def test_prepare_records_distinct_fallback_reasons(
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
    expected_reason: str,
) -> None:
    gateway = StubGateway(failure)

    with caplog.at_level("INFO", logger="app.knowledge.query"):
        plan = await QueryNormalizer(gateway).prepare(
            "充电器保修多久？",
            None,
            deadline=time.monotonic() + 1,
        )

    assert plan.fallback is True
    records = [record for record in caplog.records if record.message == "query normalization fallback"]
    assert len(records) == 1
    assert records[0].fallback_reason == expected_reason


async def test_gateway_binds_protected_transport_controls_and_sends_no_history() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_completion(
                '{"normalized":"C65-Pro 支持 PD 3.0 吗？",'
                '"synonyms":["PD3.0 兼容性"]}'
            ),
        )

    gateway, client = _gateway(handler)
    try:
        result = await gateway.normalize("C65-Pro 能用 PD 3.0 不？")
    finally:
        await client.aclose()

    assert result == NormalizationOutput(
        normalized="C65-Pro 支持 PD 3.0 吗？",
        synonyms=("PD3.0 兼容性",),
    )
    assert len(requests) == 1
    body = requests[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_completion_tokens"] == 128
    assert "tools" not in body
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "C65-Pro 能用 PD 3.0 不？"


async def test_gateway_rejects_incomplete_raw_json_even_when_shape_is_valid() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion(
                '{"normalized":"C65-Pro 支持 PD 3.0 吗？","synonyms":[]}',
                finish_reason="length",
            ),
        )

    gateway, client = _gateway(handler)
    try:
        with pytest.raises(ValueError, match="structured normalization"):
            await gateway.normalize("C65-Pro 能用 PD 3.0 不？")
    finally:
        await client.aclose()


async def test_gateway_rejects_over_budget_input_before_transport() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    settings = Settings(
        _env_file=None,
        llm_base_url="https://upstream.example/v1",
        llm_model="test-chat-model",
        llm_api_key="test-key",
        context_window_tokens=3_000,
        max_output_tokens=500,
        token_safety_margin=100,
    )
    gateway, client = _gateway(handler, settings=settings)
    try:
        with pytest.raises(ServiceError) as exc_info:
            await gateway.normalize("超长问题" * 300)
    finally:
        await client.aclose()

    assert exc_info.value.code == "INPUT_TOO_LONG"
    assert requests == []
