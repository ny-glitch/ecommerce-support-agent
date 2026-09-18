from __future__ import annotations

import asyncio
import json

import pytest

from app.db.contracts import TurnRef
from app.tools.business import ToolContext, build_registry
from app.tools.executor import RetrievalProgress, ToolExecutor, ToolOutcome
from tests.ch04_helpers import make_decision


class Unused:
    pass


def context() -> ToolContext:
    return ToolContext(TurnRef("conversation-1", "turn-1"), "demo", "原始问题", "TK-1")


async def collect(registry, args=None, *, queue=None):
    return [
        event
        async for event in ToolExecutor(timeout_seconds=0.01, max_attempts=2).run(
            {
                "name": "query_faq",
                "args": {} if args is None else args,
                "id": "knowledge-1",
                "type": "tool_call",
            },
            registry,
            deadline=asyncio.get_running_loop().time() + 1,
            progress_queue=queue,
        )
    ]


async def test_knowledge_tool_has_empty_strict_input_and_requires_dependency() -> None:
    registry = build_registry(context(), Unused(), Unused())
    schema = next(
        item["function"]["parameters"]
        for item in registry.schemas()
        if item["function"]["name"] == "query_faq"
    )
    assert schema.get("properties") == {}
    assert schema.get("additionalProperties") is False
    policy = registry.policy("query_faq")
    assert (policy.max_bytes, policy.max_attempts, policy.shared_deadline) == (
        48_000,
        1,
        True,
    )

    outcome = (await collect(registry))[-1]
    assert isinstance(outcome, ToolOutcome)
    assert json.loads(outcome.message.content) == {
        "status": "error",
        "code": "KNOWLEDGE_UNAVAILABLE",
    }


async def test_knowledge_tool_rejects_legacy_keyword_without_invocation() -> None:
    calls = 0

    async def knowledge_call() -> str:
        nonlocal calls
        calls += 1
        return json.dumps(make_decision().to_payload(), ensure_ascii=False)

    outcome = (await collect(
        build_registry(context(), Unused(), Unused(), knowledge_call=knowledge_call),
        {"keyword": "退货"},
    ))[-1]
    assert calls == 0
    assert json.loads(outcome.message.content)["code"] == "INVALID_TOOL_ARGUMENTS"


async def test_knowledge_payload_over_4kb_is_preserved_and_not_retried() -> None:
    calls = 0
    decision = make_decision(answer_size=3000)
    expected = json.dumps(decision.to_payload(), ensure_ascii=False, separators=(",", ":"))

    async def knowledge_call() -> str:
        nonlocal calls
        calls += 1
        return expected

    outcome = (await collect(
        build_registry(context(), Unused(), Unused(), knowledge_call=knowledge_call)
    ))[-1]
    assert calls == 1
    assert isinstance(outcome, ToolOutcome)
    assert len(expected.encode("utf-8")) > 4096
    assert outcome.message.content == expected
    assert outcome.terminal_status == "succeeded"


@pytest.mark.parametrize(
    ("section", "field", "invalid"),
    [
        ("query", "original", 123),
        ("query", "normalized", []),
        ("query", "synonyms", 42),
        ("query", "category", False),
        ("root", "reason_code", {"bad": "type"}),
        ("root", "refusal", ["bad type"]),
    ],
)
async def test_knowledge_payload_rejects_non_contract_field_types(
    section: str, field: str, invalid: object
) -> None:
    payload = make_decision(status="not_found").to_payload()
    target = payload["query"] if section == "query" else payload
    target[field] = invalid

    async def knowledge_call() -> str:
        return json.dumps(payload, ensure_ascii=False)

    outcome = (await collect(
        build_registry(context(), Unused(), Unused(), knowledge_call=knowledge_call)
    ))[-1]
    assert isinstance(outcome, ToolOutcome)
    assert outcome.terminal_status == "failed"
    assert json.loads(outcome.message.content)["code"] == "INVALID_TOOL_RESULT"


async def test_progress_queue_is_forwarded_while_knowledge_invocation_runs() -> None:
    queue: asyncio.Queue[RetrievalProgress] = asyncio.Queue(maxsize=4)
    release = asyncio.Event()

    async def knowledge_call() -> str:
        await queue.put(RetrievalProgress("knowledge-1", "retrieving", "正在检索知识"))
        await release.wait()
        return json.dumps(make_decision().to_payload(), ensure_ascii=False)

    generator = ToolExecutor().run(
        {"name": "query_faq", "args": {}, "id": "knowledge-1", "type": "tool_call"},
        build_registry(context(), Unused(), Unused(), knowledge_call=knowledge_call),
        deadline=asyncio.get_running_loop().time() + 1,
        progress_queue=queue,
    )
    assert (await anext(generator)).status == "running"
    progress = await anext(generator)
    assert progress == RetrievalProgress("knowledge-1", "retrieving", "正在检索知识")
    release.set()
    assert isinstance(await anext(generator), ToolOutcome)


async def test_ordinary_tool_still_uses_4kb_limit() -> None:
    registry = build_registry(context(), Unused(), Unused())
    outcome = [
        event
        async for event in ToolExecutor().run(
            {
                "name": "query_product",
                "args": {"keyword": "商品" * 30},
                "id": "product-1",
                "type": "tool_call",
            },
            registry,
            deadline=1e30,
        )
    ][-1]
    assert isinstance(outcome, ToolOutcome)
    assert len(outcome.message.content.encode("utf-8")) <= 4096
