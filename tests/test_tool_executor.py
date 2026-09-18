from __future__ import annotations

import asyncio
import json
import time

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import IntegrityError, OperationalError

from app.db.contracts import TurnRef
from app.tools.business import ToolContext, build_registry
from app.tools.executor import ToolExecutor, ToolOutcome, ToolProgress
from app.tools.registry import ToolRegistry
from app.tools.results import TransientToolError


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequiredArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


async def collect(executor: ToolExecutor, business_tool, *, deadline: float = 1e30):
    registry = ToolRegistry([business_tool])
    call = {
        "name": business_tool.name,
        "args": {},
        "id": "call-17",
        "type": "tool_call",
    }
    return [event async for event in executor.run(call, registry, deadline=deadline)]


async def test_unknown_tool_returns_correlated_error() -> None:
    ctx = ToolContext(TurnRef("c1", "t1"), "demo", "你好", "TK-test")
    registry = build_registry(ctx, faq=object(), tickets=object())
    call = {"name": "run_shell", "args": {}, "id": "c-unknown", "type": "tool_call"}

    events = [e async for e in ToolExecutor().run(call, registry, deadline=1e30)]

    result = events[-1]
    assert isinstance(result, ToolOutcome)
    assert result.message.tool_call_id == "c-unknown"
    assert json.loads(result.message.content)["code"] == "UNKNOWN_TOOL"


async def test_transient_error_retries_once_then_succeeds() -> None:
    calls = 0

    @tool(args_schema=NoArgs)
    async def flaky() -> str:
        """Fail transiently once."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientToolError("retry me")
        return json.dumps({"status": "ok", "data": {"call": calls}})

    events = await collect(ToolExecutor(max_attempts=2), flaky)

    assert calls == 2
    assert [(event.status, event.attempt) for event in events[:-1]] == [
        ("running", 1),
        ("retrying", 2),
    ]
    assert isinstance(events[-1], ToolOutcome)
    assert events[-1].attempt == 2
    assert events[-1].terminal_status == "succeeded"
    assert events[-1].message.tool_call_id == "call-17"


async def test_timeout_stops_after_configured_attempts() -> None:
    calls = 0

    @tool(args_schema=NoArgs)
    async def slow() -> str:
        """Never finish within the attempt timeout."""
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return json.dumps({"status": "ok"})

    events = await collect(ToolExecutor(timeout_seconds=0.005, max_attempts=2), slow)

    outcome = events[-1]
    assert calls == 2
    assert isinstance(outcome, ToolOutcome)
    assert outcome.attempt == 2
    assert outcome.terminal_status == "failed"
    assert json.loads(outcome.message.content)["code"] == "TOOL_TIMEOUT"


async def test_timeout_outcome_waits_for_invocation_settlement() -> None:
    cleanup_allowed = asyncio.Event()
    invocation_settled = asyncio.Event()

    @tool(args_schema=NoArgs)
    async def slow_cleanup() -> str:
        """Expose cancellation cleanup ordering."""
        try:
            await asyncio.Event().wait()
        finally:
            await cleanup_allowed.wait()
            invocation_settled.set()

    generator = ToolExecutor(timeout_seconds=0.01, max_attempts=1).run(
        {
            "name": "slow_cleanup",
            "args": {},
            "id": "settlement-1",
            "type": "tool_call",
        },
        ToolRegistry([slow_cleanup]),
        deadline=asyncio.get_running_loop().time() + 1,
    )
    assert isinstance(await anext(generator), ToolProgress)
    outcome_read = asyncio.create_task(anext(generator))
    try:
        await asyncio.sleep(0.03)
        assert not outcome_read.done()
        assert not invocation_settled.is_set()

        cleanup_allowed.set()
        outcome = await asyncio.wait_for(outcome_read, 0.5)
        assert invocation_settled.is_set()
        assert isinstance(outcome, ToolOutcome)
        assert json.loads(outcome.message.content)["code"] == "TOOL_TIMEOUT"
    finally:
        cleanup_allowed.set()
        await asyncio.gather(outcome_read, return_exceptions=True)
        await generator.aclose()


async def test_validation_error_is_not_retried() -> None:
    calls = 0

    @tool(args_schema=RequiredArgs)
    async def requires_value(value: str) -> str:
        """Require one string argument."""
        nonlocal calls
        calls += 1
        return json.dumps({"status": "ok"})

    events = await collect(ToolExecutor(max_attempts=2), requires_value)

    outcome = events[-1]
    assert calls == 0
    assert isinstance(outcome, ToolOutcome)
    assert outcome.attempt == 1
    assert json.loads(outcome.message.content)["code"] == "INVALID_TOOL_ARGUMENTS"


async def test_not_found_is_successful_execution_without_retry() -> None:
    calls = 0

    @tool(args_schema=NoArgs)
    async def missing() -> str:
        """Return a normal business miss."""
        nonlocal calls
        calls += 1
        return json.dumps({"status": "not_found", "data": []})

    events = await collect(ToolExecutor(max_attempts=2), missing)

    outcome = events[-1]
    assert calls == 1
    assert isinstance(outcome, ToolOutcome)
    assert outcome.terminal_status == "not_found"
    assert outcome.attempt == 1


async def test_cancelled_error_propagates_immediately() -> None:
    calls = 0

    @tool(args_schema=NoArgs)
    async def cancelled() -> str:
        """Propagate caller cancellation."""
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await collect(ToolExecutor(max_attempts=2), cancelled)

    assert calls == 1


async def test_expired_total_deadline_does_not_start_second_attempt() -> None:
    calls = 0

    @tool(args_schema=NoArgs)
    async def beyond_deadline() -> str:
        """Run until the total deadline expires."""
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return json.dumps({"status": "ok"})

    events = await collect(
        ToolExecutor(timeout_seconds=1, max_attempts=2),
        beyond_deadline,
        deadline=time.monotonic() + 0.01,
    )

    assert calls == 1
    assert isinstance(events[-1], ToolOutcome)
    assert json.loads(events[-1].message.content)["code"] == "TOOL_DEADLINE_EXCEEDED"


async def test_each_generator_step_can_be_driven_by_a_different_task() -> None:
    @tool(args_schema=NoArgs)
    async def simple() -> str:
        """Return immediately."""
        return json.dumps({"status": "ok"})

    generator = ToolExecutor().run(
        {"name": "simple", "args": {}, "id": "cross-task", "type": "tool_call"},
        ToolRegistry([simple]),
        deadline=1e30,
    )

    first = await asyncio.create_task(anext(generator))
    second = await asyncio.create_task(anext(generator))

    assert isinstance(first, ToolProgress)
    assert isinstance(second, ToolOutcome)
    with pytest.raises(StopAsyncIteration):
        await asyncio.create_task(anext(generator))


async def test_executor_bounds_even_nonconforming_tool_output() -> None:
    @tool(args_schema=NoArgs)
    async def oversized() -> str:
        """Return an oversized but structured payload."""
        return json.dumps(
            {
                "status": "ok",
                "order_id": "ORDER_17",
                "description": "大" * 10_000,
            },
            ensure_ascii=False,
        )

    outcome = (await collect(ToolExecutor(), oversized))[-1]

    assert isinstance(outcome, ToolOutcome)
    content = outcome.message.content
    payload = json.loads(content)
    assert len(content.encode("utf-8")) <= 4096
    assert payload["status"] == "ok"
    assert payload["order_id"] == "ORDER_17"
    assert payload["truncated"] is True


async def test_irreducible_success_payload_becomes_failed_outcome() -> None:
    @tool(args_schema=NoArgs)
    async def irreducible() -> str:
        """Return an identifier too large to preserve within the output limit."""
        return json.dumps({"status": "ok", "order_id": "x" * 5_000})

    outcome = (await collect(ToolExecutor(), irreducible))[-1]

    assert isinstance(outcome, ToolOutcome)
    assert json.loads(outcome.message.content) == {
        "status": "error",
        "code": "TOOL_RESULT_TOO_LARGE",
        "truncated": True,
    }
    assert outcome.terminal_status == "failed"
    assert outcome.message.status == "error"


@pytest.mark.parametrize(
    ("exception", "expected_calls"),
    [
        (
            OperationalError("statement", {}, Exception(1213, "deadlock")),
            2,
        ),
        (
            IntegrityError("statement", {}, Exception(1062, "duplicate")),
            1,
        ),
    ],
)
async def test_database_errors_retry_only_when_explicitly_transient(
    exception: Exception, expected_calls: int
) -> None:
    calls = 0

    @tool(args_schema=NoArgs)
    async def database_call() -> str:
        """Expose a classified database failure."""
        nonlocal calls
        calls += 1
        raise exception

    outcome = (await collect(ToolExecutor(max_attempts=2), database_call))[-1]

    assert calls == expected_calls
    assert isinstance(outcome, ToolOutcome)
    payload = json.loads(outcome.message.content)
    assert payload["status"] == "error"
    assert "deadlock" not in outcome.message.content
    assert "duplicate" not in outcome.message.content
