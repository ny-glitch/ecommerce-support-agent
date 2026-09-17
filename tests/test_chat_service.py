import asyncio
import json
import time
from dataclasses import FrozenInstanceError

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.db.contracts import StoredTurn
from app.errors import ServiceError
from tests.ch02_helpers import ChatHarness, selection


@pytest.fixture
def chat_harness():
    return ChatHarness()


async def test_one_tool_then_unbound_answer(chat_harness):
    h = chat_harness
    h.gateway.selection = selection()
    events = await h.collect("订单1001的物流到哪了")
    assert len(h.gateway.select_calls) == len(h.gateway.stream_calls) == 1
    pair = h.gateway.stream_calls[0][-2:]
    assert isinstance(pair[0], AIMessage) and isinstance(pair[1], ToolMessage)
    assert pair[1].tool_call_id == "call-1"
    names = [event.name for event in events]
    assert names[0] == "meta" and names[-1] == "done"
    assert names.index("tool_status") < names.index("token")
    assert events[0].data["turn_id"] == h.conversations.finish_calls[0][0].turn_id
    assert h.conversations.finished_status == "completed"
    assert len(h.conversations.finish_calls) == 1
    assert [op[0] for op in h.conversations.operations] == ["create", "user", "call", "result", "finish"]
    with pytest.raises(FrozenInstanceError):
        events[0].name = "other"


async def test_zero_tool_discards_first_text_and_always_requests_final(chat_harness):
    h = chat_harness
    events = await h.collect("你好，再调用工具直到满意")
    assert len(h.gateway.select_calls) == len(h.gateway.stream_calls) == 1
    assert [type(m) for m in h.gateway.stream_calls[0]][-1] is HumanMessage
    assert "discard this decision text" not in str(events)
    assert [op[0] for op in h.conversations.operations] == ["create", "user", "finish"]
    assert [e.data["content"] for e in events if e.name == "token"] == ["你好", "，这是结果"]


@pytest.mark.parametrize("kind", ["multiple", "duplicate", "empty", "invalid"])
async def test_malformed_selection_never_executes(chat_harness, kind):
    h = chat_harness
    answer = selection("create_ticket", {"issue_description": "坏了", "ticket_type": "repair"})
    if kind in {"multiple", "duplicate"}:
        answer.tool_calls.append({**answer.tool_calls[0], "id": "call-1" if kind == "duplicate" else "call-2"})
    elif kind == "empty":
        answer.tool_calls[0]["id"] = ""
    else:
        answer.invalid_tool_calls = [{"name": "create_ticket", "args": "{", "id": "bad", "error": "bad", "type": "invalid_tool_call"}]
    h.gateway.selection = answer
    events = await h.collect("帮我处理")
    assert events[-1].data["code"] == "INVALID_TOOL_CALL"
    assert not h.tickets.calls and not h.gateway.stream_calls
    assert h.conversations.finished_status == "failed"
    assert len(h.conversations.finish_calls) == 1


@pytest.mark.parametrize(("name", "args", "code"), [
    ("query_logistics", {}, "INVALID_TOOL_ARGUMENTS"),
    ("unknown", {}, "UNKNOWN_TOOL"),
    ("query_faq", {"keyword": "邮费"}, None),
])
async def test_error_or_not_found_tool_feedback_converges(chat_harness, name, args, code):
    h = chat_harness
    h.gateway.selection = selection(name, args)
    events = await h.collect("邮费是多少")
    feedback = h.gateway.stream_calls[0][-1]
    assert feedback.tool_call_id == "call-1"
    payload = json.loads(feedback.content)
    assert payload.get("code") == code
    assert payload["status"] == ("error" if code else "not_found")
    terminal = [e for e in events if e.name == "tool_status"][-1]
    assert terminal.data["status"] == ("failed" if code else "not_found")
    assert terminal.data["tool_call_id"] == "call-1"
    assert len(h.gateway.select_calls) == len(h.gateway.stream_calls) == 1
    assert events[-1].name == "done"


async def test_result_commit_precedes_terminal_and_final_model(chat_harness):
    h = chat_harness
    h.gateway.selection = selection()
    seen_terminal = False
    async with h.service.prepare("查物流", None) as p:
        async for event in h.service.stream(p):
            if event.name == "tool_status" and event.data["status"] == "succeeded":
                seen_terminal = True
                assert h.conversations.operations[-1][0] == "result"
                assert not h.gateway.stream_calls
    assert seen_terminal


async def test_result_write_failure_blocks_final_model(chat_harness):
    h = chat_harness
    h.gateway.selection = selection()
    h.conversations.result_error = RuntimeError("secret database details")
    events = await h.collect("查物流")
    assert events[-1].data["code"] == "DB_ERROR"
    assert not h.gateway.stream_calls
    assert not any(e.name == "tool_status" and e.data["status"] == "succeeded" for e in events)
    assert "secret database" not in str(events)


async def test_commit_failure_never_sends_done(chat_harness, caplog):
    h = chat_harness
    h.conversations.finish_error = RuntimeError("private database detail")
    events = await h.collect("你好")
    assert events[-1].name == "error"
    assert "done" not in [event.name for event in events]
    assert "private database detail" not in str(events) + caplog.text


async def test_final_budget_rechecked_without_request(chat_harness):
    h = chat_harness
    # Selection text is persisted alongside the tool call and must count in final input.
    h.gateway.selection = selection()
    h.gateway.selection.content = "x" * 10000
    events = await h.collect("订单1001")
    assert events[-1].data["code"] == "INPUT_TOO_LONG"
    assert not h.gateway.stream_calls
    assert h.conversations.finished_status == "failed"


async def test_preflight_rejects_before_database_writes(chat_harness):
    h = chat_harness
    with pytest.raises(ServiceError, match="上下文"):
        await h.collect("x" * 10000)
    assert not h.conversations.operations
    assert not h.gateway.select_calls


async def test_existing_owner_checked_before_start(chat_harness):
    h = chat_harness
    h.conversations.owner = None
    with pytest.raises(ServiceError) as error:
        await h.collect("你好", "unknown")
    assert error.value.code == "CONVERSATION_NOT_FOUND"
    assert [op[0] for op in h.conversations.operations] == ["get"]
    h.guard.acquire("unknown")
    h.guard.release("unknown")


async def test_completed_history_reaches_both_requests(chat_harness):
    h = chat_harness
    h.conversations.turns = [StoredTurn("old", (HumanMessage("之前"), AIMessage("回答")))]
    await h.collect("现在", "existing")
    for messages in [h.gateway.select_calls[0], h.gateway.stream_calls[0]]:
        assert [m.content for m in messages[1:]] == ["之前", "回答", "现在"]


async def test_cancel_preserves_partial_closes_upstream_and_releases_guard(chat_harness):
    h = chat_harness
    h.gateway.stream_gate = asyncio.Event()
    async with h.service.prepare("你好", None) as p:
        async def consume():
            return [event async for event in h.service.stream(p)]
        task = asyncio.create_task(consume())
        await asyncio.wait_for(h.gateway.waiting.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert h.conversations.finish_calls[0][1:] == ("你好，这是结果", "cancelled")
    assert len(h.conversations.finish_calls) == 1
    assert h.gateway.closed.is_set()
    h.guard.acquire(p.ref.conversation_id)
    h.guard.release(p.ref.conversation_id)


async def test_generator_close_and_unused_preparation_finalize_once(chat_harness):
    h = chat_harness
    async with h.service.prepare("你好", None) as p:
        stream = h.service.stream(p)
        assert (await anext(stream)).name == "meta"
        await stream.aclose()
    assert h.conversations.finished_status == "cancelled"
    assert len(h.conversations.finish_calls) == 1
    h2 = ChatHarness()
    async with h2.service.prepare("你好", None):
        pass
    assert h2.conversations.finished_status == "cancelled"


async def test_one_deadline_with_anext_driven_by_distinct_tasks(chat_harness):
    h = chat_harness
    h.gateway.stream_gate = asyncio.Event()
    async with h.service.prepare("你好", None) as p:
        p.deadline = time.monotonic() + 0.06
        stream = h.service.stream(p)
        events = []
        while True:
            try:
                events.append(await asyncio.create_task(anext(stream)))
            except StopAsyncIteration:
                break
    assert events[-1].name == "error"
    assert events[-1].data["code"] == "UPSTREAM_TIMEOUT"
    assert h.gateway.closed.is_set()
    assert h.conversations.finished_status == "failed"


async def test_expired_deadline_does_not_start_selection(chat_harness):
    h = chat_harness
    async with h.service.prepare("你好", None) as p:
        p.deadline = time.monotonic() - 1
        events = [event async for event in h.service.stream(p)]
    assert events[-1].data["code"] == "UPSTREAM_TIMEOUT"
    assert not h.gateway.select_calls


async def test_cleanup_database_wait_is_bounded_and_releases_guard(chat_harness):
    h = chat_harness
    h.conversations.finish_gate = asyncio.Event()
    started = time.monotonic()
    async with h.service.prepare("你好", None) as p:
        pass
    assert time.monotonic() - started < 2
    h.guard.acquire(p.ref.conversation_id)
    h.guard.release(p.ref.conversation_id)


async def test_cancelled_anyio_scope_still_saves_audit_and_releases_guard(chat_harness):
    import anyio
    h = chat_harness
    with anyio.CancelScope() as scope:
        async with h.service.prepare("你好", None) as p:
            scope.cancel()
            await anyio.sleep(0)
    assert h.conversations.finished_status == "cancelled"
    assert len(h.conversations.finish_calls) == 1
    h.guard.acquire(p.ref.conversation_id)
    h.guard.release(p.ref.conversation_id)


async def test_cancellation_during_selection_marks_pending_user_cancelled(chat_harness):
    h = chat_harness
    h.gateway.select_gate = asyncio.Event()
    async with h.service.prepare("你好", None) as p:
        stream = h.service.stream(p)
        await anext(stream)
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert h.conversations.finished_status == "cancelled"
    assert h.conversations.finish_calls[0][1] == ""
    assert not h.gateway.stream_calls


async def test_busy_turn_cannot_be_acquired_until_context_exits(chat_harness):
    h = chat_harness
    async with h.service.prepare("你好", "existing"):
        with pytest.raises(ServiceError) as error:
            await h.collect("重复请求", "existing")
        assert error.value.code == "SESSION_BUSY"
    assert (await h.collect("重试", "existing"))[-1].name == "done"


async def test_empty_final_stream_never_commits_completed(chat_harness):
    h = chat_harness
    h.gateway.fragments = ["", " "]
    events = await h.collect("你好")
    assert events[-1].data["code"] == "UPSTREAM_INCOMPLETE"
    assert h.conversations.finished_status == "failed"


async def test_prepare_read_uses_total_deadline_and_releases_guard(chat_harness):
    h = chat_harness
    h.settings.request_timeout_seconds = 0.04
    async def blocked_get(*args):
        await asyncio.Event().wait()
    h.conversations.get = blocked_get
    with pytest.raises(ServiceError) as error:
        await h.collect("你好", "existing")
    assert error.value.code == "UPSTREAM_TIMEOUT"
    assert not any(op[0] == "user" for op in h.conversations.operations)
    h.guard.acquire("existing")
    h.guard.release("existing")
