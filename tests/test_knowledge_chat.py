from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import time

import pytest
from langchain_core.messages import AIMessage

from app.errors import ServiceError
from app.services.chat import ChatService
from app.sessions import SessionGuard
from app.tools.executor import ToolExecutor
from tests.ch02_helpers import (
    ChatGateway,
    EmptyFaq,
    RecordingConversations,
    RecordingTickets,
    selection,
)
from tests.ch04_helpers import FakeKnowledgePipeline, RecordingLowConfidence, make_decision
from tests.helpers import settings


def service_case(decision=None, *, trace=None, configuration=None):
    gateway = ChatGateway()
    gateway.selection = selection("query_faq", {})
    gateway.fragments = ["C65-Pro支持PD 3.0[1]"]
    conversations = RecordingConversations()
    if trace is not None:
        real_append = conversations.append_result
        real_finish = conversations.finish_turn

        async def append_result(*args):
            await real_append(*args)
            trace.append("tool_saved")

        async def finish_turn(*args):
            await real_finish(*args)
            trace.append("turn_finished")

        conversations.append_result = append_result
        conversations.finish_turn = finish_turn
        real_stream = gateway.stream

        async def stream(messages):
            trace.append("model_stream")
            async for value in real_stream(messages):
                yield value

        gateway.stream = stream
    pipeline = FakeKnowledgePipeline(decision or make_decision())
    pool = RecordingLowConfidence(trace)
    service = ChatService(
        configuration or settings(), gateway, conversations, EmptyFaq(), RecordingTickets(),
        SessionGuard(100), ToolExecutor(), knowledge_pipeline=pipeline,
        low_confidence=pool,
    )
    return service, gateway, conversations, pipeline, pool


@asynccontextmanager
async def prepared_case(decision=None, *, trace=None, category="数码配件"):
    service, gateway, conversations, pipeline, pool = service_case(decision, trace=trace)
    async with service.prepare("C65-Pro支持什么协议？", None, category=category) as prepared:
        yield service, prepared, gateway, conversations, pipeline, pool


async def test_supported_knowledge_sequence_persists_before_sources_and_done() -> None:
    trace: list[str] = []
    async with prepared_case(trace=trace) as case:
        service, prepared, gateway, conversations, pipeline, pool = case
        events = [event async for event in service.stream(prepared)]

    names = [event.name for event in events]
    assert names[:2] == ["meta", "tool_status"]
    assert names[2:6] == ["retrieval_status"] * 4
    assert names[6:9] == ["tool_status", "sources", "token"]
    assert names[-1] == "done"
    assert trace.index("tool_saved") < trace.index("model_stream") < trace.index("turn_finished")
    assert pipeline.calls[0][0:2] == ("C65-Pro支持什么协议？", "数码配件")
    assert events[-1].data["refused"] is False
    assert events[-1].data["citations"] == [1]
    assert json.loads(conversations.operations[-2][2].content)["sources"][0]["number"] == 1
    assert not pool.calls


async def test_refusal_is_persisted_before_visible() -> None:
    trace: list[str] = []
    async with prepared_case(make_decision(status="not_found"), trace=trace) as case:
        service, prepared, gateway, conversations, pipeline, pool = case
        events = [event async for event in service.stream(prepared)]

    assert "pool_saved" in trace
    assert trace.index("pool_saved") < trace.index("tool_saved")
    assert [event.name for event in events].count("refusal") == 1
    assert not any(event.name == "token" for event in events)
    assert events[-1].name == "done"
    assert events[-1].data["refused"] is True
    assert events[-1].data["citations"] == []
    assert trace.count("model_stream") == 0
    assert conversations.finish_calls[-1][1] == make_decision(status="not_found").refusal


async def test_pool_write_failure_emits_only_error_after_running_progress() -> None:
    async with prepared_case(make_decision(status="not_found")) as case:
        service, prepared, gateway, conversations, pipeline, pool = case
        pool.error = ServiceError("DATABASE_ERROR", "数据库操作失败", 503)
        events = [event async for event in service.stream(prepared)]
    assert events[-1].name == "error"
    assert events[-1].data["code"] == "DATABASE_ERROR"
    assert not any(event.name in {"refusal", "done"} for event in events)
    assert not gateway.stream_calls


@pytest.mark.parametrize("answer", ["错误引用[2]", "没有引用"])
async def test_invalid_final_citation_marks_turn_failed(answer: str) -> None:
    async with prepared_case() as case:
        service, prepared, gateway, conversations, pipeline, pool = case
        gateway.fragments = [answer]
        events = [event async for event in service.stream(prepared)]
    assert events[-1].name == "error"
    assert events[-1].data["code"] == "INVALID_CITATION"
    assert not any(event.name == "done" for event in events)
    assert conversations.finished_status == "failed"


async def test_pipeline_error_is_not_recorded_or_explained_by_model() -> None:
    async with prepared_case() as case:
        service, prepared, gateway, conversations, pipeline, pool = case
        pipeline.error = ServiceError("KNOWLEDGE_UNAVAILABLE", "知识服务暂时不可用", 503)
        events = [event async for event in service.stream(prepared)]
    assert events[-1].name == "error"
    assert events[-1].data["code"] == "KNOWLEDGE_UNAVAILABLE"
    assert not pool.calls
    assert not gateway.stream_calls


async def test_cancel_during_pipeline_does_not_append_late_result() -> None:
    async with prepared_case() as case:
        service, prepared, gateway, conversations, pipeline, pool = case
        pipeline.gate = asyncio.Event()
        stream = service.stream(prepared)
        while True:
            event = await anext(stream)
            if event.name == "retrieval_status" and event.data["stage"] == "checking_evidence":
                break
        pending = asyncio.create_task(anext(stream))
        await pipeline.waiting.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        pipeline.gate.set()
    assert not any(op[0] == "result" for op in conversations.operations)
    assert conversations.finished_status == "cancelled"


async def test_knowledge_selection_extends_deadline_from_prepare_start_once() -> None:
    configuration = settings(request_timeout_seconds=1, knowledge_request_timeout_seconds=7)
    service, gateway, conversations, pipeline, pool = service_case(configuration=configuration)
    async with service.prepare("C65-Pro支持什么协议？", None) as prepared:
        original = prepared.deadline
        events = [event async for event in service.stream(prepared)]
        assert prepared.deadline == pytest.approx(prepared.started_at + 7)
    assert prepared.deadline > original
    assert events[-1].name == "done"


async def test_knowledge_deadline_covers_call_persistence() -> None:
    configuration = settings(
        request_timeout_seconds=1,
        knowledge_request_timeout_seconds=3,
    )
    service, gateway, conversations, pipeline, pool = service_case(
        configuration=configuration
    )
    real_append = conversations.append_call

    async def delayed_append(*args):
        await asyncio.sleep(0.04)
        await real_append(*args)

    conversations.append_call = delayed_append
    async with service.prepare("C65-Pro支持什么协议？", None) as prepared:
        prepared.started_at = asyncio.get_running_loop().time() - 0.98
        prepared.deadline = prepared.started_at + 1
        events = [event async for event in service.stream(prepared)]

    assert events[-1].name == "done"
    assert prepared.deadline == pytest.approx(prepared.started_at + 3)
    assert len(pipeline.calls) == 1


async def test_knowledge_call_uses_original_question_not_model_arguments() -> None:
    service, gateway, conversations, pipeline, pool = service_case()
    gateway.selection = AIMessage(content="", tool_calls=[{
        "name": "query_faq", "args": {}, "id": "knowledge-1", "type": "tool_call"
    }])
    async with service.prepare("原始售后政策问题", None, category="售后") as prepared:
        await asyncio.sleep(0)
        [event async for event in service.stream(prepared)]
    assert pipeline.calls[0][0:2] == ("原始售后政策问题", "售后")


async def test_transport_reads_the_live_extended_deadline() -> None:
    from app.api.streaming import stream_events
    from app.services.events import ChatEvent

    class ConnectedRequest:
        async def is_disconnected(self):
            return False

    deadline = [asyncio.get_running_loop().time() + 0.02]

    async def upstream():
        yield ChatEvent("meta", {})
        await asyncio.sleep(0.04)
        yield ChatEvent("done", {})

    events = stream_events(
        ConnectedRequest(), upstream(), deadline=lambda: deadline[0]
    )
    assert (await anext(events)).name == "meta"
    deadline[0] = asyncio.get_running_loop().time() + 1
    assert (await anext(events)).name == "done"
