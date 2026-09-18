"""Native socket SSE regression tests using the real service and executor."""
import asyncio
from contextlib import asynccontextmanager

import pytest

from helpers import GatedGateway, http_app, parse_sse, settings
from tests.ch02_helpers import selection
from tests.test_streaming import next_event, running_server


class ToolHttpHarness:
    def __init__(self, phase=None, blocked_close=False, timeout=5):
        self.resume_select = asyncio.Event()
        self.select_waiting = asyncio.Event()
        self.select_closed = asyncio.Event()
        self.resume_tool = asyncio.Event()
        self.tool_waiting = asyncio.Event()
        self.tool_finished = asyncio.Event()
        self.tool_closed = asyncio.Event()
        self.resume_close = asyncio.Event()
        self.close_waiting = asyncio.Event()
        owner = self

        class Gateway(GatedGateway):
            async def select(self, messages, tools):
                owner.select_waiting.set()
                try:
                    await owner.resume_select.wait()
                    return selection("create_ticket", {
                        "issue_description": "商品损坏",
                        "ticket_type": "repair",
                    })
                finally:
                    owner.select_closed.set()

            async def stream(self, messages):
                try:
                    async for fragment in super().stream(messages):
                        yield fragment
                finally:
                    if blocked_close:
                        owner.close_waiting.set()
                        await owner.resume_close.wait()

        class Tickets:
            async def create_once(self, *args):
                owner.tool_waiting.set()
                try:
                    await owner.resume_tool.wait()
                    owner.tool_finished.set()
                    return {"ticket_no": args[0], "status": "pending"}
                finally:
                    owner.tool_closed.set()

        self.gateway = Gateway()
        self.app, _, self.conversations, self.service = http_app(
            settings(max_sessions=1, request_timeout_seconds=timeout), self.gateway
        )
        self.service.tickets = Tickets()
        if phase != "select":
            self.resume_select.set()
        if phase == "final":
            self.resume_tool.set()

    def release(self):
        self.resume_select.set()
        self.resume_tool.set()
        self.gateway.resume.set()
        self.resume_close.set()

    @asynccontextmanager
    async def running_server(self, asgi_spec=None):
        try:
            async with running_server(self.app, asgi_spec) as client:
                try:
                    yield client
                finally:
                    self.release()
        finally:
            self.release()


@pytest.fixture
async def tool_http_harness():
    harness = ToolHttpHarness()
    try:
        yield harness
    finally:
        harness.release()


async def test_tool_status_is_visible_before_tool_finishes(tool_http_harness):
    h = tool_http_harness
    async with h.running_server() as client:
        async with client.stream("POST", "/api/chat", json={"message": "退货"}) as response:
            lines = response.aiter_lines()
            assert (await next_event(lines))["event"] == "meta"
            event = await next_event(lines)
            assert event["event"] == "tool_status"
            assert event["data"] == {"name": "create_ticket", "tool_call_id": "call-1", "status": "running", "attempt": 1, "message": "工具正在执行"}
            assert not h.tool_finished.is_set()
            h.resume_tool.set()
            assert (await next_event(lines))["data"]["status"] == "succeeded"
            assert (await next_event(lines))["event"] == "token"
            assert not h.gateway.completed
            h.gateway.resume.set()
            assert (await next_event(lines))["event"] == "token"
            assert (await next_event(lines))["event"] == "done"


@pytest.mark.parametrize("asgi_spec", [None, "2.4"])
@pytest.mark.parametrize("phase", ["select", "tool", "final"])
async def test_idle_disconnect_in_every_phase_releases_guard_and_audits(phase, asgi_spec):
    h = ToolHttpHarness(phase)
    async with h.running_server(asgi_spec) as client:
        async with client.stream("POST", "/api/chat", json={"message": "退货"}) as response:
            lines = response.aiter_lines()
            sid = (await next_event(lines))["data"]["session_id"]
            waiting = {"select": h.select_waiting, "tool": h.tool_waiting, "final": h.gateway.waiting}[phase]
            await asyncio.wait_for(waiting.wait(), 1)
        async with asyncio.timeout(1.5):
            while not h.conversations.finishes or sid in h.service.guard._active:
                await asyncio.sleep(0.01)
        assert len(h.conversations.finishes) == 1
        assert h.conversations.finishes[0][2] == "cancelled"
        assert h.conversations.records[sid]["turns"] == []
        assert {"select": h.select_closed, "tool": h.tool_closed, "final": h.gateway.stream_closed}[phase].is_set()
        h.release()
        retry = await client.post("/api/chat", json={"message": "再试", "session_id": sid})
        assert parse_sse(retry.text)[-1]["event"] == "done"


@pytest.mark.parametrize("asgi_spec", [None, "2.4"])
async def test_disconnect_with_blocked_ordinary_upstream_finally_is_bounded(asgi_spec):
    h = ToolHttpHarness("final", blocked_close=True)
    async with h.running_server(asgi_spec) as client:
        async with client.stream("POST", "/api/chat", json={"message": "退货"}) as response:
            lines = response.aiter_lines()
            sid = (await next_event(lines))["data"]["session_id"]
            await asyncio.wait_for(h.gateway.waiting.wait(), 1)
        await asyncio.wait_for(h.close_waiting.wait(), 1)
        async with asyncio.timeout(1.5):
            while sid in h.service.guard._active:
                await asyncio.sleep(0.01)
        assert len(h.conversations.finishes) == 1
        assert h.conversations.finishes[0][1:] == ("第一段", "cancelled")
        h.release()


async def test_selection_time_counts_toward_one_total_deadline():
    h = ToolHttpHarness("select", timeout=1)
    async with h.running_server() as client:
        start = asyncio.get_running_loop().time()
        async with client.stream("POST", "/api/chat", json={"message": "退货"}) as response:
            lines = response.aiter_lines()
            assert (await next_event(lines))["event"] == "meta"
            await asyncio.sleep(0.65)
            h.resume_select.set()
            h.resume_tool.set()
            events = []
            while not events or events[-1]["event"] != "error":
                events.append(await next_event(lines))
            assert events[-1]["data"]["code"] == "UPSTREAM_TIMEOUT"
            assert not any(item["event"] == "done" for item in events)
            assert asyncio.get_running_loop().time() - start < 1.5
        async with asyncio.timeout(1.5):
            while h.service.guard._active:
                await asyncio.sleep(0.01)
        assert len(h.conversations.finishes) == 1
        assert h.conversations.finishes[0][2] in {"failed", "cancelled"}
        h.release()


async def test_adapter_keeps_one_pending_read_across_polls_and_closes_on_disconnect():
    from app.api.streaming import stream_events
    from app.services.events import ChatEvent

    class Request:
        disconnected = False

        async def is_disconnected(self):
            return self.disconnected

    request = Request()
    waiting, closed, resume = asyncio.Event(), asyncio.Event(), asyncio.Event()
    tasks = []

    async def upstream():
        try:
            tasks.append(asyncio.current_task())
            yield ChatEvent("meta", {})
            tasks.append(asyncio.current_task())
            waiting.set()
            await resume.wait()
            yield ChatEvent("token", {"content": "late"})
        finally:
            closed.set()

    events = stream_events(request, upstream(), deadline=asyncio.get_running_loop().time() + 2)
    assert (await anext(events)).name == "meta"
    pending = asyncio.create_task(anext(events))
    try:
        await waiting.wait()
        await asyncio.sleep(0.16)
        assert not closed.is_set(), "polling cancelled the pending upstream read"
        assert tasks[0] is not tasks[1]
        request.disconnected = True
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(pending, 0.3)
        assert closed.is_set()
    finally:
        resume.set()
        await events.aclose()


async def test_adapter_does_not_start_another_read_after_deadline():
    from app.api.streaming import stream_events
    from app.services.events import ChatEvent

    class Request:
        async def is_disconnected(self):
            return False

    started = asyncio.Event()

    async def upstream():
        started.set()
        yield ChatEvent("meta", {})

    events = stream_events(Request(), upstream(), deadline=asyncio.get_running_loop().time() - 1)
    try:
        event = await anext(events)
        assert event.name == "error"
        assert event.data["code"] == "UPSTREAM_TIMEOUT"
        await asyncio.sleep(0)
        assert not started.is_set()
    finally:
        await events.aclose()
