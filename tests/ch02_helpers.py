"""Controllable model/database boundaries for the real chat orchestrator."""
import asyncio
from langchain_core.messages import AIMessage

from app.services.chat import ChatService
from app.sessions import SessionGuard
from app.tools.executor import ToolExecutor
from tests.helpers import settings


def selection(name="query_logistics", args=None, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": {"order_id": "1001"} if args is None else args,
        "id": call_id, "type": "tool_call",
    }])


class ChatGateway:
    def __init__(self):
        self.selection = AIMessage(content="discard this decision text")
        self.select_calls = []
        self.stream_calls = []
        self.fragments = ["你好", "", "，这是结果"]
        self.error = None
        self.select_gate = None
        self.stream_gate = None
        self.waiting = asyncio.Event()
        self.closed = asyncio.Event()

    async def select(self, messages, tools):
        self.select_calls.append(messages)
        if self.select_gate:
            await self.select_gate.wait()
        return self.selection

    async def stream(self, messages):
        self.stream_calls.append(messages)
        try:
            for fragment in self.fragments:
                yield fragment
            if self.stream_gate:
                self.waiting.set()
                await self.stream_gate.wait()
            if self.error:
                raise self.error
        finally:
            self.closed.set()


class RecordingConversations:
    def __init__(self):
        self.turns = []
        self.operations = []
        self.finish_error = None
        self.result_error = None
        self.finish_gate = None
        self.finished_status = None
        self.finish_calls = []
        self.owner = {"id": "existing", "user_id": "demo"}

    async def get(self, conversation_id, user_id):
        self.operations.append(("get", conversation_id, user_id))
        return self.owner

    async def create(self, conversation_id, user_id):
        self.operations.append(("create", conversation_id, user_id))

    async def history(self, conversation_id, user_id, limit):
        return self.turns

    async def start_turn(self, ref, user_id, content):
        self.operations.append(("user", ref, content))

    async def append_call(self, ref, message):
        self.operations.append(("call", ref, message))

    async def append_result(self, ref, message):
        if self.result_error:
            raise self.result_error
        self.operations.append(("result", ref, message))

    async def finish_turn(self, ref, content, status):
        self.finish_calls.append((ref, content, status))
        if self.finish_gate:
            await self.finish_gate.wait()
        if self.finish_error:
            raise self.finish_error
        self.finished_status = status
        self.operations.append(("finish", ref, content, status))


class EmptyFaq:
    async def search(self, keyword):
        return []


class RecordingTickets:
    def __init__(self):
        self.calls = []

    async def create_once(self, *args):
        self.calls.append(args)
        return {"ticket_no": args[0], "status": "pending"}


class ChatHarness:
    def __init__(self, configuration=None, repos=None):
        self.settings = configuration or settings()
        self.gateway = ChatGateway()
        self.conversations, self.faq, self.tickets = repos or (
            RecordingConversations(), EmptyFaq(), RecordingTickets()
        )
        self.guard = SessionGuard(100)
        self.service = ChatService(
            self.settings, self.gateway, self.conversations,
            self.faq, self.tickets, self.guard, ToolExecutor(),
        )

    async def collect(self, message, session_id=None):
        async with self.service.prepare(message, session_id) as prepared:
            return [event async for event in self.service.stream(prepared)]
