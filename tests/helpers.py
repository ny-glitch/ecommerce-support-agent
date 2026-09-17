"""Offline upstream doubles and SSE parsing for application boundary tests."""

import asyncio
import json

from app.config import Settings
from app.schemas import AfterSalesResult


def settings(**overrides):
    return Settings(
        _env_file=None,
        **{
            "llm_base_url": "https://upstream.example/v1",
            "llm_model": "test-model",
            "llm_api_key": "test-key",
            **overrides,
        },
    )


def parse_sse(text):
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                fields["event"] = line[7:]
            elif line.startswith("data: "):
                fields["data"] = json.loads(line[6:])
        if fields:
            events.append(fields)
    return events


class RecordingGateway:
    def __init__(self):
        self.calls = []
        self.fragments = ["你好", "", "，小林"]
        self.error = None
        self.extraction_error = None
        self.descriptions = []
        self.closed = False

    async def select(self, messages, tools):
        from langchain_core.messages import AIMessage
        return AIMessage(content="")

    async def stream(self, messages):
        self.calls.append(messages)
        for text in self.fragments:
            yield text
        if self.error:
            from app.errors import ServiceError
            if isinstance(self.error, ServiceError):
                raise self.error
            raise ServiceError("UPSTREAM_ERROR", "模型服务暂时不可用", 502) from self.error

    async def extract(self, description):
        self.descriptions.append(description)
        if self.extraction_error:
            raise self.extraction_error
        return AfterSalesResult(
            order_id="ORDER-17", request_type="refund", expected_resolution="原路退款"
        )

    async def aclose(self):
        self.closed = True


class GatedGateway(RecordingGateway):
    def __init__(self):
        super().__init__()
        self.resume = asyncio.Event()
        self.waiting = asyncio.Event()
        self.stream_closed = asyncio.Event()
        self.completed = False

    async def stream(self, messages):
        self.calls.append(messages)
        try:
            yield "第一段"
            self.waiting.set()
            await self.resume.wait()
            yield "第二段"
            self.completed = True
        finally:
            self.stream_closed.set()


class StatefulConversations:
    """Test-only repository boundary; real ChatService owns all orchestration."""
    def __init__(self):
        self.records = {}
        self.pending = {}
        self.finishes = []

    def seed(self, turns=(), user_id="demo"):
        from uuid import uuid4
        sid = str(uuid4())
        self.records[sid] = {"user_id": user_id, "turns": list(turns)}
        return sid

    async def get(self, conversation_id, user_id):
        record = self.records.get(conversation_id)
        return record if record and record["user_id"] == user_id else None

    async def create(self, conversation_id, user_id):
        self.records[conversation_id] = {"user_id": user_id, "turns": []}

    async def history(self, conversation_id, user_id, limit):
        record = await self.get(conversation_id, user_id)
        return record["turns"][-limit:]

    async def start_turn(self, ref, user_id, content):
        from langchain_core.messages import HumanMessage
        self.pending[ref.turn_id] = [HumanMessage(content=content)]

    async def append_call(self, ref, message):
        self.pending[ref.turn_id].append(message)

    async def append_result(self, ref, message):
        self.pending[ref.turn_id].append(message)

    async def finish_turn(self, ref, content, status):
        from app.db.contracts import StoredTurn
        from langchain_core.messages import AIMessage
        self.finishes.append((ref, content, status))
        messages = self.pending.pop(ref.turn_id, [])
        if status == "completed":
            self.records[ref.conversation_id]["turns"].append(
                StoredTurn(ref.turn_id, tuple([*messages, AIMessage(content=content)]))
            )


def stored_turn(user, assistant):
    from uuid import uuid4
    from langchain_core.messages import AIMessage, HumanMessage
    from app.db.contracts import StoredTurn
    return StoredTurn(str(uuid4()), (HumanMessage(content=user), AIMessage(content=assistant)))


def http_app(configuration=None, gateway=None):
    from app.main import create_app
    from tests.ch02_helpers import ChatHarness, EmptyFaq, RecordingTickets
    from app.sessions import SessionGuard
    configuration = configuration or settings()
    gateway = gateway or RecordingGateway()
    conversations = StatefulConversations()
    harness = ChatHarness(configuration, (conversations, EmptyFaq(), RecordingTickets()))
    harness.service.gateway = gateway
    harness.service.guard = SessionGuard(configuration.max_sessions)
    app = create_app(configuration, gateway, chat_service=harness.service)
    return app, gateway, conversations, harness.service
