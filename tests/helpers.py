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

    async def stream(self, messages):
        self.calls.append(messages)
        for text in self.fragments:
            yield text
        if self.error:
            raise self.error

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
