from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.knowledge.contracts import Citation
from app.knowledge.text import source_hash
from app.main import KnowledgeDependencies, create_app
from app.services.events import ChatEvent
from tests.ch02_helpers import selection
from tests.ch04_helpers import make_chunk, make_decision
from tests.helpers import RecordingGateway, parse_sse, settings
from tests.test_knowledge_chat import service_case


class ReadOnlyKnowledgeRepository:
    def __init__(self, *chunks) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}

    async def get(self, chunk_id: int):
        return self.chunks.get(chunk_id)

    async def categories(self) -> list[str]:
        return sorted({chunk.category for chunk in self.chunks.values()})


@pytest.fixture
def source_chunk():
    return make_chunk(prev_chunk_id=None, next_chunk_id=910002)


@pytest.fixture
def knowledge_api_client(source_chunk):
    service, gateway, _conversations, _pipeline, _pool = service_case()
    repository = ReadOnlyKnowledgeRepository(
        source_chunk,
        make_chunk(id=910002, category="智能家电/扫地机器人", prev_chunk_id=910001),
    )
    app = create_app(
        settings(),
        gateway,
        chat_service=service,
        knowledge_dependencies=KnowledgeDependencies(repository=repository),
    )
    with TestClient(app) as client:
        yield client


def test_categories_and_source_are_fixed_mysql_views(
    knowledge_api_client: TestClient, source_chunk
) -> None:
    categories = knowledge_api_client.get("/api/knowledge/categories")
    source = knowledge_api_client.get(f"/api/knowledge/chunks/{source_chunk.id}")

    assert categories.status_code == 200
    assert categories.json() == {
        "categories": ["数码配件/充电器", "智能家电/扫地机器人"]
    }
    assert source.status_code == 200
    assert source.json() == {
        "id": 910001,
        "category": "数码配件/充电器",
        "questions": "C65-Pro 支持什么协议？",
        "answer": "支持 PD 3.0。",
        "section_path": "商品手册/C65-Pro/协议",
        "content_type": "manual",
        "is_key_clause": False,
        "prev_chunk_id": None,
        "next_chunk_id": 910002,
        "content_hash": source_hash(source_chunk),
    }
    assert "vector" not in source.text
    assert "database" not in source.text.casefold()


def test_source_revision_conflict(knowledge_api_client: TestClient) -> None:
    response = knowledge_api_client.get(
        "/api/knowledge/chunks/910001?expected_hash=" + "0" * 64
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CHUNK_CHANGED"
    assert "answer" not in response.json()


@pytest.mark.parametrize(
    "url,status,code",
    [
        ("/api/knowledge/chunks/999999", 404, "CHUNK_NOT_FOUND"),
        ("/api/knowledge/chunks/0", 422, "INVALID_REQUEST"),
        ("/api/knowledge/chunks/-1", 422, "INVALID_REQUEST"),
        ("/api/knowledge/chunks/9223372036854775808", 422, "INVALID_REQUEST"),
        ("/api/knowledge/chunks/910001?expected_hash=not-a-hash", 422, "INVALID_REQUEST"),
    ],
)
def test_source_rejects_missing_or_invalid_identifiers(
    knowledge_api_client: TestClient, url: str, status: int, code: str
) -> None:
    response = knowledge_api_client.get(url)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_category_is_current_turn_only_and_source_frame_matches_source_api(
    source_chunk,
) -> None:
    digest = source_hash(source_chunk)
    base = make_decision()
    citation = Citation(
        number=1,
        chunk_id=source_chunk.id,
        category=source_chunk.category,
        section_path=source_chunk.section_path,
        questions=source_chunk.questions,
        answer=source_chunk.answer,
        content_hash=digest,
        url=f"/api/knowledge/chunks/{source_chunk.id}?expected_hash={digest}",
        score=0.91,
    )
    decision = replace(base, sources=(citation,))
    service, gateway, _conversations, pipeline, _pool = service_case(decision)
    pipeline.decisions.append(decision)
    gateway.selection = selection("query_faq", {})
    repository = ReadOnlyKnowledgeRepository(source_chunk)
    app = create_app(
        settings(),
        gateway,
        chat_service=service,
        knowledge_dependencies=KnowledgeDependencies(repository=repository),
    )

    with TestClient(app) as client:
        first = parse_sse(
            client.post(
                "/api/chat",
                json={"message": "协议", "category": "  数码配件/充电器  "},
            ).text
        )
        second = parse_sse(client.post("/api/chat", json={"message": "协议"}).text)
        source_response = client.get(citation.url)

    assert [call[1] for call in pipeline.calls] == ["数码配件/充电器", None]
    source_frame = next(
        item["data"]["sources"][0]
        for item in first
        if item["event"] == "sources"
    )
    second_source = next(
        item["data"]["sources"][0]
        for item in second
        if item["event"] == "sources"
    )
    assert second_source["url"] == citation.url
    assert source_frame["url"] == citation.url
    assert source_frame["content_hash"] == source_response.json()["content_hash"]
    assert {
        key: source_frame[key]
        for key in ("category", "questions", "answer", "section_path", "content_hash")
    } == {
        key: source_response.json()[key]
        for key in ("category", "questions", "answer", "section_path", "content_hash")
    }


@pytest.mark.parametrize("category", ["", " \n ", "x" * 256, 7])
def test_chat_rejects_invalid_category_without_calling_model(category) -> None:
    service, gateway, _conversations, _pipeline, _pool = service_case()
    app = create_app(
        settings(),
        gateway,
        chat_service=service,
        knowledge_dependencies=KnowledgeDependencies(
            repository=ReadOnlyKnowledgeRepository(make_chunk())
        ),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/chat", json={"message": "协议", "category": category}
        )
    assert response.status_code == 422
    assert gateway.select_calls == []


def test_http_transport_reads_live_extended_knowledge_deadline() -> None:
    class Service:
        @asynccontextmanager
        async def prepare(self, *_args, **_kwargs):
            prepared = SimpleNamespace(deadline=time.monotonic() + 0.03)
            yield prepared

        async def stream(self, prepared):
            yield ChatEvent("meta", {})
            prepared.deadline = time.monotonic() + 1
            await asyncio.sleep(0.06)
            yield ChatEvent("done", {"refused": False, "citations": []})

    gateway = RecordingGateway()
    app = create_app(
        settings(),
        gateway,
        chat_service=Service(),
        knowledge_dependencies=KnowledgeDependencies(
            repository=ReadOnlyKnowledgeRepository(make_chunk())
        ),
    )
    with TestClient(app) as client:
        events = parse_sse(client.post("/api/chat", json={"message": "协议"}).text)

    assert [event["event"] for event in events] == ["meta", "done"]
