from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, Request

from app.errors import ServiceError
from app.knowledge.contracts import KnowledgeChunk
from app.knowledge.text import source_hash


router = APIRouter(prefix="/api/knowledge")
ChunkId = Annotated[int, Path(ge=1, le=2**63 - 1)]
ExpectedHash = Annotated[
    str | None,
    Query(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]


def _repository(request: Request):
    repository = getattr(request.app.state, "knowledge_repository", None)
    if repository is None:
        raise ServiceError("KNOWLEDGE_UNAVAILABLE", "知识服务暂时不可用", 503)
    return repository


def source_response(chunk: KnowledgeChunk, digest: str) -> dict[str, object]:
    return {
        "id": chunk.id,
        "category": chunk.category,
        "questions": chunk.questions,
        "answer": chunk.answer,
        "section_path": chunk.section_path,
        "content_type": chunk.content_type,
        "is_key_clause": chunk.is_key_clause,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "content_hash": digest,
    }


@router.get("/categories")
async def categories(request: Request) -> dict[str, list[str]]:
    values = await _repository(request).categories()
    return {"categories": values}


@router.get("/chunks/{chunk_id}")
async def read_chunk(
    chunk_id: ChunkId,
    request: Request,
    expected_hash: ExpectedHash = None,
) -> dict[str, object]:
    chunk = await _repository(request).get(chunk_id)
    if chunk is None:
        raise ServiceError("CHUNK_NOT_FOUND", "知识原文不存在", 404)
    digest = source_hash(chunk)
    if expected_hash is not None and expected_hash != digest:
        raise ServiceError(
            "CHUNK_CHANGED",
            "原文已更新，请查看本轮保存的引用快照",
            409,
        )
    return source_response(chunk, digest)
