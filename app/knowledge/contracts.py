from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class KnowledgeChunk:
    id: int
    category: str
    questions: str
    answer: str
    section_path: str | None = None
    content_type: str | None = None
    is_key_clause: bool = False
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None
    vector_id: str | None = None
    vectorize_status: Literal["pending", "done"] = "pending"


@dataclass(frozen=True)
class SearchHit:
    id: int
    score: float
    source_hash: str


@dataclass(frozen=True)
class RankedChunk:
    chunk: KnowledgeChunk
    score: float
