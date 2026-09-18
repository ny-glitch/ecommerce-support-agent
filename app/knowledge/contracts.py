from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


@dataclass(frozen=True)
class QueryPlan:
    original: str
    normalized: str
    synonyms: tuple[str, ...]
    category: str | None
    fallback: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    query: QueryPlan
    strategy: str
    ranked: tuple[RankedChunk, ...]
    raw_count: int
    stale_count: int


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int = Field(ge=1, le=10)
    chunk_id: int
    category: str
    section_path: str | None
    questions: str
    answer: str
    content_hash: str
    url: str
    score: float


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sufficient: bool
    reason_code: Literal[
        "supported",
        "insufficient_evidence",
        "ambiguous_question",
    ]
    reason: str = Field(max_length=600)
    supporting_chunk_ids: list[int] = Field(max_length=10)


@dataclass(frozen=True)
class EvidencePlan:
    sources: tuple[Citation, ...]
    dropped_ids: tuple[int, ...]


@dataclass(frozen=True)
class KnowledgeDecision:
    query: QueryPlan
    status: Literal["ok", "not_found"]
    sources: tuple[Citation, ...]
    assessment: EvidenceAssessment | None
    reason_code: str | None
    refusal: str | None

    def to_payload(self) -> dict:
        payload = {
            "status": self.status,
            "query": asdict(self.query),
            "sources": [source.model_dump(mode="json") for source in self.sources],
            "assessment": (
                None
                if self.assessment is None
                else self.assessment.model_dump(mode="json")
            ),
            "reason_code": self.reason_code,
            "refusal": self.refusal,
        }
        size = len(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if size > 48_000:
            raise ValueError("knowledge payload exceeds 48000 UTF-8 bytes")
        return payload
