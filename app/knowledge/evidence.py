from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TypeVar

from langchain_core.messages import AIMessage, ToolMessage

from app.config import Settings
from app.context import build_tool_context
from app.db.contracts import StoredTurn
from app.errors import ServiceError
from app.knowledge.contracts import (
    Citation,
    EvidencePlan,
    KnowledgeDecision,
    QueryPlan,
    RankedChunk,
)
from app.knowledge.gateway import build_assessment_messages
from app.knowledge.text import source_hash
from app.prompts import knowledge_answer_system_prompt


MAX_KNOWLEDGE_RESULT_BYTES = 48_000
ASSESSMENT_RESERVE_BYTES = 4_096
_CITATION = re.compile(r"\[(\d+)\]")
_T = TypeVar("_T")


def edge_order(items: Sequence[_T]) -> list[_T]:
    return list(items[::2]) + list(reversed(items[1::2]))


def validate_citation_numbers(answer: str, allowed: set[int]) -> set[int]:
    numbers = {int(value) for value in _CITATION.findall(answer)}
    if not numbers:
        raise ValueError("citation required")
    unknown = numbers - allowed
    if unknown:
        raise ValueError(f"unknown citation numbers: {sorted(unknown)}")
    return numbers


def _citation(item: RankedChunk, number: int) -> Citation:
    digest = source_hash(item.chunk)
    return Citation(
        number=number,
        chunk_id=item.chunk.id,
        category=item.chunk.category,
        section_path=item.chunk.section_path,
        questions=item.chunk.questions,
        answer=item.chunk.answer,
        content_hash=digest,
        url=(
            f"/api/knowledge/chunks/{item.chunk.id}"
            f"?expected_hash={digest}"
        ),
        score=item.score,
    )


def _compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _tool_identity(call: AIMessage) -> tuple[str, str]:
    if len(call.tool_calls) != 1:
        raise ValueError("knowledge call must contain exactly one tool call")
    current = call.tool_calls[0]
    call_id = current.get("id")
    name = current.get("name")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("knowledge call requires a tool call id")
    if not isinstance(name, str) or not name:
        raise ValueError("knowledge call requires a tool name")
    return call_id, name


def build_knowledge_tool_message(
    call: AIMessage,
    decision: KnowledgeDecision,
) -> ToolMessage:
    call_id, name = _tool_identity(call)
    return ToolMessage(
        _compact_json(decision.to_payload()),
        tool_call_id=call_id,
        name=name,
    )


class EvidenceBudget:
    def __init__(
        self,
        settings: Settings,
        history: list[StoredTurn],
        question: str,
        call: AIMessage,
    ) -> None:
        self._settings = settings
        self._history = list(history)
        self._question = question
        self._call = call

    def select(
        self,
        ranked: tuple[RankedChunk, ...],
        query: QueryPlan,
    ) -> EvidencePlan:
        candidates = list(ranked[:10])
        while candidates:
            relevance_order = tuple(
                _citation(item, number)
                for number, item in enumerate(candidates, start=1)
            )
            sources = tuple(edge_order(relevance_order))
            if self._fits(sources, query):
                return EvidencePlan(
                    sources=sources,
                    dropped_ids=tuple(
                        item.chunk.id for item in ranked[len(candidates) :]
                    ),
                )
            candidates.pop()
        return EvidencePlan(
            sources=(),
            dropped_ids=tuple(item.chunk.id for item in ranked),
        )

    def _fits(self, sources: tuple[Citation, ...], query: QueryPlan) -> bool:
        try:
            build_assessment_messages(
                self._question,
                sources,
                normalized_question=query.normalized,
                settings=self._settings,
            )
            placeholder = KnowledgeDecision(
                query=query,
                status="ok",
                sources=sources,
                assessment=None,
                reason_code=None,
                refusal=None,
            )
            content = _compact_json(placeholder.to_payload())
            if (
                len(content.encode("utf-8")) + ASSESSMENT_RESERVE_BYTES
                > MAX_KNOWLEDGE_RESULT_BYTES
            ):
                return False
            call_id, name = _tool_identity(self._call)
            reserved_result = ToolMessage(
                content + " " * ASSESSMENT_RESERVE_BYTES,
                tool_call_id=call_id,
                name=name,
            )
            build_tool_context(
                knowledge_answer_system_prompt(),
                self._history,
                self._question,
                self._settings,
                tool_schemas=[],
                current_tool_messages=[self._call, reserved_result],
            )
            return True
        except (ServiceError, ValueError):
            return False
