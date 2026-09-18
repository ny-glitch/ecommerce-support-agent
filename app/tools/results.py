from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from pydantic import ValidationError

from app.knowledge.contracts import (
    Citation,
    EvidenceAssessment,
    KnowledgeDecision,
    QueryPlan,
)


MAX_TOOL_RESULT_BYTES = 4096
MAX_KNOWLEDGE_RESULT_BYTES = 48_000
_ESSENTIAL_KEYS = {
    "status",
    "code",
    "ticket_no",
    "order_id",
    "order_status",
    "logistics_status",
}


class InvalidToolArguments(Exception):
    pass


class TransientToolError(Exception):
    pass


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _fits(payload: dict[str, Any]) -> bool:
    return len(_serialize(payload).encode("utf-8")) <= MAX_TOOL_RESULT_BYTES


def _iter_containers(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield value, key, child
            yield from _iter_containers(child)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield value, index, child
            yield from _iter_containers(child)


def _is_essential_key(key: object) -> bool:
    return isinstance(key, str) and (
        key in _ESSENTIAL_KEYS or key.endswith("_id") or key.endswith("_no")
    )


def _shrink_lists(payload: dict[str, Any]) -> None:
    while not _fits(payload):
        lists = [
            child
            for _, _, child in _iter_containers(payload)
            if isinstance(child, list) and len(child) > 1
        ]
        if not lists:
            return
        max(lists, key=len).pop()


def _shrink_strings(payload: dict[str, Any]) -> None:
    while not _fits(payload):
        candidates = [
            (container, key, child)
            for container, key, child in _iter_containers(payload)
            if isinstance(child, str)
            and not _is_essential_key(key)
            and len(child) > 1
        ]
        if not candidates:
            return
        container, key, value = max(
            candidates, key=lambda item: len(item[2].encode("utf-8"))
        )
        container[key] = "…" if len(value) <= 4 else value[: len(value) // 2] + "…"


def _fallback(payload: dict[str, Any]) -> dict[str, Any]:
    fallback: dict[str, Any] = {
        "status": "error",
        "code": "TOOL_RESULT_TOO_LARGE",
        "truncated": True,
    }
    for _, key, value in _iter_containers(payload):
        if _is_essential_key(key) and key not in {"status", "code"}:
            fallback[str(key)] = value
    if _fits(fallback):
        return fallback
    return {
        "status": "error",
        "code": "TOOL_RESULT_TOO_LARGE",
        "truncated": True,
    }


def bounded_result(payload: dict[str, Any]) -> str:
    """Serialize a tool payload without ever producing oversized or invalid JSON."""
    if _fits(payload):
        return _serialize(payload)

    reduced = deepcopy(payload)
    reduced["truncated"] = True
    _shrink_lists(reduced)
    _shrink_strings(reduced)
    if _fits(reduced):
        return _serialize(reduced)
    return _serialize(_fallback(reduced))


def validated_knowledge_result(payload: dict[str, Any]) -> str:
    """Validate and canonically serialize the sole large-result contract."""
    if set(payload) != {
        "status",
        "query",
        "sources",
        "assessment",
        "reason_code",
        "refusal",
    }:
        raise ValueError("invalid knowledge result keys")
    try:
        query = QueryPlan(**payload["query"])
        sources = tuple(Citation.model_validate(item) for item in payload["sources"])
        assessment = (
            None
            if payload["assessment"] is None
            else EvidenceAssessment.model_validate(payload["assessment"])
        )
        decision = KnowledgeDecision(
            query=query,
            status=payload["status"],
            sources=sources,
            assessment=assessment,
            reason_code=payload["reason_code"],
            refusal=payload["refusal"],
        )
        canonical = _serialize(decision.to_payload())
    except (KeyError, TypeError, ValidationError) as exc:
        raise ValueError("invalid knowledge result") from exc
    if len(canonical.encode("utf-8")) > MAX_KNOWLEDGE_RESULT_BYTES:
        raise ValueError("knowledge result too large")
    return canonical
