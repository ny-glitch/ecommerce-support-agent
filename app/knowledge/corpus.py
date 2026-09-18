from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.knowledge.contracts import KnowledgeChunk
from app.knowledge.text import source_hash


_DOCUMENT_FIELDS = {"source_key", "source_document", "summary", "chunks"}
_CHUNK_FIELDS = {
    "id",
    "category",
    "questions",
    "answer",
    "section_path",
    "content_type",
    "is_key_clause",
    "prev_chunk_id",
    "next_chunk_id",
    "vector_id",
    "vectorize_status",
}
_CASE_FIELDS = {
    "query_id",
    "query",
    "category",
    "relevant_chunk_ids",
    "reference_answer",
    "answerable",
    "query_type",
    "difficulty",
    "rationale",
}
_SOURCE_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DIFFICULTIES = {"easy", "medium", "hard"}


def _require_exact_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"缺少 {', '.join(sorted(missing))}")
        if extra:
            details.append(f"包含未知字段 {', '.join(sorted(extra))}")
        raise ValueError(f"{label}: {'; '.join(details)}")


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    return value


def _optional_id(value: Any, label: str) -> int | None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
    ):
        raise ValueError(f"{label} 必须是正整数或 null")
    return value


def load_corpus(path: Path) -> list[KnowledgeChunk]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取语料文件 {path}: {exc}") from exc
    if not isinstance(root, dict) or set(root) != {"documents"}:
        raise ValueError("语料顶层必须且只能包含 documents")
    documents = root["documents"]
    if not isinstance(documents, list):
        raise ValueError("documents 必须是数组")

    chunks: list[KnowledgeChunk] = []
    source_keys: set[str] = set()
    for document_index, document in enumerate(documents, 1):
        label = f"document[{document_index}]"
        if not isinstance(document, dict):
            raise ValueError(f"{label} 必须是对象")
        _require_exact_fields(document, _DOCUMENT_FIELDS, label)
        source_key = _nonempty_text(document["source_key"], f"{label}.source_key")
        if not _SOURCE_KEY.fullmatch(source_key):
            raise ValueError(f"{label}.source_key 格式非法: {source_key!r}")
        if source_key in source_keys:
            raise ValueError(f"source_key 重复: {source_key}")
        source_keys.add(source_key)
        _nonempty_text(document["source_document"], f"{label}.source_document")
        _nonempty_text(document["summary"], f"{label}.summary")
        raw_chunks = document["chunks"]
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise ValueError(f"{label}.chunks 必须是非空数组")
        for chunk_index, raw in enumerate(raw_chunks, 1):
            chunk_label = f"{label}.chunks[{chunk_index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{chunk_label} 必须是对象")
            _require_exact_fields(raw, _CHUNK_FIELDS, chunk_label)
            chunk_id = raw["id"]
            if not isinstance(chunk_id, int) or isinstance(chunk_id, bool) or chunk_id <= 0:
                raise ValueError(f"{chunk_label}.id 必须是正整数")
            is_key_clause = raw["is_key_clause"]
            if not isinstance(is_key_clause, bool):
                raise ValueError(f"{chunk_label}.is_key_clause 必须是布尔值")
            vector_id = raw["vector_id"]
            if vector_id is not None and not isinstance(vector_id, str):
                raise ValueError(f"{chunk_label}.vector_id 必须是字符串或 null")
            status = raw["vectorize_status"]
            if status not in {"pending", "done"}:
                raise ValueError(f"{chunk_label}.vectorize_status 非法")
            chunks.append(
                KnowledgeChunk(
                    id=chunk_id,
                    category=_nonempty_text(raw["category"], f"{chunk_label}.category"),
                    questions=_nonempty_text(raw["questions"], f"{chunk_label}.questions"),
                    answer=_nonempty_text(raw["answer"], f"{chunk_label}.answer"),
                    section_path=_nonempty_text(
                        raw["section_path"], f"{chunk_label}.section_path"
                    ),
                    content_type=_nonempty_text(
                        raw["content_type"], f"{chunk_label}.content_type"
                    ),
                    is_key_clause=is_key_clause,
                    prev_chunk_id=_optional_id(
                        raw["prev_chunk_id"], f"{chunk_label}.prev_chunk_id"
                    ),
                    next_chunk_id=_optional_id(
                        raw["next_chunk_id"], f"{chunk_label}.next_chunk_id"
                    ),
                    vector_id=vector_id,
                    vectorize_status=status,
                )
            )
    errors = validate_corpus(chunks)
    if errors:
        raise ValueError("语料校验失败:\n- " + "\n- ".join(errors))
    return chunks


def validate_corpus(chunks: Iterable[KnowledgeChunk]) -> list[str]:
    materialized = list(chunks)
    errors: list[str] = []
    by_id: dict[int, KnowledgeChunk] = {}
    for chunk in materialized:
        if chunk.id in by_id:
            errors.append(f"chunk ID {chunk.id} 重复")
        else:
            by_id[chunk.id] = chunk
        for field in ("category", "questions", "answer", "content_type"):
            value = getattr(chunk, field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"chunk {chunk.id} 的 {field} 不能为空")
        path = chunk.section_path
        if not isinstance(path, str) or len(
            [part for part in path.split("/") if part.strip()]
        ) < 3:
            errors.append(f"chunk {chunk.id} 的 section_path 必须包含至少三级完整路径")

    for chunk in materialized:
        for direction, neighbor_id, inverse in (
            ("prev_chunk_id", chunk.prev_chunk_id, "next_chunk_id"),
            ("next_chunk_id", chunk.next_chunk_id, "prev_chunk_id"),
        ):
            if neighbor_id is None:
                continue
            neighbor = by_id.get(neighbor_id)
            if neighbor is None:
                errors.append(f"chunk {chunk.id} 的 {direction} 指向不存在的 {neighbor_id}")
            elif neighbor_id == chunk.id:
                errors.append(f"chunk {chunk.id} 的 {direction} 不能指向自身")
            elif getattr(neighbor, inverse) != chunk.id:
                errors.append(
                    f"chunk {chunk.id} 与 {neighbor_id} 的相邻指针缺少反向 {inverse}"
                )
            elif neighbor.category != chunk.category:
                errors.append(f"chunk {chunk.id} 与 {neighbor_id} 的相邻指针跨越 category")
    return errors


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"无法读取标注文件 {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON: {exc}") from exc
        label = f"{path}:{line_number}"
        if not isinstance(case, dict):
            raise ValueError(f"{label} 必须是对象")
        _require_exact_fields(case, _CASE_FIELDS, label)
        query_id = _nonempty_text(case["query_id"], f"{label}.query_id")
        if query_id in seen_ids:
            raise ValueError(f"query_id {query_id} 重复")
        seen_ids.add(query_id)
        _nonempty_text(case["query"], f"{label}.query")
        category = case["category"]
        if category is not None:
            _nonempty_text(category, f"{label}.category")
        ids = case["relevant_chunk_ids"]
        if not isinstance(ids, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in ids
        ):
            raise ValueError(f"{label}.relevant_chunk_ids 必须是正整数数组")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label}.relevant_chunk_ids 不得重复")
        answerable = case["answerable"]
        if not isinstance(answerable, bool):
            raise ValueError(f"{label}.answerable 必须是布尔值")
        reference_answer = case["reference_answer"]
        if not isinstance(reference_answer, str):
            raise ValueError(f"{label}.reference_answer 必须是字符串")
        if answerable and (not ids or not reference_answer.strip()):
            raise ValueError(f"{label} 可回答项必须有 relevant_chunk_ids 和 reference_answer")
        if not answerable and ids:
            raise ValueError(f"{label} 无答案项的 relevant_chunk_ids 必须为空")
        if not answerable and reference_answer.strip():
            raise ValueError(f"{label} 无答案项的 reference_answer 必须为空")
        _nonempty_text(case["query_type"], f"{label}.query_type")
        if case["difficulty"] not in _DIFFICULTIES:
            raise ValueError(f"{label}.difficulty 必须是 easy、medium 或 hard")
        _nonempty_text(case["rationale"], f"{label}.rationale")
        cases.append(case)
    return cases


def validate_cases(
    cases: Iterable[dict[str, Any]],
    chunks: Iterable[KnowledgeChunk],
) -> list[str]:
    by_id = {chunk.id: chunk for chunk in chunks}
    errors: list[str] = []
    for case in cases:
        query_id = case.get("query_id", "<unknown>")
        category = case.get("category")
        for chunk_id in case.get("relevant_chunk_ids", []):
            chunk = by_id.get(chunk_id)
            if chunk is None:
                errors.append(f"{query_id} 引用的 chunk {chunk_id} 不存在")
            elif category is not None and chunk.category != category:
                errors.append(
                    f"{query_id} 引用的 chunk {chunk_id} category 为"
                    f" {chunk.category!r}，与标注 {category!r} 不符"
                )
    return errors


def corpus_fingerprint(chunks: Iterable[KnowledgeChunk]) -> str:
    values = [
        f"{chunk.id}:{source_hash(chunk)}"
        for chunk in sorted(chunks, key=lambda item: item.id)
    ]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
