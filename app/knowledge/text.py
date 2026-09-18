from __future__ import annotations

import hashlib
import json

from app.knowledge.contracts import KnowledgeChunk


def embedding_text(chunk: KnowledgeChunk) -> str:
    return (
        f"分类：{chunk.category}\n"
        f"问法：{chunk.questions}\n"
        f"答案：{chunk.answer}"
    )


def source_hash(chunk: KnowledgeChunk) -> str:
    fields = (
        "category",
        "questions",
        "answer",
        "section_path",
        "content_type",
        "is_key_clause",
        "prev_chunk_id",
        "next_chunk_id",
    )
    value = {name: getattr(chunk, name) for name in fields}
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
