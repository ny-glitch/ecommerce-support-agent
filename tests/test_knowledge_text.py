from __future__ import annotations

from dataclasses import replace

from app.knowledge.text import embedding_text, source_hash
from tests.ch04_helpers import make_chunk


def test_metadata_does_not_enter_embedding_but_changes_source_revision() -> None:
    a = make_chunk()
    b = replace(
        a,
        section_path="商品手册/修订版/协议",
        vector_id="910001",
        vectorize_status="done",
    )

    assert embedding_text(a) == embedding_text(b)
    assert embedding_text(a) == (
        "分类：数码配件/充电器\n"
        "问法：C65-Pro 支持什么协议？\n"
        "答案：支持 PD 3.0。"
    )
    assert source_hash(a) != source_hash(b)
    assert source_hash(a) == source_hash(
        replace(a, vector_id="910001", vectorize_status="done")
    )


def test_source_hash_covers_each_authoritative_source_field() -> None:
    chunk = make_chunk()
    changes = (
        {"category": "通用/协议"},
        {"questions": "支持哪些快充协议？"},
        {"answer": "支持 PD 2.0。"},
        {"section_path": None},
        {"content_type": "faq"},
        {"is_key_clause": True},
        {"prev_chunk_id": 910000},
        {"next_chunk_id": 910002},
    )

    assert all(
        source_hash(chunk) != source_hash(replace(chunk, **change))
        for change in changes
    )
