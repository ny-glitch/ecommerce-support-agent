from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.knowledge.corpus import (
    corpus_fingerprint,
    load_cases,
    load_corpus,
    validate_cases,
    validate_corpus,
)
from tests.ch04_helpers import make_chunk


def _write_json(tmp_path, value):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _document(*chunks, source_key="charger-guide"):
    return {
        "source_key": source_key,
        "source_document": "本店演示资料：充电器使用指南",
        "summary": "虚构充电器的规格和使用限制。",
        "chunks": list(chunks),
    }


def _chunk(**changes):
    value = {
        "id": 910001,
        "category": "数码配件/充电器",
        "questions": "C65-Pro 支持什么协议？",
        "answer": "C65-Pro 支持 PD 3.0。",
        "section_path": "充电器使用指南/C65-Pro/协议",
        "content_type": "manual",
        "is_key_clause": True,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "vector_id": None,
        "vectorize_status": "pending",
    }
    value.update(changes)
    return value


def test_load_corpus_parses_explicit_chunk_fields(tmp_path):
    path = _write_json(tmp_path, {"documents": [_document(_chunk())]})

    assert load_corpus(path) == [
        make_chunk(
            answer="C65-Pro 支持 PD 3.0。",
            section_path="充电器使用指南/C65-Pro/协议",
            is_key_clause=True,
        )
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"source_key": ""}, "source_key"),
        ({"source_key": "charger guide"}, "source_key"),
        ({"unexpected": "field"}, "unexpected"),
    ],
)
def test_load_corpus_rejects_malformed_document_metadata(
    tmp_path, mutation, message
):
    document = _document(_chunk())
    document.update(mutation)
    path = _write_json(tmp_path, {"documents": [document]})

    with pytest.raises(ValueError, match=message):
        load_corpus(path)


def test_load_corpus_rejects_missing_or_extra_chunk_fields(tmp_path):
    missing = _chunk()
    del missing["answer"]
    path = _write_json(tmp_path, {"documents": [_document(missing)]})
    with pytest.raises(ValueError, match="answer"):
        load_corpus(path)

    path = _write_json(
        tmp_path,
        {"documents": [_document(_chunk(search_aliases=["快充头"]))]},
    )
    with pytest.raises(ValueError, match="search_aliases"):
        load_corpus(path)


def test_duplicate_chunk_id_is_rejected():
    errors = validate_corpus([make_chunk(), make_chunk(answer="另一份原文")])
    assert any("910001" in error and "重复" in error for error in errors)


def test_missing_neighbor_is_rejected():
    errors = validate_corpus([replace(make_chunk(), next_chunk_id=999999)])
    assert any("999999" in error for error in errors)


def test_non_reciprocal_neighbor_is_rejected():
    first = replace(make_chunk(), next_chunk_id=910002)
    second = replace(make_chunk(id=910002), prev_chunk_id=None)
    errors = validate_corpus([first, second])
    assert any("910002" in error and "反向" in error for error in errors)


def test_blank_required_fields_and_incomplete_section_path_are_rejected():
    chunk = replace(make_chunk(), questions=" ", section_path="协议")
    errors = validate_corpus([chunk])
    assert any("questions" in error for error in errors)
    assert any("section_path" in error for error in errors)


def test_load_cases_requires_exact_schema_and_consistent_answerability(tmp_path):
    valid = {
        "query_id": "test-model-01",
        "query": "C65-Pro 能用 PD 3.0 吗？",
        "category": "数码配件/充电器",
        "relevant_chunk_ids": [910001],
        "reference_answer": "C65-Pro 支持 PD 3.0。",
        "answerable": True,
        "query_type": "model",
        "difficulty": "easy",
        "rationale": "型号和协议均在 910001 原文中明确出现。",
    }
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(valid, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_cases(path) == [valid]

    invalid = {**valid, "answerable": False, "reference_answer": "", "relevant_chunk_ids": [910001]}
    path.write_text(json.dumps(invalid, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        load_cases(path)


def test_load_cases_rejects_duplicate_ids_and_unknown_fields(tmp_path):
    case = {
        "query_id": "cal-literal-01",
        "query": "这个充电器支持什么协议？",
        "category": None,
        "relevant_chunk_ids": [910001],
        "reference_answer": "支持 PD 3.0。",
        "answerable": True,
        "query_type": "literal",
        "difficulty": "easy",
        "rationale": "协议在原文中明确出现。",
    }
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [json.dumps(case, ensure_ascii=False), json.dumps(case, ensure_ascii=False)]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cal-literal-01.*重复"):
        load_cases(path)

    path.write_text(
        json.dumps({**case, "expected_rank": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_rank"):
        load_cases(path)


def test_corpus_fingerprint_is_order_independent_and_content_sensitive():
    first = make_chunk()
    second = make_chunk(id=910002, questions="C65 支持什么协议？")

    assert corpus_fingerprint([first, second]) == corpus_fingerprint([second, first])
    assert corpus_fingerprint([first, second]) != corpus_fingerprint(
        [first, replace(second, answer="仅支持 QC 3.0。")]
    )


def test_validate_cases_rejects_missing_and_cross_category_targets():
    base = {
        "query_id": "test-model-01",
        "query": "目标条目是否匹配？",
        "category": "家居用品/保温杯",
        "relevant_chunk_ids": [910001, 999999],
        "reference_answer": "有答案。",
        "answerable": True,
        "query_type": "model",
        "difficulty": "hard",
        "rationale": "用于校验标注引用。",
    }

    errors = validate_cases([base], [make_chunk()])

    assert any("999999" in error and "不存在" in error for error in errors)
    assert any("910001" in error and "category" in error for error in errors)
