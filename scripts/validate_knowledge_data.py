from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from app.knowledge.corpus import (
    corpus_fingerprint,
    load_cases,
    load_corpus,
    validate_cases,
    validate_corpus,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data/knowledge/ch04/chunks.json"
MANIFEST_PATH = ROOT / "data/knowledge/ch04/manifest.json"
CALIBRATION_PATH = ROOT / "evals/ch04/calibration.jsonl"
TEST_PATH = ROOT / "evals/ch04/test.jsonl"

EXPECTED_CATEGORIES = {
    "数码配件/充电器": 24,
    "智能家电/扫地机器人": 24,
    "个护电器/电动牙刷": 24,
    "家居用品/保温杯": 24,
    "通用/售后与配送": 24,
}
EXPECTED_CALIBRATION_BUCKETS = {
    "literal": 4,
    "model": 4,
    "colloquial": 4,
    "synonym": 4,
    "category_filter": 4,
    "unanswerable": 10,
}
EXPECTED_TEST_BUCKETS = {
    "literal": 10,
    "model": 10,
    "colloquial": 10,
    "synonym": 10,
    "category_filter": 10,
    "unanswerable": 10,
}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_manifest(manifest: dict, chunks) -> list[str]:
    errors: list[str] = []
    expected_fields = {
        "manifest_version",
        "dataset",
        "purpose",
        "disclaimer",
        "corpus_fingerprint",
        "documents",
    }
    if set(manifest) != expected_fields:
        errors.append("manifest 顶层字段不符合冻结格式")
        return errors
    if manifest["manifest_version"] != 1:
        errors.append("manifest_version 必须为 1")
    if "演示" not in manifest["disclaimer"] or "不代表" not in manifest["disclaimer"]:
        errors.append("manifest 必须明确资料为演示内容且不代表真实承诺")
    fingerprint = corpus_fingerprint(chunks)
    if manifest["corpus_fingerprint"] != fingerprint:
        errors.append("manifest corpus_fingerprint 与语料不符")

    corpus_root = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    raw_documents = corpus_root["documents"]
    expected_mapping = {
        document["source_key"]: {
            "source_document": document["source_document"],
            "summary": document["summary"],
            "chunk_ids": [chunk["id"] for chunk in document["chunks"]],
            "category": document["chunks"][0]["category"],
        }
        for document in raw_documents
    }
    actual_mapping = {}
    for document in manifest["documents"]:
        if not isinstance(document, dict) or set(document) != {
            "source_key",
            "source_document",
            "summary",
            "category",
            "chunk_ids",
        }:
            errors.append("manifest document 字段不符合冻结格式")
            continue
        key = document["source_key"]
        if key in actual_mapping:
            errors.append(f"manifest source_key 重复: {key}")
        actual_mapping[key] = {
            name: document[name] for name in document if name != "source_key"
        }
    if actual_mapping != expected_mapping:
        errors.append("manifest 的 source_key/source_document/chunk ID 映射与语料不符")
    return errors


def _distribution(cases, field: str) -> dict[str, int]:
    return dict(sorted(Counter(case[field] for case in cases).items()))


def main() -> int:
    errors: list[str] = []
    try:
        chunks = load_corpus(CORPUS_PATH)
        calibration = load_cases(CALIBRATION_PATH)
        test = load_cases(TEST_PATH)
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    errors.extend(validate_corpus(chunks))
    errors.extend(validate_cases(calibration, chunks))
    errors.extend(validate_cases(test, chunks))
    errors.extend(_validate_manifest(manifest, chunks))

    category_counts = Counter(chunk.category for chunk in chunks)
    if len(chunks) != 120:
        errors.append(f"语料数量应为 120，实际为 {len(chunks)}")
    if dict(category_counts) != EXPECTED_CATEGORIES:
        errors.append(f"语料分类分布不符: {dict(category_counts)}")
    ids = sorted(chunk.id for chunk in chunks)
    if ids != list(range(910001, 910121)):
        errors.append("语料 ID 必须连续固定为 910001..910120")
    if len(calibration) != 30:
        errors.append(f"校准集数量应为 30，实际为 {len(calibration)}")
    if len(test) != 60:
        errors.append(f"正式集数量应为 60，实际为 {len(test)}")
    calibration_buckets = Counter(case["query_type"] for case in calibration)
    test_buckets = Counter(case["query_type"] for case in test)
    if dict(calibration_buckets) != EXPECTED_CALIBRATION_BUCKETS:
        errors.append(f"校准集桶分布不符: {dict(calibration_buckets)}")
    if dict(test_buckets) != EXPECTED_TEST_BUCKETS:
        errors.append(f"正式集桶分布不符: {dict(test_buckets)}")

    calibration_ids = {case["query_id"] for case in calibration}
    test_ids = {case["query_id"] for case in test}
    duplicate_ids = sorted(calibration_ids & test_ids)
    if duplicate_ids:
        errors.append(f"跨集合 query_id 重复: {duplicate_ids}")
    calibration_queries = {case["query"] for case in calibration}
    test_queries = {case["query"] for case in test}
    duplicate_queries = sorted(calibration_queries & test_queries)
    if duplicate_queries:
        errors.append(f"跨集合原话重复: {duplicate_queries}")
    for label, cases in (("校准集", calibration), ("正式集", test)):
        queries = [case["query"] for case in cases]
        if len(queries) != len(set(queries)):
            errors.append(f"{label}内存在重复原话")
    source_questions = {chunk.questions for chunk in chunks}
    copied = sorted((calibration_queries | test_queries) & source_questions)
    if copied:
        errors.append(f"评估问题直接复制 questions 字段: {copied}")

    print(
        f"corpus: {len(chunks)} chunks; "
        f"categories={dict(sorted(category_counts.items()))}"
    )
    print(
        "calibration: "
        f"{len(calibration)} cases; buckets={dict(sorted(calibration_buckets.items()))}; "
        f"difficulty={_distribution(calibration, 'difficulty')}"
    )
    print(
        "test: "
        f"{len(test)} cases; buckets={dict(sorted(test_buckets.items()))}; "
        f"difficulty={_distribution(test, 'difficulty')}"
    )
    print(
        "cross_set_duplicates: "
        f"query_ids={len(duplicate_ids)}; queries={len(duplicate_queries)}"
    )
    reference_count = sum(len(case["relevant_chunk_ids"]) for case in calibration + test)
    cross_chunk_count = sum(
        len(case["relevant_chunk_ids"]) > 1 for case in calibration + test
    )
    print(
        f"annotations: references={reference_count}; "
        f"cross_chunk_cases={cross_chunk_count}"
    )
    print(f"corpus_fingerprint: {corpus_fingerprint(chunks)}")
    for path in (CORPUS_PATH, MANIFEST_PATH, CALIBRATION_PATH, TEST_PATH):
        print(f"sha256 {path.relative_to(ROOT)}: {_file_hash(path)}")
    if errors:
        print("validation: FAILED")
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
