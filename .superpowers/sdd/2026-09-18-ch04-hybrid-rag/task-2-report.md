# Task 2 implementation report

## Status

DONE

## What was implemented

- Added a traced 120-chunk ecommerce demo corpus with fixed IDs `910001..910120`, five independent 24-entry source chapters, stable category paths, explicit source metadata, complete section paths, content types, key-clause flags, and reciprocal neighbor pointers.
- Added a separate frozen manifest mapping `source_key`, `source_document`, category, summary, and exact chunk IDs without adding source metadata to the authoritative MySQL DDL or `KnowledgeChunk` storage contract.
- Added 30 independently phrased calibration cases and 60 held-out test cases. The calibration distribution is five answerable buckets × 4 plus 10 unanswerable; the test distribution is six buckets × 10. Queries and rationales were hand composed rather than produced by numeric substitutions.
- Added strict corpus and JSONL case loaders, corpus/case validators, and stable ID/source-hash fingerprinting.
- Added a validation CLI that reports counts, category/bucket/difficulty distributions, cross-set duplicates, annotation references, file hashes, and corpus fingerprint, and exits nonzero for any inconsistency.
- Added an idempotent knowledge-only import CLI. It explicitly runs `create_schema()`, then calls `KnowledgeRepository.insert_seed()`; it does not invoke legacy FAQ seeding and does not clear existing data.
- Added a per-category assistant data review documenting source grounding, preserved conditions, unknown-item review, apparent-conflict resolution, and rewrites. It explicitly states that the review is assistant review rather than user human review or human ground truth.
- Consulted current SQLAlchemy 2.0 asyncio documentation through Context7 before using the concrete async session/engine APIs in the import path.

## TDD evidence

### RED 1 — loader and validation interfaces

Command:

```bash
.venv/bin/python -m pytest tests/test_knowledge_corpus.py -q
```

Expected failure before production implementation:

```text
ImportError while importing test module ...
ModuleNotFoundError: No module named 'app.knowledge.corpus'
1 error in 0.11s
```

The failure was expected because the new loader/validator module did not yet exist.

### GREEN 1

After implementing the strict loaders, corpus validation, and fingerprint function, the focused suite passed after correcting one test fixture's hand-written section path expectation:

```text
............                                                             [100%]
12 passed in 0.04s
```

### RED 2 — annotation target validation

Command:

```bash
.venv/bin/python -m pytest tests/test_knowledge_corpus.py -q
```

Expected failure:

```text
ImportError: cannot import name 'validate_cases' from 'app.knowledge.corpus'
1 error in 0.10s
```

The new test required missing-target and cross-category target validation before that function existed.

### GREEN 2 and final focused run

```text
.............                                                            [100%]
13 passed in 0.03s
```

The tests exercise real parsing and validation behavior with hand-checked fixtures: exact fields, source-key format, missing/extra fields, duplicate IDs, missing and nonreciprocal neighbors, incomplete section paths, case schema/answerability, duplicate query IDs, target existence/category, and fingerprint stability.

## Data validation evidence

Command:

```bash
.venv/bin/python scripts/validate_knowledge_data.py
```

Final result:

```text
corpus: 120 chunks; categories={'个护电器/电动牙刷': 24, '家居用品/保温杯': 24, '数码配件/充电器': 24, '智能家电/扫地机器人': 24, '通用/售后与配送': 24}
calibration: 30 cases; buckets={'category_filter': 4, 'colloquial': 4, 'literal': 4, 'model': 4, 'synonym': 4, 'unanswerable': 10}; difficulty={'easy': 11, 'hard': 4, 'medium': 15}
test: 60 cases; buckets={'category_filter': 10, 'colloquial': 10, 'literal': 10, 'model': 10, 'synonym': 10, 'unanswerable': 10}; difficulty={'easy': 23, 'hard': 9, 'medium': 28}
cross_set_duplicates: query_ids=0; queries=0
annotations: references=76; cross_chunk_cases=6
corpus_fingerprint: 843ba4d251f560b2ed61717f727c4fc46a34061d0a9f39f2ff098c1c3078e619
sha256 data/knowledge/ch04/chunks.json: 4915922a9912c643efd9dc655a1b5cb609a23d68e23e78e82259ee4cb4a8c9ec
sha256 data/knowledge/ch04/manifest.json: 724c6ff0641fe7e23c518660d01073c2b579ca321b6f05e6db9f46f2888c33dd
sha256 evals/ch04/calibration.jsonl: e5bf83c93712e3cd858bb22a42670ae0d90412277a1764d1f2cb8834b7c4d6c3
sha256 evals/ch04/test.jsonl: 8db5db6543b8b516902535a897a0ceaa414eb2307983af5cb5cddd21d307b87f
validation: OK
```

The assistant reviewed all 90 annotations against their cited chunks during authoring and recorded per-category results in `dev-notes/ch04-data-evaluation.md`. This is assistant review, not human ground truth.

## Actual configured MySQL import evidence

The sandbox initially denied localhost access. The same authorized command was rerun with localhost access, without changing the configured DSN or displaying credentials.

First run:

```text
knowledge import: corpus=120; matched=120; total_before=0; total_after=120
corpus_fingerprint: 843ba4d251f560b2ed61717f727c4fc46a34061d0a9f39f2ff098c1c3078e619
```

Second run, without clearing data:

```text
knowledge import: corpus=120; matched=120; total_before=120; total_after=120
corpus_fingerprint: 843ba4d251f560b2ed61717f727c4fc46a34061d0a9f39f2ff098c1c3078e619
```

This proves the repository import is idempotent for the frozen corpus and preserved existing data.

## Full verification

The first sandboxed full-suite attempt could not bind ephemeral localhost sockets or connect to the isolated test MySQL port 13307. Rerunning the unchanged suite with localhost access produced:

```text
268 passed, 1 warning in 26.22s
```

The sole warning is an existing Starlette/AnyIO `BlockingPortal` deprecation warning. `git diff --check` was clean before commit. No process was started or stopped on ports 8000 or 8001.

## Files changed

- `app/knowledge/corpus.py`
- `data/knowledge/ch04/chunks.json`
- `data/knowledge/ch04/manifest.json`
- `evals/ch04/calibration.jsonl`
- `evals/ch04/test.jsonl`
- `scripts/init_knowledge.py`
- `scripts/validate_knowledge_data.py`
- `dev-notes/ch04-data-evaluation.md`
- `tests/test_knowledge_corpus.py`

Implementation commit: `70d2c91 data: add traced ecommerce corpus and held-out evaluation cases`

## Self-review findings

- Rechecked the Task 2 brief and approved spec sections 2 and 11 against the final files: counts, IDs, category paths, bucket isolation, no direct question copying, cross-chunk coverage, unanswerable shape, manifest separation, and import behavior are present.
- Confirmed source metadata is consumed only while parsing and is not added to `KnowledgeChunk` or the authoritative MySQL schema.
- Confirmed every chapter has exactly one head/tail and reciprocal internal neighbor pointers.
- Confirmed the import script calls no legacy seed and performs no delete/truncate operation.
- Corrected trailing blank lines found by `git diff --check`, then reran the focused tests and data validator.

## Concerns

- The full test suite retains one pre-existing dependency deprecation warning from Starlette/AnyIO; it is unrelated to Task 2.
- The first real import emitted a MySQL integer display-width deprecation warning during schema reflection/creation; the second import was clean and both imports succeeded.

## Fix round 1 — annotation review corrections

The accepted review findings were verified against the frozen source rows and corrected as data-only changes. No corpus row, manifest mapping, loader, validator, or database content changed.

### Reviewed cases before and after

1. `test-literal-06`
   - Before: `relevant_chunk_ids=[910057]`, rationale claimed `910057` contained the three-month brush-head replacement rule.
   - After: `relevant_chunk_ids=[910058]`; the answer remains “建议每三个月更换；刷毛散开、褪色或变形时应提前更换。” because `910058` is the actual supporting row.
2. `test-literal-02`
   - Before: “不附送。标准包装只有充电器和说明卡，套装版所含线材会在商品页标注。” The word “只有” incorrectly presented the cited row as a complete packaging list and conflicted with the separate warranty-card row.
   - After: “不附送。C65-Pro 标准包装不含充电线；套装版所含线材以商品页标注为准。” This answers the cable question without inventing a complete list.
3. `test-model-09`
   - Before: a general invoice-reissue question labeled `model`, with an ambiguous comparison between “one month” and 30 days.
   - After: “C65 普通款有几个充电接口，单口最高能输出多少瓦？” citing `910005`, with answer “C65 只有一个 USB-C 口，最高输出 65W。”
4. `test-model-10`
   - Before: a general exchange-warranty question labeled `model`.
   - After: “R8-Max 在硬质地面开标准档最长能扫多久，换强力档会一样吗？” citing `910039`, with answer “标准档在硬质地面上最长约 150 分钟；强力档会缩短续航。”

The two replacement model queries are distinct from corpus `questions`, calibration queries, and every other held-out query. The formal test set remains six mutually exclusive buckets of ten cases.

### Annotated sample review

Command:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from app.knowledge.corpus import load_cases, load_corpus
chunks={c.id:c for c in load_corpus(Path('data/knowledge/ch04/chunks.json'))}
cases={c['query_id']:c for c in load_cases(Path('evals/ch04/test.jsonl'))}
for query_id in ('test-literal-02','test-literal-06','test-model-09','test-model-10'):
    case=cases[query_id]
    print(f"{query_id}: {case['query']}")
    print(f"  answer: {case['reference_answer']}")
    for chunk_id in case['relevant_chunk_ids']:
        print(f"  source {chunk_id}: {chunks[chunk_id].answer}")
PY
```

Output:

```text
test-literal-02: C65-Pro 标准版随盒附送 USB-C 线吗？
  answer: 不附送。C65-Pro 标准包装不含充电线；套装版所含线材以商品页标注为准。
  source 910006: C65-Pro 标准包装含充电器和说明卡，不含充电线；套装版会在商品页明确标注所含线材。
test-literal-06: T3 系列刷头建议使用几个月后更换？
  answer: 建议每三个月更换；刷毛散开、褪色或变形时应提前更换。
  source 910058: 建议每三个月更换，刷毛散开、褪色或变形时应提前更换。
test-model-09: C65 普通款有几个充电接口，单口最高能输出多少瓦？
  answer: C65 只有一个 USB-C 口，最高输出 65W。
  source 910005: C65 只有一个最高 65W 的 USB-C 口；C65-Pro 有两个 USB-C 口和一个 USB-A 口，并支持 PPS。
test-model-10: R8-Max 在硬质地面开标准档最长能扫多久，换强力档会一样吗？
  answer: 标准档在硬质地面上最长约 150 分钟；强力档会缩短续航。
  source 910039: 标准档在硬质地面上最长约 150 分钟；强力档、拖地和复杂路线会缩短续航。
```

### Fix-round verification

Command:

```bash
.venv/bin/python scripts/validate_knowledge_data.py
```

Output:

```text
corpus: 120 chunks; categories={'个护电器/电动牙刷': 24, '家居用品/保温杯': 24, '数码配件/充电器': 24, '智能家电/扫地机器人': 24, '通用/售后与配送': 24}
calibration: 30 cases; buckets={'category_filter': 4, 'colloquial': 4, 'literal': 4, 'model': 4, 'synonym': 4, 'unanswerable': 10}; difficulty={'easy': 11, 'hard': 4, 'medium': 15}
test: 60 cases; buckets={'category_filter': 10, 'colloquial': 10, 'literal': 10, 'model': 10, 'synonym': 10, 'unanswerable': 10}; difficulty={'easy': 23, 'hard': 9, 'medium': 28}
cross_set_duplicates: query_ids=0; queries=0
annotations: references=75; cross_chunk_cases=5
corpus_fingerprint: 843ba4d251f560b2ed61717f727c4fc46a34061d0a9f39f2ff098c1c3078e619
sha256 data/knowledge/ch04/chunks.json: 4915922a9912c643efd9dc655a1b5cb609a23d68e23e78e82259ee4cb4a8c9ec
sha256 data/knowledge/ch04/manifest.json: 724c6ff0641fe7e23c518660d01073c2b579ca321b6f05e6db9f46f2888c33dd
sha256 evals/ch04/calibration.jsonl: e5bf83c93712e3cd858bb22a42670ae0d90412277a1764d1f2cb8834b7c4d6c3
sha256 evals/ch04/test.jsonl: 13a4a5af0896915e00ee11ed7446ff806a388a2da2b85bb3a0ecbd95a020cd00
validation: OK
```

The corpus fingerprint and source-file hashes remain unchanged. The formal-test hash is now frozen at `13a4a5af0896915e00ee11ed7446ff806a388a2da2b85bb3a0ecbd95a020cd00`. This remains an assistant annotation review rather than human ground truth.
