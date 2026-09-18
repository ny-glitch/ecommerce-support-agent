# Chapter 4 evaluation status

## Status

The evaluator, calibration CLI, resumable artifacts, metrics, report generation,
and controlled local tests are implemented. Real calibration and comparison are
pending because `external-evaluation-authorization.md` is still
`PENDING_USER_REPLY`. No Chapter 4 question, excerpt, generated answer, or judge
request was sent to the configured DeepSeek endpoint during this task.

There are no production quality numbers or production calibration artifact yet.
Fixtures in the automated tests are synthetic and must not be cited as product
quality evidence.

## Commands to run after authorization

```bash
.venv/bin/python scripts/evaluate_knowledge.py calibrate \
  --cases evals/ch04/calibration.jsonl \
  --output-dir evals/reports/ch04/calibration

.venv/bin/python scripts/evaluate_knowledge.py compare \
  --cases evals/ch04/test.jsonl \
  --calibration evals/reports/ch04/calibration/calibration.json \
  --output-dir evals/reports/ch04/comparison
```

The calibration command requires the exact 30-case split: four cases in each
of the five answerable buckets and ten unanswerable cases. The comparison
command requires ten cases in each of the six main buckets. `--limit` is a
comparison-only smoke option; its manifest status and report are labelled
`smoke` and cannot be presented as a full comparison.

## Artifact contract

Each run writes `manifest.json`, `normalizations.json`, `results.jsonl`,
`report.md`, and `calibration.json`. Per-query atomic cache files support a
same-configuration resume. A changed configuration hash is rejected. The
manifest stores only an explicit safe allowlist: code and dependency versions,
model revisions, hashes, token limits, threshold, strategy, and run mode. It
does not store the API key, DSN, base URL, or full settings object.

The CLI validates the local file corpus against the read-only SQL corpus before
the run and checks the SQL corpus fingerprint again after the run. A change
during evaluation marks the materialized report `invalid`. `compare` validates
the frozen Task 8 `CalibrationArtifact` against current corpus, prompt, and
model provenance and never recalibrates from the formal test set.

The Markdown report includes full denominators, Recall@5/10/50, MRR@50,
retrievable count, actual retained-context coverage, faithfulness and its
sample count, answer/refusal/wrong-answer rates, generation count, failures,
and retrieval/rerank/total timing by strategy, main bucket, and difficulty.
Faithfulness is a separate structured call to the same configured model; it is
explicitly described as model judgment, not a human gold standard. The raw
structured judge response and validated atomic claims remain in per-case
artifacts.

## Remaining real gates

- Run the 30-case calibration and freeze a production-ready artifact with zero
  technical failures and unknown false-accept rate at or below 10%.
- Run all 60 formal cases across all four strategies with no `--limit`.
- Review at least 12 case artifacts, at least two per main bucket, against the
  source text. Record the reviewer as assistant review unless a human performs
  it.
- Use the real normalization observations from calibration to close the pending
  Task 5 quality check. Use overlapping real evidence/answer artifacts for Task
  6 only where source conditions match; run only the missing probes.
- Copy the accepted production calibration to the configured
  `knowledge_calibration_path` and verify startup against its provenance.
