# Chapter 4 evaluation status

## Current status — 2026-09-22

Real evaluations now exist, but there is **no accepted calibration or four-strategy comparison report yet**. The user authorized the demonstration-data transfer to DeepSeek (`api.deepseek.com`); the measured endpoint used model `deepseek-flash`, thinking disabled, zero automatic retries. No real customer data was included.

| Run | Coverage | Result |
| --- | --- | --- |
| `evals/reports/ch04/2026-09-22-calibration` | 30/30 | Incomplete: two judge responses confused display reference numbers with source chunk IDs. |
| `evals/reports/ch04/2026-09-22-calibration-v2` | 30/30 | Incomplete: one query-normalization `invalid_response` (`cal-literal-01`); judge namespace fix already applied. |

The second run measured Recall@5/10/50 = **1.0**, MRR@50 = **0.9167**, retained evidence coverage = **0.95**, answerable answer rate = **0.80**, unknown refusal = **0.90** (one false accept in ten), and model-judged Faithfulness = **0.988235 over 17 valid judgments**. These are calibration-split observations for its executed retrieval strategy, **not** four-strategy comparison numbers. The candidate threshold `0.5767565140084168` remains unaccepted because a technical failure is present. Provider token usage was not captured by this evaluator and is unknown.

The false accept `cal-none-04` infers a V500-Lock 750 ml answer from 500 ml evidence. Four answerable cases were refused (`cal-model-02`, `cal-colloquial-04`, `cal-synonym-03`, `cal-category-01`). `cal-colloquial-01` scored 0.8 in the model faithfulness judgment, including an unsupported charger recommendation. These quality defects remain recorded; labels and source material were not altered to hide them.

The judge Prompt fix (`606a31e`) passed four fixed real probes and independent scoped review. One later diagnostic of the normalization case returned valid JSON in one request (400 tokens), but does not explain or clear the original failure: its raw failed provider response was unavailable. The prepared instrumented calibration follow-up has not run.

The 60-query × four-strategy comparison has **0/240 executions** because no calibration has passed its gate. Twelve manual-review queries were preselected (two per bucket); source review remains pending actual comparison artifacts. No synthetic test metrics are presented as real quality results.

## Evaluation commands (comparison requires accepted calibration)

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
