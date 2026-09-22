# Chapter 4 evaluation status

## Current status — 2026-09-22

A **production-ready calibration is now available; all 240 four-strategy attempts have finished with one judge technical failure**. The user authorized the demonstration-data transfer to DeepSeek (`api.deepseek.com`); the measured endpoint used model `deepseek-flash`, thinking disabled, zero automatic retries. No real customer data was included.

| Run | Coverage | Result |
| --- | --- | --- |
| `evals/reports/ch04/2026-09-22-calibration` | 30/30 | Incomplete: two judge responses confused display reference numbers with source chunk IDs. |
| `evals/reports/ch04/2026-09-22-calibration-v2` | 30/30 | Incomplete: one query-normalization `invalid_response` (`cal-literal-01`); judge namespace fix already applied. |
| `evals/reports/ch04/2026-09-22-calibration-diagnostic` | 30/30 | Complete, production-ready, zero technical failures; one unknown false accept in ten, meeting the configured upper bound. |

The historical second run measured Recall@5/10/50 = **1.0**, MRR@50 = **0.9167**, retained evidence coverage = **0.95**, answerable answer rate = **0.80**, unknown refusal = **0.90** (one false accept in ten), and model-judged Faithfulness = **0.988235 over 17 valid judgments**. These are calibration-split observations for its executed retrieval strategy, **not** four-strategy comparison numbers. That second run could not accept threshold `0.5767565140084168` because of its technical failure; the later diagnostic calibration below has accepted the same threshold. Provider token usage was not captured by this evaluator and is unknown.

The false accept `cal-none-04` infers a V500-Lock 750 ml answer from 500 ml evidence. Four answerable cases were refused (`cal-model-02`, `cal-colloquial-04`, `cal-synonym-03`, `cal-category-01`). `cal-colloquial-01` scored 0.8 in the model faithfulness judgment, including an unsupported charger recommendation. These quality defects remain recorded; labels and source material were not altered to hide them.

The judge Prompt fix (`606a31e`) passed four fixed real probes and independent scoped review. One later diagnostic of the normalization case returned valid JSON in one request (400 tokens), but does not explain or clear the original failure: its raw failed provider response was unavailable. The instrumented 30-case follow-up has now completed: all 30 raw normalization observations are valid, so the old format error did not recur. This remains non-reproduction evidence, not an identified normalization fix.

The 60-query × four-strategy comparison has finished all **240 attempts** in `evals/reports/ch04/2026-09-22-comparison`. Its manifest remains **incomplete** because one judge response failed validation after successful generation (`test-model-04/hybrid`: supported=true with empty source IDs). Original answer and raw judge response are retained; its Faithfulness is null. Fixed assistant source review now covers 12 preselected queries × four strategies plus seven additional failure/low-score records; review is complete (see `manual-review.md` in that run directory). No labels were changed and no failed row was retried.

| Strategy | Recall@5 | Recall@10 | Recall@50 | MRR@50 | Answer rate (50 answerable) | Model Faithfulness (valid n) | Technical failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| dense | .95 | 1.00 | 1.00 | .9060 | .90 | .9796 (45) | 0 |
| BM25 | .88 | .98 | .98 | .7666 | .76 | .9868 (38) | 0 |
| hybrid | .97 | 1.00 | 1.00 | .8556 | .90 | .9848 (44) | 1 |
| hybrid + rerank | .99 | 1.00 | 1.00 | .9767 | .82 | .9939 (41) | 0 |

All 60 test queries specify a category (13 charger, 13 vacuum, 12 toothbrush, 12 cup, 10 general-policy cases). Thus these retrieval scores describe category-filtered search, not the chat page’s default all-category search. All four strategies refused the same ten unknown test queries. Only hybrid_rerank uses the frozen score threshold, so answer-rate differences are not an isolated causal estimate of reranking. Its mean local rerank time was 16.47 s and total time 19.13 s. The full run took 1703.9 s, started at commit b5ac0e1, and normalization accepted 48/60 proposals with 12 protected-term fallbacks. BM25's model bucket has Recall@10/50=1.0; `test-model-01` specifically retrieved C65-Pro chunks 910002/910012 at ranks 1/6. The model judge is not a human gold standard; no provider token total was captured.


Latest diagnostic calibration: Recall@5/10/50 1.0, MRR@50 0.9167, context coverage 0.95, answerable answer rate 0.90, unknown refusal 0.90, Faithfulness 1.0 over 19 valid same-model judgments. `cal-none-04` is still an unknown false accept despite a perfect judge score; `cal-colloquial-04` and `cal-synonym-03` are false refusals. The gate accepts the threshold 0.5767565140084168 at the maximum allowed false-accept rate of 10%; it does not certify perfect knowledge coverage. Runtime 531.5 s, run-start commit d2cd025, configuration SHA 50be6aecab8367ee364e1cdbf70e2dd48b2ce534ebd1fec14e5bdcb4b68c8d70. Accepted normalization proposals: 21/30; protected-term fallback: 9/30. No provider token total is claimed.

Assistant source review completed all 55 records and 40 originals: the fixed 48 contained 39 supported answers, 8 correct refusals and 1 incorrect refusal (`test-category-10/hybrid_rerank`, chunk 910114 discarded by the Chapter 4 threshold). Seven follow-ups identified one judge under-score, one technically invalid judge response for a supported answer, and five answers whose main claims were correct but included small unsupported additions. Two judge semantic inconsistencies were recorded. These findings do not rewrite the formal metrics and are not a human gold standard.

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

- Completed: the 30-case diagnostic calibration is production-ready with zero
  technical failures and unknown false-accept rate of 10%.
- All 60 formal cases × four strategies were attempted without `--limit`; retain
  the one judge technical failure and the incomplete manifest, not a zero-error pass.
- Completed: independent assistant reviewed 12 preselected queries × four
  strategies plus seven follow-ups against 40 originals; full verdicts and
  limits are in the comparison manual-review.md.
- Use the real normalization observations from calibration to close the pending
  Task 5 quality check. Use overlapping real evidence/answer artifacts for Task
  6 only where source conditions match; run only the missing probes.
- Completed: installed the accepted artifact at `.cache/ch04/calibration.json`
  and passed the existing runtime loader against fixed-model hashes, knowledge
  Prompt hashes and the previously database-verified corpus fingerprint. This
  local check made no model request; app cutover is a separate gate.
