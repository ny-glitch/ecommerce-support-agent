# Task 6 report: evidence budgets, one-shot sufficiency, and controlled refusal

Status: `DONE_WITH_CONCERNS`. The production implementation, focused TDD
suite, mock-transport request validation, and complete required offline/local
suite are complete. Real DeepSeek prompt-quality probes remain pending explicit
external data-transfer authorization and are not reported as passed.

## Implementation

- Added immutable evidence contracts. `Citation.number` is restricted to
  1..10; `EvidenceAssessment` restricts the decision codes, reason length, and
  supporting-ID count; `KnowledgeDecision.to_payload()` validates the compact
  UTF-8 serialization against the independent 48,000-byte tool-result limit.
- Citation numbers are assigned in relevance order before source layout.
  `edge_order` places the strongest source first and the second-strongest at
  the final edge. Citations contain only current MySQL chunk snapshots, their
  content hashes, stable source URLs, and reranker scores; the LLM never fills
  provenance fields.
- `EvidenceBudget` considers at most the top ten ranked chunks and removes only
  complete lowest-ranked chunks. It first lets the existing context builder
  remove old complete turns. A candidate must fit both the actual assessment
  request and the actual final-generation message structure. The latter uses
  the real knowledge system prompt, current user message, supplied assistant
  tool call, and a valid `ToolMessage`, with 4,096 UTF-8 bytes reserved for the
  bounded assessment/result fields. The 48,000-byte result limit is checked
  separately from the model input estimate. No configured context limit is
  raised.
- Added a shared assessment-message builder used by budgeting and invocation.
  It includes the schema, original utterance, normalized question, and the
  selected source snapshots, and includes no history. `KnowledgeGateway`
  builds one JSON-mode structured assessor from the existing shared model and
  binds the protected `chat_extra_body` at runnable construction. It rechecks
  finish reason, raw content, raw JSON schema validation, parsing error, and
  parsed/raw equality. Transport failures and invalid outputs use controlled
  `KNOWLEDGE_UNAVAILABLE` or `EVIDENCE_ASSESSMENT_ERROR` errors.
- Added the fixed one-pass pipeline: one normalizer call, one
  `hybrid_rerank` retrieval, and at most one assessment call. Zero hits, stale
  results, low relevance, and no-fit context return their exact fixed refusal
  without assessment. Supporting chunk IDs must be a non-empty subset of the
  selected evidence for a supported decision; unknown IDs or internally
  inconsistent assessments are technical errors and never default to an
  answer. Baseline evaluations can reuse `decide_evidence(...,
  threshold=None)`.
- Pipeline progress is restricted to `normalizing`, `retrieving`,
  `reranking`, and `checking_evidence`. The absolute deadline covers progress
  callbacks and the complete pipeline. The pipeline has no database writes or
  agent loop.
- Added separate assessment and grounded-answer prompt templates. Both treat
  source text as untrusted data. The assessment prompt explicitly covers exact
  model/condition matching, negation, conflicts, multi-source questions,
  conditional arrival timing, ambiguous current questions, and embedded
  instructions. The answer prompt requires valid current-turn citation
  numbers and exact refusal passthrough.

## RED evidence

Initial required focused run after adding tests:

```text
$ .venv/bin/python -m pytest tests/test_knowledge_evidence.py \
  tests/test_knowledge_pipeline.py -q
E ImportError: cannot import name 'Citation' from 'app.knowledge.contracts'
E ImportError: cannot import name 'EvidenceAssessment' from 'app.knowledge.contracts'
2 errors in 0.61s
```

The first implementation run reached 16 passing tests. Its three failures
showed that the test fixtures' context limits were smaller than the explicit
4,096-byte reserve, so those fixtures were corrected to exercise the intended
all-fit and partial-fit boundaries without changing production behavior.

A separate deadline regression was then added and observed failing because a
slow progress callback consumed the deadline before `decide_evidence` returned
a service error:

```text
$ .venv/bin/python -m pytest \
  tests/test_knowledge_pipeline.py::test_pipeline_deadline_includes_progress_callbacks -q
F [100%]
E app.errors.ServiceError: 知识服务暂时不可用，请稍后重试
1 failed in 1.03s
```

This drove an outer absolute timeout around the complete pipeline, including
all progress callbacks.

## GREEN and covering commands

Focused GREEN after the final deadline change:

```text
$ .venv/bin/python -m pytest tests/test_knowledge_evidence.py \
  tests/test_knowledge_pipeline.py -q
.................... [100%]
20 passed in 0.81s
```

The first full required command inside the restricted sandbox was not a valid
integration result: localhost access was denied for MySQL 13307, Milvus, and
ephemeral ASGI sockets. It completed with 292 passed, 17 failed, and 49 errors;
all failures/errors showed `PermissionError: [Errno 1] Operation not
permitted`. No Task 6 focused test failed.

The identical command was rerun with localhost permission while Hugging Face
and Transformers remained offline:

```text
$ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest \
  --require-milvus --require-mysql --require-local-models -q
........................................................................ [ 20%]
........................................................................ [ 40%]
........................................................................ [ 60%]
........................................................................ [ 80%]
......................................................................   [100%]
358 passed, 1 warning in 67.27s (0:01:07)
```

The warning is the existing Starlette deprecation warning for
`anyio.abc.BlockingPortal`. `git diff --check` and `compileall` for all Task 6
Python files also exited 0.

## API provenance

The supplied fresh Context7 record documents `PromptTemplate.from_template`
and structured-output `include_raw`. The supplied installed-source note and
mock transport probe establish that `ChatOpenAI.with_structured_output`
accepts `method="json_mode"`, `include_raw=True`, and extra binding kwargs,
and that protected request controls must be bound while constructing the
structured runnable rather than only passed to its outer `ainvoke`.

Read-only installed signature confirmation before implementation:

```text
langchain_core 1.6.3
PromptTemplate.from_template(template, *, template_format='f-string',
                             partial_variables=None, **kwargs)
langchain_openai 1.6.2
ChatOpenAI.with_structured_output(self, schema=None, *, method='json_schema',
                                  include_raw=False, strict=None, tools=None,
                                  **kwargs)
AIMessage(..., tool_calls=..., invalid_tool_calls=...)
ToolMessage(..., tool_call_id: str, name: str | None = None, ...)
```

The focused assessment transport test uses real installed `ChatOpenAI` request
construction with `httpx.MockTransport` and a fake key. It confirms one request
with `response_format={"type":"json_object"}`, disabled thinking, the protected
output-token field, and exactly system/user messages containing the original
question, normalized question, and selected source. It makes no network call.

## Files

Task-owned implementation and tests:

- `app/context.py`
- `app/knowledge/contracts.py`
- `app/knowledge/evidence.py`
- `app/knowledge/gateway.py`
- `app/knowledge/pipeline.py`
- `app/prompts.py`
- `app/prompts/evidence_assessment.txt`
- `app/prompts/knowledge_answer.txt`
- `tests/test_knowledge_evidence.py`
- `tests/test_knowledge_pipeline.py`

This report is also Task 6-owned. The root-owned `app/web/index.html`, the
controller-owned dev notes/plan, corpus, calibration labels, `.env`, service
configuration, and ports were not edited or staged. Port 8001 was preserved;
port 8000 was never used.

## Self-review

- Mutation check: wrong citation bounds, absent/unknown citation numbers,
  relevance numbering after edge layout, partial source truncation, failure to
  reserve assessment bytes, assessment on an early refusal, repeated
  assessment, unsupported/unknown IDs, inconsistent sufficient/reason states,
  non-default retrieval strategy, extra/missing progress stages, and an
  unbounded progress callback are each covered by an observable test.
- Model-input budgeting uses the same assessment-message helper as transport
  invocation and the same final prompt/context builder that Task 7 can reuse.
  It does not equate the 48,000-byte tool-result ceiling with model context.
- Source numbering stays attached to relevance rank while layout changes only
  position. Dropping always removes the current relevance tail and never
  slices source text.
- The assessor receives only the current original and normalized questions;
  history is available solely to the separately budgeted final-generation
  context.
- Technical failures are not converted to a refusal or a guessed answer.

## Pending real prompt-quality gate and resumable plan

The controller's authorization record remains `PENDING_USER_REPLY` after
automatic approval rejected sending demo questions and source excerpts to the
configured DeepSeek endpoint. No real external LLM call, retry, proxy, or
alternate route was attempted. Mocked structured outputs validate code and
transport shape only; they are not claimed as real prompt-quality results.

After the controller records explicit authorization, resume from
`task-6-probe-design.md` and the frozen calibration corpus. For every listed
case, invoke this same `KnowledgeGateway.assess` with the explicit original
question, normalized question, and exact selected source snapshots; record
provided source IDs, expected label, actual structured decision, and a short
reason summary. Run the sufficient cases through the same answer prompt and
validate citations with `validate_citation_numbers`. Include no reference
answers in evaluator inputs and no real customer/order data. Record external
transport or budget failures as technical failures rather than prompt passes.
The required set includes no evidence, exact and nearby models, negation,
embedded instructions, conditional arrival timing, full and missing
cross-chunk evidence, unresolved referent, and no-fit context. Task 9 remains
responsible for the full four-strategy metrics.

## Fix round 1: authoritative tool name and cropped-assessment regression

Base commit: `0590a1b2d93f07c066c5be262706668bdea99b6a`.

The controller-verified review had exactly two Important findings:

1. The final answer prompt named `query_knowledge`, but the binding tool and
   existing registry use `query_faq`.
2. The purported cropped-assessment regression supplied one already-fitting
   source, so it did not prove that assessment receives the post-budget set.

The prompt now names `query_faq`, and both evidence/pipeline call fixtures use
that authoritative name. This is a direct contract correction; no brittle
source-text assertion was added and no prompt-quality claim is inferred.

The assessment regression now supplies three ranked chunks with 650-character
answers under a 12,000-token configured context. The real `EvidenceBudget`
removes the relevance tail and retains chunk IDs `910001` and `910002`.
Assertions verify that the decision and the single gateway assessment receive
exactly those two IDs and that dropped ID `910003` is absent.

Because production pruning was already correct and the defect was missing
coverage, the new regression initially passed against the base implementation:

```text
$ .venv/bin/python -m pytest \
  tests/test_knowledge_pipeline.py::test_assessment_runs_once_on_the_actual_cropped_sources -q
. [100%]
1 passed in 0.69s
```

Mutation evidence then temporarily changed `EvidenceBudget._fits` to always
return true, disabling pruning. The regression failed on the exact behavioral
boundary:

```text
$ .venv/bin/python -m pytest \
  tests/test_knowledge_pipeline.py::test_assessment_runs_once_on_the_actual_cropped_sources -q
F [100%]
E assert (910001, 910003, 910002) == (910001, 910002)
1 failed in 0.57s
```

The mutation was fully reverted before covering verification. Final focused
command:

```text
$ .venv/bin/python -m pytest tests/test_knowledge_evidence.py \
  tests/test_knowledge_pipeline.py -q
.................... [100%]
20 passed in 0.85s
```

Self-review confirmed the fixture uses real budget selection rather than a
stubbed/cropped source list, the input has strictly more IDs than the retained
set, the edge-ordered retained IDs are asserted exactly, the dropped ID cannot
reach assessment, and assessment call count remains one. The deferred minor
empty-schema accounting concern in `app/context.py` was not changed. Per the
fix brief, the unchanged full integration suite was not repeated. Root-owned
frontend/dev-note/plan changes remain untouched. No external LLM call was
made, and the real prompt-quality gate remains pending explicit authorization.
