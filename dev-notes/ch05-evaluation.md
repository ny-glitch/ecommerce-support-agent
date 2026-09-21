# Current status — 2026-09-22

Implementation and scoped backend reviews are complete at `d2b8799`. The final required local suite is **807 passed, 1 known deprecation warning (84.32 s)**, using real isolated MySQL, PostgreSQL, Milvus and cached local models. Real-model acceptance is **not complete**; results below include failures.

The user authorized demonstration questions, necessary knowledge, simulated tool results and generated answers to DeepSeek (`api.deepseek.com`). The actual configured model is `deepseek-flash`, thinking disabled, with zero automatic retries. A later, separate three-case Agent-control diagnostic was rejected by automatic approval review before process creation; a specific follow-up authorization question is pending. Zero requests from that diagnostic were sent. Original failed replies were not retained, so their cause remains unproved.

| Actual run | Coverage and execution status | Selected measured results |
| --- | --- | --- |
| `evals/reports/ch05/2026-09-22-intents` | 35/35 attempted, incomplete; 4 technical failures | Classification and route 31/35 (88.57%); business flag 29/35; business-to-knowledge misroutes 0/15. 90 calls / 67,017 provider tokens. |
| `evals/reports/ch05/2026-09-22-evidence` | 12/12, complete; zero technical failures | Classification, route and sufficiency 12/12; refusal 3/3; business flag 11/12; exact support IDs 10/12; exact citation IDs 7/9. 43 calls / 45,250 tokens. |
| `evals/reports/ch05/2026-09-22-assessor` | 2/2, complete | Both injection/conflicting-source cases correctly insufficient with no supporting IDs. 2 calls / 1,437 tokens. |

Total for these three runs: **135 calls / 113,704 provider tokens**. `complete` means execution coverage, not perfect answer quality. Structural ID metrics do not prove semantic faithfulness. Fixed labels were not edited to improve scores.

`intent-009`, `intent-010` and `intent-022` failed with `AGENT_CONTROL_ERROR`; `intent-016` reached its conservative input-token guard and returned the bounded fallback, but is still counted as an evaluation failure (`INPUT_TOO_LONG`). Cases 009/010/022 remain awaiting the blocked diagnostic. The successful control probes in separate work do not replace these failures.

Actual browser checks on isolated port 8002 have verified policy retrieval/clickable source text, unknown refusal and a low-confidence pool row, complaint actions without automatic execution, cancellation before ticket creation, one confirmed ticket, repeated confirmation before/after graceful restart without duplicates, front-end feedback locking, and the logistics tool badge. The policy save conflict observed earlier was fixed and passed a fresh browser re-test. PostgreSQL retained the completed policy checkpoint after restart. Multi-step browsing, formal quality gates, the final 8001 cutover and finish remain separate pending work. Artifacts: `evals/reports/ch05/2026-09-22-acceptance/`.

The sections below retain the earlier local-only chronology and dataset rationale. Their pending-authorization and not-yet-run statements describe those earlier stages, not the current status above.

# Chapter 5 intent evaluation status

## Scope and authorization boundary

`evals/ch05/intents.jsonl` contains 35 manually labelled intent cases: five for each of the seven fixed labels. The case file is local evaluation input only and has not been sent to DeepSeek or any other model endpoint. External model-data authorization remains `PENDING_USER_REPLY`.

The MockTransport tests prove request and response protocol behavior: one classification request, JSON mode, strict application-side DTO validation, disabled thinking controls, zero retries in the shared owner, and safe failure handling. They do not measure prompt accuracy. No classification accuracy, confusion matrix, or model-quality score is claimed in Task 4; that real prompt gate belongs to Task 12 after explicit authorization.

## History wire contract

Every `history` value is a JSON list. Empty history is `[]`. Non-empty history reuses Task 2's completed-turn wire exactly:

```json
[
  {
    "turn_id": "stable-turn-id",
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]
```

Only structurally complete turns are present. The future evaluator must load this through Task 2's `load_turns` contract rather than inventing a second history format. The current question remains unchanged and separate from history.

## Manual label review

The review applies these rules: greetings do not override a business request; explicit complaints stay complaint; return policy and order-specific eligibility stay return/refund so the knowledge gate is preserved; repair/exchange progress is after-sales; `needs_business_data` marks the need for concrete order, logistics, current price, or stock facts and never changes the fixed route.

| ID | Label | Business data | Manual rationale |
| --- | --- | --- | --- |
| intent-001 | logistics | yes | Greeting plus order-specific delivery location; business content wins. |
| intent-002 | logistics | yes | Current parcel location requires a logistics lookup; missing ID remains missing. |
| intent-003 | logistics | yes | Explicit order delivery-trace request. |
| intent-004 | logistics | yes | Current signed/unsigned parcel state is logistics data. |
| intent-005 | logistics | yes | Completed history identifies the prior order context; current delivery ETA remains logistics. |
| intent-006 | order | yes | Current state of a named order. |
| intent-007 | order | yes | Cancellation concerns the user's current order; no ID is invented. |
| intent-008 | order | yes | Contents of a named order require order data. |
| intent-009 | order | yes | Current payment state is an order fact. |
| intent-010 | order | yes | Completed history supplies the referenced order; current processing state is order data. |
| intent-011 | product | no | Static model/protocol compatibility requires knowledge evidence. |
| intent-012 | product | no | Static usage instructions require knowledge evidence. |
| intent-013 | product | yes | Current stock requires product business data after the knowledge policy path where applicable. |
| intent-014 | product | yes | Combined current price and static protocol question stays product and preserves the knowledge gate. |
| intent-015 | product | no | History resolves the product model; laptop compatibility is static product knowledge. |
| intent-016 | return_refund | yes | Named order eligibility must first pass return-policy evidence, then use order facts. |
| intent-017 | return_refund | no | Pure return-policy question. |
| intent-018 | return_refund | no | General refund timing is policy, not a named refund status. |
| intent-019 | return_refund | yes | Status of a named return application is business data. |
| intent-020 | return_refund | yes | History supplies the order and condition; eligibility still preserves the policy gate before order facts. |
| intent-021 | after_sales | no | Generic defective-on-arrival after-sales request has no current record lookup yet. |
| intent-022 | after_sales | yes | Current progress of a named repair record. |
| intent-023 | after_sales | yes | Current exchange progress requires business data. |
| intent-024 | after_sales | no | Static warranty/repair policy, classified by after-sales semantics. |
| intent-025 | after_sales | no | History supplies the defect and the user requests exchange handling; no current record fact is requested. |
| intent-026 | complaint | no | Explicit complaint takes the fixed complaint exit. |
| intent-027 | complaint | no | Explicit service-attitude complaint. |
| intent-028 | complaint | yes | Explicit complaint remains complaint; the named order fact sets the flag but cannot reroute it to Agent. |
| intent-029 | complaint | no | Explicit request for the complaint channel. |
| intent-030 | complaint | no | History gives context, while the current utterance explicitly asks to complain. |
| intent-031 | chitchat | no | Assistant identity question without a business task. |
| intent-032 | chitchat | no | Greeting only. |
| intent-033 | chitchat | no | Thanks only. |
| intent-034 | chitchat | no | Non-business conversational request. |
| intent-035 | chitchat | no | Non-business small talk. |

All 35 rows were checked for the exact five-field shape (`id`, `question`, `history`, `expected_intent`, `needs_business_data`), valid strict DTO values, unique IDs, and five cases per label. This is a human label review, not a review of model predictions.

## Evidence and routing labels — corrected corpus-grounded set

The first Task 5 version embedded synthetic 510x/520x `sources` in the formal rows. The real workflow runner never receives those values: it receives only `question`, `history`, and `category`, then retrieves from `data/knowledge/ch04/chunks.json`. The formal 12 rows now use only those runner inputs and expected labels grounded in the complete 910001–910120 corpus. They contain no source payload or score.

| ID | Coverage | Expected | Business data | Corpus-grounded manual rationale |
| --- | --- | --- | --- | --- |
| evidence-001 | Complete product fact | sufficient: 910001 | no | 910001 states USB-C PD 3.0/PPS and USB-A QC 3.0 for C65-Pro. |
| evidence-002 | Model mismatch | insufficient | no | The full corpus contains C65/C65-Pro but no C65-Air fact. |
| evidence-003 | No answer | insufficient | no | The full corpus contains no Z99-Pro or matching earphone material. |
| evidence-004 | Missing policy conditions | insufficient | no | Neither V500 nor general return policy covers customized engraving; “刻字” is absent. |
| evidence-005 | Cross-chunk support | sufficient: 910071, 910110 | no | 910071 covers the unopened T3 set and seven-day conditions; 910110 supplies return postage ownership. |
| evidence-006 | Colloquial low lexical overlap | sufficient: 910030 | no | “断网、按机身键、扫完整屋、回充” is fully covered by 910030; no retrieval score is claimed. |
| evidence-007 | Policy before order facts | sufficient: 910015 | yes | 910015 covers the stated C65-Pro return conditions; order 1001 facts still require a post-gate tool lookup. |
| evidence-008 | Supported negative answer | sufficient: 910082 | no | 910082 explicitly prohibits microwave heating. |
| evidence-009 | Conditional limitation | sufficient: 910088 | no | 910088 gives the seal conditions and warns that vigorous shaking or aging may still leak. |
| evidence-010 | Cross-chunk product answer | sufficient: 910007, 910006 | no | 910007 covers laptop compatibility and limits; 910006 says the standard package has no cable. |
| evidence-011 | Completed history plus current evidence | sufficient: 910053 | no | 910053 states IPX7, daily rinsing only, and prohibits prolonged immersion or underwater charging. |
| evidence-012 | Product policy exception | sufficient: 910070 | no | 910070 explicitly excludes opened sanitary brush heads from no-reason returns, except quality issues. |

The automated data check loads the authoritative corpus, requires exactly IDs 910001–910120, verifies every supporting ID and literal fact fragment against the actual answer, checks matching categories, validates completed-history wire through `load_turns`, and proves the three refusal-specific terms are absent from the full corpus. This replaces the earlier row-internal membership check that could not detect unreachable synthetic evidence.

## Assessor-only adversarial fixtures

`evals/ch05/evidence_adversarial.jsonl` contains exactly two controlled cases. Each is marked `evaluation_scope=assessor_fixture` and separates the `input` object (`question`, strict `intent`, explicit Citation sources) from the `expected` labels. Their Citation scores are fixed protocol fields for the controlled request and are expressly not measured retrieval scores.

| ID | Boundary | Expected | Manual rationale |
| --- | --- | --- | --- |
| evidence-adversarial-001 | Source prompt injection | insufficient, no support IDs | Its sole source is an instruction to violate the evidence policy and contains no 100W product fact. |
| evidence-adversarial-002 | Conflicting sources | insufficient, no support IDs | The two explicit sources disagree on the same C1 maximum-power fact, 65W versus 45W. |

These fixtures are inputs only to the future assessor-scope runner using the same `WorkflowGateway.assess`; they are excluded from retrieval, routing, band and end-to-end citation metrics. All 12 formal labels and both assessor fixtures were manually re-read against the files after the correction. None has entered a model request. Real evidence-gate quality remains `PENDING_USER_REPLY` under the existing data-send gate.

## Task 12 local implementation and evidence status — 2026-09-21

The evaluator and demo are now implemented locally. Existing 35 intent, 12 corpus-grounded evidence, and two assessor fixture labels are unchanged. `evaluate_workflow` clones only allowed runtime inputs; expected labels stay in the grading/artifact layer. Actual graph execution uses production graph/nodes/Agent, with isolated audit, action-offer and low-confidence sinks. It uses no saver and does not replace the actual MySQL/PostgreSQL integration gate. The business tools retain their existing random simulated data.

Rows are atomically saved after each attempted case, including safe error codes, actual score/band, routes, node/tool paths, observed pre-dispatch model attempt counts, reservations, optional provider usage and isolated effects. A failed evidence assessment still retains already-retrieved scores. Graph-completed budget fallback is distinguished from evaluation completion (`graph_status=completed`, evaluation failure). Unknown usage is null. Cancellation leaves the run running with already-persisted rows; the interrupted row is eligible to run again. Cached failures are retained and require a new output directory for a new attempt.

Run identity includes exact input and labels, execution scope, effective request controls, threshold settings, Prompt bundle, local model manifest, corpus, current source and installed dependency fingerprints. Configuration artifacts contain hashes rather than arbitrary settings/URLs/credentials. The same model with changed credentials is not a new quality configuration. `--limit` always yields smoke; fewer cases than expected yields partial; technical failures yield incomplete. Changed or unverifiable corpus yields invalid, which cannot resume. Full `complete` records execution coverage only, never automatically asserts quality.

Classification and route correctness include failed cases in labeled denominators. Confusion marks failures explicitly; business-to-knowledge misroutes have a separate denominator; sufficiency, exact supporting-ID match, refusal and exact citation-ID correctness are distinct metrics. Citation-ID correctness is structural and label-based, not a semantic faithfulness judge. Failed examples include misroutes, unsupported answers and any unauthorized ticket creation observation, with no average hiding these cases. Actual score values determine bands, including 0.7 and 0.8 in middle. Reports never count assessor Citation fixture scores as actual reranker output.

Assessor reports contain sufficiency, supporting IDs, errors and failure examples only. The same WorkflowGateway builds the evidence Prompt and enforces the same structured protocol and cumulative budget. The CLI receives the two explicit sources separately from labels; it does not inject these into normal retrieval.

Demo tests exercise terminal SSE framing, EOF/error rejection, source numeric-ID/hash validation, fixed same-origin source paths, redirect refusal, scoped session/turn/action identity, strict empty-JSON confirmation and replay numbering. The default complaint run never confirms. These tests use controlled HTTP transports; they are not the seven real browser acceptance checks.

External transfer remains **PENDING_USER_REPLY**. No Task 12 CLI real-model evaluation or real demo has been run. No true classification accuracy, evidence quality, route-quality score or real-model token total is reported. Root retains the prior automatic-review rejection and will ask for explicit destination/data-scope permission after local work and review are concrete. The whole-branch backend review and its single scoped fix review are now accepted at `4a5641d`, with no open mandatory backend finding. Browser acceptance, demo-database migration, old 8001 handover, finish and integration choice remain release gates.

Local verification results: evaluator/demo + Chapter 4 evaluator regression **69 passed** (2.08 s); both CLI help commands passed; `pip check` found no broken requirements; offline/no-deps wheel contained all 11 prompts and both new evaluation modules, and import caused zero socket connections. The final full required suite with all four dependency flags ran against actual isolated MySQL 13307/PostgreSQL 15433/Milvus and cached offline models: **782 passed, 4 failed, 1 existing Starlette warning** (80.91 s). All four failures were stale pre-Chapter-5 test fixtures in two legacy schema/init files, aligned under an explicitly approved two-file scope extension; the affected real-MySQL rerun was **10 passed** (0.30 s), covering all four failed tests. After that full run, evaluator production fix `00c7ad6` changed technical-failure observation handling; its complete evaluator-file covering run was **25 passed** (1.83 s). The final review then required a separate recovery compatibility fix: only provably migrated, uniformly completed, structurally incomplete legacy audit turns are excluded from graph history, while current corruption and checkpoint references remain strict. Its complete affected recovery/repository/service files passed **84 tests** (8.87 s) against isolated MySQL 13307 and PostgreSQL 15433 with scripted models. These are revision-specific covering runs; no subsequent all-green full-suite run was performed. The earlier wheel was not rebuilt after either production fix, so its packaging result is evidence for its earlier revision only. An earlier default-sandbox attempt failed localhost sockets and is recorded in the task report, not treated as acceptance.


## Fresh final required-suite result — 2026-09-22

On `d2b8799`, the required suite with all four database/Milvus/local-model flags completed with **807 passed, 1 existing Starlette/AnyIO deprecation warning in 84.32 seconds (exit 0)**. This is a new full run after the historical stale-fixture corrections, evaluator/recovery fixes, Prompt namespace fix and score-roundtrip fix. The raw output is `evals/reports/ch05/2026-09-22-validation/pytest-required.txt`. It supersedes the prior absence of a later all-green full test run; that earlier chronology remains below as history. No new wheel build is claimed, and real model/browser gates are separate.
