# Current status — 2026-09-22

Backend implementation and its scoped reviews are complete at `d2b8799`; Agent control candidate `45ca47a` completed its frozen v3 run with no control-protocol failures; answer-only citation clarification `3cc14e4` passed three targeted answer probes and actual browser multi-step validation, followed by independent scoped approval. The backend required suite at `d2b8799` is **807 passed, 1 known deprecation warning (84.32 s)**, using real isolated MySQL, PostgreSQL, Milvus and cached local models. Final Prompt budget/gateway regression at `3cc14e4` is **42 passed (2.04 s)**. Functional browser acceptance and cutover are complete; the formal model reports below retain their failures and do not constitute a full-corpus zero-error pass.

The user authorized demonstration questions, necessary knowledge, simulated tool results and generated answers to DeepSeek (`api.deepseek.com`). The actual configured model is `deepseek-flash`, thinking disabled, with zero automatic retries. A later, separate three-case Agent-control diagnostic was rejected by automatic approval review before process creation; the user then explicitly authorized it and exactly three requests completed (3,066 provider tokens). The old control error did not recur; one native call proposed an invalid $ORDER_ID placeholder and therefore does not establish end-to-end success. Original failed replies were not retained, so their cause remains unproved.

| Actual run | Coverage and execution status | Selected measured results |
| --- | --- | --- |
| `evals/reports/ch05/2026-09-22-intents` | 35/35 attempted, incomplete; 4 technical failures | Classification and route 31/35 (88.57%); business flag 29/35; business-to-knowledge misroutes 0/15. 90 calls / 67,017 provider tokens. |
| `evals/reports/ch05/2026-09-22-intents-v2` | 35/35 attempted, incomplete; 2 technical failures | Classification and route 33/35 (94.29%); business flag 31/35; business-to-knowledge misroutes 0/15. 96 calls / 74,202 provider tokens. |
| `evals/reports/ch05/2026-09-22-intents-v3` | 35/35 attempted, incomplete; 2 final-answer citation failures | Classification and route 33/35; business flag 31/35; business-to-knowledge misroutes 0/15. 92 calls / 65,776 provider tokens. |
| `evals/reports/ch05/2026-09-22-evidence` | 12/12, complete; zero technical failures | Classification, route and sufficiency 12/12; refusal 3/3; business flag 11/12; exact support IDs 10/12; exact citation IDs 7/9. 43 calls / 45,250 tokens. |
| `evals/reports/ch05/2026-09-22-assessor` | 2/2, complete | Both injection/conflicting-source cases correctly insufficient with no supporting IDs. 2 calls / 1,437 tokens. |

Total for these five formal runs: **323 calls / 253,682 provider tokens** (the separate diagnostic is excluded). `complete` means execution coverage, not perfect answer quality. Structural ID metrics do not prove semantic faithfulness. Fixed labels were not edited to improve scores.

`intent-009`, `intent-010` and `intent-022` failed with `AGENT_CONTROL_ERROR`; `intent-016` reached its conservative input-token guard and returned the bounded fallback, but is still counted as an evaluation failure (`INPUT_TOO_LONG`). The original failures in cases 009/010/022 remain retained; the separate three-request diagnostic did not reproduce their error. The successful control probes in separate work do not replace these failures.

The v2 intent run has raw observations: 002 returned prose followed by JSON, and 023 returned only prose, so strict control parsing rejected both. Case 009 reached a valid final clarification only after its placeholder tool argument was blocked; its formal passing grade does not close the missing-argument Prompt quality gap. Cases 010, 016 and 022 did not repeat their earlier technical failures. The second bounded Agent Prompt candidate `45ca47a` then passed all control-protocol checks in v3: 002/009/023 clarified directly without tool proposals. Cases 005/010 failed later when final generation invented [1] despite sources=[]; the strict citation validator correctly rejected them. This is not proof that the Agent Prompt caused the separate answer-stage error. A bounded answer-only clarification is committed as `3cc14e4`; no parser, label, budget or business-tool change was made. Case 016 took insufficient-evidence fallback in v3, so this does not revalidate its earlier budget path.

Actual browser checks on isolated port 8002 have verified policy retrieval/clickable source text, unknown refusal and a low-confidence pool row, complaint actions without automatic execution, cancellation before ticket creation, one confirmed ticket, repeated confirmation before/after graceful restart without duplicates, front-end feedback locking, and the logistics tool badge. The policy save conflict observed earlier was fixed and passed a fresh browser re-test. PostgreSQL retained the completed policy checkpoint after restart. Subsequent real browser multi-step execution completed query_order then query_logistics, three Agent decisions, validated answering and MySQL/PostgreSQL persistence. New-message recovery in the old failed session also completed. The new single-worker service now runs on 8001; 8002 is stopped. Final branch integration still requires the user’s choice. Artifacts: `evals/reports/ch05/2026-09-22-acceptance/`.

The first attempt to start the three-answer check was rejected before sending any request. After the user explicitly confirmed the exact demo payload and DeepSeek destination, the same frozen harness completed once: three requests / 2,430 provider tokens, strict citation and independent factual review 3/3 passed. This separate check does not change the formal 35-case result or the five-run totals. The two Prompt tasks subsequently received scoped quality approval with a minor timestamp presentation limitation (UTC omitted in one answer).

The sections below retain the earlier local-only chronology and dataset rationale. Their pending-authorization and not-yet-run statements describe those earlier stages, not the current status above. Final artifacts are in `2026-09-22-answer-citations/` and `2026-09-22-acceptance/`; the demo migration preserved all 67 original messages and both original tickets before the final two chat checks.

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

All 35 rows were checked for the exact five-field shape (`id`, `question`, `history`, `expected_intent`, `needs_business_data`), valid strict DTO values, unique IDs, and five cases per label. This is an assistant review of reference labels, not a human gold-standard review or a review of model predictions.

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


### 2026-09-22 · Agent Prompt 第一候选真实复验结束，进入第二次有界修正
- 用户关键原话：“缺信息就追问用户”“允许”；沿用已批准的演示评估授权，不重跑已完成的三次专门诊断。
- 关键产出：`evals/reports/ch05/2026-09-22-intents-v2/` 已跑 35/35，执行状态 incomplete；分类/路由 33/35，业务数据标记 31/35，96 次模型调用、74,202 provider tokens。原 010 已实际完成订单→物流两步；原 016 预算错误及 022 控制格式错本轮未复现。
- 拒绝或纠偏：不能把最终追问或 formal grade=true 等同于没有无效工具申请。009 先申请 order_id="?"，Schema 阻止实际业务函数，随后 clarify；缺参直接追问的候选目标未达到。002 输出解释文字+JSON，023 只输出追问文字，严格控制协议拒绝均有原始观测为证。
- 翻车与返工：第一候选 b5ac0e1 的 25 项本地测试不证明 Prompt 质量。保留所有失败，不改解析器、标签或预算；第二次修正只强化控制层身份及无工具时整条消息仅为 JSON，泛化缺参分支。四策略比较使用独立知识 Prompt 与已完成校准，可并行继续。


### 2026-09-22 · 第三轮 35 条执行结束及引用错误定位
- 用户关键原话：“缺信息就追问用户”“回答带引用编号，编号能映射回来源 chunk”。
- 关键产出：`2026-09-22-intents-v3` 在 45ca47a 上 35/35 尝试，92 次调用/65,776 provider tokens。原 002/009/023 均直接 clarify、没有工具申请，Agent 控制格式错误本轮为零；整体仍 incomplete，005/010 因最终引用错误失败，分类/路由计分 33/35、业务标记 31/35。
- 拒绝或纠偏：005/010 的 sources=[]，合法工具执行后生成却加 [1]，严格未知编号校验正确拒绝。不能归因于新增 Agent Prompt，因为最终回答 Prompt 未修改且无因果对照；不能放宽校验或伪造来源。016 本轮走证据不足兜底，未复验原预算路径。
- 翻车与返工：安排仅 `workflow_answer.txt` 的简短编号命名空间澄清，以两条无知识源业务回答加一条有知识源正例做固定生成验证，并验真实页面多步。不重跑整套 35/240 刷绿；保留完整基线失败，定向样例绝不冒充全量通过。
