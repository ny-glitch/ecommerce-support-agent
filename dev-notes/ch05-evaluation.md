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

## Evidence and routing labels

`evals/ch05/evidence.jsonl` contains 12 locally authored cases. Each row has the question, intent, candidate source snapshots, `expected_sufficient`, `supporting_chunk_ids`, `needs_business_data`, and a manual reason. It deliberately has no retrieval score field: phrases such as “low relevance” describe the controlled coverage category and do not claim a measured reranker result.

| ID | Coverage | Expected | Business data | Manual rationale |
| --- | --- | --- | --- | --- |
| evidence-001 | Complete product fact | sufficient | no | Same model; protocol and power limit are explicit. |
| evidence-002 | Model mismatch | insufficient | no | C65-Air evidence cannot support a C65-Pro answer. |
| evidence-003 | No answer | insufficient | no | Empty candidates cannot support a model answer. |
| evidence-004 | Missing policy conditions | insufficient | no | Time limit alone omits opened/used eligibility and exceptions. |
| evidence-005 | Cross-chunk support | sufficient | no | Two cited blocks jointly cover eligibility and application steps. |
| evidence-006 | Low-relevance complete fact | sufficient | no | Text fully covers object, compatibility, action, and power limit; no score is invented. |
| evidence-007 | Source prompt injection | insufficient | no | The source contains only a model-directed instruction, not a product fact. |
| evidence-008 | Conflicting sources | insufficient | no | 65W and 45W conflict on the requested key fact. |
| evidence-009 | Policy then business lookup | sufficient | yes | Static policy is complete; order date/state must be queried only after the gate. |
| evidence-010 | Ambiguous object | insufficient | no | The unresolved pronoun maps to candidates with opposite facts. |
| evidence-011 | Supported negative limit | sufficient | no | The evidence explicitly states both the unsupported port and supported alternative. |
| evidence-012 | Unknown policy exception | insufficient | no | Ordinary-goods policy does not cover customized engraving. |

All supporting IDs were manually checked against the candidate list. Sufficient rows name at least one supporting source; refusal rows name none. These labels have not entered any model request. Real evidence-gate accuracy and routing measurements remain `PENDING_USER_REPLY` with the existing external data-send gate.
