# Authentic Agent Execution Example

This example references the unmodified local workflow trace:

`outputs/controlled_full_bid1/workflow_trace_6612d7b1-51fb-4a61-8a1a-389490ea1a54.json`

The identical submission copy is `submission/workflow_trace_example.json`.

Run ID: `6612d7b1-51fb-4a61-8a1a-389490ea1a54`

The run processed `data\Bid1` in extraction mode with query rewriting and
semantic validation disabled, zero field retries, and a goal of extracting and
validating all official fields. It is used here because it records every major
workflow stage and 60 individual agent traces. It predates later robustness
corrections and its historical result was only 2 passed and 18 not-found; it is
an execution-trace example, not the final submission-quality Bid1 result.

## Recorded sequence

1. **Planning** — one `orchestrator_planner/create_execution_plan` trace and one
   `plan` event created the field workflow.
2. **Index readiness** — one `ensure_index` event checked the existing bid
   index before retrieval.
3. **Retrieval and extraction** — 19 `retrieve_extract_reconcile` events routed
   the 19 non-summary fields. The trace records 38 retrieval calls and 19
   extraction calls.
4. **Addendum reconciliation** — two recorded reconciliation attempts were
   made for extracted candidates requiring amendment review.
5. **Validation** — 19 `validate_field` events recorded field decisions.
6. **Report generation** — one `finalize_report` event assembled the historical
   20-field result.

## Recorded agent activity

| Agent | Calls | Success | Error | Summed recorded latency (ms) | Recorded tokens |
|---|---:|---:|---:|---:|---:|
| orchestrator_planner | 1 | 1 | 0 | 1,670.88 | 550 |
| retrieval_agent | 38 | 27 | 11 | 132,898.54 | 0 |
| extraction_agent | 19 | 2 | 17 | 8,597.30 | 3,891 |
| addendum_reconciliation | 2 | 0 | 2 | 1,120.76 | 0 |

These are sums of the trace's recorded `latency_ms` and `tokens_used` fields.
Zero means no token value was recorded; it is not an estimate.

## Recorded tool calls

- `google.genai.generate_content:gemini-2.5-flash-lite`
- `rag.cohere_reranker.retrieve_and_rerank_cohere`

The trace contains 41 workflow events, 60 agent traces, eight warnings, and no
top-level errors. The historical final validation summary was 2 passed, 0
failed, and 18 not-found. The later completed Bid1 JSON and its validation
summary are separate artifacts under `outputs/controlled_full_bid1_resume/`.
