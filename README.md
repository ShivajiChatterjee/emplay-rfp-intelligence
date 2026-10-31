# Emplay RFP Intelligence Platform

Evidence-grounded RFP extraction and Q&A using LangGraph, the native Gemini
SDK, Qdrant hybrid dense/sparse retrieval, Cohere reranking, Pydantic contracts,
local resumable checkpoints, and optional LangSmith observability.

The official extraction contract contains exactly 20 fields. Document layouts,
product categories, identifiers, contacts, and addenda are discovered from each
bid's indexed evidence; they are not encoded for a particular solicitation.

## Setup

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env` with your own provider credentials. Never commit `.env`.
Qdrant must contain (or be able to create) the configured collection before a
live extraction or Q&A run.

### Environment variables

| Variable | Purpose |
|---|---|
| `QDRANT_URL` | Qdrant service URL |
| `QDRANT_API_KEY` | Qdrant credential |
| `COHERE_API_KEY` | Preferred reranker credential |
| `COHERE_RERANK_MODEL` | Cohere reranking model |
| `GOOGLE_API_KEY` | Native Gemini SDK credential |
| `GEMINI_MODEL` | Explicit Gemini model; no automatic model fallback |
| `LANGSMITH_TRACING` | Enable or disable LangSmith tracing |
| `LANGSMITH_API_KEY` | LangSmith credential when tracing is enabled |
| `LANGSMITH_PROJECT` | LangSmith project name |
| `GEMINI_MIN_INTERVAL_SECONDS` | Shared Gemini request pacing |
| `COHERE_MIN_INTERVAL_SECONDS` | Shared Cohere request pacing |
| `EMPLAY_MAX_CONCURRENCY` | Bounded workflow concurrency |

Optional controls include `GEMINI_TIMEOUT_MS`,
`EMPLAY_MAX_PRODUCT_STRUCTURE_CHUNKS`, and
`EMPLAY_MAX_CONTACT_STRUCTURE_CHUNKS`. See `.env.example` for a conservative
single-worker configuration.

API controls include upload limits, bounded job workers, cache TTL/storage and
optional pricing metadata. Pricing values are per one million tokens. If usage
or pricing is unavailable, the API reports cost as unavailable rather than
inventing an estimate.

## Windows local web application

The FastAPI application serves the genuine HTML/CSS/JavaScript frontend and
calls the same retrieval, Q&A and LangGraph extraction workflows used by the
CLI. It does not contain sample-response shortcuts.

```powershell
cd C:\Users\wwwri\vscode\emplay-rfp-v2
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000`. The OpenAPI contract is available at
`http://127.0.0.1:8000/docs`.

The local UI provides:

- a bid dashboard and safe PDF/HTML upload/index workflow;
- Qdrant hybrid search with Cohere reranking metadata and original evidence;
- cited Q&A through the production LangGraph workflow;
- viewing and starting/resuming the official 20-field extraction;
- bid-to-bid field comparison with confidence and citations;
- extracted-answer/original-document side-by-side inspection; and
- run metrics and authentic local trace inspection.

Background indexing and extraction jobs are deliberately bounded. Repeated
grounded questions use a conservative SQLite cache scoped by bid identity,
local document/index version and retrieval configuration. Numeric/model
qualifiers and negation must match before reuse, and only answers with fresh
citations are cached. Reindexing invalidates only that bid's entries.

Application-originated upload, index, Q&A and extraction activity is persisted
in `outputs/api_activity.sqlite3` (or `EMPLAY_ACTIVITY_PATH`). The Runs & Traces
screen keeps these operations separate from preserved CLI/development history.
Historical runs without post-API telemetry remain inspectable and explicitly
report that metrics were not recorded; missing token or cost data is never
inferred.

## One-command CLI

Ingest a bid folder:

```powershell
.\.venv\Scripts\python.exe main.py --bid data\Bid1 --mode ingest --output outputs
```

Run or resume extraction:

```powershell
.\.venv\Scripts\python.exe main.py --bid data\Bid1 --mode extract --output outputs\controlled_full_bid1_resume --resume --checkpoint outputs\controlled_full_bid1_resume\Bid1_field_checkpoints.json --max-retries 0 --disable-query-rewrite --deterministic-validation-only
```

Run an unseen bid folder with the same production workflow:

```powershell
.\.venv\Scripts\python.exe main.py --bid C:\path\to\UnseenRfpFolder --mode extract --output outputs\unseen_rfp
```

The folder name becomes the bid ID; no production rule depends on `Bid1` or
`Bid2`.

Ask a cited question:

```powershell
.\.venv\Scripts\python.exe main.py --bid data\Bid1 --mode ask --question "What is the final submission deadline after addenda?" --output outputs\qa
```

Compare two bids:

```powershell
.\.venv\Scripts\python.exe main.py --bid data\Bid1 --bid data\Bid2 --mode compare --question "Compare delivery and warranty requirements." --output outputs\comparison
```

## Design and safety decisions

- Extract planning is deterministic because the assignment fixes the 20-field
  registry and specialist routing. Ask/compare planning remains goal-specific.
- Each bid is filtered independently in Qdrant. Dense and sparse results are
  fused with RRF; Cohere is preferred for reranking, with an explicitly marked
  RRF fallback when reranking is rate-limited.
- Values and short quotations are checked against original indexed passages.
  Provider failures are not interpreted as documentary absence.
- Addenda are reconciled before validation and validated amendment records are
  retained across resume runs.
- Only terminal evidence-based fields are reusable. Checkpoints include bid,
  document, index/config, and model fingerprints.
- A final JSON is published only after all 20 fields satisfy the Pydantic output
  contract and the publication guard.

See [docs/architecture.md](docs/architecture.md) for the component and workflow
diagram.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The evaluation question set is in `evaluation/questions.json`. Existing
retrieval/reranker reports are in `outputs/`. Generated measurements and Q&A
answers are never synthesized offline; run the evaluation or Q&A workflow with
configured providers to create genuine artifacts.

Ten genuine cited Q&A examples are available in `outputs/qa_examples.json` and
`outputs/qa_examples.md`. The resumable bounded generator is:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_qa_examples
```

For controlled maintenance, `scripts/invalidate_checkpoint_fields.py` removes
only explicitly named checkpoint fields after validating their official labels
and records the supplied reason. It is not part of normal extraction or resume
execution and should be used only when a known checkpoint is proven stale.

An annotated authentic workflow trace is documented in
`docs/agent_execution_example.md`; its unchanged submission copy is
`submission/workflow_trace_example.json`.

## Current artifact status

- Bid1 and Bid2 each have published 20-field JSON outputs with 17 passed, three
  justified not-found, and zero failed fields.
- Existing local workflow traces remain under `outputs/`; LangSmith transport is
  optional and non-fatal.

## Local application limitations

- The API is intended for a trusted local workstation. It does not implement
  multi-user authentication or authorization; do not expose it directly to the
  public internet.
- Background job state is in process. Existing workflow checkpoints remain
  durable, but the API job-progress record itself does not survive a server
  restart.
- The semantic cache intentionally favors false misses over unsafe reuse and is
  not a replacement for the Qdrant document index.
- PDF page navigation uses the browser's standard `#page=` support. Passage
  text is shown beside the document when trustworthy highlight coordinates are
  unavailable.
- Gemini token splits, Cohere usage and embedding usage are shown only when the
  providers recorded them. Cost estimates require explicit pricing variables
  and a pricing date.
- Docker packaging is intentionally omitted for this local Windows submission;
  the assignment's optional container path is not required to run the app.
