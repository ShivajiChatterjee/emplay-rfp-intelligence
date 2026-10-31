from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.app as app_module
from agents.schemas import AgentTrace, AnswerCitation, CitedAnswer
from api.app import create_app
from api.activity import ActivityStore
from api.models import ExtractRequest
from api.semantic_cache import SemanticAnswerCache, questions_match
from api.telemetry import TelemetryStore
from agents.ingestion_agent import IndexStatus


@pytest.fixture()
def api_workspace(tmp_path, monkeypatch):
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    data.mkdir()
    outputs.mkdir()
    monkeypatch.setenv("EMPLAY_DATA_DIR", str(data))
    monkeypatch.setenv("EMPLAY_OUTPUT_DIR", str(outputs))
    monkeypatch.setattr(app_module, "answer_cache", SemanticAnswerCache(tmp_path / "cache.sqlite3"))
    monkeypatch.setattr(app_module, "activity_store", ActivityStore(tmp_path / "activities.sqlite3"))
    monkeypatch.setattr(app_module, "telemetry", TelemetryStore(tmp_path / "telemetry"))
    return data, outputs


@pytest.fixture()
def client(api_workspace):
    return TestClient(create_app())


def _write_bid(data: Path, bid_id: str, file_name: str = "request.pdf") -> Path:
    folder = data / bid_id
    folder.mkdir()
    (folder / file_name).write_bytes(b"%PDF-1.4 synthetic offline fixture")
    return folder


def _field_record(value):
    return {"value": value, "sources": [{"file": "request.pdf", "page": 1}], "confidence": 0.9, "notes": ""}


def test_health_does_not_probe_external_providers(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["external_checks_performed"] is False


def test_frontend_is_served_and_calls_production_api_routes(client):
    page = client.get("/")
    script = client.get("/assets/app.js")
    styles = client.get("/assets/styles.css")
    assert page.status_code == script.status_code == styles.status_code == 200
    assert "evidence-dialog" in page.text
    assert "screen-compare" in page.text
    assert "/api/search" in script.text
    assert "/api/ask" in script.text
    assert "/api/extract" in script.text
    assert "/api/compare" in script.text
    assert "Application runs" in page.text
    assert "Historical runs" in page.text
    assert "No application runs yet." in script.text
    assert client.get("/favicon.ico").status_code == 200
    assert 'id="source-files"' in page.text
    assert 'accept=".pdf,.html,.htm"' in page.text
    assert "multiple" in page.text
    assert "addEventListener('change'" in script.text
    assert "addEventListener('drop'" in script.text
    assert "body.append('files', file, file.name)" in script.text
    assert "data-remove-file" in script.text
    assert "state.uploadFiles" in script.text


def test_fresh_application_activity_excludes_preserved_history(client, api_workspace):
    _, outputs = api_workspace
    run_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    (outputs / f"run_manifest_{run_id}.json").write_text(
        json.dumps({
            "run_id": run_id,
            "bid_id": "HistoricalBid",
            "run_status": "partial",
            "final_submission_published": False,
        }),
        encoding="utf-8",
    )
    application = client.get("/api/runs")
    historical = client.get("/api/runs", params={"scope": "historical"})
    assert application.status_code == historical.status_code == 200
    assert application.json()["runs"] == []
    assert [item["run_id"] for item in historical.json()["runs"]] == [run_id]


def test_uploaded_activity_persists_across_store_instances(client, api_workspace):
    response = client.post(
        "/api/bids/upload",
        data={"bid_id": "Persistent-RFP"},
        files={"files": ("request.pdf", b"%PDF-1.4 offline", "application/pdf")},
    )
    assert response.status_code == 201
    activity_id = response.json()["activity_id"]
    listed = client.get("/api/runs").json()["runs"]
    assert listed[0]["activity_id"] == activity_id
    assert listed[0]["operation"] == "upload"
    assert listed[0]["status"] == "succeeded"
    reopened = ActivityStore(app_module.activity_store.path)
    assert reopened.get(activity_id)["status"] == "succeeded"


def test_historical_metrics_absence_is_an_available_state_not_404(client, api_workspace):
    _, outputs = api_workspace
    run_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    (outputs / f"workflow_trace_{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "events": [{"agent": "planner", "action": "plan"}]}),
        encoding="utf-8",
    )
    response = client.get(f"/api/runs/{run_id}/metrics")
    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "source": "historical",
        "reason": "Metrics were not recorded for this historical run.",
    }


def test_trace_normalization_preserves_order_and_redacts_secrets(client, api_workspace):
    _, outputs = api_workspace
    run_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    trace = {
        "run_id": run_id,
        "agent_traces": [
            {
                "agent": "retrieval_search", "action": "retrieve", "status": "success",
                "started_at": "2026-10-31T10:00:00+00:00", "latency_ms": 12,
                "input_summary": {"bid_id": "Synthetic"},
            },
            {
                "agent": "validator_critic", "action": "validate", "status": "error",
                "started_at": "2026-10-31T10:00:01+00:00", "latency_ms": 4,
                "metadata": {"api_key": "must-not-leak", "retry": 1},
            },
        ],
        "retry_counts": {"Due Date": 1},
        "errors": ["Validation failed"],
    }
    (outputs / f"workflow_trace_{run_id}.json").write_text(json.dumps(trace), encoding="utf-8")
    response = client.get(f"/api/runs/{run_id}/trace")
    assert response.status_code == 200
    payload = response.json()
    assert [step["operation"] for step in payload["steps"]] == ["retrieve", "validate"]
    assert [step["order"] for step in payload["steps"]] == [1, 2]
    assert payload["steps"][1]["details"]["metadata"]["api_key"] == "[redacted]"
    assert payload["context"]["retry_counts"] == {"Due Date": 1}


def test_application_list_preserves_partial_and_failed_statuses(client):
    partial = app_module.activity_store.create("extract", ["Synthetic"], status="partial")
    failed = app_module.activity_store.create("index", ["Synthetic"], status="failed")
    runs = client.get("/api/runs").json()["runs"]
    statuses = {item["activity_id"]: item["status"] for item in runs}
    assert statuses[partial["activity_id"]] == "partial"
    assert statuses[failed["activity_id"]] == "failed"


def test_upload_and_document_navigation_are_path_safe(client, api_workspace):
    response = client.post(
        "/api/bids/upload",
        data={"bid_id": "Unseen-RFP"},
        files={"files": ("request.pdf", b"%PDF-1.4 offline", "application/pdf")},
    )
    assert response.status_code == 201
    document = response.json()["documents"][0]
    preview = client.get(f"/api/bids/Unseen-RFP/documents/{document['document_id']}")
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("application/pdf")
    assert client.get("/api/bids/../documents/../../.env").status_code in {404, 422}


def test_multiple_documents_upload_atomically_with_original_names_and_content(
    client, api_workspace
):
    data, _ = api_workspace
    documents = [
        ("files", ("main-rfp.pdf", b"%PDF-1.4 main document", "application/pdf")),
        ("files", ("addendum-1.pdf", b"%PDF-1.4 addendum", "application/pdf")),
        ("files", ("portal.html", b"<html><body>Portal</body></html>", "text/html")),
        ("files", ("terms.htm", b"<html>Terms</html>", "text/html")),
    ]
    response = client.post(
        "/api/bids/upload", data={"bid_id": "Arbitrary-2027"}, files=documents
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["uploaded_count"] == 4
    assert payload["upload_status"] == "uploaded"
    assert {item["name"] for item in payload["documents"]} == {
        "main-rfp.pdf", "addendum-1.pdf", "portal.html", "terms.htm"
    }
    folder = data / "Arbitrary-2027"
    assert (folder / "main-rfp.pdf").read_bytes() == b"%PDF-1.4 main document"
    assert (folder / "addendum-1.pdf").read_bytes() == b"%PDF-1.4 addendum"
    assert (folder / "portal.html").read_bytes() == b"<html><body>Portal</body></html>"


@pytest.mark.parametrize(
    ("bid_id", "files", "expected_status"),
    [
        ("Empty-RFP", [("files", ("empty.pdf", b"", "application/pdf"))], 422),
        (
            "Duplicate-RFP",
            [
                ("files", ("Terms.pdf", b"%PDF first", "application/pdf")),
                ("files", ("terms.PDF", b"%PDF second", "application/pdf")),
            ],
            409,
        ),
        (
            "Mixed-RFP",
            [
                ("files", ("valid.pdf", b"%PDF valid", "application/pdf")),
                ("files", ("script.exe", b"MZ", "application/octet-stream")),
            ],
            415,
        ),
    ],
)
def test_invalid_multi_uploads_are_rejected_without_partial_bid_folder(
    client, api_workspace, bid_id, files, expected_status
):
    data, _ = api_workspace
    response = client.post("/api/bids/upload", data={"bid_id": bid_id}, files=files)
    assert response.status_code == expected_status
    assert not (data / bid_id).exists()


def test_upload_limits_and_bid_identifier_validation(client, api_workspace, monkeypatch):
    data, _ = api_workspace
    monkeypatch.setenv("EMPLAY_MAX_UPLOAD_BYTES", "8")
    oversized = client.post(
        "/api/bids/upload",
        data={"bid_id": "Large-RFP"},
        files={"files": ("large.pdf", b"%PDF-too-large", "application/pdf")},
    )
    invalid_bid = client.post(
        "/api/bids/upload",
        data={"bid_id": "../escape"},
        files={"files": ("valid.pdf", b"%PDF", "application/pdf")},
    )
    assert oversized.status_code == 413
    assert invalid_bid.status_code in {400, 422}
    assert not (data / "Large-RFP").exists()


def test_upload_rejects_unsafe_names_and_extensions(client):
    unsafe = client.post(
        "/api/bids/upload",
        data={"bid_id": "Unsafe"},
        files={"files": ("../escape.pdf", b"%PDF", "application/pdf")},
    )
    assert unsafe.status_code == 400
    executable = client.post(
        "/api/bids/upload",
        data={"bid_id": "Executable"},
        files={"files": ("payload.exe", b"MZ", "application/octet-stream")},
    )
    assert executable.status_code == 415


def test_search_uses_existing_retrieval_adapter(client, api_workspace, monkeypatch):
    data, _ = api_workspace
    _write_bid(data, "FutureBid")
    evidence = SimpleNamespace(
        model_dump=lambda mode=None: {
            "bid_id": "FutureBid", "field_name": "Q&A search", "question": "delivery",
            "status": "success", "passages": [], "warnings": [], "trace": {},
        },
        status="success",
    )
    monkeypatch.setattr(app_module, "search_evidence", lambda request: evidence)
    response = client.get("/api/search", params={"bid_id": "FutureBid", "query": "delivery"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_index_job_uses_existing_index_service_and_records_activity(
    client, api_workspace, monkeypatch
):
    data, _ = api_workspace
    _write_bid(data, "Index-RFP")
    calls = []

    def fake_index(folder, output):
        calls.append((folder.name, output.name))
        return IndexStatus(bid_id="Index-RFP", status="indexed", documents=1, chunks=3)

    monkeypatch.setattr(app_module, "ensure_bid_indexed", fake_index)
    response = client.post("/api/bids/Index-RFP/index")
    assert response.status_code == 202
    job = response.json()
    for _ in range(100):
        current = client.get(f"/api/jobs/{job['job_id']}").json()
        if current["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.01)
    assert current["status"] == "succeeded"
    assert calls == [("Index-RFP", "outputs")]
    activity = app_module.activity_store.get(job["activity_id"])
    assert activity["operation"] == "index"
    assert activity["status"] == "succeeded"
    assert activity["metrics_available"] is True
    assert client.get(f"/api/runs/{job['activity_id']}/trace").json()["step_count"] == 1


def test_extraction_runner_remains_connected_to_production_workflow(
    api_workspace, monkeypatch
):
    data, _ = api_workspace
    _write_bid(data, "Extract-RFP")
    requests = []

    def fake_workflow(request):
        requests.append(request)
        return {
            "run_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
            "run_manifest": SimpleNamespace(
                run_status="partial", final_submission_published=False
            ),
            "agent_traces": [],
            "retry_counts": {},
        }

    monkeypatch.setattr(app_module, "run_workflow", fake_workflow)
    result = app_module._run_extract(ExtractRequest(bid_id="Extract-RFP"))
    assert result["run_status"] == "partial"
    assert len(requests) == 1
    assert requests[0].mode == "extract"
    assert requests[0].resume is True


def test_ask_returns_real_workflow_shape_and_then_cache_hit(client, api_workspace, monkeypatch):
    data, _ = api_workspace
    _write_bid(data, "FutureBid")
    answer = CitedAnswer(
        answer="Delivery is required within thirty days.",
        sources=[AnswerCitation(bid_id="FutureBid", file="request.pdf", page=1, chunk_id="chunk-1")],
        confidence=0.8,
    )
    calls = []

    def fake_run(_request):
        calls.append(1)
        return {"run_id": "11111111-1111-1111-1111-111111111111", "qa_answer": answer, "agent_traces": [], "retry_counts": {}}

    monkeypatch.setattr(app_module, "run_workflow", fake_run)
    payload = {"bid_id": "FutureBid", "question": "What is the delivery requirement?"}
    first = client.post("/api/ask", json=payload)
    second = client.post("/api/ask", json=payload)
    assert first.status_code == 200 and first.json()["cache"]["hit"] is False
    assert second.status_code == 200 and second.json()["cache"]["hit"] is True
    assert len(calls) == 1


def test_compare_supports_synthetic_unseen_bids(client, api_workspace):
    data, outputs = api_workspace
    official = list(app_module.OFFICIAL_FIELD_LABELS)
    for bid_id, due in (("Alpha-RFP", "2027-01-01"), ("Beta-RFP", "2027-02-01")):
        _write_bid(data, bid_id)
        fields = {name: _field_record(due if name == "Due Date" else None) for name in official}
        (outputs / f"{bid_id}_extraction.json").write_text(json.dumps({"bid_id": bid_id, "fields": fields}), encoding="utf-8")
    response = client.post("/api/compare", json={"bid_ids": ["Alpha-RFP", "Beta-RFP"], "fields": ["Due Date"]})
    assert response.status_code == 200
    row = response.json()["fields"][0]
    assert row["different"] is True
    assert row["bids"]["Alpha-RFP"]["sources"][0]["page"] == 1


def test_semantic_cache_blocks_cross_bid_negation_numbers_and_versions(tmp_path):
    cache = SemanticAnswerCache(tmp_path / "qa.sqlite3", ttl_seconds=60)
    config = {"reranker": "test"}
    scope_a_v1 = cache.scope("A", "v1", config)
    scope_b_v1 = cache.scope("B", "v1", config)
    scope_a_v2 = cache.scope("A", "v2", config)
    answer = {"answer": "Supported", "sources": [{"file": "a.pdf", "page": 1, "chunk_id": "c1"}]}
    cache.put(scope_a_v1, "Is model 5400 required?", answer, 10)
    fresh = lambda _answer: True
    assert cache.get(scope_a_v1, "Is model 5400 required?", evidence_is_fresh=fresh)
    assert cache.get(scope_b_v1, "Is model 5400 required?", evidence_is_fresh=fresh) is None
    assert cache.get(scope_a_v2, "Is model 5400 required?", evidence_is_fresh=fresh) is None
    assert cache.get(scope_a_v1, "Is model 5500 required?", evidence_is_fresh=fresh) is None
    assert cache.get(scope_a_v1, "Is model 5400 not required?", evidence_is_fresh=fresh) is None
    cache.put(scope_b_v1, "Is model 5400 required?", answer, 10)
    assert cache.invalidate_bid("A") == 1
    assert cache.get(scope_b_v1, "Is model 5400 required?", evidence_is_fresh=fresh)


def test_question_similarity_is_conservative_for_material_qualifiers():
    assert questions_match("What is the delivery date?", "What is the delivery date?")
    assert not questions_match("Is installation required?", "Is installation not required?")
    assert not questions_match("Does model 7420 qualify?", "Does model 7430 qualify?")


def test_telemetry_marks_unknown_cost_and_usage(tmp_path):
    store = TelemetryStore(tmp_path)
    trace = AgentTrace(agent="qa_report", action="answer", status="success", latency_ms=12)
    metrics = store.record(
        "22222222-2222-2222-2222-222222222222",
        mode="ask", bid_ids=["Synthetic"], latency_ms=20, state={"agent_traces": [trace], "retry_counts": {}},
    )
    assert metrics["usage"]["gemini"]["total_tokens"] is None
    assert metrics["estimated_cost"]["amount"] is None
    assert metrics["estimated_cost"]["is_estimate"] is True


def test_telemetry_aggregates_tokens_and_configured_estimated_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_INPUT_COST_PER_MILLION", "0.10")
    monkeypatch.setenv("GEMINI_OUTPUT_COST_PER_MILLION", "0.40")
    monkeypatch.setenv("EMPLAY_PRICING_DATE", "2026-10-01")
    trace = AgentTrace(
        agent="extractor", action="extract", status="success", latency_ms=10,
        tokens_used=150, metadata={"token_usage": {"prompt_tokens": 100, "output_tokens": 50, "total_tokens": 150}},
    )
    metrics = TelemetryStore(tmp_path).record(
        "33333333-3333-3333-3333-333333333333",
        mode="extract", bid_ids=["Synthetic"], latency_ms=10,
        state={"agent_traces": [trace], "retry_counts": {}},
    )
    assert metrics["usage"]["gemini"]["input_tokens"] == 100
    assert metrics["usage"]["gemini"]["output_tokens"] == 50
    assert metrics["estimated_cost"]["amount"] == pytest.approx(0.00003)
    assert metrics["estimated_cost"]["pricing_date"] == "2026-10-01"


def test_ci_runs_offline_checks_without_provider_credentials():
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pytest -q" in ci
    assert "compileall" in ci
    assert "GOOGLE_API_KEY" not in ci and "COHERE_API_KEY" not in ci
