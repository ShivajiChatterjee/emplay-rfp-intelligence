"""FastAPI application wired to the validated Emplay services."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from agents.field_specs import OFFICIAL_FIELD_LABELS
from agents.ingestion_agent import ensure_bid_indexed
from agents.retrieval_agent import RetrievalRequest, search_evidence
from agents.schemas import AgentTrace
from api.activity import activity_store
from api.catalog import (
    CatalogError,
    bid_directory,
    data_root,
    document_records,
    extraction_evidence,
    find_trace,
    historical_runs,
    index_version,
    list_bids,
    load_extraction,
    output_root,
    public_payload,
    resolve_document,
    validate_bid_id,
)
from api.jobs import JobQueueFull, jobs
from api.models import AskRequest, CompareRequest, ExtractRequest, JobView, SearchQuery
from api.semantic_cache import answer_cache
from api.run_inspection import normalize_trace
from api.telemetry import Timer, telemetry
from rag.cohere_reranker import MODEL as COHERE_MODEL
from rag.vector_store import COLLECTION, DENSE_MODEL, SPARSE_MODEL
from workflow import WorkflowRequest, run_workflow


ALLOWED_EXTENSIONS = {".pdf", ".html", ".htm"}


def _http_catalog_error(exc: CatalogError) -> HTTPException:
    status = 404 if "not found" in str(exc).casefold() or "not available" in str(exc).casefold() else 400
    return HTTPException(status_code=status, detail=str(exc))


def _retrieval_config() -> dict[str, Any]:
    return {
        "collection": COLLECTION,
        "dense_model": DENSE_MODEL,
        "sparse_model": SPARSE_MODEL,
        "reranker": COHERE_MODEL,
        "gemini_model": os.getenv("GEMINI_MODEL", ""),
    }


def _sources_fresh(bid_id: str, answer: dict[str, Any]) -> bool:
    known = {item["name"] for item in document_records(bid_id)}
    sources = answer.get("sources", [])
    return bool(sources) and all(
        source.get("file") in known
        and isinstance(source.get("page"), int)
        and source.get("page", 0) >= 1
        and bool(source.get("chunk_id"))
        for source in sources
    )


def _run_index(bid_id: str, run_id: str) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    with Timer() as timer:
        status = ensure_bid_indexed(bid_directory(bid_id), output_root())
    trace = AgentTrace(
        agent="ingestion",
        action="ensure_bid_indexed",
        status="error" if status.status == "error" else "success",
        started_at=started,
        latency_ms=timer.latency_ms,
        tool_calls=["agents.ingestion_agent.ensure_bid_indexed"],
        input_summary={"bid_id": bid_id},
        output_summary={
            "status": status.status,
            "documents": status.documents,
            "chunks": status.chunks,
            "warning_count": len(status.warnings),
            "error_count": len(status.errors),
        },
        error="Indexing failed" if status.status == "error" else None,
    )
    telemetry.record(
        run_id,
        mode="index",
        bid_ids=[bid_id],
        latency_ms=timer.latency_ms,
        state={"agent_traces": [trace], "retry_counts": {}},
    )
    answer_cache.invalidate_bid(bid_id)
    return {
        "run_id": run_id,
        "run_status": status.status,
        "metrics_available": True,
        "trace": [trace.model_dump(mode="json")],
        "result": public_payload(status.model_dump(mode="json")),
    }


def _run_extract(payload: ExtractRequest) -> dict[str, Any]:
    request = WorkflowRequest(
        mode="extract",
        bid_folders=(bid_directory(payload.bid_id),),
        output_dir=output_root() / payload.bid_id,
        max_retries=payload.max_retries,
        allow_query_rewrite=payload.allow_query_rewrite,
        semantic_validation=payload.semantic_validation,
        resume=payload.resume,
    )
    with Timer() as timer:
        state = run_workflow(request)
    run_id = state["run_id"]
    metrics = telemetry.record(
        run_id,
        mode="extract",
        bid_ids=[payload.bid_id],
        latency_ms=timer.latency_ms,
        state=state,
    )
    manifest = state.get("run_manifest")
    return {
        "run_id": run_id,
        "run_status": manifest.run_status if manifest else "incomplete",
        "final_submission_published": bool(manifest and manifest.final_submission_published),
        "metrics_available": bool(metrics),
    }


def _comparison_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Bid comparison", "", "| Field | " + " | ".join(payload["bid_ids"]) + " |", "|---|" + "|".join("---" for _ in payload["bid_ids"]) + "|"]
    for field in payload["fields"]:
        cells = []
        for bid_id in payload["bid_ids"]:
            value = field["bids"][bid_id]["value"]
            cells.append(json.dumps(value, ensure_ascii=False).replace("|", "\\|"))
        lines.append(f"| {field['field_name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _application_status(run_status: str) -> str:
    if run_status in {"complete", "ready", "indexed", "success"}:
        return "succeeded"
    if run_status in {"partial", "incomplete"}:
        return "partial"
    return "failed"


def create_app() -> FastAPI:
    app = FastAPI(title="Emplay RFP Intelligence", version="1.0.0")
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    app.mount("/assets", StaticFiles(directory=frontend), name="assets")

    @app.exception_handler(CatalogError)
    async def catalog_exception_handler(_request, exc: CatalogError):
        response = _http_catalog_error(exc)
        return JSONResponse(status_code=response.status_code, content={"detail": response.detail})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return FileResponse(frontend / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse(frontend / "favicon.svg", media_type="image/svg+xml")

    @app.get("/api/health")
    def health():
        return {
            "status": "ready",
            "dependencies": {
                "qdrant": "configured" if os.getenv("QDRANT_URL") else "not_configured",
                "cohere": "configured" if os.getenv("COHERE_API_KEY") else "not_configured",
                "gemini": "configured" if os.getenv("GOOGLE_API_KEY") else "not_configured",
                "langsmith": "configured" if os.getenv("LANGSMITH_API_KEY") else "not_configured",
            },
            "external_checks_performed": False,
            "cache": answer_cache.stats(),
        }

    @app.get("/api/bids")
    def bids():
        return {"bids": list_bids(), "application_runs": activity_store.list(limit=8)}

    @app.get("/api/runs")
    def runs(scope: str = Query(default="application", pattern="^(application|historical)$")):
        if scope == "application":
            return {"scope": scope, "runs": activity_store.list()}
        records = historical_runs(
            exclude_run_ids=activity_store.workflow_run_ids()
        )
        return {"scope": scope, "runs": records}

    @app.post("/api/bids/upload", status_code=201)
    async def upload_bid(
        bid_id: Annotated[str, Form(min_length=1, max_length=80)],
        files: Annotated[list[UploadFile], File(min_length=1)],
    ):
        validate_bid_id(bid_id)
        maximum = max(1, min(
            int(os.getenv("EMPLAY_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))),
            1024 * 1024 * 1024,
        ))
        total_maximum = max(maximum, min(
            int(os.getenv("EMPLAY_MAX_UPLOAD_TOTAL_BYTES", str(200 * 1024 * 1024))),
            2 * 1024 * 1024 * 1024,
        ))
        maximum_files = max(1, min(
            int(os.getenv("EMPLAY_MAX_UPLOAD_FILES", "100")), 500
        ))
        if not files:
            raise HTTPException(status_code=422, detail="Select at least one document")
        if len(files) > maximum_files:
            raise HTTPException(
                status_code=413,
                detail=f"Upload contains more than {maximum_files} documents",
            )
        validated: list[tuple[UploadFile, str]] = []
        seen_names: set[str] = set()
        known_total = 0
        for upload in files:
            supplied = upload.filename or ""
            name = Path(supplied).name
            if supplied != name or name in {"", ".", ".."}:
                raise HTTPException(status_code=400, detail="Unsafe upload filename")
            if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported document extension: {Path(name).suffix or '(none)'}",
                )
            normalized_name = name.casefold()
            if normalized_name in seen_names:
                raise HTTPException(
                    status_code=409,
                    detail=f"Duplicate filename in upload: {name}",
                )
            seen_names.add(normalized_name)
            if upload.size is not None:
                if upload.size == 0:
                    raise HTTPException(
                        status_code=422, detail=f"Empty document is not allowed: {name}"
                    )
                if upload.size > maximum:
                    raise HTTPException(
                        status_code=413, detail=f"Document exceeds upload limit: {name}"
                    )
                known_total += upload.size
            validated.append((upload, name))
        if known_total > total_maximum:
            raise HTTPException(status_code=413, detail="Combined upload size limit exceeded")
        root = data_root()
        root.mkdir(parents=True, exist_ok=True)
        target = bid_directory(bid_id, must_exist=False)
        if target.exists() and any(target.iterdir()):
            raise HTTPException(status_code=409, detail="Bid already exists")
        activity = activity_store.create("upload", [bid_id], status="running")
        temporary = Path(tempfile.mkdtemp(prefix=f".{bid_id}-", dir=root))
        total = 0
        try:
            for upload, name in validated:
                written = 0
                with (temporary / name).open("xb") as destination:
                    while chunk := await upload.read(1024 * 1024):
                        written += len(chunk)
                        total += len(chunk)
                        if written > maximum or total > total_maximum:
                            raise HTTPException(status_code=413, detail="Upload size limit exceeded")
                        destination.write(chunk)
                if written == 0:
                    raise HTTPException(
                        status_code=422, detail=f"Empty document is not allowed: {name}"
                    )
            if target.exists():
                target.rmdir()
            temporary.replace(target)
        except Exception as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            activity_store.update(
                activity["activity_id"],
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: upload failed",
            )
            raise
        records = document_records(bid_id)
        activity_store.update(
            activity["activity_id"],
            status="succeeded",
            finished_at=datetime.now(timezone.utc).isoformat(),
            details={"document_count": len(records), "index_status": "not_indexed"},
        )
        return {
            "bid_id": bid_id,
            "documents": records,
            "uploaded_count": len(records),
            "upload_status": "uploaded",
            "index_status": "not_indexed",
            "activity_id": activity["activity_id"],
        }

    @app.post("/api/bids/{bid_id}/index", response_model=JobView, status_code=202)
    def index_bid(bid_id: str):
        bid_directory(bid_id)
        run_id = str(uuid.uuid4())
        activity = activity_store.create(
            "index", [bid_id], status="queued", workflow_run_id=run_id
        )

        def task():
            activity_store.update(activity["activity_id"], status="running")
            try:
                result = _run_index(bid_id, run_id)
                status = _application_status(result["run_status"])
                activity_store.update(
                    activity["activity_id"],
                    status=status,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    metrics_available=True,
                    trace_available=True,
                    details={"trace": result.pop("trace"), "result": result["result"]},
                )
                if status == "failed":
                    raise RuntimeError("Indexing failed")
                return {**result, "activity_id": activity["activity_id"]}
            except Exception as exc:
                current = activity_store.get(activity["activity_id"])
                if current and current["status"] != "failed":
                    activity_store.update(
                        activity["activity_id"],
                        status="failed",
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        error=f"{type(exc).__name__}: indexing failed",
                    )
                raise
        try:
            record = jobs.submit(
                "index", bid_id, task, activity_id=activity["activity_id"]
            )
            activity_store.update(activity["activity_id"], job_id=record["job_id"])
            return record
        except JobQueueFull as exc:
            activity_store.update(
                activity["activity_id"],
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="Background job capacity is full",
            )
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.get("/api/search")
    def search(
        bid_id: str,
        query: str = Query(min_length=2, max_length=2000),
        top_k: int = Query(default=5, ge=1, le=20),
        candidate_k: int = Query(default=20, ge=1, le=100),
        doc_type: str | None = None,
        addendum_number: int | None = Query(default=None, ge=1),
        file_name: str | None = None,
    ):
        params = SearchQuery(
            bid_id=bid_id,
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            doc_type=doc_type,
            addendum_number=addendum_number,
            file_name=file_name,
        )
        bid_directory(params.bid_id)
        result = search_evidence(
            RetrievalRequest(
                bid_id=params.bid_id,
                field_name="Q&A search",
                question=params.query,
                top_k=params.top_k,
                candidate_k=params.candidate_k,
                doc_type=params.doc_type,
                addendum_number=params.addendum_number,
                file_name=params.file_name,
            )
        )
        if result.status == "error":
            raise HTTPException(status_code=503, detail="Retrieval service unavailable")
        return result.model_dump(mode="json")

    @app.post("/api/ask")
    def ask(payload: AskRequest):
        folder = bid_directory(payload.bid_id)
        activity = activity_store.create("ask", [payload.bid_id], status="running")
        scope = answer_cache.scope(payload.bid_id, index_version(payload.bid_id), _retrieval_config())
        if payload.use_cache:
            cached = answer_cache.get(
                scope,
                payload.question,
                evidence_is_fresh=lambda answer: _sources_fresh(payload.bid_id, answer),
            )
            if cached:
                activity_store.update(
                    activity["activity_id"],
                    status="succeeded",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    cache_hit=True,
                    details={"grounded": True, "cache_hit": True},
                )
                return {**cached, "activity_id": activity["activity_id"]}
        request = WorkflowRequest(
            mode="ask",
            bid_folders=(folder,),
            question=payload.question,
            output_dir=output_root() / payload.bid_id,
        )
        try:
            with Timer() as timer:
                state = run_workflow(request)
            answer = state.get("qa_answer")
            if answer is None or not answer.sources:
                activity_store.update(
                    activity["activity_id"],
                    status="failed",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    workflow_run_id=state.get("run_id"),
                    trace_available=bool(state.get("workflow_trace_path")),
                    error="No grounded answer could be verified",
                )
                raise HTTPException(status_code=422, detail="No grounded answer could be verified")
        except HTTPException:
            raise
        except Exception as exc:
            activity_store.update(
                activity["activity_id"],
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: Q&A workflow failed",
            )
            raise HTTPException(status_code=503, detail="Q&A workflow failed") from exc
        answer_payload = answer.model_dump(mode="json")
        evidence_payload = [
            {
                "bid_id": bid_id,
                "file": item.file_name,
                "page": item.page_number,
                "chunk_id": item.chunk_id,
                "quote": item.quote,
            }
            for key, items in state.get("evidence", {}).items()
            for bid_id in [key.split("::", 1)[0]]
            for item in items
        ]
        cached_payload = {**answer_payload, "evidence": evidence_payload}
        answer_cache.put(scope, payload.question, cached_payload, timer.latency_ms)
        telemetry.record(
            state["run_id"],
            mode="ask",
            bid_ids=[payload.bid_id],
            latency_ms=timer.latency_ms,
            state=state,
        )
        activity_store.update(
            activity["activity_id"],
            status="succeeded",
            finished_at=datetime.now(timezone.utc).isoformat(),
            workflow_run_id=state["run_id"],
            metrics_available=True,
            trace_available=bool(state.get("workflow_trace_path")),
            details={"grounded": True, "citation_count": len(answer.sources)},
        )
        return {
            "answer": cached_payload,
            "run_id": state["run_id"],
            "activity_id": activity["activity_id"],
            "cache": {"hit": False},
        }

    @app.post("/api/extract", response_model=JobView, status_code=202)
    def extract(payload: ExtractRequest):
        bid_directory(payload.bid_id)
        activity = activity_store.create("extract", [payload.bid_id], status="queued")

        def task():
            activity_store.update(activity["activity_id"], status="running")
            try:
                result = _run_extract(payload)
                status = _application_status(result["run_status"])
                activity_store.update(
                    activity["activity_id"],
                    status=status,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    workflow_run_id=result["run_id"],
                    metrics_available=result["metrics_available"],
                    trace_available=True,
                    details={
                        "run_status": result["run_status"],
                        "final_submission_published": result["final_submission_published"],
                    },
                )
                return {**result, "activity_id": activity["activity_id"]}
            except Exception as exc:
                activity_store.update(
                    activity["activity_id"],
                    status="failed",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    error=f"{type(exc).__name__}: extraction workflow failed",
                )
                raise
        try:
            record = jobs.submit(
                "extract", payload.bid_id, task, activity_id=activity["activity_id"]
            )
            activity_store.update(activity["activity_id"], job_id=record["job_id"])
            return record
        except JobQueueFull as exc:
            activity_store.update(
                activity["activity_id"],
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="Background job capacity is full",
            )
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}", response_model=JobView)
    def job(job_id: str):
        if not re.fullmatch(r"[0-9a-f-]{36}", job_id):
            raise HTTPException(status_code=400, detail="Invalid job identifier")
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return record

    @app.get("/api/bids/{bid_id}/fields")
    def fields(bid_id: str):
        payload = load_extraction(bid_id)
        payload["_evidence"] = extraction_evidence(bid_id)
        return payload

    @app.get("/api/bids/{bid_id}/documents/{requested_document_id}")
    def document(bid_id: str, requested_document_id: str):
        path = resolve_document(bid_id, requested_document_id)
        media_type = "application/pdf" if path.suffix.lower() == ".pdf" else "text/html"
        response = FileResponse(path, media_type=media_type, filename=path.name, content_disposition_type="inline")
        response.headers["X-Content-Type-Options"] = "nosniff"
        if media_type == "text/html":
            response.headers["Content-Security-Policy"] = "sandbox; default-src 'none'; img-src data:"
        return response

    @app.post("/api/compare")
    def compare(payload: CompareRequest):
        selected = payload.fields or list(OFFICIAL_FIELD_LABELS)
        unknown = sorted(set(selected) - set(OFFICIAL_FIELD_LABELS))
        if unknown:
            raise HTTPException(status_code=422, detail={"unknown_fields": unknown})
        extractions = {bid_id: load_extraction(bid_id) for bid_id in payload.bid_ids}
        rows = []
        for field_name in selected:
            values = {
                bid_id: extractions[bid_id]["fields"][field_name]
                for bid_id in payload.bid_ids
            }
            normalized = {json.dumps(item.get("value"), sort_keys=True, ensure_ascii=False) for item in values.values()}
            rows.append({"field_name": field_name, "different": len(normalized) > 1, "bids": values})
        result = {"bid_ids": payload.bid_ids, "fields": rows}
        if payload.format == "markdown":
            return PlainTextResponse(_comparison_markdown(result), media_type="text/markdown")
        return result

    @app.get("/api/runs/{run_id}/metrics")
    def run_metrics(run_id: str):
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", run_id):
            raise HTTPException(status_code=400, detail="Invalid run identifier")
        activity = activity_store.get(run_id)
        resolved_run_id = activity.get("workflow_run_id") if activity else run_id
        metrics = telemetry.get(resolved_run_id) if resolved_run_id else None
        if metrics is not None:
            return {
                "available": True,
                "source": "application" if activity else "historical",
                "metrics": metrics,
            }
        if activity is not None:
            return {
                "available": False,
                "source": "application",
                "reason": "Metrics are unavailable for this application operation.",
            }
        try:
            find_trace(run_id)
        except CatalogError as exc:
            if not any(item["run_id"] == run_id for item in historical_runs()):
                raise HTTPException(status_code=404, detail="Run not found") from exc
        return {
            "available": False,
            "source": "historical",
            "reason": "Metrics were not recorded for this historical run.",
        }

    @app.get("/api/runs/{run_id}/trace")
    def run_trace(run_id: str):
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", run_id):
            raise HTTPException(status_code=400, detail="Invalid run identifier")
        activity = activity_store.get(run_id)
        resolved_run_id = activity.get("workflow_run_id") if activity else run_id
        payload = None
        if resolved_run_id:
            try:
                path = find_trace(resolved_run_id)
                payload = json.loads(path.read_text(encoding="utf-8"))
            except CatalogError:
                payload = None
        if payload is None and activity:
            local_trace = activity.get("details", {}).get("trace", [])
            if local_trace:
                payload = {"run_id": resolved_run_id or run_id, "agent_traces": local_trace}
        if payload is None:
            if activity:
                return {
                    "run_id": resolved_run_id or run_id,
                    "available": False,
                    "step_count": 0,
                    "steps": [],
                    "agent_traces": [],
                    "workflow_events": [],
                    "context": {},
                    "reason": "No execution trace was recorded for this application operation.",
                    "source": "application",
                }
            if any(item["run_id"] == run_id for item in historical_runs()):
                return {
                    "run_id": run_id,
                    "available": False,
                    "step_count": 0,
                    "steps": [],
                    "agent_traces": [],
                    "workflow_events": [],
                    "context": {},
                    "reason": "No execution trace was recorded for this historical run.",
                    "source": "historical",
                }
            raise HTTPException(status_code=404, detail="Run trace not found")
        normalized = normalize_trace(
            resolved_run_id or run_id,
            public_payload(payload),
        )
        normalized["source"] = "application" if activity else "historical"
        return normalized

    return app


app = create_app()
