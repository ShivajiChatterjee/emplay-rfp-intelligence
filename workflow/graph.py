"""Compiled LangGraph and reusable execution entrypoint."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from workflow.nodes import build_nodes
from workflow.routing import (
    route_after_ingestion,
    route_after_planner,
    route_after_validation,
)
from workflow.services import WorkflowServices
from workflow.state import GraphState, WorkflowRequest
from workflow.checkpoints import (
    checkpoint_addendum_changes,
    load_checkpoint,
    normalize_checkpoint_values,
    repair_checkpoint_contact_values,
    repair_checkpoint_citations,
    recover_addendum_changes_from_traces,
    revalidate_checkpoint_fields,
    reusable_checkpoint_state,
    workflow_fingerprint,
)
from agents.schemas import CriticDecision, Evidence, ExtractedField, FieldRunRecord, RunManifest
from agents.field_specs import OFFICIAL_FIELD_LABELS
from agents.llm_client import configured_gemini_model


def build_workflow(services: WorkflowServices | None = None):
    """Build a graph with adaptive plan, mode, and critic-retry routing."""

    active_services = services or WorkflowServices.default()
    nodes = build_nodes(active_services)
    builder = StateGraph(GraphState)

    for name, node in nodes.items():
        builder.add_node(name, node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"ingest": "ingest", "failure": "failure"},
    )
    builder.add_conditional_edges("ingest", route_after_ingestion)
    builder.add_edge("field_worker", "validate")
    builder.add_conditional_edges("validate", route_after_validation)
    builder.add_edge("report", END)
    builder.add_edge("qa", END)
    builder.add_edge("failure", END)

    return builder.compile()


def _json_default(value: Any):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def export_local_trace(
    *,
    state: GraphState,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"workflow_trace_{state['run_id']}.json"
    statuses = state.get("field_statuses", {})
    status_counts = {
        status: sum(item.status == status for item in statuses.values())
        for status in (
            "passed", "not_found_evidence", "provider_error", "retrieval_error",
            "reconciliation_incomplete", "validation_failed",
        )
        if any(item.status == status for item in statuses.values())
    }
    payload = {
        "run_id": state["run_id"],
        "request": state["request"],
        "plan": state.get("plan"),
        "index_status": state.get("index_status", {}),
        "events": state.get("events", []),
        "agent_traces": state.get("agent_traces", []),
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
        "retry_counts": state.get("retry_counts", {}),
        "validation": state.get("validation", {}),
        "field_statuses": state.get("field_statuses", {}),
        "operational_status_counts": status_counts,
        "checkpoint_fingerprint": state.get("checkpoint_fingerprint"),
        "retrieval_sources": {
            key: [
                {
                    "file": item.file_name,
                    "page": item.page_number,
                    "chunk_id": item.chunk_id,
                }
                for item in items
            ]
            for key, items in state.get("evidence", {}).items()
        },
        "final_output": state.get("final_output"),
        "qa_answer": state.get("qa_answer"),
    }
    target.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return target


def run_workflow(
    request: WorkflowRequest,
    *,
    services: WorkflowServices | None = None,
    export: bool = True,
) -> GraphState:
    """Run one bounded workflow with LangSmith correlation metadata."""

    load_dotenv()
    run_id = str(uuid.uuid4())
    request_payload = {
        "mode": request.mode,
        "goal": request.resolved_goal,
        "question": request.question,
        "bid_folders": [str(path) for path in request.bid_folders],
        "bid_ids": list(request.bid_ids),
        "output_dir": str(request.output_dir),
        "max_retries": request.max_retries,
        "allow_query_rewrite": request.allow_query_rewrite,
        "semantic_validation": request.semantic_validation,
        "resume": request.resume,
    }
    fingerprint = workflow_fingerprint(
        request.bid_folders[0],
        {
            "gemini_model": configured_gemini_model(),
            "allow_query_rewrite": request.allow_query_rewrite,
            "semantic_validation": request.semantic_validation,
            "max_retries": request.max_retries,
        },
    ) if request.mode == "extract" else "not-applicable"
    checkpoint_path = request.checkpoint_path or (
        request.output_dir / f"{request.bid_ids[0]}_field_checkpoints.json"
    )
    request_payload["checkpoint_path"] = str(checkpoint_path)
    checkpoint_records = {}
    if request.mode == "extract" and request.resume and checkpoint_path.exists():
        saved = load_checkpoint(checkpoint_path, fingerprint)
        recovery_candidates = {
            name
            for name, record in saved.get("fields", {}).items()
            if record.get("operational", {}).get("reusable") is not True
            and record.get("operational", {}).get("reconciliation_verified") is True
        }
        normalized = normalize_checkpoint_values(
            checkpoint_path, fingerprint, recovery_candidates
        )
        contact_repaired = repair_checkpoint_contact_values(
            checkpoint_path, fingerprint, recovery_candidates
        )
        repaired = repair_checkpoint_citations(
            checkpoint_path, fingerprint, recovery_candidates
        )
        completeness_candidates = {
            name
            for name, record in saved.get("fields", {}).items()
            if record.get("operational", {}).get("status") == "validation_failed"
            and record.get("operational", {}).get("reconciliation_verified") is True
            and any(
                "omits available evidence components" in reason
                for reason in record.get("operational", {}).get("reasons", [])
            )
        }
        revalidate_checkpoint_fields(
            checkpoint_path,
            fingerprint,
            normalized | contact_repaired | repaired | completeness_candidates,
        )
    if request.mode == "extract" and request.resume:
        checkpoint_records = reusable_checkpoint_state(load_checkpoint(checkpoint_path, fingerprint))
    restored_fields = {
        name: ExtractedField.model_validate(record["field"])
        for name, record in checkpoint_records.items()
    }
    restored_evidence = {
        name: [Evidence.model_validate(item) for item in record.get("evidence", [])]
        for name, record in checkpoint_records.items()
    }
    restored_validation = {
        name: CriticDecision.model_validate(record["decision"])
        for name, record in checkpoint_records.items()
    }
    restored_statuses = {
        name: FieldRunRecord.model_validate(record["operational"])
        for name, record in checkpoint_records.items()
    }
    restored_changes = checkpoint_addendum_changes(checkpoint_records)
    if request.mode == "extract" and request.resume:
        restored_changes = recover_addendum_changes_from_traces(
            request.output_dir,
            fingerprint,
            checkpoint_records,
        )
    graph = build_workflow(services)
    max_concurrency = max(
        1,
        min(int(os.getenv("EMPLAY_MAX_CONCURRENCY", "3")), 6),
    )
    result = graph.invoke(
        {
            "request": request_payload,
            "run_id": run_id,
            "index_status": {},
            "evidence": restored_evidence,
            "fields": restored_fields,
            "validation": restored_validation,
            "validation_attempts": {},
            "retry_counts": {},
            "addendum_changes": restored_changes,
            "warnings": [],
            "errors": [],
            "events": [],
            "agent_traces": [],
            "field_statuses": restored_statuses,
            "checkpoint_fingerprint": fingerprint,
        },
        config={
            "run_name": f"emplay-{request.mode}",
            "tags": ["emplay-rfp", request.mode],
            "metadata": {
                "correlation_id": run_id,
                "bid_ids": list(request.bid_ids),
                "mode": request.mode,
            },
            "max_concurrency": max_concurrency,
            "recursion_limit": 50,
        },
    )

    if export:
        trace_path = export_local_trace(
            state=result,
            output_dir=request.output_dir,
        )
        result.setdefault("events", []).append(
            {
                "run_id": run_id,
                "agent": "orchestrator_planner",
                "action": "trace_exported",
                "summary": {"path": str(trace_path)},
            }
        )
        statuses = result.get("field_statuses", {})
        complete = (
            request.mode == "extract"
            and set(statuses) == set(OFFICIAL_FIELD_LABELS)
            and all(item.terminal and item.status in {"passed", "not_found_evidence"} for item in statuses.values())
        )
        manifest = None
        if request.mode == "extract":
            manifest = RunManifest(
                run_id=run_id,
                bid_id=request.bid_ids[0],
                run_status="complete" if complete else ("partial" if statuses else "incomplete"),
                final_submission_published=complete,
                checkpoint_fingerprint=fingerprint,
                field_statuses=statuses,
                errors=result.get("errors", []),
                warnings=result.get("warnings", []),
                output_path=(str(request.output_dir / f"{request.bid_ids[0]}_extraction.json") if complete else None),
            )
            manifest_path = request.output_dir / f"run_manifest_{run_id}.json"
            manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            result["run_manifest"] = manifest
            result["run_manifest_path"] = manifest_path
        result["workflow_trace_path"] = trace_path
        if result.get("final_output") is not None and complete:
            output_path = request.output_dir / f"{request.bid_ids[0]}_extraction.json"
            output_path.write_text(
                result["final_output"].model_dump_json(indent=2),
                encoding="utf-8",
            )
        elif result.get("qa_answer") is not None:
            output_path = request.output_dir / f"qa_{run_id}.json"
            output_path.write_text(
                result["qa_answer"].model_dump_json(indent=2),
                encoding="utf-8",
            )
    return result
