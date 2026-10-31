"""Single-responsibility LangGraph nodes and specialist worker."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from types import SimpleNamespace

from agents.field_specs import get_field_spec
from agents.field_completeness import (
    contact_completeness_errors,
    enrich_contact_value_from_evidence,
)
from agents.llm_client import configured_gemini_model
from agents.schemas import (
    AgentTrace,
    CriticDecision,
    Evidence,
    EvidenceSearchRequest,
    ExtractedField,
    FieldRunRecord,
)
from workflow.checkpoints import save_field_checkpoint
from workflow.services import WorkflowServices
from workflow.state import GraphState


def _event(state: GraphState, agent: str, action: str, **summary) -> dict:
    return {
        "run_id": state["run_id"],
        "agent": agent,
        "action": action,
        "summary": summary,
    }


def _as_dict(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


def _merge_retrieval_results(primary, supplemental):
    passages = []
    seen = set()
    for item in [*primary.passages, *supplemental.passages]:
        key = item.evidence.chunk_id
        if key in seen:
            continue
        seen.add(key)
        passages.append(
            item.model_copy(update={"rank": len(passages) + 1})
            if hasattr(item, "model_copy")
            else SimpleNamespace(
                rank=len(passages) + 1,
                score=getattr(item, "score", None),
                evidence=item.evidence,
            )
        )
    updates = {
        "question": supplemental.question,
        "status": "success" if passages else "empty",
        "passages": passages,
        "warnings": [
            *getattr(primary, "warnings", []),
            *getattr(supplemental, "warnings", []),
        ],
    }
    if hasattr(primary, "model_copy"):
        return primary.model_copy(update=updates)
    return SimpleNamespace(
        bid_id=primary.bid_id,
        field_name=primary.field_name,
        trace=None,
        **updates,
    )


def build_nodes(services: WorkflowServices) -> dict[str, Any]:
    def planner_node(state: GraphState) -> dict:
        try:
            planning_result = services.plan(state["request"])
            if hasattr(planning_result, "plan"):
                plan = planning_result.plan
                planner_trace = planning_result.trace.model_copy(
                    update={"run_id": state["run_id"]}
                )
                traces = [planner_trace]
                telemetry = planner_trace.metadata
            else:
                plan = planning_result
                traces = []
                telemetry = {}
            return {
                "plan": plan,
                "planner_failed": False,
                "agent_traces": traces,
                "events": [
                    _event(
                        state,
                        "orchestrator_planner",
                        "plan",
                        mode=plan.mode,
                        task_count=len(plan.tasks),
                        selected_fields=list(plan.selected_fields),
                        selected_specialist_groups=(
                            planner_trace.output_summary.get(
                                "selected_specialist_groups", []
                            )
                            if traces
                            else []
                        ),
                        model=telemetry.get("model"),
                        finish_reason=telemetry.get("finish_reason"),
                        token_usage=telemetry.get("token_usage"),
                        latency_ms=(planner_trace.latency_ms if traces else None),
                        retry_attempts=telemetry.get("retry_attempts", 0),
                        structured_output_validation=telemetry.get(
                            "structured_output_validation"
                        ),
                    )
                ],
            }
        except Exception as exc:
            message = f"Planner failed: {type(exc).__name__}: {exc}"
            usage = getattr(exc, "usage", None)
            metadata = {
                "model": configured_gemini_model(),
                "finish_reason": getattr(exc, "finish_reason", None),
                "token_usage": (
                    usage.model_dump() if hasattr(usage, "model_dump") else {}
                ),
                "retry_attempts": getattr(exc, "retry_count", 0),
                "failure_classification": getattr(
                    exc, "classification", type(exc).__name__
                ),
                "structured_output_validation": "failed",
            }
            error_trace = AgentTrace(
                agent="orchestrator_planner",
                action="create_execution_plan",
                status="error",
                started_at=datetime.now(timezone.utc),
                latency_ms=getattr(exc, "latency_ms", 0),
                input_summary={"mode": state["request"].get("mode")},
                output_summary={"validation_status": "failed"},
                tokens_used=getattr(usage, "total_tokens", None),
                error=message,
                run_id=state["run_id"],
                metadata=metadata,
            )
            return {
                "planner_failed": True,
                "errors": [message],
                "agent_traces": [error_trace],
                "events": [
                    _event(
                        state,
                        "orchestrator_planner",
                        "plan_error",
                        error=message,
                        **metadata,
                    )
                ],
            }

    def ingestion_node(state: GraphState) -> dict:
        statuses = {}
        warnings = []
        errors = []
        events = []
        for raw_path in state["request"]["bid_folders"]:
            path = Path(raw_path)
            try:
                status = services.ensure_index(
                    path,
                    Path(state["request"]["output_dir"]),
                )
                payload = _as_dict(status)
            except Exception as exc:
                payload = {
                    "bid_id": path.name,
                    "status": "error",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "warnings": [],
                }
            statuses[path.name] = payload
            warnings.extend(payload.get("warnings", []))
            errors.extend(payload.get("errors", []))
            events.append(
                _event(
                    state,
                    "ingestion",
                    "ensure_index",
                    bid_id=path.name,
                    status=payload.get("status"),
                    documents=payload.get("documents", 0),
                    chunks=payload.get("chunks", 0),
                )
            )
        return {
            "index_status": statuses,
            "warnings": warnings,
            "errors": errors,
            "events": events,
        }

    def field_worker_node(state: GraphState) -> dict:
        field_name = state["worker_field_name"]
        bid_id = state["worker_bid_id"]
        attempt = state["worker_attempt"]
        spec = get_field_spec(field_name)
        warnings = []
        agent_traces = []
        try:
            request = EvidenceSearchRequest(
                bid_id=bid_id,
                field_name=field_name,
                question=state["worker_question"],
                top_k=5,
                candidate_k=20,
            )
            retrieval = services.retrieve(
                request,
                allow_rewrite=(
                    attempt == 0
                    and state["request"].get("allow_query_rewrite", True)
                ),
            )
            contact_supplemented = False
            if field_name == "contact_info":
                supplemental = services.retrieve(
                    EvidenceSearchRequest(
                        bid_id=bid_id,
                        field_name=field_name,
                        question=spec.questions[1],
                        top_k=8,
                        candidate_k=30,
                    ),
                    allow_rewrite=False,
                )
                if getattr(supplemental, "trace", None):
                    agent_traces.append(supplemental.trace)
                warnings.extend(getattr(supplemental, "warnings", []))
                if getattr(supplemental, "status", None) == "error":
                    message = "; ".join(supplemental.warnings) or "Contact supplemental retrieval failed"
                    return {
                        "evidence": {field_name: [item.evidence for item in retrieval.passages]},
                        "fields": {field_name: ExtractedField(field_name=field_name, status="error", reasoning=message)},
                        "field_statuses": {field_name: FieldRunRecord(status="retrieval_error", reasons=[message])},
                        "retry_counts": {field_name: attempt},
                        "errors": [f"{field_name}: {message}"],
                        "agent_traces": agent_traces,
                        "events": [_event(state, spec.specialist_group, "supplemental_retrieval_error", field_name=field_name, error=message)],
                    }
                retrieval = _merge_retrieval_results(retrieval, supplemental)
                contact_supplemented = True
            evidence = [item.evidence for item in retrieval.passages]
            if getattr(retrieval, "trace", None):
                agent_traces.append(retrieval.trace)
            warnings.extend(getattr(retrieval, "warnings", []))

            if getattr(retrieval, "status", None) == "error":
                message = "; ".join(getattr(retrieval, "warnings", [])) or "Retrieval failed"
                field = ExtractedField(field_name=field_name, status="error", reasoning=message)
                return {
                    "evidence": {field_name: evidence},
                    "fields": {field_name: field},
                    "field_statuses": {field_name: FieldRunRecord(
                        status="retrieval_error", reasons=[message], terminal=False, reusable=False
                    )},
                    "retry_counts": {field_name: attempt},
                    "errors": [f"{field_name}: {message}"],
                    "agent_traces": agent_traces,
                    "events": [_event(state, spec.specialist_group, "retrieval_error", field_name=field_name, error=message)],
                }

            empty_retrieval = getattr(retrieval, "status", None) == "empty" and not evidence
            if empty_retrieval:
                field = ExtractedField(
                    field_name=field_name,
                    status="not_found",
                    value=None,
                    reasoning="No supporting evidence was returned for this retrieval question",
                )
                extraction_trace = AgentTrace(
                    agent="extraction_agent", action="extract_field", status="skipped",
                    input_summary={"field_name": field_name},
                    output_summary={"field_status": "not_found", "reason": "empty_retrieval"},
                )
                agent_traces.append(extraction_trace)
            else:
                field, extraction_trace = services.extract(retrieval)
                agent_traces.append(extraction_trace)

            if field.status == "error":
                classification = extraction_trace.metadata.get("failure_classification")
                status = (
                    "provider_error"
                    if classification in {
                        "quota_or_rate_limit", "timeout", "transport",
                        "transient_unavailable", "authentication",
                    }
                    else "validation_failed"
                )
                message = field.reasoning or "Extraction failed"
                return {
                    "evidence": {field_name: evidence},
                    "fields": {field_name: field},
                    "field_statuses": {field_name: FieldRunRecord(status=status, reasons=[message])},
                    "retry_counts": {field_name: attempt},
                    "errors": [f"{field_name}: {message}"],
                    "agent_traces": agent_traces,
                    "events": [_event(state, spec.specialist_group, "extraction_error", field_name=field_name, status=status, error=message)],
                }

            incomplete_contact = (
                field_name == "contact_info"
                and bool(contact_completeness_errors(field.value, evidence))
            )
            if (
                (field.status == "not_found" or incomplete_contact)
                and not contact_supplemented
                and len(spec.questions) > 1
            ):
                supplemental = services.retrieve(
                    EvidenceSearchRequest(
                        bid_id=bid_id,
                        field_name=field_name,
                        question=spec.questions[1],
                        top_k=5,
                        candidate_k=20,
                    ),
                    allow_rewrite=False,
                )
                if getattr(supplemental, "trace", None):
                    agent_traces.append(supplemental.trace)
                warnings.extend(getattr(supplemental, "warnings", []))
                if getattr(supplemental, "status", None) == "error":
                    message = "; ".join(supplemental.warnings) or "Supplemental retrieval failed"
                    return {
                        "evidence": {field_name: evidence},
                        "fields": {field_name: field},
                        "field_statuses": {field_name: FieldRunRecord(
                            status="retrieval_error", reasons=[message]
                        )},
                        "retry_counts": {field_name: attempt},
                        "errors": [f"{field_name}: {message}"],
                        "agent_traces": agent_traces,
                        "events": [_event(state, spec.specialist_group, "supplemental_retrieval_error", field_name=field_name, error=message)],
                    }
                retrieval = _merge_retrieval_results(retrieval, supplemental)
                empty_retrieval = not retrieval.passages
                evidence = [item.evidence for item in retrieval.passages]
                if retrieval.passages:
                    field, supplemental_extraction_trace = services.extract(retrieval)
                    agent_traces.append(supplemental_extraction_trace)
                    if field.status == "error":
                        classification = supplemental_extraction_trace.metadata.get("failure_classification")
                        status = (
                            "provider_error"
                            if classification in {
                                "quota_or_rate_limit", "timeout", "transport",
                                "transient_unavailable", "authentication",
                            }
                            else "validation_failed"
                        )
                        message = field.reasoning or "Supplemental extraction failed"
                        return {
                            "evidence": {field_name: evidence},
                            "fields": {field_name: field},
                            "field_statuses": {field_name: FieldRunRecord(status=status, reasons=[message])},
                            "retry_counts": {field_name: attempt},
                            "errors": [f"{field_name}: {message}"],
                            "agent_traces": agent_traces,
                            "events": [_event(state, spec.specialist_group, "supplemental_extraction_error", field_name=field_name, status=status, error=message)],
                        }

            if empty_retrieval and attempt < state["request"]["max_retries"]:
                reconciled = SimpleNamespace(
                    field=field, evidence=evidence, change=None, warning=None,
                    traces=[], verified=True,
                )
            else:
                reconciled = services.reconcile(
                    bid_id=bid_id,
                    field_spec=spec,
                    field=field,
                    evidence=evidence,
                )
            agent_traces.extend(getattr(reconciled, "traces", []))
            evidence = reconciled.evidence
            field = reconciled.field
            if field_name == "contact_info":
                revised_value, contact_enriched = enrich_contact_value_from_evidence(
                    field.value, evidence
                )
                if contact_enriched:
                    field = field.model_copy(update={"value": revised_value})
                    agent_traces.append(AgentTrace(
                        agent="evidence_verification",
                        action="enrich_submission_contact_from_evidence",
                        status="success",
                        input_summary={"field_name": field_name},
                        output_summary={"evidence_derived": True},
                    ))
            from agents.evidence_verification import reattribute_supported_citations
            field, citations_reattributed = reattribute_supported_citations(field, evidence)
            if citations_reattributed:
                agent_traces.append(AgentTrace(
                    agent="evidence_verification",
                    action="reattribute_supported_citations",
                    status="success",
                    input_summary={"field_name": field_name, "evidence_count": len(evidence)},
                    output_summary={"citation_count": len(field.citations)},
                ))
            changes = [reconciled.change] if reconciled.change else []
            if reconciled.warning:
                warnings.append(reconciled.warning)

            reconciliation_verified = getattr(reconciled, "verified", True)
            operational = FieldRunRecord(
                status=(
                    getattr(reconciled, "failure_status", None)
                    or ("reconciliation_incomplete" if not reconciliation_verified else "validation_failed")
                ),
                reasons=([reconciled.warning or getattr(reconciled, "verification_reason", None)] if (reconciled.warning or getattr(reconciled, "verification_reason", None)) else []),
                reconciliation_verified=reconciliation_verified,
            )

            token_usage = sum(
                trace.tokens_used or 0 for trace in agent_traces
            )
            latency_ms = round(
                sum(trace.latency_ms for trace in agent_traces),
                2,
            )

            return {
                "evidence": {field_name: evidence},
                "fields": {field_name: field},
                "retry_counts": {field_name: attempt},
                "addendum_changes": changes,
                "warnings": warnings,
                "field_statuses": {field_name: operational},
                "errors": ([f"{field_name}: {reconciled.warning}"] if reconciled.warning and not reconciliation_verified else []),
                "agent_traces": agent_traces,
                "events": [
                    _event(
                        state,
                        spec.specialist_group,
                        "retrieve_extract_reconcile",
                        bid_id=bid_id,
                        field_name=field_name,
                        attempt=attempt,
                        evidence_count=len(evidence),
                        field_status=field.status,
                        addendum_change=bool(changes),
                        token_usage=token_usage,
                        latency_ms=latency_ms,
                        sources=[
                            {
                                "file": item.file_name,
                                "page": item.page_number,
                                "chunk_id": item.chunk_id,
                            }
                            for item in evidence
                        ],
                    )
                ],
            }
        except Exception as exc:
            message = (
                f"{field_name} attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            field = ExtractedField(
                field_name=field_name,
                status="error",
                value=None,
                confidence=0.0,
                reasoning=message,
            )
            return {
                "evidence": {field_name: []},
                "fields": {field_name: field},
                "retry_counts": {field_name: attempt},
                "errors": [message],
                "field_statuses": {field_name: FieldRunRecord(
                    status=("provider_error" if getattr(exc, "classification", None) == "quota_or_rate_limit" else "retrieval_error"),
                    reasons=[message], terminal=False, reusable=False,
                )},
                "events": [
                    _event(
                        state,
                        spec.specialist_group,
                        "worker_error",
                        field_name=field_name,
                        attempt=attempt,
                        error=message,
                    )
                ],
            }

    def validator_node(state: GraphState) -> dict:
        updates = {}
        attempts = {}
        events = []
        errors = []
        for field_name, field in state.get("fields", {}).items():
            if field_name == "Bid Summary":
                continue
            existing_status = state.get("field_statuses", {}).get(field_name)
            if existing_status and existing_status.terminal and existing_status.reusable:
                continue
            attempt = state.get("retry_counts", {}).get(field_name, 0)
            if state.get("validation_attempts", {}).get(field_name) == attempt:
                continue
            current = state.get("field_statuses", {}).get(field_name)
            if current and current.status in {"provider_error", "retrieval_error", "reconciliation_incomplete"}:
                decision = CriticDecision(
                    field_name=field_name,
                    status=current.status,
                    reasons=current.reasons,
                    retry_required=False,
                    confidence=0.0,
                )
                updates[field_name] = decision
                attempts[field_name] = attempt
                events.append(_event(state, "validator_critic", "validation_blocked", field_name=field_name, status=current.status))
                continue
            try:
                decision = services.validate(
                    field=field,
                    field_spec=get_field_spec(field_name),
                    evidence=state.get("evidence", {}).get(field_name, []),
                    retry_count=attempt,
                    max_retries=state["request"]["max_retries"],
                    semantic_validation=state["request"].get(
                        "semantic_validation",
                        True,
                    ),
                )
            except Exception as exc:
                decision = CriticDecision(
                    field_name=field_name,
                    status=(
                        "failed"
                        if attempt < state["request"]["max_retries"]
                        else "validation_failed"
                    ),
                    reasons=[f"Validator error: {type(exc).__name__}: {exc}"],
                    revised_question=get_field_spec(field_name).questions[0],
                    retry_required=attempt < state["request"]["max_retries"],
                    confidence=0.0,
                )
                errors.extend(decision.reasons)
            # Python owns retry bounds even if an LLM asks for more.
            if attempt >= state["request"]["max_retries"]:
                decision.retry_required = False
                if decision.status == "failed":
                    decision.status = "validation_failed"
            updates[field_name] = decision
            operational_status = {
                "not_found": "not_found_evidence",
                "failed": "validation_failed",
            }.get(decision.status, decision.status)
            terminal = operational_status in {"passed", "not_found_evidence"}
            record = FieldRunRecord(
                status=operational_status,
                terminal=terminal,
                reusable=terminal,
                reasons=decision.reasons,
                reconciliation_verified=(state.get("field_statuses", {}).get(field_name).reconciliation_verified if state.get("field_statuses", {}).get(field_name) else False),
            )
            try:
                checkpoint_path = state["request"].get("checkpoint_path")
                fingerprint = state.get("checkpoint_fingerprint")
                if checkpoint_path and fingerprint:
                    field_changes = [
                        item.model_dump(mode="json")
                        for item in state.get("addendum_changes", [])
                        if item.field_name == field_name
                    ]
                    save_field_checkpoint(
                        Path(checkpoint_path), fingerprint, field_name,
                        {
                            "field": field.model_dump(mode="json"),
                            "evidence": [item.model_dump(mode="json") for item in state.get("evidence", {}).get(field_name, [])],
                            "decision": decision.model_dump(mode="json"),
                            "operational": record.model_dump(mode="json"),
                            "retry_count": attempt,
                            "addendum_changes": field_changes,
                        },
                    )
            except Exception as exc:
                errors.append(f"Checkpoint write failed for {field_name}: {exc}")
            state.setdefault("field_statuses", {})[field_name] = record
            attempts[field_name] = attempt
            events.append(
                _event(
                    state,
                    "validator_critic",
                    "validate_field",
                    field_name=field_name,
                    attempt=attempt,
                    status=decision.status,
                    retry_required=decision.retry_required,
                    reasons=decision.reasons,
                )
            )
        return {
            "validation": updates,
            "validation_attempts": attempts,
            "errors": errors,
            "field_statuses": {
                name: state.get("field_statuses", {}).get(name)
                for name in updates
                if state.get("field_statuses", {}).get(name) is not None
            },
            "events": events,
        }

    def qa_node(state: GraphState) -> dict:
        question = state["request"]["question"]
        evidence_by_bid: dict[str, list[Evidence]] = {}
        warnings = []
        events = []
        for bid_id in state["request"]["bid_ids"]:
            try:
                result = services.retrieve(
                    EvidenceSearchRequest(
                        bid_id=bid_id,
                        field_name="Q&A",
                        question=question,
                        top_k=5,
                        candidate_k=20,
                    ),
                    allow_rewrite=state["request"].get(
                        "allow_query_rewrite",
                        True,
                    ),
                )
                evidence_by_bid[bid_id] = [
                    passage.evidence for passage in result.passages
                ]
                warnings.extend(getattr(result, "warnings", []))
            except Exception as exc:
                evidence_by_bid[bid_id] = []
                warnings.append(
                    f"Q&A retrieval failed for {bid_id}: {type(exc).__name__}: {exc}"
                )
            events.append(
                _event(
                    state,
                    "retrieval_search",
                    "qa_retrieve",
                    bid_id=bid_id,
                    evidence_count=len(evidence_by_bid[bid_id]),
                )
            )
        try:
            answer = services.answer(
                question=question,
                evidence_by_bid=evidence_by_bid,
            )
        except Exception as exc:
            from agents.schemas import CitedAnswer

            answer = CitedAnswer(
                answer="Not found in documents",
                confidence=0.0,
                notes=f"Answer generation failed: {type(exc).__name__}: {exc}",
            )
        return {
            "evidence": {
                f"{bid_id}::qa": items
                for bid_id, items in evidence_by_bid.items()
            },
            "qa_answer": answer,
            "warnings": warnings,
            "events": [
                *events,
                _event(
                    state,
                    "qa_report",
                    "answer",
                    cited_sources=len(answer.sources),
                    confidence=answer.confidence,
                ),
            ],
        }

    def report_node(state: GraphState) -> dict:
        fields = dict(state.get("fields", {}))
        decisions = dict(state.get("validation", {}))
        evidence = [
            item
            for field_items in state.get("evidence", {}).values()
            for item in field_items
        ]
        unresolved = {
            name: record
            for name, record in state.get("field_statuses", {}).items()
            if not record.terminal
        }
        errors = []
        if unresolved or len(state.get("field_statuses", {})) < 19:
            message = "Bid Summary deferred because one or more source fields are incomplete"
            summary = ExtractedField(
                field_name="Bid Summary", status="error", value=None, reasoning=message
            )
            summary_decision = CriticDecision(
                field_name="Bid Summary", status="validation_failed",
                reasons=[message], retry_required=False, confidence=0.0,
            )
            summary_status = FieldRunRecord(
                status="validation_failed", reasons=[message], terminal=False, reusable=False
            )
            errors.append(message)
        else:
          try:
            summary = services.summarize(
                fields=fields,
                decisions=decisions,
            )
            summary_decision = services.validate(
                field=summary,
                field_spec=get_field_spec("Bid Summary"),
                evidence=evidence,
                retry_count=0,
                max_retries=0,
                semantic_validation=False,
            )
          except Exception as exc:
            summary = ExtractedField(
                field_name="Bid Summary",
                status="error",
                value=None,
                reasoning=f"Summary generation failed: {type(exc).__name__}: {exc}",
            )
            summary_decision = CriticDecision(
                field_name="Bid Summary",
                status="provider_error" if getattr(exc, "classification", None) == "quota_or_rate_limit" else "validation_failed",
                reasons=[summary.reasoning],
                retry_required=False,
                confidence=0.0,
            )
          summary_status = FieldRunRecord(
              status=({"not_found": "not_found_evidence", "failed": "validation_failed"}.get(summary_decision.status, summary_decision.status)),
              terminal=summary_decision.status in {"passed", "not_found", "not_found_evidence"},
              reusable=summary_decision.status in {"passed", "not_found", "not_found_evidence"},
              reasons=summary_decision.reasons,
              reconciliation_verified=True,
          )
          if not summary_status.terminal:
              errors.extend(
                  f"Bid Summary: {reason}" for reason in summary_status.reasons
              )
        fields["Bid Summary"] = summary
        decisions["Bid Summary"] = summary_decision
        all_statuses = {**state.get("field_statuses", {}), "Bid Summary": summary_status}
        publishable = (
            len(all_statuses) == 20
            and all(
                item.terminal and item.status in {"passed", "not_found_evidence"}
                for item in all_statuses.values()
            )
        )
        output = (
            services.assemble(
                bid_id=state["request"]["bid_ids"][0],
                fields=fields,
                decisions=decisions,
                changes=state.get("addendum_changes", []),
                trace=state.get("agent_traces", []),
            )
            if publishable else None
        )
        return {
            "fields": {"Bid Summary": summary},
            "validation": {"Bid Summary": summary_decision},
            "field_statuses": {"Bid Summary": summary_status},
            "errors": errors,
            "final_output": output,
            "events": [
                _event(
                    state,
                    "qa_report",
                    "finalize_report",
                    publishable=publishable,
                    operational_status_counts={
                        status: sum(item.status == status for item in all_statuses.values())
                        for status in {
                            "passed", "not_found_evidence", "provider_error", "retrieval_error",
                            "reconciliation_incomplete", "validation_failed",
                        }
                        if any(item.status == status for item in all_statuses.values())
                    },
                )
            ],
        }

    def failure_node(state: GraphState) -> dict:
        return {
            "events": [
                _event(
                    state,
                    "orchestrator_planner",
                    "stop_failure",
                    error_count=len(state.get("errors", [])),
                )
            ]
        }

    return {
        "planner": planner_node,
        "ingest": ingestion_node,
        "field_worker": field_worker_node,
        "validate": validator_node,
        "qa": qa_node,
        "report": report_node,
        "failure": failure_node,
    }
