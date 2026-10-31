"""Sequential, budget-enforced partial extraction smoke runner."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.extractor import extract_field
from agents.field_specs import OFFICIAL_FIELD_LABELS, get_field_spec
from agents.ingestion_agent import ensure_bid_indexed
from agents.llm_client import GeminiClient, get_gemini_client
from agents.reconciler import reconcile_field
from agents.retrieval_agent import search_evidence
from agents.schemas import (
    AddendumChange,
    AgentTrace,
    CriticDecision,
    EvidenceSearchRequest,
    ExtractedField,
    PartialExtractionOutput,
    PartialFieldRecord,
    PartialUsage,
    SourceCitation,
    ValidationSummary,
)
from agents.validator import validate_field


MAX_PARTIAL_FIELDS = 3
MAX_GEMINI_CALLS = 6
MAX_RETRIEVAL_CALLS = 6


class ApiCallLimitError(RuntimeError):
    """Raised before an external call would exceed the smoke-test budget."""


class _BudgetedGeminiClient:
    def __init__(self, client: GeminiClient, maximum: int) -> None:
        self._client = client
        self.maximum = maximum
        self.calls = 0

    @property
    def model(self) -> str:
        return self._client.model

    def generate_structured(self, *args, **kwargs):
        if self.calls >= self.maximum:
            raise ApiCallLimitError(
                f"Gemini call limit reached ({self.maximum})"
            )
        self.calls += 1
        return self._client.generate_structured(*args, **kwargs)


class _RetrievalBudget:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.calls = 0

    def search(self, request):
        if self.calls >= self.maximum:
            raise ApiCallLimitError(
                f"Qdrant/Cohere query limit reached ({self.maximum})"
            )
        self.calls += 1
        return search_evidence(request)


def _source_records(field: ExtractedField) -> list[SourceCitation]:
    unique: dict[tuple[str, int], SourceCitation] = {}
    for citation in field.citations:
        key = (citation.file_name, citation.page_number)
        unique[key] = SourceCitation(file=key[0], page=key[1])
    return list(unique.values())


def _token_usage(traces: list[AgentTrace], gemini_calls: int) -> PartialUsage:
    prompt = output = total = 0
    calls_with_metadata = 0
    for trace in traces:
        usage = trace.metadata.get("token_usage") if trace.metadata else None
        if not usage:
            continue
        calls_with_metadata += 1
        prompt += usage.get("prompt_tokens") or 0
        output += usage.get("output_tokens") or 0
        total += usage.get("total_tokens") or 0
    return PartialUsage(
        gemini_calls=gemini_calls,
        prompt_tokens=prompt if calls_with_metadata else None,
        output_tokens=output if calls_with_metadata else None,
        total_tokens=total if calls_with_metadata else None,
        calls_without_token_metadata=max(gemini_calls - calls_with_metadata, 0),
    )


def _record_for(
    field: ExtractedField,
    decision: CriticDecision,
) -> PartialFieldRecord:
    if (
        decision.status == "passed"
        and field.value is not None
        and field.citations
    ):
        return PartialFieldRecord(
            value=field.value,
            sources=_source_records(field),
            confidence=min(field.confidence, decision.confidence),
            notes=field.reasoning or "; ".join(decision.reasons),
            validation_status="passed",
            validation_reasons=decision.reasons,
            retry_count=0,
        )
    prefix = (
        "Validation failed"
        if decision.status == "failed"
        else "Not found in documents"
    )
    return PartialFieldRecord(
        value=None,
        sources=[],
        confidence=0.0,
        notes=f"{prefix}: {'; '.join(decision.reasons)}",
        validation_status=decision.status,
        validation_reasons=decision.reasons,
        retry_count=0,
    )


def run_partial_extraction_smoke(
    *,
    bid_dir: Path,
    selected_fields: tuple[str, ...],
    output_dir: Path,
    gemini_client: GeminiClient | None = None,
) -> tuple[PartialExtractionOutput, Path]:
    """Run only selected fields with no rewrites, retries, or summary call."""

    if not 1 <= len(selected_fields) <= MAX_PARTIAL_FIELDS:
        raise ValueError("partial extraction requires between one and three fields")
    if len(set(selected_fields)) != len(selected_fields):
        raise ValueError("partial extraction fields must be unique")
    unknown = set(selected_fields) - set(OFFICIAL_FIELD_LABELS)
    if unknown:
        raise ValueError(f"Unknown official fields: {sorted(unknown)}")
    if "Bid Summary" in selected_fields:
        raise ValueError("Bid Summary is not supported by the partial smoke runner")

    started = time.perf_counter()
    traces: list[AgentTrace] = []
    warnings: list[str] = []
    errors: list[str] = []
    fields: dict[str, ExtractedField] = {}
    decisions: dict[str, CriticDecision] = {}
    changes: list[AddendumChange] = []
    gemini = _BudgetedGeminiClient(
        gemini_client or get_gemini_client(),
        MAX_GEMINI_CALLS,
    )
    retrieval = _RetrievalBudget(MAX_RETRIEVAL_CALLS)

    index_started = datetime.now(timezone.utc)
    index_timer = time.perf_counter()
    status = ensure_bid_indexed(bid_dir, output_dir)
    index_payload = status.model_dump(mode="json")
    traces.append(
        AgentTrace(
            agent="ingestion",
            action="ensure_index",
            status="error" if status.status == "error" else "success",
            started_at=index_started,
            latency_ms=round((time.perf_counter() - index_timer) * 1000, 2),
            tool_calls=["qdrant.index_readiness"],
            output_summary=index_payload,
            error="; ".join(status.errors) or None,
        )
    )
    warnings.extend(status.warnings)
    errors.extend(status.errors)

    if status.status != "error":
        for field_name in selected_fields:
            spec = get_field_spec(field_name)
            try:
                result = retrieval.search(
                    EvidenceSearchRequest(
                        bid_id=bid_dir.name,
                        field_name=field_name,
                        question=spec.questions[0],
                        top_k=5,
                        candidate_k=20,
                    )
                )
                traces.append(result.trace)
                warnings.extend(result.warnings)
                field, extraction_trace = extract_field(
                    result,
                    llm_client=gemini,
                )
                traces.append(extraction_trace)
                outcome = reconcile_field(
                    bid_id=bid_dir.name,
                    field_spec=spec,
                    field=field,
                    evidence=[item.evidence for item in result.passages],
                    llm_client=gemini,
                    retrieval_fn=retrieval.search,
                    search_addenda=True,
                )
                traces.extend(outcome.traces)
                if outcome.warning:
                    warnings.append(outcome.warning)
                if outcome.change:
                    changes.append(outcome.change)
                field = outcome.field
                validation_started = datetime.now(timezone.utc)
                validation_timer = time.perf_counter()
                decision = validate_field(
                    field=field,
                    field_spec=spec,
                    evidence=outcome.evidence,
                    retry_count=0,
                    max_retries=0,
                    semantic_validation=False,
                )
                traces.append(
                    AgentTrace(
                        agent="validator_critic",
                        action="validate_field",
                        status="success",
                        started_at=validation_started,
                        latency_ms=round(
                            (time.perf_counter() - validation_timer) * 1000,
                            2,
                        ),
                        input_summary={"field_name": field_name},
                        output_summary={
                            "status": decision.status,
                            "reasons": decision.reasons,
                        },
                    )
                )
            except Exception as exc:
                message = f"{field_name}: {type(exc).__name__}: {exc}"
                errors.append(message)
                field = ExtractedField(
                    field_name=field_name,
                    status="error",
                    value=None,
                    confidence=0.0,
                    reasoning=message,
                )
                decision = CriticDecision(
                    field_name=field_name,
                    status="not_found",
                    reasons=[message],
                    retry_required=False,
                    confidence=0.0,
                )
            fields[field_name] = field
            decisions[field_name] = decision

    for field_name in selected_fields:
        if field_name not in fields:
            message = "Not found in documents: indexing was unavailable"
            fields[field_name] = ExtractedField(
                field_name=field_name,
                status="error",
                reasoning=message,
            )
            decisions[field_name] = CriticDecision(
                field_name=field_name,
                status="not_found",
                reasons=[message],
                confidence=0.0,
            )

    records = {
        name: _record_for(fields[name], decisions[name])
        for name in selected_fields
    }
    summary = ValidationSummary(
        passed=sum(item.status == "passed" for item in decisions.values()),
        failed=sum(item.status == "failed" for item in decisions.values()),
        not_found=sum(item.status == "not_found" for item in decisions.values()),
    )
    usage = _token_usage(traces, gemini.calls)
    usage.qdrant_cohere_queries = retrieval.calls
    artifact = PartialExtractionOutput(
        bid_id=bid_dir.name,
        selected_fields=selected_fields,
        api_limits={
            "max_fields": MAX_PARTIAL_FIELDS,
            "max_gemini_calls": MAX_GEMINI_CALLS,
            "max_qdrant_cohere_queries": MAX_RETRIEVAL_CALLS,
            "max_retries_per_field": 0,
            "max_parallelism": 1,
        },
        index_status=index_payload,
        fields=records,
        addendum_changes=changes,
        validation=summary,
        errors=errors,
        warnings=warnings,
        usage=usage,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        trace=traces,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{bid_dir.name}_partial_extraction.json"
    target.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return artifact, target
