"""Grounded Q&A and final 20-field report assembly."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from agents.evidence_verification import claim_evidence_errors
from agents.field_specs import OFFICIAL_FIELD_LABELS
from agents.llm_client import (
    GenerationSettings,
    GeminiClient,
    GeminiGenerationError,
    get_gemini_client,
)
from agents.prompts import QA_REPORT_PROMPT
from agents.schemas import (
    AddendumChange,
    AgentTrace,
    AnswerCitation,
    CitedAnswer,
    CriticDecision,
    Evidence,
    ExtractedField,
    FinalExtractionOutput,
    FinalFieldRecord,
    SourceCitation,
    ValidationSummary,
)


class SummaryResponse(BaseModel):
    value: str | None = None
    source_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    notes: str = ""


def _sentence_count(value: str) -> int:
    return len([
        part for part in value.replace("!", ".").replace("?", ".").split(".")
        if part.strip()
    ])


def _claim_attributed_citations(
    value: str,
    citations: list[Evidence],
) -> tuple[list[Evidence], list[str]]:
    """Select authentic citations that support every summary claim."""
    diagnostics, errors = claim_evidence_errors(value, citations)
    if errors:
        return [], errors
    supporting_ids = {
        item["evidence_id"]
        for diagnostic in diagnostics
        for item in diagnostic.get("supporting_evidence", [])
    }
    selected: list[Evidence] = []
    seen: set[tuple[str, str, int, str]] = set()
    for citation in citations:
        # One indexed chunk can support several fields through different
        # authentic short spans. Preserve those spans here; final report source
        # records are deduplicated by file/page during assembly.
        key = (
            citation.chunk_id, citation.file_name,
            citation.page_number, citation.quote,
        )
        if citation.chunk_id in supporting_ids and key not in seen:
            seen.add(key)
            selected.append(citation)
    return selected, []


def _deterministic_summary_fallback(
    *, fields: dict[str, ExtractedField], decisions: dict[str, CriticDecision],
) -> ExtractedField | None:
    """Build a summary only from exact values that already passed validation."""
    priority = (
        "Title", "Product", "Term of Bid", "Due Date", "Bid Submission Type",
        "Installation", "Delivery Date", "Payment Terms", "company_name",
    )
    ordered = [*priority, *[name for name in fields if name not in priority]]
    selected_fields: list[tuple[str, ExtractedField]] = []
    for name in ordered:
        field = fields.get(name)
        decision = decisions.get(name)
        if (
            name == "Bid Summary"
            or field is None
            or decision is None
            or decision.status != "passed"
            or not isinstance(field.value, str)
            or not field.value.strip()
            or not field.citations
            or _sentence_count(field.value) != 1
        ):
            continue
        selected_fields.append((name, field))
        if len(selected_fields) == 6:
            break
    if len(selected_fields) < 3:
        return None

    value = " ".join(
        f"{field.value.strip().rstrip('.')} .".replace(" .", ".")
        for _, field in selected_fields
    )
    citations = [
        citation
        for _, field in selected_fields
        for citation in field.citations
    ]
    attributed, errors = _claim_attributed_citations(value, citations)
    if errors or not attributed or not 3 <= _sentence_count(value) <= 6:
        return None
    confidence = min(
        min(field.confidence, decisions[name].confidence)
        for name, field in selected_fields
    )
    return ExtractedField(
        field_name="Bid Summary",
        status="found",
        value=value,
        confidence=confidence,
        citations=attributed,
        reasoning=(
            "Deterministic fallback assembled from exact validated field values "
            "after the generated summary failed claim-level attribution."
        ),
    )


def _source_records(evidence: list[Evidence]) -> list[SourceCitation]:
    seen: set[tuple[str, int]] = set()
    result = []
    for item in evidence:
        key = (item.file_name, item.page_number)
        if key not in seen:
            seen.add(key)
            result.append(SourceCitation(file=key[0], page=key[1]))
    return result


def answer_question(
    *,
    question: str,
    evidence_by_bid: dict[str, list[Evidence]],
    llm_client: GeminiClient | None = None,
) -> CitedAnswer:
    """Answer from separated per-bid evidence and verify every citation."""

    if not any(evidence_by_bid.values()):
        return CitedAnswer(
            answer="Not found in documents",
            confidence=0.0,
            notes="Retrieval returned no supporting evidence.",
        )

    payload = {
        bid_id: [
            {
                "file": item.file_name,
                "page": item.page_number,
                "chunk_id": item.chunk_id,
                "text": item.quote[:3_000],
            }
            for item in evidence
        ]
        for bid_id, evidence in evidence_by_bid.items()
    }
    client = llm_client or get_gemini_client()
    answer = client.generate_structured(
        json.dumps(
            {
                "question": question,
                "evidence_grouped_by_bid": payload,
                "instruction": (
                    "Keep evidence from different bids separate. Cite only the "
                    "provided bid_id, file, page, and chunk_id values."
                ),
            },
            ensure_ascii=False,
        ),
        CitedAnswer,
        system_instruction=QA_REPORT_PROMPT,
        generation=GenerationSettings(temperature=0, max_output_tokens=900),
    ).output

    lookup = {
        (bid_id, item.file_name, item.page_number, item.chunk_id)
        for bid_id, items in evidence_by_bid.items()
        for item in items
    }
    invalid = [
        source
        for source in answer.sources
        if (source.bid_id, source.file, source.page, source.chunk_id) not in lookup
    ]
    if invalid or (answer.answer != "Not found in documents" and not answer.sources):
        return CitedAnswer(
            answer="Not found in documents",
            confidence=0.0,
            notes="Generated answer contained unsupported or missing citations.",
        )
    return answer


def generate_bid_summary(
    *,
    fields: dict[str, ExtractedField],
    decisions: dict[str, CriticDecision],
    llm_client: GeminiClient | None = None,
) -> ExtractedField:
    supported = {
        name: {
            "value": field.value,
            "sources": [
                {
                    "file": citation.file_name,
                    "page": citation.page_number,
                    "chunk_id": citation.chunk_id,
                }
                for citation in field.citations
            ],
        }
        for name, field in fields.items()
        if name != "Bid Summary"
        and decisions.get(name)
        and decisions[name].status == "passed"
        and field.value is not None
    }
    if not supported:
        return ExtractedField(
            field_name="Bid Summary",
            status="not_found",
            value=None,
            reasoning="Not found in documents: no validated fields were available.",
        )

    client = llm_client or get_gemini_client()
    try:
        response = client.generate_structured(
            json.dumps(
                {
                    "validated_fields": supported,
                    "instruction": "Write a grounded 3-6 sentence bid summary.",
                },
                ensure_ascii=False,
            ),
            SummaryResponse,
            system_instruction=QA_REPORT_PROMPT,
            generation=GenerationSettings(temperature=0, max_output_tokens=500),
        ).output
    except GeminiGenerationError as exc:
        if exc.classification not in {
            "transient_unavailable", "timeout", "transport", "quota_or_rate_limit"
        }:
            raise
        fallback = _deterministic_summary_fallback(
            fields=fields, decisions=decisions
        )
        if fallback is None:
            raise
        return fallback.model_copy(update={
            "reasoning": (
                "Deterministic fallback assembled from exact validated field "
                f"values after summary provider failure ({exc.classification})."
            )
        })
    if response.value is None:
        return ExtractedField(
            field_name="Bid Summary",
            status="not_found",
            value=None,
            reasoning=response.notes or "Not found in documents",
        )

    # The model's source_fields list is advisory. Ground every claim against
    # all citations from validated fields, then retain only actual supporters.
    citation_pool = [
        citation
        for name in supported
        for citation in fields[name].citations
    ]
    citations, attribution_errors = _claim_attributed_citations(
        response.value, citation_pool
    )
    sentence_count = _sentence_count(response.value)
    if attribution_errors:
        fallback = _deterministic_summary_fallback(
            fields=fields, decisions=decisions
        )
        if fallback is not None:
            return fallback
    if not citations or attribution_errors or not 3 <= sentence_count <= 6:
        return ExtractedField(
            field_name="Bid Summary",
            status="error",
            value=None,
            reasoning=(
                "Summary lacked complete claim-level source attribution or did "
                "not contain 3-6 sentences."
            ),
        )
    return ExtractedField(
        field_name="Bid Summary",
        status="found",
        value=response.value,
        confidence=response.confidence,
        citations=citations,
        reasoning=response.notes,
    )


def assemble_final_output(
    *,
    bid_id: str,
    fields: dict[str, ExtractedField],
    decisions: dict[str, CriticDecision],
    changes: list[AddendumChange],
    trace: list[AgentTrace] | None = None,
) -> FinalExtractionOutput:
    """Mechanically assemble exactly the official output contract."""

    records: dict[str, FinalFieldRecord] = {}
    passed = failed = not_found = 0
    for name in OFFICIAL_FIELD_LABELS:
        field = fields.get(name)
        decision = decisions.get(name)
        status = decision.status if decision else "not_found"
        if status == "passed" and field and field.value is not None and field.citations:
            passed += 1
            records[name] = FinalFieldRecord(
                value=field.value,
                sources=_source_records(field.citations),
                confidence=min(field.confidence, decision.confidence),
                # Free-form model reasoning is diagnostic and is not itself
                # claim-validated. Publish only validator-controlled notes.
                notes="; ".join(decision.reasons),
            )
        elif status == "failed":
            failed += 1
            records[name] = FinalFieldRecord(
                value=None,
                confidence=0.0,
                notes="Validation failed: " + "; ".join(decision.reasons),
            )
        else:
            not_found += 1
            notes = "Not found in documents"
            if decision and decision.reasons:
                notes += ": " + "; ".join(decision.reasons)
            records[name] = FinalFieldRecord(
                value=None,
                confidence=0.0,
                notes=notes,
            )

    # Deduplicate changes across retry rounds while keeping the latest result.
    change_by_key: dict[tuple[str, int | None], AddendumChange] = {}
    for change in changes:
        decision = decisions.get(change.field_name)
        field = fields.get(change.field_name)
        if (
            decision is not None
            and decision.status == "passed"
            and field is not None
            and field.value is not None
        ):
            change_by_key[(change.field_name, change.addendum_number)] = change

    return FinalExtractionOutput(
        bid_id=bid_id,
        fields=records,
        addendum_changes=list(change_by_key.values()),
        validation=ValidationSummary(
            passed=passed,
            failed=failed,
            not_found=not_found,
        ),
        trace=trace or [],
    )
