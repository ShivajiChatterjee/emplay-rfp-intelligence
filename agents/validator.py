"""Deterministic citation checks plus selective Gemini semantic criticism."""

from __future__ import annotations

import json
import re

from agents.llm_client import GenerationSettings, GeminiClient, get_gemini_client
from agents.evidence_verification import claim_evidence_errors, quote_matches
from agents.field_completeness import (
    contact_completeness_errors,
    documentation_structure_errors,
    product_coverage_errors,
)
from agents.prompts import VALIDATOR_CRITIC_PROMPT
from agents.schemas import CriticDecision, Evidence, ExtractedField, FieldSpec


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _retry_or_terminal(
    *,
    field_spec: FieldSpec,
    reasons: list[str],
    retry_count: int,
    max_retries: int,
    confidence: float = 0.0,
) -> CriticDecision:
    can_retry = retry_count < max_retries
    return CriticDecision(
        field_name=field_spec.field_name,
        status="validation_failed" if not can_retry else "failed",
        reasons=reasons,
        corrective_instruction=(
            "Retrieve targeted evidence and extract this field again."
            if can_retry
            else "Validation did not establish a terminal evidence-based result."
        ),
        revised_question=field_spec.questions[
            min(retry_count + 1, len(field_spec.questions) - 1)
        ],
        retry_required=can_retry,
        confidence=confidence,
    )


def _matching_evidence(
    citation: Evidence,
    evidence: list[Evidence],
) -> Evidence | None:
    return next(
        (
            item
            for item in evidence
            if item.chunk_id == citation.chunk_id
            and item.file_name == citation.file_name
            and item.page_number == citation.page_number
        ),
        None,
    )


def validate_field(
    *,
    field: ExtractedField,
    field_spec: FieldSpec,
    evidence: list[Evidence],
    retry_count: int,
    max_retries: int = 2,
    llm_client: GeminiClient | None = None,
    semantic_validation: bool = True,
) -> CriticDecision:
    """Validate grounding and type; never let the model bypass hard checks."""

    if field.status == "error":
        reason = field.reasoning or "Field processing failed"
        folded = reason.casefold()
        status = (
            "provider_error"
            if any(x in folded for x in (
                "quota_or_rate_limit", "429", "resource_exhausted", "daily quota",
                "transient_unavailable", "transport", "timeout", "503", "504",
            ))
            else "retrieval_error" if "retrieval" in folded else "validation_failed"
        )
        return CriticDecision(
            field_name=field_spec.field_name,
            status=status,
            reasons=[reason],
            retry_required=False,
            confidence=0.0,
        )

    if field.status == "not_found" or field.value is None:
        can_retry = retry_count < max_retries
        return CriticDecision(
            field_name=field_spec.field_name,
            status="failed" if can_retry else "not_found_evidence",
            reasons=[field.reasoning or "No supported value was found in retrieved evidence"],
            corrective_instruction=("Retrieve targeted evidence again" if can_retry else None),
            revised_question=field_spec.questions[min(retry_count + 1, len(field_spec.questions) - 1)],
            retry_required=can_retry,
            confidence=0.0,
        )

    if not field.citations:
        return _retry_or_terminal(
            field_spec=field_spec,
            reasons=["Non-null value has no citation"],
            retry_count=retry_count,
            max_retries=max_retries,
        )

    citation_errors = []
    for citation in field.citations:
        source = _matching_evidence(citation, evidence)
        if source is None:
            citation_errors.append(
                f"Citation not present in retrieved evidence: {citation.chunk_id}"
            )
        elif not quote_matches(citation.quote, source.quote):
            citation_errors.append(
                f"Citation quote is not contained in chunk: {citation.chunk_id}"
            )
    if citation_errors:
        return _retry_or_terminal(
            field_spec=field_spec,
            reasons=citation_errors,
            retry_count=retry_count,
            max_retries=max_retries,
        )

    matched_sources = [
        source for citation in field.citations
        if (source := _matching_evidence(citation, evidence)) is not None
    ]
    claim_diagnostics, support_errors = claim_evidence_errors(
        field.value, field.citations
    )
    if support_errors:
        return _retry_or_terminal(
            field_spec=field_spec,
            reasons=[
                "Extracted value has unsupported material claims: " + item
                for item in support_errors
            ],
            retry_count=retry_count,
            max_retries=max_retries,
            confidence=field.confidence,
        )

    if field_spec.field_name == "Bid Bond Requirement":
        cited_text = " ".join(
            _normalize(citation.quote) for citation in field.citations
        )
        indirect_references = (
            "as noted elsewhere",
            "as specified elsewhere",
            "if required elsewhere",
            "where required elsewhere",
        )
        if any(marker in cited_text for marker in indirect_references):
            return _retry_or_terminal(
                field_spec=field_spec,
                reasons=[
                    "The citation only cross-references possible bond requirements "
                    "elsewhere and does not establish the actual bid-bond requirement"
                ],
                retry_count=retry_count,
                max_retries=max_retries,
                confidence=field.confidence,
            )

    if _normalize(field.value) == "not required" and not any(
        "not required" in _normalize(citation.quote)
        or re.search(
            r"\bno\b.{0,80}\brequired\b",
            _normalize(citation.quote),
        )
        for citation in field.citations
    ):
        return _retry_or_terminal(
            field_spec=field_spec,
            reasons=[
                "'Not required' is not explicitly supported by the citations"
            ],
            retry_count=retry_count,
            max_retries=max_retries,
            confidence=field.confidence,
        )

    value = field.value
    type_valid = {
        "string": isinstance(value, str),
        "datetime": isinstance(value, str),
        "list": isinstance(value, list),
        "object": isinstance(value, dict),
        "summary": isinstance(value, str),
    }[field_spec.expected_type]
    if not type_valid:
        return _retry_or_terminal(
            field_spec=field_spec,
            reasons=[
                f"Expected {field_spec.expected_type}, got {type(value).__name__}"
            ],
            retry_count=retry_count,
            max_retries=max_retries,
            confidence=field.confidence,
        )

    if field_spec.expected_type == "summary":
        sentence_count = len(
            [item for item in re.split(r"(?<=[.!?])\s+", value) if item.strip()]
        )
        if not 3 <= sentence_count <= 6:
            return _retry_or_terminal(
                field_spec=field_spec,
                reasons=["Bid Summary must contain 3-6 sentences"],
                retry_count=retry_count,
                max_retries=max_retries,
                confidence=field.confidence,
            )

    completeness_errors: list[str] = []
    if field_spec.field_name == "Product Specification":
        completeness_errors.extend(product_coverage_errors(value, evidence))
    elif field_spec.field_name == "contact_info":
        completeness_errors.extend(contact_completeness_errors(value, evidence))
    elif field_spec.field_name == "Any Additional Documentation Required":
        completeness_errors.extend(documentation_structure_errors(value, evidence))
    if completeness_errors:
        return _retry_or_terminal(
            field_spec=field_spec,
            reasons=completeness_errors,
            retry_count=retry_count,
            max_retries=max_retries,
            confidence=field.confidence,
        )

    # Semantic criticism is reserved for complex values and reconciled/uncertain
    # results. Mechanical string fields remain deterministic to control cost.
    use_semantic = semantic_validation and (
        field_spec.expected_type in {"list", "object", "summary"}
        or field.status == "conflict"
        or field.confidence < 0.6
    )
    if use_semantic:
        try:
            client = llm_client or get_gemini_client()
            prompt = json.dumps(
                {
                    "field": field_spec.model_dump(),
                    "candidate": field.model_dump(mode="json"),
                    "retrieved_evidence": [
                        {
                            "file": item.file_name,
                            "page": item.page_number,
                            "chunk_id": item.chunk_id,
                            "text": item.quote[:2_500],
                        }
                        for item in evidence
                    ],
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                },
                ensure_ascii=False,
            )
            decision = client.generate_structured(
                prompt,
                CriticDecision,
                system_instruction=VALIDATOR_CRITIC_PROMPT,
                generation=GenerationSettings(
                    temperature=0,
                    max_output_tokens=500,
                ),
            ).output
            if decision.field_name != field_spec.field_name:
                raise ValueError("Critic returned a different field name")
            if decision.status == "passed":
                decision.retry_required = False
            elif retry_count >= max_retries:
                decision.status = "validation_failed"
                decision.retry_required = False
            return decision
        except Exception as exc:
            if getattr(exc, "classification", None) == "quota_or_rate_limit":
                return CriticDecision(
                    field_name=field_spec.field_name,
                    status="provider_error",
                    reasons=[f"Semantic validation provider failure: {exc}"],
                    retry_required=False,
                    confidence=field.confidence,
                )
            return _retry_or_terminal(
                field_spec=field_spec,
                reasons=[f"Semantic validation failed: {type(exc).__name__}: {exc}"],
                retry_count=retry_count,
                max_retries=max_retries,
                confidence=field.confidence,
            )

    return CriticDecision(
        field_name=field_spec.field_name,
        status="passed",
        reasons=["Deterministic citation and type checks passed"],
        retry_required=False,
        confidence=field.confidence,
    )
