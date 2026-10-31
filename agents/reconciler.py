"""Evidence-grounded addendum reconciliation with proactive addendum search."""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone

from agents.llm_client import GenerationSettings, GeminiClient, get_gemini_client
from agents.evidence_verification import claim_evidence_errors
from agents.prompts import ADDENDUM_RECONCILIATION_PROMPT
from agents.schemas import (
    AddendumChange,
    AgentTrace,
    Evidence,
    EvidenceSearchRequest,
    ExtractedField,
    FieldCandidate,
    FieldSpec,
    ProductSpecificationReconciliationDecision,
    ReconciliationDecision,
    SourceCitation,
)


@dataclass
class ReconciliationOutcome:
    field: ExtractedField
    evidence: list[Evidence]
    change: AddendumChange | None = None
    warning: str | None = None
    traces: list[AgentTrace] = dataclass_field(default_factory=list)
    verified: bool = False
    verification_reason: str | None = None
    failure_status: str | None = None


def _unique_evidence(items: list[Evidence]) -> list[Evidence]:
    unique: dict[str, Evidence] = {}
    for item in items:
        unique[item.chunk_id] = item
    return list(unique.values())


def _addendum_may_change_field(field_spec: FieldSpec, evidence: list[Evidence]) -> bool:
    """Conservative gate: skip Gemini only when no amendment language exists."""
    change_markers = (
        "amend", "revis", "replace", "supersed", "change", "extend",
        "shall now", "new deadline", "corrected", "updated",
    )
    field_terms = {
        token.casefold().strip("?:,.")
        for question in (field_spec.field_name, *field_spec.questions)
        for token in question.split()
        if len(token.strip("?:,.")) >= 4
    }
    for item in evidence:
        text = item.quote.casefold()
        if any(marker in text for marker in change_markers) and any(
            term in text for term in field_terms
        ):
            return True
    return False


def _deduplicate_values(values: list) -> list:
    seen = set()
    result = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False).casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _explicit_replacement(evidence: Evidence) -> bool:
    text = evidence.quote.casefold()
    return any(marker in text for marker in (
        "replace in its entirety", "replaces the requirement", "supersedes",
        "in lieu of", "is hereby deleted", "all prior requirements are cancelled",
    ))


def _affirmative_bid_bond_requirement(evidence: list[Evidence]) -> Evidence | None:
    """Return only evidence that affirmatively requires proposal security.

    Generic insurance/bond clauses, post-award bonding, joint-venture capacity,
    and cross-references to requirements "elsewhere" do not establish a bid
    bond requirement or amount for this solicitation.
    """
    for item in evidence:
        text = " ".join(item.quote.casefold().split())
        if not any(term in text for term in ("bid bond", "proposal bond", "proposal security", "bid guarantee")):
            continue
        if any(marker in text for marker in (
            "as noted elsewhere", "as specified elsewhere", "if required",
            "joint venture", "after award", "insurance and/or bonds",
        )):
            continue
        if re.search(
            r"(?:shall|must|required to|requirement is).{0,100}(?:bid bond|proposal bond|proposal security|bid guarantee)|"
            r"(?:bid bond|proposal bond|proposal security|bid guarantee).{0,100}(?:shall|must|required)",
            text,
        ):
            return item
    return None


def _normalized_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _product_addendum_batches(evidence: list[Evidence], size: int = 3) -> list[list[Evidence]]:
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.addendum_number if item.addendum_number is not None else 10**9,
            item.document_date or "",
            item.file_name,
            item.page_number,
            item.chunk_id,
        ),
    )
    batches: list[list[Evidence]] = []
    for addendum_number in dict.fromkeys(item.addendum_number for item in ordered):
        group = [item for item in ordered if item.addendum_number == addendum_number]
        batches.extend(group[index:index + size] for index in range(0, len(group), size))
    return batches


# Product reconciliation emits only compact operations, not a regenerated
# specification. Use the client's supported ceiling for the initial compact
# patch attempt; a MAX_TOKENS result still falls back to bounded subdivision.
PRODUCT_PATCH_MAX_OUTPUT_TOKENS = 4_096
PRODUCT_PATCH_MAX_PROVIDER_CALLS = 10
PRODUCT_PATCH_MAX_SUBDIVISION_DEPTH = 2


def _reconcile_product_specification(
    *,
    bid_id: str,
    field: ExtractedField,
    all_evidence: list[Evidence],
    addendum_evidence: list[Evidence],
    llm_client: GeminiClient | None,
    prior_traces: list[AgentTrace],
) -> ReconciliationOutcome:
    """Apply compact category/attribute patches without regenerating the object."""

    if not isinstance(field.value, dict):
        return ReconciliationOutcome(
            field=field,
            evidence=all_evidence,
            warning="Product Specification reconciliation requires an object value",
            traces=prior_traces,
            verified=False,
            failure_status="reconciliation_incomplete",
        )

    client = llm_client or get_gemini_client()
    structure = {
        category: list(attributes) if isinstance(attributes, dict) else []
        for category, attributes in field.value.items()
    }
    decisions: list[tuple[ProductSpecificationReconciliationDecision, list[Evidence]]] = []
    traces = list(prior_traces)
    provider_calls = 0

    def review_batch(
        batch: list[Evidence],
        batch_label: str,
        subdivision_depth: int = 0,
    ) -> tuple[
        list[tuple[ProductSpecificationReconciliationDecision, list[Evidence]]],
        str | None,
        str | None,
    ]:
        """Review one batch, subdividing only a truncated multi-passage call."""

        nonlocal provider_calls
        if provider_calls >= PRODUCT_PATCH_MAX_PROVIDER_CALLS:
            return (
                [],
                f"Product Specification reconciliation stopped after the bounded "
                f"limit of {PRODUCT_PATCH_MAX_PROVIDER_CALLS} provider calls",
                "truncated_output",
            )

        evidence_payload = [
            {
                "rank": rank,
                "file": item.file_name,
                "page": item.page_number,
                "addendum_number": item.addendum_number,
                "text": item.quote[:2_000],
            }
            for rank, item in enumerate(batch, start=1)
        ]
        prompt = f"""
Determine only the explicit product-specification changes in this addendum batch.
Return compact patches; never repeat or regenerate the complete specification.
Use an existing category and attribute name exactly. Use category "*" only when
the cited addendum explicitly applies the same attribute change to every product
category. Add or remove a category only when the addendum explicitly does so.
If this batch contains no explicit change, amended=false and patches=[].

FIELD: Product Specification
CATEGORY/ATTRIBUTE INDEX: {json.dumps(structure, ensure_ascii=False)}
ADDENDUM EVIDENCE:
{json.dumps(evidence_payload, ensure_ascii=False)}
""".strip()
        started = datetime.now(timezone.utc)
        timer = time.perf_counter()
        provider_calls += 1
        try:
            generation = client.generate_structured(
                prompt,
                ProductSpecificationReconciliationDecision,
                system_instruction=ADDENDUM_RECONCILIATION_PROMPT,
                generation=GenerationSettings(
                    temperature=0,
                    max_output_tokens=PRODUCT_PATCH_MAX_OUTPUT_TOKENS,
                ),
            )
            decision = generation.output
            usage = getattr(generation, "usage", None)
            traces.append(AgentTrace(
                agent="addendum_reconciliation",
                action="reconcile_product_specification_batch",
                status="success",
                started_at=started,
                latency_ms=getattr(generation, "latency_ms", round((time.perf_counter() - timer) * 1000, 2)),
                tool_calls=[f"google.genai.generate_content:{getattr(client, 'model', 'injected-client')}"],
                input_summary={
                    "bid_id": bid_id,
                    "field_name": field.field_name,
                    "batch": batch_label,
                    "subdivision_depth": subdivision_depth,
                    "category_count": len(structure),
                    "evidence_count": len(batch),
                    "prompt_characters": len(prompt),
                },
                output_summary={"amended": decision.amended, "patch_count": len(decision.patches)},
                tokens_used=getattr(usage, "total_tokens", None),
                metadata={
                    "token_usage": usage.model_dump() if hasattr(usage, "model_dump") else {},
                    "max_output_tokens": PRODUCT_PATCH_MAX_OUTPUT_TOKENS,
                    "provider_call": provider_calls,
                },
            ))
            return [(decision, batch)], None, None
        except Exception as exc:
            classification = getattr(exc, "classification", None)
            can_subdivide = (
                classification == "truncated_output"
                and len(batch) > 1
                and subdivision_depth < PRODUCT_PATCH_MAX_SUBDIVISION_DEPTH
                and provider_calls < PRODUCT_PATCH_MAX_PROVIDER_CALLS
            )
            traces.append(AgentTrace(
                agent="addendum_reconciliation",
                action="reconcile_product_specification_batch",
                status="error",
                started_at=started,
                latency_ms=round((time.perf_counter() - timer) * 1000, 2),
                tool_calls=[f"google.genai.generate_content:{getattr(client, 'model', 'injected-client')}"],
                input_summary={
                    "bid_id": bid_id,
                    "field_name": field.field_name,
                    "batch": batch_label,
                    "subdivision_depth": subdivision_depth,
                    "category_count": len(structure),
                    "evidence_count": len(batch),
                    "prompt_characters": len(prompt),
                },
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "failure_classification": classification,
                    "max_output_tokens": PRODUCT_PATCH_MAX_OUTPUT_TOKENS,
                    "provider_call": provider_calls,
                    "will_subdivide": can_subdivide,
                },
            ))
            if not can_subdivide:
                detail = (
                    f"single-passage batch remains truncated: {type(exc).__name__}: {exc}"
                    if classification == "truncated_output" and len(batch) == 1
                    else f"{type(exc).__name__}: {exc}"
                )
                return [], (
                    f"Product Specification reconciliation batch {batch_label} failed: {detail}"
                ), classification

            split_at = (len(batch) + 1) // 2
            reviewed: list[tuple[ProductSpecificationReconciliationDecision, list[Evidence]]] = []
            for suffix, child in enumerate((batch[:split_at], batch[split_at:]), start=1):
                child_decisions, child_error, child_classification = review_batch(
                    child,
                    f"{batch_label}.{suffix}",
                    subdivision_depth + 1,
                )
                if child_error:
                    return [], child_error, child_classification
                reviewed.extend(child_decisions)
            return reviewed, None, None

    for batch_number, batch in enumerate(_product_addendum_batches(addendum_evidence), start=1):
        batch_decisions, batch_error, classification = review_batch(batch, str(batch_number))
        if batch_error:
            return ReconciliationOutcome(
                field=field,
                evidence=all_evidence,
                warning=batch_error,
                traces=traces,
                verified=False,
                failure_status=(
                    "provider_error"
                    if classification in {"quota_or_rate_limit", "timeout", "transport", "transient_unavailable", "authentication"}
                    else "reconciliation_incomplete"
                ),
            )
        decisions.extend(batch_decisions)

    revised_value = deepcopy(field.value)
    category_lookup = {_normalized_key(key): key for key in revised_value}
    applied: dict[tuple[int, str, str], object] = {}
    changed_originals: dict[str, object] = {}
    used_evidence: list[Evidence] = []
    explanations: list[str] = []
    patch_candidates: list[FieldCandidate] = []
    try:
        for decision, batch in decisions:
            for patch in decision.patches:
                if patch.evidence_rank > len(batch):
                    raise ValueError("Patch evidence rank is outside its addendum batch")
                source = batch[patch.evidence_rank - 1]
                if source.addendum_number is None:
                    raise ValueError("Product specification patch lacks numbered addendum evidence")
                if patch.operation != "remove_category":
                    _, support_errors = claim_evidence_errors(patch.value, [source])
                    if support_errors:
                        raise ValueError(
                            "Product specification patch is unsupported: " + "; ".join(support_errors)
                        )

                if patch.operation == "set_attribute":
                    if patch.category == "*":
                        if not re.search(r"\b(?:all|every|each)\b", source.quote, re.I):
                            raise ValueError("Wildcard patch lacks explicit all/every/each language")
                        target_categories = list(revised_value)
                    else:
                        category = category_lookup.get(_normalized_key(patch.category))
                        if category is None:
                            raise ValueError(f"Patch targets unknown category: {patch.category}")
                        target_categories = [category]
                    for category in target_categories:
                        attributes = revised_value.get(category)
                        if not isinstance(attributes, dict):
                            raise ValueError(f"Category is not an object: {category}")
                        attribute_lookup = {_normalized_key(key): key for key in attributes}
                        attribute = attribute_lookup.get(_normalized_key(patch.attribute or ""), patch.attribute)
                        if not attribute:
                            raise ValueError("Patch attribute is blank")
                        key = (source.addendum_number, _normalized_key(category), _normalized_key(attribute))
                        if key in applied and applied[key] != patch.value:
                            raise ValueError(f"Conflicting patches for {category}.{attribute} in one addendum")
                        applied[key] = patch.value
                        path = f"{category}.{attribute}"
                        changed_originals.setdefault(path, attributes.get(attribute))
                        attributes[attribute] = patch.value
                elif patch.operation == "add_category":
                    normalized = _normalized_key(patch.category)
                    if normalized in category_lookup:
                        raise ValueError(f"Add-category patch duplicates existing category: {patch.category}")
                    revised_value[patch.category] = deepcopy(patch.value)
                    category_lookup[normalized] = patch.category
                    changed_originals[f"{patch.category}.__category__"] = None
                else:
                    category = category_lookup.get(_normalized_key(patch.category))
                    if category is None:
                        raise ValueError(f"Remove-category patch targets unknown category: {patch.category}")
                    if not _explicit_replacement(source):
                        raise ValueError("Remove-category patch lacks explicit deletion/supersession language")
                    changed_originals[f"{category}.__category__"] = deepcopy(revised_value[category])
                    revised_value.pop(category)
                    category_lookup.pop(_normalized_key(category))

                used_evidence.append(source)
                target = patch.category
                if patch.attribute:
                    target = f"{target}.{patch.attribute}"
                explanations.append(
                    f"Applied {patch.operation} to {target} from Addendum "
                    f"{source.addendum_number}, {source.file_name} page {source.page_number}."
                )
                patch_candidates.append(FieldCandidate(
                    value=patch.value,
                    evidence=source,
                    confidence=field.confidence,
                ))
    except Exception as exc:
        return ReconciliationOutcome(
            field=field,
            evidence=all_evidence,
            warning=f"Product Specification patch validation failed: {type(exc).__name__}: {exc}",
            traces=traces,
            verified=False,
            failure_status="reconciliation_incomplete",
        )

    if not used_evidence:
        return ReconciliationOutcome(
            field=field,
            evidence=all_evidence,
            traces=traces,
            verified=True,
            verification_reason="All addendum batches were checked; no explicit product specification change was found.",
        )

    # A set-attribute reconciliation must never lose or duplicate categories.
    structural_operations = any(
        patch.operation in {"add_category", "remove_category"}
        for decision, _ in decisions for patch in decision.patches
    )
    if not structural_operations and list(revised_value) != list(field.value):
        return ReconciliationOutcome(
            field=field,
            evidence=all_evidence,
            warning="Product Specification patch merge changed category coverage unexpectedly",
            traces=traces,
            verified=False,
            failure_status="reconciliation_incomplete",
        )

    unique_used = _unique_evidence(used_evidence)
    notes = " ".join(dict.fromkeys(explanations))
    revised = field.model_copy(update={
        "value": revised_value,
        "confidence": field.confidence,
        "citations": _unique_evidence([*field.citations, *unique_used]),
        "candidates": [*field.candidates, *patch_candidates],
        "reasoning": notes,
    })
    latest_addendum = max(item.addendum_number for item in unique_used if item.addendum_number is not None)
    change = AddendumChange(
        field_name=field.field_name,
        original_value=changed_originals,
        updated_value=revised_value,
        addendum_number=latest_addendum,
        addendum_date=next(
            (item.document_date for item in reversed(unique_used) if item.addendum_number == latest_addendum and item.document_date),
            None,
        ),
        sources=[SourceCitation(file=item.file_name, page=item.page_number) for item in unique_used],
        notes=notes,
    )
    return ReconciliationOutcome(
        field=revised,
        evidence=all_evidence,
        change=change,
        traces=traces,
        verified=True,
        verification_reason=notes,
    )


def reconcile_field(
    *,
    bid_id: str,
    field_spec: FieldSpec,
    field: ExtractedField,
    evidence: list[Evidence],
    llm_client: GeminiClient | None = None,
    retrieval_fn=None,
    search_addenda: bool = True,
) -> ReconciliationOutcome:
    """Apply an amendment only when Gemini selects valid addendum evidence."""

    all_evidence = list(evidence)
    traces: list[AgentTrace] = []
    if search_addenda:
        try:
            if retrieval_fn is None:
                from agents.retrieval_agent import search_evidence

                active_retrieval = search_evidence
            else:
                active_retrieval = retrieval_fn
            addendum_result = active_retrieval(
                EvidenceSearchRequest(
                    bid_id=bid_id,
                    field_name=field_spec.field_name,
                    question=(
                        field_spec.questions[1]
                        if len(field_spec.questions) > 1
                        else field_spec.questions[0]
                    ),
                    top_k=5,
                    candidate_k=20,
                    doc_type="addendum",
                )
            )
            if getattr(addendum_result, "trace", None):
                traces.append(addendum_result.trace)
            if getattr(addendum_result, "status", None) == "error":
                return ReconciliationOutcome(
                    field=field,
                    evidence=_unique_evidence(all_evidence),
                    warning="Addendum search failed; reconciliation is incomplete",
                    traces=traces,
                    verified=False,
                )
            all_evidence.extend(
                passage.evidence for passage in addendum_result.passages
            )
        except Exception as exc:
            return ReconciliationOutcome(
                field=field,
                evidence=_unique_evidence(all_evidence),
                warning=f"Addendum search failed: {type(exc).__name__}: {exc}",
                traces=traces,
            )

    all_evidence = _unique_evidence(all_evidence)
    addendum_evidence = [
        item
        for item in all_evidence
        if item.doc_type == "addendum" or item.addendum_number is not None
    ]
    if field.status == "not_found":
        if not addendum_evidence:
            return ReconciliationOutcome(
                field=field,
                evidence=all_evidence,
                traces=traces,
                verified=True,
                verification_reason=(
                    "Base extraction found no directly supported value and the "
                    "successful addendum search returned no addendum evidence."
                ),
            )
        if (
            field_spec.field_name == "Bid Bond Requirement"
            and _affirmative_bid_bond_requirement(all_evidence) is None
        ):
            field.reasoning = (
                "A complete review of the retrieved base and addendum evidence "
                "found only generic bond cross-references, post-award insurance/"
                "bond language, or joint-venture bonding capacity. No passage "
                "affirmatively states a bid-bond requirement, amount, percentage, "
                "or proposal-security submission condition."
            )
            return ReconciliationOutcome(
                field=field,
                evidence=all_evidence,
                traces=traces,
                verified=True,
                verification_reason=field.reasoning,
            )
        try:
            from agents.extractor import extract_field
            from agents.retrieval_agent import RankedEvidence, RetrievalResult

            review = RetrievalResult(
                bid_id=bid_id,
                field_name=field_spec.field_name,
                question=" ".join(field_spec.questions),
                status="success",
                passages=[
                    RankedEvidence(rank=index, evidence=item)
                    for index, item in enumerate(all_evidence, start=1)
                ],
                trace=AgentTrace(
                    agent="retrieval_agent",
                    action="combined_base_addendum_evidence",
                    status="success",
                ),
            )
            reviewed, review_trace = extract_field(
                review, llm_client=llm_client or get_gemini_client()
            )
            traces.append(review_trace)
            if reviewed.status == "error":
                classification = review_trace.metadata.get("failure_classification")
                return ReconciliationOutcome(
                    field=field, evidence=all_evidence, traces=traces,
                    warning="Combined base/addendum evidence review failed: " + reviewed.reasoning,
                    verified=False,
                    verification_reason=reviewed.reasoning,
                    failure_status=(
                        "provider_error"
                        if classification in {
                            "quota_or_rate_limit", "timeout", "transport",
                            "transient_unavailable", "authentication",
                        }
                        else "reconciliation_incomplete"
                    ),
                )
            if reviewed.status == "found":
                return ReconciliationOutcome(
                    field=reviewed, evidence=all_evidence, traces=traces,
                    verified=True,
                    verification_reason=(
                        "A directly supported value was recovered by reviewing "
                        "the combined base and addendum evidence."
                    ),
                )
            return ReconciliationOutcome(
                field=reviewed, evidence=all_evidence, traces=traces,
                verified=True,
                verification_reason=(
                    reviewed.reasoning or
                    "Gemini reviewed combined base and addendum evidence and found no directly supported value."
                ),
            )
        except Exception as exc:
            return ReconciliationOutcome(
                field=field, evidence=all_evidence, traces=traces,
                warning=f"Combined base/addendum evidence review failed: {type(exc).__name__}: {exc}",
                verified=False,
                verification_reason=f"{type(exc).__name__}: {exc}",
            )

    if not addendum_evidence or field.status not in {"found", "conflict"}:
        return ReconciliationOutcome(
            field=field,
            evidence=all_evidence,
            traces=traces,
            verified=(not addendum_evidence and field.status in {"found", "conflict"}),
            verification_reason=(
                "Successful addendum search returned no addendum evidence."
                if not addendum_evidence else "Field status was not reconcilable."
            ),
        )

    if not _addendum_may_change_field(field_spec, addendum_evidence):
        traces.append(AgentTrace(
            agent="addendum_reconciliation",
            action="reconcile_field",
            status="skipped",
            input_summary={"bid_id": bid_id, "field_name": field_spec.field_name},
            output_summary={
                "amended": False,
                "verification": "deterministic_no_field_specific_amendment_language",
            },
        ))
        return ReconciliationOutcome(
            field=field, evidence=all_evidence, traces=traces, verified=True
        )

    if field_spec.field_name == "Product Specification":
        return _reconcile_product_specification(
            bid_id=bid_id,
            field=field,
            all_evidence=all_evidence,
            addendum_evidence=addendum_evidence,
            llm_client=llm_client,
            prior_traces=traces,
        )

    ranked = list(enumerate(all_evidence, start=1))
    evidence_payload = [
        {
            "rank": rank,
            "file": item.file_name,
            "page": item.page_number,
            "doc_type": item.doc_type,
            "addendum_number": item.addendum_number,
            "text": item.quote[:2_500],
        }
        for rank, item in ranked
    ]
    prompt = f"""
Determine whether an addendum explicitly changes this extracted field. A later
document is not automatically an amendment. Select evidence ranks only from the
provided list.

FIELD: {field_spec.field_name}
CURRENT VALUE: {json.dumps(field.value, ensure_ascii=False)}
EVIDENCE:
{json.dumps(evidence_payload, ensure_ascii=False)}
""".strip()

    try:
        client = llm_client or get_gemini_client()
        llm_started = datetime.now(timezone.utc)
        llm_timer = time.perf_counter()
        generation = client.generate_structured(
            prompt,
            ReconciliationDecision,
            system_instruction=ADDENDUM_RECONCILIATION_PROMPT,
            generation=GenerationSettings(
                temperature=0,
                max_output_tokens=700,
            ),
        )
        decision = generation.output
        usage = getattr(generation, "usage", None)
        usage_payload = (
            usage.model_dump()
            if hasattr(usage, "model_dump")
            else {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        )
        traces.append(
            AgentTrace(
                agent="addendum_reconciliation",
                action="reconcile_field",
                status="success",
                started_at=llm_started,
                latency_ms=getattr(
                    generation,
                    "latency_ms",
                    round((time.perf_counter() - llm_timer) * 1000, 2),
                ),
                tool_calls=[
                    "google.genai.generate_content:"
                    f"{getattr(client, 'model', 'injected-client')}"
                ],
                input_summary={
                    "bid_id": bid_id,
                    "field_name": field_spec.field_name,
                    "evidence_count": len(all_evidence),
                },
                output_summary={"amended": decision.amended},
                tokens_used=getattr(usage, "total_tokens", None),
                metadata={"token_usage": usage_payload},
            )
        )
        if decision.field_name != field_spec.field_name:
            raise ValueError("Reconciler returned a different field name")
        if not decision.amended:
            return ReconciliationOutcome(
                field=field,
                evidence=all_evidence,
                traces=traces,
                verified=True,
            )

        if decision.updated_evidence_rank is None:
            raise ValueError("Amendment lacks updated evidence rank")
        updated = all_evidence[decision.updated_evidence_rank - 1]
        if updated not in addendum_evidence:
            raise ValueError("Updated evidence is not from an addendum")
        if updated.addendum_number is None:
            raise ValueError("Updated evidence has no addendum number")

        supporting = [updated]
        if decision.original_evidence_rank is not None:
            original = all_evidence[decision.original_evidence_rank - 1]
            supporting.insert(0, original)

        revised_value = decision.updated_value
        revised_citations = [updated]
        explanation = decision.explanation
        if isinstance(field.value, list) and isinstance(decision.updated_value, list):
            replace_allowed = decision.change_mode == "replace" and _explicit_replacement(updated)
            if replace_allowed:
                revised_value = _deduplicate_values(decision.updated_value)
            else:
                revised_value = _deduplicate_values([*field.value, *decision.updated_value])
                revised_citations = _unique_evidence([*field.citations, updated])
                if decision.change_mode == "replace":
                    explanation = (
                        decision.explanation
                        + " Replacement was not applied because the cited addendum "
                        "did not explicitly supersede the original list; new requirements were appended."
                    ).strip()

        revised = field.model_copy(
            update={
                "status": "found",
                "value": revised_value,
                "confidence": decision.confidence,
                "citations": revised_citations,
                "candidates": [
                    *field.candidates,
                    FieldCandidate(
                        value=revised_value,
                        evidence=updated,
                        confidence=decision.confidence,
                    ),
                ],
                "reasoning": explanation,
            }
        )
        change = AddendumChange(
            field_name=field_spec.field_name,
            original_value=decision.original_value,
            updated_value=revised_value,
            addendum_number=updated.addendum_number,
            addendum_date=(
                decision.addendum_date or updated.document_date
            ),
            sources=[
                SourceCitation(file=item.file_name, page=item.page_number)
                for item in supporting
            ],
            notes=explanation,
        )
        return ReconciliationOutcome(
            field=revised,
            evidence=all_evidence,
            change=change,
            traces=traces,
            verified=True,
            verification_reason=explanation,
        )
    except Exception as exc:
        classification = getattr(exc, "classification", None)
        if "llm_started" in locals() and not any(
            trace.action == "reconcile_field" for trace in traces
        ):
            traces.append(
                AgentTrace(
                    agent="addendum_reconciliation",
                    action="reconcile_field",
                    status="error",
                    started_at=llm_started,
                    latency_ms=round(
                        (time.perf_counter() - llm_timer) * 1000,
                        2,
                    ),
                    tool_calls=[
                        "google.genai.generate_content:"
                        f"{getattr(client, 'model', 'injected-client')}"
                    ],
                    input_summary={
                        "bid_id": bid_id,
                        "field_name": field_spec.field_name,
                    },
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=(
                        {"failure_classification": classification}
                        if classification else {}
                    ),
                )
            )
        return ReconciliationOutcome(
            field=field,
            evidence=all_evidence,
            warning=f"Reconciliation unresolved: {type(exc).__name__}: {exc}",
            traces=traces,
            verified=False,
            failure_status=(
                "provider_error"
                if classification in {
                    "quota_or_rate_limit", "timeout", "transport",
                    "transient_unavailable", "authentication",
                }
                else "reconciliation_incomplete"
            ),
        )
