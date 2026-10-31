"""Conservative quote and value verification tolerant of harmless OCR drift."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from collections.abc import Iterable
from typing import Any

from agents.schemas import Evidence, ExtractedField, FieldCandidate


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "will", "with",
}


def normalize_evidence_text(value: str) -> str:
    """Normalize presentation noise, not substantive words or numbers."""
    text = unicodedata.normalize("NFKC", value or "")
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\ufb01": "fi",
        "\ufb02": "fl",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    # Join words split only by an OCR/PDF line-break hyphen.
    text = re.sub(r"(?<=\w)-\s*[\r\n]+\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _token_text(value: str) -> str:
    tokens = (
        token.strip("-./")
        for token in re.findall(r"[\w@./%+#-]+", normalize_evidence_text(value))
    )
    return " ".join(token for token in tokens if token)


def quote_matches(quote: str, source: str) -> bool:
    """Accept exact content with harmless spacing/Unicode/punctuation drift."""
    candidate = normalize_evidence_text(quote)
    document = normalize_evidence_text(source)
    if not candidate:
        return False
    if candidate in document:
        return True
    token_candidate = _token_text(quote)
    token_document = _token_text(source)
    return bool(token_candidate and token_candidate in token_document)


def find_quote_evidence(quote: str, passages: Iterable[Any], selected_rank: int):
    """Find the quote on the selected chunk or a sibling chunk on its page."""
    items = list(passages)
    selected = next((item for item in items if item.rank == selected_rank), None)
    if selected is None:
        return None, "The model selected an evidence rank that was not retrieved."
    candidates = [selected]
    candidates.extend(
        item for item in items
        if item is not selected
        and item.evidence.file_name == selected.evidence.file_name
        and item.evidence.page_number == selected.evidence.page_number
    )
    for item in candidates:
        if quote_matches(quote, item.evidence.quote):
            return item, None
    return None, (
        "Supporting quote was not found on the selected source page after "
        "Unicode, OCR whitespace, punctuation, and line-break normalization: "
        f"{selected.evidence.file_name}, page {selected.evidence.page_number}."
    )


def material_claims(value: Any, path: str = "value") -> list[dict[str, str]]:
    """Flatten a composite value into independently auditable material claims."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [
            claim
            for key, nested in value.items()
            for claim in (
                [{
                    "path": f"{path}.{key}",
                    "claim": f"{key}: {nested.strip()}",
                }]
                if isinstance(nested, str)
                and nested.strip().casefold().rstrip(".") in {"unspecified", "unknown"}
                else material_claims(nested, f"{path}.{key}")
            )
        ]
    if isinstance(value, (list, tuple)):
        return [
            claim
            for index, nested in enumerate(value)
            for claim in material_claims(nested, f"{path}[{index}]")
        ]
    text = str(value).strip()
    if not text:
        return []
    # Sentences and semicolon-delimited contractual clauses commonly carry
    # independent obligations. Never split on commas: technical configurations
    # use commas inside CPU/cache/core/thread tuples, memory and storage
    # attributes, display/camera descriptions, and product-qualified lists.
    # Splitting those lists loses attribute/value and product association.
    parts = [
        item.strip(" -")
        for item in re.split(
            r"(?<=[.!?])\s+|\s*;\s*|\s+including\s+",
            text,
            flags=re.I,
        )
        if item.strip(" -")
    ]
    if len(parts) > 1 and parts[0].casefold().rstrip(".") in {"yes", "no"}:
        parts[1] = f"{parts[0]}, {parts[1]}"
        parts = parts[1:]
    return [
        {"path": path if len(parts) == 1 else f"{path}.claim[{index}]", "claim": item}
        for index, item in enumerate(parts)
    ]


_UNKNOWN_ATTRIBUTE_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9 /_-]{1,80}:\s*(?:unspecified|unknown)\.?$",
    re.I,
)


def is_unknown_attribute_marker(claim: str) -> bool:
    """Recognize constrained metadata for an unestablished sub-attribute."""
    return bool(_UNKNOWN_ATTRIBUTE_RE.fullmatch((claim or "").strip()))


def normalize_unestablished_composite_claims(
    *, field_name: str, value: Any, diagnostics: list[dict[str, Any]],
    passages: Iterable[Any],
) -> tuple[Any, list[str]]:
    """Replace only recognized unsupported sub-attribute conclusions.

    Field labels encode reusable RFP semantics, not answers for a particular
    bid. Positive or unrecognized unsupported claims are never removed. A value
    consisting only of unknown metadata is left to the not-found workflow.
    """
    if not isinstance(value, str):
        return value, []
    supported = [item for item in diagnostics if item.get("supported") is True]
    unsupported = [item for item in diagnostics if item.get("supported") is False]
    if not supported or not unsupported:
        return value, []

    rules: dict[str, tuple[re.Pattern[str], str]] = {
        "Pre Bid Meeting": (
            re.compile(r"\b(?:not mandatory|optional|attendance status|participation status)\b", re.I),
            "Attendance requirement: unspecified",
        ),
        "Delivery Date": (
            re.compile(r"\b(?:no fixed delivery date|delivery date (?:is )?(?:not specified|unknown)|does not specify (?:a |the )?delivery date)\b", re.I),
            "Exact delivery date: unspecified",
        ),
        "Payment Terms": (
            re.compile(r"\b(?:(?:documents?|solicitation|rfp).{0,60}(?:do not|does not|not).{0,40}(?:payment terms?|district.{0,20}prime)|district.to.prime.{0,40}(?:not specified|unknown))\b", re.I),
            "District-to-prime payment terms: unspecified",
        ),
    }
    rule = rules.get(field_name)
    if rule is None or any(not rule[0].search(item["claim"]) for item in unsupported):
        return value, []

    retained = [item["claim"].rstrip(". ") for item in supported]
    annotations: list[str] = []
    for item in unsupported:
        positive = re.sub(
            r"\s*,?\s*but\s+not\s+mandatory\.?$", "", item["claim"], flags=re.I
        )
        if positive != item["claim"] and any(
            claim_support_error(positive, passage.evidence.quote) is None
            for passage in passages
        ):
            retained.append(positive.rstrip(". "))
        if rule[1] not in annotations:
            annotations.append(rule[1])

    if not retained:
        return value, []
    return ". ".join([*retained, *annotations]) + ".", annotations


def _support_tokens(value: str) -> list[str]:
    def stem(token: str) -> str:
        if any(char.isdigit() for char in token) or "@" in token or "/" in token:
            return token
        # Deterministic inflection normalization, not fuzzy matching. Preserve
        # exact identifiers above, then align common noun/verb forms so a
        # passive paraphrase can be checked against the same source concepts.
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith(
            ("ches", "shes", "sses", "xes", "zes")
        ):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) > 5 and token.endswith("ated"):
            token = token[:-1]
        elif len(token) > 5 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 6 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 5 and token.endswith("ly"):
            token = token[:-2]
        if token == "delivery":
            return "deliver"
        return token

    tokens = []
    for token in _token_text(value).split():
        parts = (
            token.split("-")
            if "-" in token and not any(char.isdigit() for char in token)
            else [token]
        )
        tokens.extend(stem(part) for part in parts if part and part not in _STOPWORDS)
    return tokens


def claim_support_error(claim: str, source: str) -> str | None:
    """Require strong lexical, identifier, and negation coverage for a claim."""
    normalized_claim = normalize_evidence_text(claim)
    normalized_source = normalize_evidence_text(source)
    if normalized_claim and normalized_claim in normalized_source:
        return None
    iso_match = re.search(
        r"\b(\d{4})-(\d{2})-(\d{2})t(\d{2}):(\d{2})(?::\d{2})?\s*([a-z]{2,5})?",
        normalized_claim,
    )
    if iso_match:
        year, month, day, hour, minute, zone = iso_match.groups()
        month_name = datetime(2000, int(month), 1).strftime("%B").casefold()
        hour_24 = int(hour)
        hour_12 = hour_24 % 12 or 12
        meridiem = "am" if hour_24 < 12 else "pm"
        date_supported = re.search(
            rf"\b{month_name}\s+0?{int(day)},?\s+{year}\b", normalized_source
        )
        time_supported = re.search(
            rf"\b{hour_12}:?{minute}\s*{meridiem}\b", normalized_source
        )
        zone_supported = not zone or zone in normalized_source
        if date_supported and time_supported and zone_supported:
            return None
    absence_pattern = re.compile(
        r"(?:documents?|solicitation|rfp).{0,40}(?:do not|does not|not).{0,25}"
        r"(?:specif|state|provide|include)",
        re.I,
    )
    if absence_pattern.search(claim) and not absence_pattern.search(source):
        return "documentary-absence claim is not explicitly stated in evidence"
    claim_tokens = _support_tokens(claim)
    source_tokens = set(_support_tokens(source))
    if not claim_tokens:
        return "claim has no auditable content tokens"
    identifiers = [
        token for token in claim_tokens
        if any(char.isdigit() for char in token) or "@" in token or "/" in token
    ]
    missing_ids = sorted({token for token in identifiers if token not in source_tokens})
    if missing_ids:
        return f"identifiers absent from evidence: {missing_ids}"
    negative_markers = (" no ", " not ", "do not", "does not", "without", "none")
    padded_claim = f" {normalized_claim} "
    if any(marker in padded_claim for marker in negative_markers) and not any(
        marker in f" {normalized_source} " for marker in negative_markers
    ):
        return "negative or absence claim is not explicitly stated in evidence"
    overlap = sum(token in source_tokens for token in claim_tokens)
    required = 1 if len(claim_tokens) <= 2 else max(2, int(len(claim_tokens) * 0.65 + 0.999))
    if overlap < required:
        return f"insufficient lexical coverage ({overlap}/{len(claim_tokens)} tokens; {required} required)"
    return None


def evidence_claim_support_error(claim: str, evidence: Evidence) -> str | None:
    """Evaluate a claim using text plus trustworthy indexed metadata."""
    error = claim_support_error(claim, evidence.quote)
    if error is None:
        return None
    addendum_claim = re.fullmatch(
        r"signed\s+(?:copy\s+of\s+)?addendum\s+(\d+)\.?",
        normalize_evidence_text(claim),
    )
    if (
        addendum_claim
        and evidence.addendum_number == int(addendum_claim.group(1))
        and re.search(r"\b(?:please\s+)?sign\s+this\s+addendum\b", evidence.quote, re.I)
        and re.search(r"\bsubmit\b", evidence.quote, re.I)
    ):
        return None
    return error


def _short_supported_quote(claim: str, source: str) -> str | None:
    """Return a verbatim, bounded source span that supports one claim."""
    blocks = [item.strip() for item in re.split(r"\n\s*\n", source) if item.strip()]
    # Keep character offsets so a multi-line candidate remains an authentic
    # substring of indexed text. Joining split sentences with spaces can make a
    # semantically equivalent quote that never literally appeared in the chunk.
    boundaries = list(re.finditer(r"(?<=[.!?;])\s+|\n+", source))
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        if source[start:boundary.start()].strip():
            spans.append((start, boundary.start()))
        start = boundary.end()
    if source[start:].strip():
        spans.append((start, len(source)))
    candidates = blocks + [
        source[spans[offset][0]:spans[offset + width - 1][1]].strip()
        for width in range(1, min(6, len(spans)) + 1)
        for offset in range(0, len(spans) - width + 1)
    ]
    supported = [
        item for item in candidates
        # Extraction receives at most 3,500 source characters per passage.
        # Keep the same hard bound when a dense technical configuration needs
        # the complete authentic span to preserve product/attribute context.
        if len(item) <= 3_500
        and claim_support_error(claim, item) is None
    ]
    return min(supported, key=len) if supported else None


def _short_supported_evidence_quote(claim: str, evidence: Evidence) -> str | None:
    quote = _short_supported_quote(claim, evidence.quote)
    if quote is not None:
        return quote
    if evidence_claim_support_error(claim, evidence) is None:
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.!?;])\s+|\n+", evidence.quote)
            if item.strip()
        ]
        anchored = [
            item for item in sentences
            if re.search(r"\bsign\s+this\s+addendum\b", item, re.I)
            and re.search(r"\bsubmit\b", item, re.I)
        ]
        return min(anchored, key=len) if anchored else None
    return None


def resolve_claim_evidence(
    *, value: Any, passages: Iterable[Any], selected_rank: int,
    evidence_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[Any, str]], str | None]:
    """Attribute every material claim to its own authentic indexed passage."""
    items = list(passages)
    selected = next(
        (item for item in items if evidence_id and item.evidence.chunk_id == evidence_id),
        None,
    ) or next((item for item in items if item.rank == selected_rank), None)
    if selected is None:
        return [], [], "The model selected an evidence rank or ID that was not retrieved."
    same_page = [
        item for item in items
        if item is not selected
        and item.evidence.file_name == selected.evidence.file_name
        and item.evidence.page_number == selected.evidence.page_number
    ]
    ordered = [selected, *same_page, *[
        item for item in items if item is not selected and item not in same_page
    ]]
    diagnostics: list[dict[str, Any]] = []
    matches: list[tuple[Any, str]] = []
    for claim_record in material_claims(value):
        if is_unknown_attribute_marker(claim_record["claim"]):
            diagnostics.append({
                **claim_record,
                "supported": None,
                "evidence_status": "unknown_attribute",
                "candidate_evidence": [],
                "corrected_citation_found": False,
                "reason": "Attribute is marked unspecified; no factual absence is asserted.",
            })
            continue
        failures = []
        found = None
        for item in ordered:
            reason = evidence_claim_support_error(claim_record["claim"], item.evidence)
            if reason is None:
                authentic_quote = _short_supported_evidence_quote(
                    claim_record["claim"], item.evidence
                )
                if authentic_quote is not None:
                    found = (item, authentic_quote)
                    break
                reason = "no bounded authentic source span supports the complete claim"
            failures.append({
                "evidence_id": item.evidence.chunk_id,
                "file": item.evidence.file_name,
                "page": item.evidence.page_number,
                "reason": reason,
            })
        if found is None:
            diagnostics.append({
                **claim_record,
                "supported": False,
                "candidate_evidence": failures,
                "corrected_citation_found": False,
                "reason": "No retrieved passage independently supports this material claim.",
            })
            continue
        item, authentic_quote = found
        relocated = (
            item.evidence.file_name != selected.evidence.file_name
            or item.evidence.page_number != selected.evidence.page_number
        )
        matches.append((item, authentic_quote))
        diagnostics.append({
            **claim_record,
            "supported": True,
            "candidate_evidence": failures,
            "evidence_id": item.evidence.chunk_id,
            "file": item.evidence.file_name,
            "page": item.evidence.page_number,
            "quote": authentic_quote,
            "corrected_citation_found": relocated,
            "reason": None,
        })
    unsupported = [item for item in diagnostics if item["supported"] is False]
    error = None
    if unsupported:
        details = "; ".join(
            f"{item['path']}: {item['claim']!r}" for item in unsupported
        )
        error = f"Material claims are not supported: {details}"
    return diagnostics, matches, error


def resolve_value_evidence(
    *,
    value: Any,
    passages: Iterable[Any],
    selected_rank: int,
    evidence_id: str | None = None,
) -> tuple[Any | None, str | None, bool, str | None]:
    """Resolve a model selection to authentic indexed text.

    The model selects an evidence rank or chunk ID, but it never supplies the
    published quote. If the selected page does not support the value, other
    retrieved passages may establish the value; in that case the citation is
    explicitly relocated to the supporting file/page.
    """
    diagnostics, matches, error = resolve_claim_evidence(
        value=value, passages=passages, selected_rank=selected_rank,
        evidence_id=evidence_id,
    )
    if error or not matches:
        return None, None, False, error or "No material claim was supported."
    item, quote = matches[0]
    selected_id = evidence_id or next(
        (entry.evidence.chunk_id for entry in passages if entry.rank == selected_rank), None
    )
    return item, quote, item.evidence.chunk_id != selected_id, None


def claim_evidence_errors(value: Any, citations: list[Evidence]) -> tuple[list[dict[str, Any]], list[str]]:
    """Revalidate that every material claim has at least one cited source."""
    diagnostics = []
    errors = []
    for claim_record in material_claims(value):
        if is_unknown_attribute_marker(claim_record["claim"]):
            diagnostics.append({
                **claim_record,
                "supported": None,
                "evidence_status": "unknown_attribute",
                "supporting_evidence": [],
                "candidate_evidence": [],
            })
            continue
        supporting = []
        failures = []
        for citation in citations:
            reason = evidence_claim_support_error(claim_record["claim"], citation)
            if reason is None:
                supporting.append(citation)
            else:
                failures.append({
                    "evidence_id": citation.chunk_id,
                    "file": citation.file_name,
                    "page": citation.page_number,
                    "reason": reason,
                })
        diagnostic = {
            **claim_record,
            "supported": bool(supporting),
            "supporting_evidence": [
                {"evidence_id": item.chunk_id, "file": item.file_name, "page": item.page_number}
                for item in supporting
            ],
            "candidate_evidence": failures,
        }
        diagnostics.append(diagnostic)
        if not supporting:
            errors.append(
                f"{claim_record['path']} is unsupported: {claim_record['claim']!r}"
            )
    return diagnostics, errors


def reattribute_supported_citations(
    field: ExtractedField,
    evidence: list[Evidence],
) -> tuple[ExtractedField, bool]:
    """Repair missing citations only when every claim has indexed support."""

    _, existing_errors = claim_evidence_errors(field.value, field.citations)
    if not existing_errors or not evidence:
        return field, False

    class _Ranked:
        def __init__(self, rank: int, item: Evidence):
            self.rank = rank
            self.evidence = item

    ranked = [_Ranked(index, item) for index, item in enumerate(evidence, start=1)]
    diagnostics, matches, error = resolve_claim_evidence(
        value=field.value,
        passages=ranked,
        selected_rank=1,
    )
    if error or not matches:
        return field, False

    citations_by_chunk: dict[str, Evidence] = {}
    for matched_item, authentic_quote in matches:
        source = matched_item.evidence
        existing = citations_by_chunk.get(source.chunk_id)
        citations_by_chunk[source.chunk_id] = source.model_copy(update={
            "quote": source.quote if existing is not None else authentic_quote,
        })
    candidates = [
        FieldCandidate(
            value=item["claim"],
            evidence=citations_by_chunk[item["evidence_id"]],
            confidence=field.confidence,
        )
        for item in diagnostics
        if item.get("supported") is True
    ]
    return field.model_copy(update={
        "citations": list(citations_by_chunk.values()),
        "candidates": candidates,
    }), True


def _leaf_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _leaf_strings(nested)]
    if isinstance(value, (list, tuple, set)):
        return [item for nested in value for item in _leaf_strings(nested)]
    return [str(value)]


def value_support_errors(value: Any, sources: list[Evidence]) -> list[str]:
    """Reject values with no meaningful lexical or exact-identifier support."""
    evidence_text = _token_text(" ".join(item.quote for item in sources))
    evidence_tokens = set(evidence_text.split())
    errors: list[str] = []
    for leaf in _leaf_strings(value):
        normalized = _token_text(leaf)
        if not normalized:
            continue
        if normalized in evidence_text:
            continue
        tokens = [token for token in normalized.split() if token not in _STOPWORDS]
        if not tokens:
            continue
        identifiers = [
            token for token in tokens
            if any(char.isdigit() for char in token) or "@" in token or "/" in token
        ]
        missing_ids = [token for token in identifiers if token not in evidence_tokens]
        overlap = sum(token in evidence_tokens for token in tokens)
        required = 1 if len(tokens) <= 2 else max(2, int(len(tokens) * 0.35 + 0.999))
        if missing_ids or overlap < required:
            detail = f"unsupported value component {leaf!r}"
            if missing_ids:
                detail += f"; identifiers absent from evidence: {missing_ids}"
            errors.append(detail)
    return errors
