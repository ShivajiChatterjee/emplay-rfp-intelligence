"""Deterministic completeness checks for composite official fields.

These checks do not invent field values.  They only compare an extracted value
with structure or contact details that are already present in retrieved,
citable evidence.
"""

from __future__ import annotations

import re
from copy import deepcopy
from collections.abc import Iterable
from typing import Any

from agents.schemas import Evidence


_GENERIC_SPEC_HEADINGS = {
    "scope and specifications",
    "scope and specifications of the proposal",
    "specifications",
    "technical specifications",
    "proposal",
    "equipment",
    "product",
    "item",
    "device",
}
_SPEC_SUFFIX = re.compile(
    r"\s+(?:minimum\s+)?(?:technical\s+)?(?:requirements|specifications)\s*$",
    re.IGNORECASE,
)
_SECTION_LINE = re.compile(r"^\s*Section:\s*(.+?)\s*$", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]\d{4}(?:\s*(?:x|ext\.?)[\s]*\d+)?(?!\d)",
    re.IGNORECASE,
)
_ADDRESS_HINT = re.compile(
    r"\b(?:address|mail(?:ing)?|deliver(?:y|ed)?|street|suite|ste\.?|"
    r"avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|highway|hwy\.?|"
    r"drive|dr\.?|lane|ln\.?|building|floor|postal|zip)\b",
    re.IGNORECASE,
)
_CONDITION = re.compile(
    r"\b(?:if|when|where|unless|except|may|as applicable|if applicable|"
    r"for\s+(?:proposing\s+)?(?:vendors?|bidders?|resellers?)\b)\b",
    re.IGNORECASE,
)

_INDIVIDUAL_CONTACT = re.compile(
    r"(?im)^\s*(?:buyer|contact\s+(?:person|name)|contracting\s+officer|"
    r"procurement\s+contact)\s*(?::|-|\s)\s*(\S[^\r\n]{1,79})\s*$"
)
_GENERIC_CONTACT_NAMES = {
    "contact", "information", "office", "department", "services",
    "procurement", "purchasing", "staff", "member", "bid", "number",
    "name", "representative",
}


def _clean_category(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" :-–—\t")
    value = _SPEC_SUFFIX.sub("", value).strip(" :-–—\t")
    return value


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _looks_like_category_heading(value: str) -> bool:
    cleaned = _clean_category(value)
    folded = _normalized(cleaned)
    if not folded or folded in _GENERIC_SPEC_HEADINGS:
        return False
    original = value.casefold()
    has_requirement_label = bool(
        re.search(r"\b(?:requirements?|specifications?)\b", original)
    )
    administrative = re.search(
        r"\b(?:insurance|submission|proposal|bidder|contract|payment|"
        r"documentation|delivery|installation|general)\b",
        cleaned,
        re.IGNORECASE,
    )
    if administrative:
        return False
    # "Minimum Requirements" headings within specification sections commonly
    # use arbitrary domain nouns (scanner, inverter, analyzer, furniture, etc.).
    # Avoid a fixed product vocabulary so unseen procurements remain supported.
    has_product_label = bool(
        re.search(
            r"\b(?:product|item|device|equipment|system|unit|tier|model|category|service|"
            r"laptop|desktop|tablet|monitor|server|printer|camera|vehicle)\b",
            original,
        )
    )
    if has_requirement_label and "minimum requirement" in original:
        return True
    if re.match(r"^\s*\d+(?:\.\d+)*\s+line\b", cleaned, re.I):
        return False
    device_label = re.search(
        r"\b(?:laptop|desktop|tablet|monitor|server|printer|camera|vehicle)\b",
        original,
    )
    return bool(has_product_label and device_label and len(cleaned.split()) <= 8)


def _table_categories(text: str) -> set[str]:
    """Discover category values from an explicit product/item table column."""

    rows = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(rows) < 2:
        return set()
    cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
    header_index = next((
        row_index
        for row_index, row in enumerate(cells)
        if any(
            re.search(r"\b(?:product|item|category|device|equipment|service)\b", value, re.I)
            for value in row
        )
        and any(
            re.search(
                r"\b(?:specification|requirement|description|configuration|model|sku|part|quantity|qty)\b",
                value,
                re.I,
            )
            for value in row
        )
    ), None)
    if header_index is None:
        discovered = set()
        for row in cells:
            if not row:
                continue
            value = _clean_category(row[0])
            repeated = sum(_normalized(cell) == _normalized(value) for cell in row)
            if (
                value
                and not re.match(r"^\d+[.)]", value)
                and not re.match(
                    r"^(?:section\b|functional\s+area\b|fa\s+[ivxlcdm]+\b)",
                    value,
                    re.I,
                )
                and repeated >= 2
                and re.search(
                    r"\b(?:warranty|service|support|maintenance|license|subscription)\b",
                    value,
                    re.I,
                )
            ):
                discovered.add(re.sub(r"\s*\([^)]*\)\s*$", "", value).strip())
        return discovered
    header = cells[header_index]
    category_column = next(
        index
        for index, value in enumerate(header)
        if re.search(r"\b(?:product|item|category|device|equipment|service)\b", value, re.I)
    )
    discovered: set[str] = set()
    for row in cells[header_index + 1:]:
        if category_column >= len(row):
            continue
        value = _clean_category(row[category_column])
        value = re.sub(r"^\d+[.)]\s*", "", value)
        if (
            not value
            or re.fullmatch(r"[-: ]+", value)
            or _normalized(value) == _normalized(header[category_column])
            or _normalized(value) in {"product", "item", "category", "device", "equipment", "service"}
        ):
            continue
        discovered.add(value)
    return discovered


def discover_product_categories(evidence: Iterable[Evidence]) -> tuple[str, ...]:
    """Return distinct product categories declared by sections or tables."""

    categories: dict[str, str] = {}
    for item in evidence:
        headings: list[str] = []
        section = getattr(item, "section", None)
        if section:
            headings.append(str(section))
        for line in item.quote.splitlines()[:4]:
            match = _SECTION_LINE.match(line)
            if match:
                headings.append(match.group(1))
        for heading in headings:
            if _looks_like_category_heading(heading):
                cleaned = _clean_category(heading)
                categories.setdefault(_normalized(cleaned), cleaned)
        for category in _table_categories(item.quote):
            categories.setdefault(_normalized(category), category)
        # Some price sheets and technical narratives contain no useful
        # heading/table metadata.  Accept only explicit labels or a leading
        # alphanumeric SKU followed by a product description; numbers or prose
        # alone are intentionally insufficient.
        for line in item.quote.splitlines():
            cleaned_line = re.sub(r"^\s*[-*]\s*", "", line).strip()
            labelled = re.match(
                r"^(?:product|item|equipment|service)\s+"
                r"(?:name|description)\s*[:=-]\s*(.{2,120})$",
                cleaned_line,
                re.IGNORECASE,
            )
            if labelled:
                candidate = _clean_category(labelled.group(1).split(";")[0])
                if _looks_like_category_heading(candidate):
                    categories.setdefault(_normalized(candidate), candidate)
                continue
            sku = re.match(
                r"^(?:\d+[.)]\s*)?(?:sku\s*[:#-]?\s*)?"
                r"(?P<sku>(?=[A-Z0-9._/-]{4,}\b)(?=[A-Z0-9._/-]*[A-Z])"
                r"(?=[A-Z0-9._/-]*\d)[A-Z0-9._/-]+)\s+"
                r"(?P<name>[A-Z][^;|]{2,100})(?:[;|]|$)",
                cleaned_line,
            )
            if sku:
                candidate = _clean_category(sku.group("name"))
                if len(re.findall(r"[A-Za-z]+", candidate)) >= 2:
                    categories.setdefault(_normalized(candidate), candidate)
    return tuple(categories.values())


def _category_is_covered(category: str, extracted_keys: Iterable[str]) -> bool:
    expected = _normalized(category)
    expected_tokens = set(expected.split())
    for key in extracted_keys:
        actual = _normalized(key)
        if expected == actual or expected in actual or actual in expected:
            return True
        actual_tokens = set(actual.split())
        if expected_tokens and len(expected_tokens & actual_tokens) / len(expected_tokens | actual_tokens) >= 0.65:
            return True
    return False


def product_coverage_errors(value: Any, evidence: list[Evidence]) -> list[str]:
    categories = discover_product_categories(evidence)
    if not isinstance(value, dict):
        return ["Product Specification must be an object keyed by discovered product category"]
    if not categories:
        if not value:
            return ["Product Specification is empty and no product inventory could be established"]
        source_text = _normalized(" ".join(item.quote for item in evidence))
        unsupported = [key for key in value if _normalized(str(key)) not in source_text]
        return ([
            "Product Specification contains category keys absent from retrieved evidence: "
            + "; ".join(unsupported)
        ] if unsupported else [])
    missing = [name for name in categories if not _category_is_covered(name, value)]
    if not missing:
        return []
    return [
        "Product Specification omits discovered categories: " + "; ".join(missing)
    ]


def available_contact_components(evidence: Iterable[Evidence]) -> set[str]:
    """Identify contact components visibly present in retrieved evidence."""

    available: set[str] = set()
    for item in evidence:
        text = item.quote
        if _EMAIL.search(text):
            available.add("email")
        if _PHONE.search(text):
            available.add("phone")
        if _ADDRESS_HINT.search(text) and (
            re.search(r"\b\d{1,6}\s+[A-Za-z0-9]", text)
            or re.search(r"\b(?:address|mailing address)\s*:\s*\S+", text, re.I)
        ):
            available.add("mailing_address")
        if _has_populated_individual_contact(text):
            available.add("name")
        if re.search(r"\b(?:questions?|communications?|inquiries)\b", text, re.I):
            available.add("contact_instructions")
    return available


def _has_populated_individual_contact(text: str) -> bool:
    """Distinguish a named person from an office, department, or blank form."""

    for match in _INDIVIDUAL_CONTACT.finditer(text):
        candidate = match.group(1).strip(" :-\t")
        if not candidate or "@" in candidate or re.search(r"\d", candidate):
            continue
        tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", candidate.casefold())
        if len(tokens) < 2:
            continue
        if set(tokens) <= _GENERIC_CONTACT_NAMES:
            continue
        if any(marker in candidate.casefold() for marker in (
            "provide your answer", "if different", "city, state", "email address"
        )):
            continue
        return True
    return False


def contact_completeness_errors(value: Any, evidence: list[Evidence]) -> list[str]:
    if not isinstance(value, dict):
        return ["contact_info must be an object"]
    available = available_contact_components(evidence)
    aliases = {
        "name": ("name", "contact_name", "contact_person"),
        "email": ("email", "email_address"),
        "phone": ("phone", "phone_number", "telephone"),
        "mailing_address": ("mailing_address", "address", "postal_address"),
        "contact_instructions": (
            "contact_instructions", "contact_instruction", "instructions",
            "communication_instructions", "communication_instruction",
        ),
    }
    missing = []
    for component in sorted(available):
        keys = aliases[component]
        if not _nested_component_present(value, keys):
            missing.append(component)
    return (["contact_info omits available evidence components: " + ", ".join(missing)] if missing else [])


def _nested_component_present(value: Any, aliases: tuple[str, ...]) -> bool:
    expected = {_normalized(item) for item in aliases}
    if isinstance(value, (list, tuple)):
        return any(_nested_component_present(item, aliases) for item in value)
    if not isinstance(value, dict):
        return False
    for key, nested in value.items():
        if _normalized(str(key)) in expected and nested not in (None, "", [], {}):
            return True
        if isinstance(nested, (dict, list, tuple)) and _nested_component_present(nested, aliases):
            return True
    return False


_SUBMISSION_ADDRESS_ANCHOR = re.compile(
    r"(?P<context>(?:the\s+)?(?:offeror|vendor|bidder|proposer|manual\s+submission|"
    r"paper\s+(?:proposal|response)|mail(?:ed|ing)?\s+(?:proposal|response|submission)s?)"
    r".{0,400}?\b(?:at|to)\s+(?:the\s+)?(?:following\s+)?"
    r"(?:mailing\s+)?address)\s*:\s*",
    re.IGNORECASE,
)
_SUBMISSION_ADDRESS_STOP = re.compile(
    r"\s+(?=(?:does\s+the\s+(?:vendor|offeror|bidder)|questions?\b|"
    r"circle\s+one\b|response\s+values?\b|type\b\s*[-:]))",
    re.IGNORECASE,
)


def enrich_contact_value_from_evidence(
    value: Any,
    evidence: Iterable[Evidence],
) -> tuple[Any, bool]:
    """Add one unambiguous submission address using verbatim indexed text.

    This is deliberately narrower than extraction: it never infers a general
    office address, and it does nothing when multiple different submission
    addresses are present.  The returned context and address are normalized
    only for whitespace, so ordinary claim/citation verification still applies.
    """

    if not isinstance(value, dict) or _nested_component_present(
        value, ("mailing_address", "postal_address", "submission_address")
    ):
        return value, False

    candidates: dict[str, tuple[str, str]] = {}
    for item in evidence:
        text = re.sub(r"\s+", " ", item.quote).strip()
        for match in _SUBMISSION_ADDRESS_ANCHOR.finditer(text):
            tail = text[match.end():]
            stop = _SUBMISSION_ADDRESS_STOP.search(tail)
            address = (tail[:stop.start()] if stop else tail[:400]).strip(" .;:-")
            # A citable postal destination must contain a street number and an
            # address signal.  This excludes blank forms and email addresses.
            if not re.search(r"\b\d{1,6}\s+[A-Za-z0-9]", address):
                continue
            if not _ADDRESS_HINT.search(address):
                continue
            normalized = _normalized(address)
            if normalized:
                candidates.setdefault(normalized, (match.group("context").strip(), address))

    if len(candidates) != 1:
        return value, False

    context, address = next(iter(candidates.values()))
    revised = deepcopy(value)
    key = "submission_delivery"
    suffix = 2
    while key in revised:
        key = f"submission_delivery_{suffix}"
        suffix += 1
    revised[key] = {
        "mailing_address": address,
        "submission_context": context,
    }
    return revised, True


def documentation_structure_errors(value: Any, evidence: list[Evidence]) -> list[str]:
    """Detect flattened conditions/alternatives without rewriting requirements."""

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return []
    errors: list[str] = []
    normalized_items = [(_normalized(item), item) for item in value]
    for source in evidence:
        clauses = re.split(r"(?<=[.;])\s+|\n+", source.quote)
        for clause in clauses:
            folded = clause.casefold()
            has_alternative = " or " in folded
            has_condition = bool(_CONDITION.search(clause))
            if not (has_alternative or has_condition):
                continue
            clause_tokens = set(_normalized(clause).split())
            matched = [
                original
                for normalized, original in normalized_items
                if len(set(normalized.split()) & clause_tokens) >= min(3, max(1, len(set(normalized.split()))))
            ]
            alternative_matches: list[str] = []
            if has_alternative:
                sides = re.split(r"\b(?:and/)?or\b", clause, flags=re.I)
                for normalized, original in normalized_items:
                    item_tokens = set(normalized.split())
                    if not item_tokens:
                        continue
                    if any(
                        len(item_tokens & set(_normalized(side).split())) / len(item_tokens) >= 0.6
                        for side in sides
                    ):
                        alternative_matches.append(original)
            if len(alternative_matches) >= 2 and not any(" or " in item.casefold() for item in alternative_matches):
                errors.append(
                    "Alternative documentation requirements were flattened into separate mandatory items: "
                    + " | ".join(alternative_matches)
                )
            if has_condition:
                condition_matched = []
                for normalized, original in normalized_items:
                    item_tokens = set(normalized.split())
                    if item_tokens and len(item_tokens & clause_tokens) / len(item_tokens) >= 0.6:
                        condition_matched.append(original)
                for item in condition_matched:
                    if not _CONDITION.search(item) and "applicable" not in item.casefold():
                        errors.append(
                            "Conditional documentation requirement lost its condition: " + item
                        )
    return list(dict.fromkeys(errors))
