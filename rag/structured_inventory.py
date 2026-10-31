"""Deterministic structured-evidence inventory over an existing Qdrant index."""

from __future__ import annotations

import os
import re

from agents.field_completeness import available_contact_components, discover_product_categories
from agents.schemas import Evidence
from rag.retrieval import build_filter
from rag.vector_store import COLLECTION, get_client


def _format_point(point, rank: int) -> dict:
    payload = point.payload or {}
    return {
        "rank": rank,
        "chunk_id": str(point.id),
        "score": getattr(point, "score", None),
        "bid_id": payload.get("bid_id"),
        "file_name": payload.get("file_name"),
        "doc_type": payload.get("doc_type"),
        "addendum_number": payload.get("addendum_number"),
        "document_date": payload.get("document_date"),
        "page_number": payload.get("page_number"),
        "section": payload.get("section"),
        "chunk_type": payload.get("chunk_type"),
        "table_id": payload.get("table_id"),
        "text": payload.get("text", ""),
    }


def structured_product_inventory(
    bid_id: str,
    *,
    client=None,
    max_chunks: int | None = None,
) -> list[dict]:
    """Return every section/table chunk that declares a product category."""

    client = client or get_client()
    if not client.collection_exists(COLLECTION):
        raise RuntimeError(f"Qdrant collection {COLLECTION} does not exist")
    maximum = max_chunks or int(os.getenv("EMPLAY_MAX_PRODUCT_STRUCTURE_CHUNKS", "60"))
    points = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=build_filter(bid_id=bid_id),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(page)
        if offset is None:
            break

    ordered = sorted(
        points,
        key=lambda point: (
            str((point.payload or {}).get("file_name") or ""),
            int((point.payload or {}).get("page_number") or 0),
            int((point.payload or {}).get("chunk_index") or 0),
        ),
    )
    category_indexes: set[int] = set()
    for index, point in enumerate(ordered):
        payload = point.payload or {}
        evidence = Evidence(
            file_name=str(payload.get("file_name") or "unknown"),
            page_number=max(1, int(payload.get("page_number") or 1)),
            quote=str(payload.get("text", "")),
            chunk_id=str(point.id),
            doc_type=payload.get("doc_type"),
            addendum_number=payload.get("addendum_number"),
            document_date=payload.get("document_date"),
            section=payload.get("section"),
            chunk_type=payload.get("chunk_type"),
            table_id=payload.get("table_id"),
        )
        text = evidence.quote
        section = str(evidence.section or "")
        table_signal = bool(
            re.search(r"\b(?:product|item|model|sku|part)\b", text, re.I)
            and re.search(
                r"\b(?:description|specification|requirement|configuration|quantity|qty|deliverable)\b",
                text,
                re.I,
            )
        )
        description_signal = bool(re.search(
            r"\b(?:sku|item|product|equipment|service|configuration|bill\s+of\s+materials)\s+description\b",
            section,
            re.I,
        ))
        specification_density = len(re.findall(
            r"\b(?:\d+(?:\.\d+)?\s*(?:gb|tb|mb|ghz|mhz|w|v|hz|inch|hours?|years?)|"
            r"processor|operating\s+system|warranty|memory|storage|resolution|interface)\b",
            text,
            re.I,
        ))
        identifier_density = len(re.findall(
            r"\b(?=[A-Z0-9-]{5,}\b)(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b",
            text,
        ))
        content_signal = specification_density >= 3 and identifier_density >= 1
        if (
            discover_product_categories([evidence])
            or table_signal
            or description_signal
            or content_signal
        ):
            category_indexes.add(index)

    selected_indexes = set(category_indexes)
    for index in sorted(category_indexes):
        base = ordered[index].payload or {}
        for next_index in range(index + 1, len(ordered)):
            if next_index in category_indexes:
                break
            candidate = ordered[next_index].payload or {}
            if candidate.get("file_name") != base.get("file_name"):
                break
            if int(candidate.get("page_number") or 0) > int(base.get("page_number") or 0) + 1:
                break
            section = str(candidate.get("section") or "").casefold()
            text = str(candidate.get("text") or "")
            generic_continuation = section in {
                "proposal",
                "scope and specifications",
                "scope and specifications of the proposal",
                "specifications",
                "technical specifications",
            }
            specification_like = (
                len(re.findall(r"(?m)^\s*-\s+", text)) >= 2
                or bool(re.search(r"\b(?:sku|model|part|warranty|configuration)\b", text, re.I))
            )
            if not (generic_continuation and specification_like):
                break
            selected_indexes.add(next_index)

    selected = [ordered[index] for index in sorted(selected_indexes)]
    if len(selected) > maximum:
        raise RuntimeError(
            f"Structured product inventory found {len(selected)} chunks, exceeding safe limit {maximum}"
        )
    return [_format_point(point, rank) for rank, point in enumerate(selected, start=1)]


def structured_contact_inventory(
    bid_id: str,
    *,
    client=None,
    max_chunks: int | None = None,
) -> list[dict]:
    """Return populated procurement-contact chunks, excluding blank response forms."""

    client = client or get_client()
    if not client.collection_exists(COLLECTION):
        raise RuntimeError(f"Qdrant collection {COLLECTION} does not exist")
    maximum = max_chunks or int(os.getenv("EMPLAY_MAX_CONTACT_STRUCTURE_CHUNKS", "12"))
    points = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=build_filter(bid_id=bid_id),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(page)
        if offset is None:
            break

    selected = []
    for point in sorted(
        points,
        key=lambda item: (
            str((item.payload or {}).get("file_name") or ""),
            int((item.payload or {}).get("page_number") or 0),
            int((item.payload or {}).get("chunk_index") or 0),
        ),
    ):
        payload = point.payload or {}
        text = str(payload.get("text") or "")
        if not re.search(
            r"\b(?:buyer|procurement|purchasing|solicitation|bidder\s+questions?|"
            r"manual\s+submissions?|offer\s+submittal|contact\s+information)\b",
            text,
            re.I,
        ):
            continue
        evidence = Evidence(
            file_name=str(payload.get("file_name") or "unknown"),
            page_number=max(1, int(payload.get("page_number") or 1)),
            quote=text,
            chunk_id=str(point.id),
            doc_type=payload.get("doc_type"),
            addendum_number=payload.get("addendum_number"),
            document_date=payload.get("document_date"),
            section=payload.get("section"),
            chunk_type=payload.get("chunk_type"),
            table_id=payload.get("table_id"),
        )
        components = available_contact_components([evidence])
        strong_contact = (
            "name" in components
            or (
                "email" in components
                and bool(re.search(r"\b(?:buyer|procurement|questions?|contact\s+information)\b", text, re.I))
            )
            or (
                "mailing_address" in components
                and bool(re.search(r"\b(?:manual\s+submission|offer\s+submittal|procurement|place\s+of\s+performance)\b", text, re.I))
            )
        )
        if strong_contact:
            selected.append(point)

    if len(selected) > maximum:
        raise RuntimeError(
            f"Structured contact inventory found {len(selected)} chunks, exceeding safe limit {maximum}"
        )
    return [_format_point(point, rank) for rank, point in enumerate(selected, start=1)]
