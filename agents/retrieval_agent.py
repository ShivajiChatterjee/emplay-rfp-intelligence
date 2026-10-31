
import argparse
import time
import threading
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from agents.schemas import AgentTrace, Evidence, EvidenceSearchRequest
from rag.cohere_reranker import retrieve_and_rerank_cohere
from rag.structured_inventory import structured_contact_inventory, structured_product_inventory


_CACHE: dict[tuple, "RetrievalResult"] = {}
_CACHE_LOCK = threading.Lock()


class RetrievalRequest(EvidenceSearchRequest):
    """Backward-compatible public retrieval request."""


class RankedEvidence(BaseModel):
    rank: int
    score: float | None = None
    evidence: Evidence


class RetrievalResult(BaseModel):
    bid_id: str
    field_name: str
    question: str
    status: Literal["success", "empty", "error"]
    passages: list[RankedEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    trace: AgentTrace


def get_value(item: dict, *keys):
    """Read a field from the result or its nested metadata."""
    metadata = item.get("metadata") or {}
    payload = item.get("payload") or {}

    for key in keys:
        for source in (item, metadata, payload):
            if isinstance(source, dict):
                value = source.get(key)
                if value is not None:
                    return value

    return None


def convert_passage(item: dict, rank: int) -> RankedEvidence:
    """Convert a reranked result into our evidence schema."""
    if not isinstance(item, dict):
        raise ValueError("Expected each reranked passage to be a dict")

    source = get_value(
        item, "file_name", "source_file", "source"
    )

    if isinstance(source, dict):
        source = source.get("file_name") or source.get("name")

    page = get_value(item, "page_number", "page")
    chunk_id = get_value(item, "chunk_id", "id")
    text = get_value(
        item, "text", "chunk_text", "content", "evidence"
    )

    if not source or page is None or not chunk_id or not text:
        raise ValueError(
            "Reranker result missing source, page, "
            "chunk_id, or text. Available keys: "
            f"{list(item.keys())}"
        )

    addendum = get_value(item, "addendum_number")

    evidence = Evidence(
        file_name=str(source),
        page_number=int(page),
        quote=str(text),
        chunk_id=str(chunk_id),
        doc_type=get_value(item, "doc_type"),
        addendum_number=(
            int(addendum) if addendum is not None else None
        ),
        document_date=get_value(item, "document_date"),
        section=get_value(item, "section"),
        chunk_type=get_value(item, "chunk_type"),
        table_id=get_value(item, "table_id"),
    )

    score = get_value(
        item, "cohere_score", "rerank_score",
        "relevance_score", "score"
    )

    return RankedEvidence(
        rank=rank,
        score=float(score) if score is not None else None,
        evidence=evidence,
    )


def search_evidence(request: RetrievalRequest) -> RetrievalResult:
    cache_key = (
        request.bid_id, request.question, request.top_k, request.candidate_k,
        request.doc_type, request.addendum_number, request.file_name,
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached.model_copy(deep=True)
    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    warnings = []
    passages = []
    error_message = None

    try:
        raw = retrieve_and_rerank_cohere(
            question=request.question,
            bid_id=request.bid_id,
            top_k=request.top_k,
            candidate_k=max(
                request.candidate_k, request.top_k
            ),
            doc_type=request.doc_type,
            addendum_number=request.addendum_number,
            file_name=request.file_name,
        )

        retrieval_metadata = {}
        if isinstance(raw, dict):
            warnings.extend(raw.get("warnings", []))
            retrieval_metadata = {
                "ranking_method": raw.get("ranking_method", "cohere_rerank"),
                "rerank_applied": raw.get("rerank_applied", True),
                "rerank_fallback_reason": raw.get("rerank_fallback_reason"),
            }
            raw = raw.get("results", raw.get("passages"))

        if not isinstance(raw, list):
            raise ValueError(
                "Unexpected reranker response type: "
                f"{type(raw).__name__}"
            )

        for item in raw:
            try:
                passage = convert_passage(
                    item, len(passages) + 1
                )
                passages.append(passage)
            except (ValueError, TypeError) as exc:
                warnings.append(f"Skipped invalid passage: {exc}")

        if request.field_name == "Product Specification" and request.doc_type != "addendum":
            try:
                structured = structured_product_inventory(request.bid_id)
                existing = {item.evidence.chunk_id for item in passages}
                for item in structured:
                    if str(get_value(item, "chunk_id", "id")) in existing:
                        continue
                    passage = convert_passage(item, len(passages) + 1)
                    passages.append(passage)
                    existing.add(passage.evidence.chunk_id)
            except Exception as exc:
                status = "error"
                error_message = f"Structured product inventory failed: {exc}"
                warnings.append(error_message)

        if request.field_name == "contact_info" and request.doc_type != "addendum":
            try:
                structured = structured_contact_inventory(request.bid_id)
                existing = {item.evidence.chunk_id for item in passages}
                for item in structured:
                    if str(get_value(item, "chunk_id", "id")) in existing:
                        continue
                    passage = convert_passage(item, len(passages) + 1)
                    passages.append(passage)
                    existing.add(passage.evidence.chunk_id)
            except Exception as exc:
                status = "error"
                error_message = f"Structured contact inventory failed: {exc}"
                warnings.append(error_message)

        if "status" not in locals() or status != "error":
            status = "success" if passages else "empty"

        if not passages:
            warnings.append(
                "No usable evidence found for this question."
            )

    except Exception as exc:
        status = "error"
        error_message = str(exc)
        warnings.append(f"Retrieval failed: {error_message}")

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    trace = AgentTrace(
        agent="retrieval_agent",
        action="search_evidence",
        status="error" if status == "error" else "success",
        started_at=started,
        latency_ms=round(latency_ms, 2),
        tool_calls=[
            "rag.cohere_reranker.retrieve_and_rerank_cohere"
        ],
        input_summary={
            "bid_id": request.bid_id,
            "field_name": request.field_name,
            "question": request.question,
            "top_k": request.top_k,
            "candidate_k": request.candidate_k,
            "doc_type": request.doc_type,
            "addendum_number": request.addendum_number,
            "file_name": request.file_name,
        },
        output_summary={
            "status": status,
            "passages_returned": len(passages),
            "warnings": len(warnings),
            **(retrieval_metadata if "retrieval_metadata" in locals() else {}),
        },
        tokens_used=None,
        error=error_message,
    )

    result = RetrievalResult(
        bid_id=request.bid_id,
        field_name=request.field_name,
        question=request.question,
        status=status,
        passages=passages,
        warnings=warnings,
        trace=trace,
    )
    if result.status in {"success", "empty"}:
        with _CACHE_LOCK:
            _CACHE[cache_key] = result.model_copy(deep=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bid", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--doc-type", default=None)

    args = parser.parse_args()

    request = RetrievalRequest(
        bid_id=args.bid,
        field_name=args.field,
        question=args.question,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        doc_type=args.doc_type,
    )

    result = search_evidence(request)

    print("\n=== RETRIEVAL AGENT ===")
    print("Status:", result.status)
    print("Field:", result.field_name)
    print("Passages:", len(result.passages))

    for passage in result.passages:
        e = passage.evidence

        print("\n" + "-" * 60)
        print("Rank:", passage.rank)
        print("Score:", passage.score)
        print("Source:", e.file_name)
        print("Page:", e.page_number)
        print("Chunk ID:", e.chunk_id)
        print("Evidence:", e.quote[:350])

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print("-", warning)

    print(
        "\nAgent latency:",
        round(result.trace.latency_ms / 1000, 2),
        "seconds",
    )


if __name__ == "__main__":
    main()
