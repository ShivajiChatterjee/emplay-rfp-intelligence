
"""Hybrid RFP retrieval using Qdrant Cloud.

Features:
- Dense semantic search
- BM25 sparse keyword search
- Reciprocal Rank Fusion (RRF)
- Query expansion
- Bid/document/addendum metadata filtering
- Source citations and full chunk text
- Separate retrieval modes for evaluation

Run:
    python -m rag.retrieval --bid Bid1 --question "..."
"""

from __future__ import annotations

import argparse
import json

from qdrant_client import models

from rag.query_processor import build_query_plan
from rag.vector_store import (
    COLLECTION,
    DENSE_MODEL,
    DENSE_VECTOR,
    SPARSE_MODEL,
    SPARSE_VECTOR,
    get_client,
)


# Number of candidates retrieved before final ranking.
DEFAULT_CANDIDATES = 30
DEFAULT_TOP_K = 5


# --------------------------------------------------
# 1. METADATA FILTERING
# --------------------------------------------------

def build_filter(
    bid_id: str,
    doc_type: str | None = None,
    addendum_number: int | None = None,
    file_name: str | None = None,
) -> models.Filter:
    """Build server-side Qdrant metadata filters."""

    if not bid_id or not bid_id.strip():
        raise ValueError("bid_id is required.")

    conditions = [
        models.FieldCondition(
            key="bid_id",
            match=models.MatchValue(value=bid_id),
        )
    ]

    if doc_type:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=doc_type),
            )
        )

    if addendum_number is not None:
        conditions.append(
            models.FieldCondition(
                key="addendum_number",
                match=models.MatchValue(
                    value=addendum_number
                ),
            )
        )

    if file_name:
        conditions.append(
            models.FieldCondition(
                key="file_name",
                match=models.MatchValue(value=file_name),
            )
        )

    return models.Filter(must=conditions)


# --------------------------------------------------
# 2. DENSE SEMANTIC RETRIEVAL
# --------------------------------------------------

def dense_prefetch(
    question: str,
    filters: models.Filter,
    limit: int,
) -> models.Prefetch:

    return models.Prefetch(
        query=models.Document(
            text=question,
            model=DENSE_MODEL,
        ),
        using=DENSE_VECTOR,
        filter=filters,
        limit=limit,
    )


# --------------------------------------------------
# 3. BM25 KEYWORD RETRIEVAL
# --------------------------------------------------

def bm25_prefetches(
    questions: list[str],
    filters: models.Filter,
    limit: int,
) -> list[models.Prefetch]:
    """Create a separate sparse search for each variant."""

    return [
        models.Prefetch(
            query=models.Document(
                text=question,
                model=SPARSE_MODEL,
            ),
            using=SPARSE_VECTOR,
            filter=filters,
            limit=limit,
        )
        for question in questions
    ]


def bm25_group(
    questions: list[str],
    filters: models.Filter,
    limit: int,
) -> models.Prefetch:
    """Combine expanded BM25 queries into one ranked list.

    This prevents multiple synonym queries from
    unfairly outweighing the single dense retriever.
    """

    prefetches = bm25_prefetches(
        questions,
        filters,
        limit,
    )

    if len(prefetches) == 1:
        return prefetches[0]

    return models.Prefetch(
        prefetch=prefetches,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        limit=limit,
    )


# --------------------------------------------------
# 4. FORMAT RESULTS WITH CITATIONS
# --------------------------------------------------

def format_results(points: list) -> list[dict]:
    """Preserve the complete source evidence."""

    results = []

    for rank, point in enumerate(points, start=1):

        payload = point.payload or {}

        results.append({
            "rank": rank,
            "chunk_id": str(point.id),
            "score": point.score,
            "bid_id": payload.get("bid_id"),
            "file_name": payload.get("file_name"),
            "doc_type": payload.get("doc_type"),
            "addendum_number": payload.get(
                "addendum_number"
            ),
            "document_date": payload.get("document_date"),
            "page_number": payload.get(
                "page_number"
            ),
            "section": payload.get("section"),
            "chunk_type": payload.get(
                "chunk_type"
            ),
            "table_id": payload.get("table_id"),
            "text": payload.get("text", ""),
        })

    return results


# --------------------------------------------------
# 5. MAIN RETRIEVAL FUNCTION
# --------------------------------------------------

def search(
    question: str,
    bid_id: str,
    doc_type: str | None = None,
    addendum_number: int | None = None,
    file_name: str | None = None,
    mode: str = "hybrid",
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATES,
    expand: bool = True,
    client=None,
) -> dict:
    """Retrieve cited evidence from one bid.

    mode:
        dense  -> semantic search only
        bm25   -> sparse keyword search only
        hybrid -> dense + BM25 + RRF
    """

    if mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError(
            "mode must be dense, bm25, or hybrid"
        )

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1 or candidate_k < top_k:
        raise ValueError(
            "Require 1 <= top_k <= candidate_k."
        )

    if client is None:
        client = get_client()

    if not client.collection_exists(COLLECTION):
        raise RuntimeError(
            f"Qdrant collection {COLLECTION} "
            "does not exist. Index documents first."
        )

    filters = build_filter(
        bid_id=bid_id,
        doc_type=doc_type,
        addendum_number=addendum_number,
        file_name=file_name,
    )

    # ------------------------------------------------
    # Query understanding
    # ------------------------------------------------

    plan = build_query_plan(
        question,
        max_variants=4 if expand else 1,
    )

    dense_query = plan["dense_query"]
    keyword_queries = plan["keyword_queries"]

    # ------------------------------------------------
    # Dense-only mode
    # ------------------------------------------------

    if mode == "dense":

        response = client.query_points(
            collection_name=COLLECTION,
            query=models.Document(
                text=dense_query,
                model=DENSE_MODEL,
            ),
            using=DENSE_VECTOR,
            query_filter=filters,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    # ------------------------------------------------
    # BM25-only mode
    # ------------------------------------------------

    elif mode == "bm25":

        if len(keyword_queries) == 1:

            response = client.query_points(
                collection_name=COLLECTION,
                query=models.Document(
                    text=keyword_queries[0],
                    model=SPARSE_MODEL,
                ),
                using=SPARSE_VECTOR,
                query_filter=filters,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

        else:

            response = client.query_points(
                collection_name=COLLECTION,
                prefetch=bm25_prefetches(
                    keyword_queries,
                    filters,
                    candidate_k,
                ),
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

    # ------------------------------------------------
    # Hybrid mode: dense + BM25 + RRF
    # ------------------------------------------------

    else:

        dense = dense_prefetch(
            dense_query,
            filters,
            candidate_k,
        )

        sparse = bm25_group(
            keyword_queries,
            filters,
            candidate_k,
        )

        response = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                dense,
                sparse,
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    return {
        "question": question,
        "mode": mode,
        "filters": {
            "bid_id": bid_id,
            "doc_type": doc_type,
            "addendum_number": addendum_number,
            "file_name": file_name,
        },
        "query_plan": plan,
        "results": format_results(response.points),
    }


# --------------------------------------------------
# 6. CLI
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Search indexed RFP documents"
    )

    parser.add_argument(
        "--question",
        required=True,
    )

    parser.add_argument(
        "--bid",
        required=True,
    )

    parser.add_argument(
        "--doc-type",
        default=None,
    )

    parser.add_argument(
        "--addendum-number",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--file-name",
        default=None,
    )

    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--no-expand",
        action="store_true",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON results",
    )

    args = parser.parse_args()

    output = search(
        question=args.question,
        bid_id=args.bid,
        doc_type=args.doc_type,
        addendum_number=args.addendum_number,
        file_name=args.file_name,
        mode=args.mode,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        expand=not args.no_expand,
    )

    if args.json:

        print(
            json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print("\nQuestion:", output["question"])
    print("Mode:", output["mode"])
    print("Filters:", output["filters"])
    print(
        "Query variants:",
        len(output["query_plan"]["keyword_queries"])
    )

    results = output["results"]

    print("\nRetrieved chunks:", len(results))

    for result in results:

        print("\n" + "=" * 70)

        print(
            f"Rank: {result['rank']} "
            f"| Score: {result['score']:.4f}"
        )

        print("Bid:", result["bid_id"])
        print("Source:", result["file_name"])
        print("Page:", result["page_number"])
        print("Document type:", result["doc_type"])
        print(
            "Addendum:",
            result["addendum_number"]
        )
        print("Section:", result["section"])
        print("Chunk ID:", result["chunk_id"])

        print("\nEvidence:")
        print(result["text"])

    if not results:
        print(
            "\nNo matching evidence found. "
            "Check the query and metadata filters."
        )


if __name__ == "__main__":
    main()
