
"""Cross-Encoder reranking for retrieved RFP evidence.

Pipeline:
    Query understanding
        -> Dense + BM25 retrieval
        -> RRF
        -> Cross-Encoder reranking
        -> Top-k cited evidence

Run:
    python -m rag.reranker --bid Bid1 --question "..."
"""

from __future__ import annotations

import argparse
from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag.retrieval import search


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

DEFAULT_CANDIDATES = 20
DEFAULT_TOP_K = 5


# --------------------------------------------------
# 1. LOAD THE RERANKER
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load model once and reuse it for later queries."""

    print(f"Loading reranker: {RERANK_MODEL}")

    return CrossEncoder(
        RERANK_MODEL,
        max_length=512,
    )


# --------------------------------------------------
# 2. RERANK EXISTING EVIDENCE
# --------------------------------------------------

def rerank_results(
    question: str,
    results: list[dict],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Rerank candidates without losing citation metadata."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not results:
        return []

    model = get_reranker()

    pairs = [
        (question, result["text"])
        for result in results
    ]

    scores = model.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    reranked = []

    for result, score in zip(results, scores):

        item = result.copy()

        # Preserve the original fusion result
        # for comparisons during evaluation.
        item["retrieval_rank"] = result["rank"]
        item["retrieval_score"] = result["score"]

        # Cross-Encoder scores are relevance
        # scores, not probabilities.
        item["rerank_score"] = float(score)

        reranked.append(item)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    final = reranked[:top_k]

    for index, result in enumerate(final, start=1):
        result["rank"] = index

    return final


# --------------------------------------------------
# 3. COMPLETE RETRIEVAL PIPELINE
# --------------------------------------------------

def retrieve_and_rerank(
    question: str,
    bid_id: str,
    doc_type: str | None = None,
    addendum_number: int | None = None,
    file_name: str | None = None,
    mode: str = "hybrid",
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATES,
    expand: bool = True,
) -> dict:
    """Retrieve candidates and apply Cross-Encoder reranking."""

    if top_k < 1 or candidate_k < top_k:
        raise ValueError(
            "Require 1 <= top_k <= candidate_k."
        )

    # First-stage retrieval.
    # Retrieve more candidates than we intend to return.
    output = search(
        question=question,
        bid_id=bid_id,
        doc_type=doc_type,
        addendum_number=addendum_number,
        file_name=file_name,
        mode=mode,
        top_k=candidate_k,
        candidate_k=max(30, candidate_k),
        expand=expand,
    )

    candidates = output["results"]

    # Second-stage reranking.
    output["results"] = rerank_results(
        question=question,
        results=candidates,
        top_k=top_k,
    )

    output["reranker_model"] = RERANK_MODEL
    output["candidates_retrieved"] = len(candidates)
    output["reranking_enabled"] = True

    return output


# --------------------------------------------------
# 4. COMMAND-LINE TEST
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Retrieve and rerank RFP evidence"
    )

    parser.add_argument(
        "--bid",
        required=True,
    )

    parser.add_argument(
        "--question",
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
        default=20,
    )

    parser.add_argument(
        "--no-expand",
        action="store_true",
    )

    args = parser.parse_args()

    output = retrieve_and_rerank(
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

    print("\nQuestion:", output["question"])
    print("Retrieval mode:", output["mode"])
    print("Reranker:", output["reranker_model"])
    print(
        "Candidates retrieved:",
        output["candidates_retrieved"]
    )

    for result in output["results"]:

        print("\n" + "=" * 70)

        print(
            f"Final rank: {result['rank']} "
            f"| Previous rank: {result['retrieval_rank']}"
        )

        print(
            f"RRF score: {result['retrieval_score']:.4f}"
        )

        print(
            f"Reranker score: {result['rerank_score']:.4f}"
        )

        print("Source:", result["file_name"])
        print("Page:", result["page_number"])
        print("Document type:", result["doc_type"])
        print("Addendum:", result["addendum_number"])
        print("Chunk ID:", result["chunk_id"])

        print("\nEvidence:")
        print(result["text"])

    if not output["results"]:
        print("No matching evidence found.")


if __name__ == "__main__":
    main()
