
"""Cohere reranking of Qdrant hybrid retrieval results."""

import argparse
import os

import cohere
from dotenv import load_dotenv

from rag.retrieval import search
from providers.rate_limit import cohere_limiter


MODEL = "rerank-v5.0-pro"


def retrieve_and_rerank_cohere(
    question: str,
    bid_id: str,
    top_k: int = 5,
    candidate_k: int = 20,
    doc_type: str | None = None,
    addendum_number: int | None = None,
    file_name: str | None = None,
):
    load_dotenv()

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise ValueError("COHERE_API_KEY is missing from .env")

    if not 1 <= top_k <= candidate_k:
        raise ValueError("Require 1 <= top_k <= candidate_k")

    # Retrieve the same first-stage candidates
    # used by our local MS MARCO reranker.
    output = search(
        question=question,
        bid_id=bid_id,
        doc_type=doc_type,
        addendum_number=addendum_number,
        file_name=file_name,
        mode="hybrid",
        top_k=candidate_k,
        candidate_k=max(30, candidate_k),
    )

    candidates = output["results"]

    if not candidates:
        output["results"] = []
        return output

    # Include source metadata so the reranker
    # can distinguish original RFPs and addendums.
    documents = []

    for item in candidates:
        document = (
            f"Source: {item['file_name']}\n"
            f"Document type: {item['doc_type']}\n"
            f"Addendum number: {item['addendum_number']}\n"
            f"Section: {item['section']}\n\n"
            f"{item['text']}"
        )
        documents.append(document)

    # One retry owner only: the shared application limiter below.
    client = cohere.ClientV2(api_key=api_key, max_retries=0)

    def rate_limited(exc):
        return any(
            marker in f"{type(exc).__name__}: {exc}".casefold()
            for marker in ("429", "rate limit", "too many requests", "resource_exhausted")
        )

    def retryable(exc):
        text = f"{type(exc).__name__}: {exc}".casefold()
        return rate_limited(exc) or "timeout" in text

    try:
        response = cohere_limiter().call(
            "cohere",
            lambda: client.rerank(
                model=MODEL,
                query=question,
                documents=documents,
                top_n=min(top_k, len(documents)),
            ),
            retryable=retryable,
        )
    except Exception as exc:
        if not rate_limited(exc):
            raise
        # The hybrid search result is already Qdrant RRF ranked. Preserve its
        # payload verbatim and make the absence of Cohere reranking explicit.
        fallback = []
        for rank, candidate in enumerate(candidates[:top_k], start=1):
            item = candidate.copy()
            item["retrieval_rank"] = item.get("rank", rank)
            item["retrieval_score"] = item.get("score")
            item["rank"] = rank
            item["rerank_score"] = None
            fallback.append(item)
        output["results"] = fallback
        output["reranker_model"] = None
        output["ranking_method"] = "qdrant_hybrid_rrf_fallback"
        output["rerank_applied"] = False
        output["rerank_fallback_reason"] = "cohere_rate_limit"
        output["warnings"] = [
            "Cohere rate limit exhausted; used Qdrant hybrid RRF order without reranking."
        ]
        output["candidates_retrieved"] = len(candidates)
        return output

    reranked = []

    for rank, match in enumerate(response.results, start=1):
        item = candidates[match.index].copy()

        item["retrieval_rank"] = item["rank"]
        item["retrieval_score"] = item["score"]
        item["rerank_score"] = float(match.relevance_score)
        item["rank"] = rank

        reranked.append(item)

    output["results"] = reranked
    output["reranker_model"] = MODEL
    output["ranking_method"] = "cohere_rerank"
    output["rerank_applied"] = True
    output["candidates_retrieved"] = len(candidates)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bid", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--doc-type")
    parser.add_argument("--addendum-number", type=int)
    parser.add_argument("--file-name")

    args = parser.parse_args()

    output = retrieve_and_rerank_cohere(
        question=args.question,
        bid_id=args.bid,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        doc_type=args.doc_type,
        addendum_number=args.addendum_number,
        file_name=args.file_name,
    )

    print("\nModel:", output.get("reranker_model", MODEL))
    print("Question:", output["question"])
    print("Candidates:", output.get("candidates_retrieved", 0))

    for item in output["results"]:
        print("\n" + "=" * 65)
        print(
            f"Final rank: {item['rank']} "
            f"| Previous rank: {item['retrieval_rank']}"
        )
        print(f"Cohere score: {item['rerank_score']:.4f}")
        print("Source:", item["file_name"])
        print("Page:", item["page_number"])
        print("Addendum:", item["addendum_number"])
        print("Evidence:\n", item["text"])


if __name__ == "__main__":
    main()
