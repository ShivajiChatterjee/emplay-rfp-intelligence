
"""Evaluate RFP retrieval against annotated source passages."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from rag.retrieval import search
from rag.vector_store import COLLECTION, get_client


MODES = (
    "dense",
    "bm25",
    "hybrid",
    "ms_marco",
    "cohere",
)


def normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def is_relevant(item: dict, example: dict) -> bool:
    """Match source, physical page, and passage evidence."""

    gold = example["gold"]

    return (
        item.get("bid_id") == example["bid_id"]
        and normalize(gold["file_contains"])
        in normalize(item.get("file_name"))
        and item.get("page_number") == gold["page"]
        and normalize(gold["anchor"])
        in normalize(item.get("text"))
        and all(
            item.get(key) == value
            for key, value in example.get("filters", {}).items()
        )
    )


def validate_gold(questions: list[dict], client) -> None:
    """Ensure ground-truth passages exist in Qdrant."""

    points = []
    offset = None

    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points.extend(
            point.payload or {}
            for point in batch
        )

        if offset is None:
            break

    missing = [
        question["id"]
        for question in questions
        if not any(
            is_relevant(point, question)
            for point in points
        )
    ]

    if missing:
        raise SystemExit(
            "Gold passages missing from Qdrant: "
            + ", ".join(missing)
            + "\nCheck metadata, indexed chunks, and evidence anchors."
        )

    print(
        f"Verified {len(questions)} gold passages "
        f"against {len(points)} Qdrant points."
    )


def retrieve(
    mode: str,
    example: dict,
    client,
    top_k: int,
    candidate_k: int,
) -> list[dict]:

    arguments = {
        "question": example["question"],
        "bid_id": example["bid_id"],
        "doc_type": example.get("filters", {}).get("doc_type"),
        "addendum_number": example.get("filters", {}).get(
            "addendum_number"
        ),
        "top_k": top_k,
        "candidate_k": candidate_k,
    }

    if mode in ("dense", "bm25", "hybrid"):
        output = search(
            mode=mode,
            client=client,
            **arguments,
        )
        return output["results"]

    if mode == "ms_marco":
        from rag.reranker import retrieve_and_rerank

        output = retrieve_and_rerank(
            mode="hybrid",
            **arguments,
        )
        return output["results"]

    if mode == "cohere":
        from rag.cohere_reranker import retrieve_and_rerank_cohere

        output = retrieve_and_rerank_cohere(
            **arguments,
        )
        return output["results"]

    raise ValueError(f"Unsupported mode: {mode}")


def run_evaluation(
    questions: list[dict],
    modes: list[str],
    client,
    top_k: int,
    candidate_k: int,
    cohere_delay: float,
) -> dict:

    reports = {}

    if "ms_marco" in modes:
        from rag.reranker import get_reranker

        # Preload once, excluding model startup from query latency.
        get_reranker()

    for mode in modes:
        print(f"\n=== Evaluating {mode} ===")
        rows = []

        for index, example in enumerate(questions, start=1):
            start = time.perf_counter()

            results = retrieve(
                mode=mode,
                example=example,
                client=client,
                top_k=top_k,
                candidate_k=candidate_k,
            )

            latency = time.perf_counter() - start

            # First relevant result inside top-k.
            relevant_rank = next(
                (
                    rank
                    for rank, result in enumerate(
                        results[:top_k],
                        start=1,
                    )
                    if is_relevant(result, example)
                ),
                None,
            )

            rows.append({
                "id": example["id"],
                "bid_id": example["bid_id"],
                "question": example["question"],
                "first_relevant_rank": relevant_rank,
                "latency_s": round(latency, 3),
                "top_sources": [
                    {
                        "file": result.get("file_name"),
                        "page": result.get("page_number"),
                        "chunk_id": result.get("chunk_id"),
                    }
                    for result in results[:top_k]
                ],
            })

            position = (
                str(relevant_rank)
                if relevant_rank is not None
                else "MISS"
            )

            print(
                f"{index:02d}/{len(questions)} "
                f"{example['id']} | "
                f"Gold rank: {position} | "
                f"{latency:.2f}s"
            )

            # Cohere trial keys have request-rate limits.
            # Delay is excluded from latency measurement.
            if (
                mode == "cohere"
                and index < len(questions)
                and cohere_delay > 0
            ):
                time.sleep(cohere_delay)

        count = len(rows)

        recall_at_1 = sum(
            row["first_relevant_rank"] == 1
            for row in rows
        ) / count

        recall_at_k = sum(
            row["first_relevant_rank"] is not None
            for row in rows
        ) / count

        mrr_at_k = sum(
            1 / row["first_relevant_rank"]
            if row["first_relevant_rank"]
            else 0
            for row in rows
        ) / count

        reports[mode] = {
            "n_questions": count,
            "recall_at_1": round(recall_at_1, 4),
            f"recall_at_{top_k}": round(recall_at_k, 4),
            f"mrr_at_{top_k}": round(mrr_at_k, 4),
            "mean_latency_s": round(
                statistics.mean(
                    row["latency_s"]
                    for row in rows
                ),
                3,
            ),
            "queries": rows,
        }

    return reports


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate RFP retrieval quality"
    )

    parser.add_argument(
        "--questions",
        default="evaluation/questions.json",
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
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
        "--limit",
        type=int,
        default=None,
        help="Evaluate a subset for smoke testing",
    )

    parser.add_argument(
        "--cohere-delay",
        type=float,
        default=7.0,
    )

    parser.add_argument(
        "--out",
        default="outputs/retrieval_evaluation.json",
    )

    args = parser.parse_args()

    if not 1 <= args.top_k <= args.candidate_k:
        parser.error(
            "Require 1 <= top-k <= candidate-k"
        )

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    all_questions = json.loads(
        Path(args.questions).read_text(
            encoding="utf-8"
        )
    )

    questions = (
        all_questions[:args.limit]
        if args.limit
        else all_questions
    )

    client = get_client()

    # Prevent misleading evaluation scores caused by
    # missing or incorrectly indexed gold passages.
    validate_gold(questions, client)

    reports = run_evaluation(
        questions=questions,
        modes=args.modes,
        client=client,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        cohere_delay=args.cohere_delay,
    )

    output = {
        "questions_evaluated": len(questions),
        "total_gold_questions": len(all_questions),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "metric_note": (
            "One annotated evidence passage per question. "
            "Recall@k is equivalent to Hit@k."
        ),
        "modes": reports,
    }

    destination = Path(args.out)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    k = args.top_k

    lines = [
        "# RFP Retrieval Evaluation",
        "",
        (
            f"Evaluated {len(questions)} of "
            f"{len(all_questions)} labeled questions."
        ),
        "",
        f"| Mode | Recall@1 | Recall@{k} | MRR@{k} | Mean latency (s) |",
        "|---|---:|---:|---:|---:|",
    ]

    for mode, metrics in reports.items():

        lines.append(
            f"| {mode} "
            f"| {metrics['recall_at_1']:.3f} "
            f"| {metrics[f'recall_at_{k}']:.3f} "
            f"| {metrics[f'mrr_at_{k}']:.3f} "
            f"| {metrics['mean_latency_s']:.2f} |"
        )

    lines.extend([
        "",
        (
            "**Method:** Each question has one annotated evidence "
            "passage identified by source document, physical page, "
            "and an evidence phrase. A hit occurs when any top-k "
            "chunk contains that evidence at the specified source. "
            "With one gold evidence unit per question, Recall@k "
            "is equivalent to Hit@k. MRR@k uses the first matching rank."
        ),
        "",
        (
            "**Latency:** Includes retrieval, network and reranking "
            "inference; excludes one-time MS MARCO model loading "
            "and intentional Cohere rate-limit delays."
        ),
    ])

    markdown_path = destination.with_suffix(".md")

    markdown_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n=== FINAL RESULTS ===")

    for line in lines:
        print(line)

    print("\nJSON report:", destination)
    print("Markdown report:", markdown_path)


if __name__ == "__main__":
    main()
