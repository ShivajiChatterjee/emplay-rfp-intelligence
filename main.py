"""Command-line entrypoints for ingestion, extraction, and grounded Q&A.

Existing ingestion usage remains valid because ``--mode`` defaults to ingest:
    python main.py --bid data/Bid1 --pages 1:3

Agentic workflows:
    python main.py --bid data/Bid1 --mode extract
    python main.py --bid data/Bid1 --mode partial-extract --field "Due Date"
    python main.py --bid data/Bid1 --mode ask --question "..."
    python main.py --bid data/Bid1 --bid data/Bid2 --mode compare --question "..."
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ingestion import ingest_bid


def parse_range(value: str) -> tuple[int, int]:
    try:
        start, end = map(int, value.split(":"))
        if start < 1 or end < start:
            raise ValueError()
        return start, end
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use --pages START:END, e.g. 1:3"
        ) from exc


def run_ingestion(
    bid_dir: Path,
    *,
    page_range: tuple[int, int] | None,
    output_dir: Path,
) -> Path:
    result = ingest_bid(bid_dir, page_range=page_range)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{bid_dir.name}_ingested.json"
    target.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {target}")
    print("Documents:", len(result["documents"]))
    print("Pages:", sum(len(doc["pages"]) for doc in result["documents"]))
    print("Tables:", sum(len(doc["tables"]) for doc in result["documents"]))
    print("Warnings:", sum(len(doc["warnings"]) for doc in result["documents"]))
    print("Errors:", len(result["errors"]))
    for error in result["errors"]:
        print("  FAILED:", error["file_name"], "-", error["error"])
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emplay RFP ingestion and intelligence workflow"
    )
    parser.add_argument(
        "--bid",
        type=Path,
        action="append",
        required=True,
        help="Bid directory; repeat for compare mode",
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "extract", "partial-extract", "ask", "compare"],
        default="ingest",
    )
    parser.add_argument("--question")
    parser.add_argument("--goal")
    parser.add_argument(
        "--field",
        action="append",
        help="Official field label; partial-extract only, maximum three",
    )
    parser.add_argument("--pages", type=parse_range)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--max-retries", type=int, choices=[0, 1, 2])
    parser.add_argument(
        "--disable-query-rewrite",
        action="store_true",
        help="Disable the one-shot Gemini retrieval rewrite for cost control",
    )
    parser.add_argument(
        "--deterministic-validation-only",
        action="store_true",
        help="Run citation/type/conservative validation without semantic critic calls",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse fingerprint-matching terminal field checkpoints; extract mode only",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint JSON path (defaults inside the selected output directory)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.mode == "ingest":
        if args.resume or args.checkpoint:
            parser.error("--resume/--checkpoint are supported only in extract mode")
        if len(args.bid) != 1:
            parser.error("ingest mode accepts exactly one --bid")
        run_ingestion(
            args.bid[0],
            page_range=args.pages,
            output_dir=args.output,
        )
        return

    if args.pages:
        parser.error("--pages is supported only in ingest mode")
    if (args.resume or args.checkpoint) and args.mode != "extract":
        parser.error("--resume/--checkpoint are supported only in extract mode")

    if args.mode == "partial-extract":
        if len(args.bid) != 1:
            parser.error("partial-extract accepts exactly one --bid")
        if not args.field:
            parser.error("partial-extract requires at least one --field")
        if len(args.field) > 3:
            parser.error("partial-extract accepts at most three --field values")
        if args.max_retries not in {None, 0}:
            parser.error("partial-extract fixes --max-retries at 0")
        from workflow.partial_smoke import run_partial_extraction_smoke

        try:
            artifact, target = run_partial_extraction_smoke(
                bid_dir=args.bid[0],
                selected_fields=tuple(args.field),
                output_dir=args.output,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print("Saved partial smoke artifact:", target)
        print(artifact.model_dump_json(indent=2))
        return

    if args.field:
        parser.error("--field is supported only in partial-extract mode")

    from workflow import WorkflowRequest, run_workflow

    try:
        request = WorkflowRequest(
            mode=args.mode,
            bid_folders=tuple(args.bid),
            goal=args.goal,
            question=args.question,
            output_dir=args.output,
            max_retries=args.max_retries if args.max_retries is not None else 2,
            allow_query_rewrite=not args.disable_query_rewrite,
            semantic_validation=not args.deterministic_validation_only,
            resume=args.resume,
            checkpoint_path=args.checkpoint,
        )
    except ValueError as exc:
        parser.error(str(exc))

    result = run_workflow(request)
    print("Run ID:", result["run_id"])
    manifest = result.get("run_manifest")
    if manifest is not None:
        print("Run status:", manifest.run_status)
        print("Run manifest:", result.get("run_manifest_path"))
        print("Workflow trace:", result.get("workflow_trace_path"))
    if result.get("final_output") is not None:
        if manifest is not None and manifest.run_status != "complete":
            print("Final submission was not published; resume from the checkpoint.")
        else:
            print(result["final_output"].model_dump_json(indent=2))
    elif result.get("qa_answer") is not None:
        print(result["qa_answer"].model_dump_json(indent=2))
    else:
        if manifest is not None:
            print("No official extraction JSON was published; the incomplete run is resumable.")
        else:
            print("Workflow stopped without a final result.")
        for error in result.get("errors", []):
            print("ERROR:", error)


if __name__ == "__main__":
    main()
