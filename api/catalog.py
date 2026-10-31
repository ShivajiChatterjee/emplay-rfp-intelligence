"""Safe, bid-agnostic discovery of local bids, documents, and artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from ingestion import SUPPORTED_SUFFIXES


BID_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
RUN_ID_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}$")


class CatalogError(ValueError):
    """Raised when a catalog identifier is invalid or unavailable."""


def data_root() -> Path:
    return Path(os.getenv("EMPLAY_DATA_DIR", "data")).resolve()


def output_root() -> Path:
    return Path(os.getenv("EMPLAY_OUTPUT_DIR", "outputs")).resolve()


def validate_bid_id(bid_id: str) -> str:
    if not BID_ID_PATTERN.fullmatch(bid_id):
        raise CatalogError("Invalid bid identifier")
    return bid_id


def bid_directory(bid_id: str, *, must_exist: bool = True) -> Path:
    bid_id = validate_bid_id(bid_id)
    root = data_root()
    candidate = (root / bid_id).resolve()
    if candidate.parent != root:
        raise CatalogError("Invalid bid identifier")
    if must_exist and not candidate.is_dir():
        raise CatalogError("Bid not found")
    return candidate


def _documents(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def document_id(folder: Path, path: Path) -> str:
    relative = path.resolve().relative_to(folder.resolve()).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]


def document_records(bid_id: str) -> list[dict[str, Any]]:
    folder = bid_directory(bid_id)
    return [
        {
            "document_id": document_id(folder, path),
            "name": path.name,
            "media_type": "application/pdf" if path.suffix.lower() == ".pdf" else "text/html",
            "size_bytes": path.stat().st_size,
        }
        for path in _documents(folder)
    ]


def resolve_document(bid_id: str, requested_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{24}", requested_id):
        raise CatalogError("Invalid document identifier")
    folder = bid_directory(bid_id)
    for path in _documents(folder):
        if document_id(folder, path) == requested_id:
            return path
    raise CatalogError("Document not found")


def index_version(bid_id: str) -> str:
    """Fingerprint the local document identity without reading document content."""

    folder = bid_directory(bid_id)
    digest = hashlib.sha256()
    for path in _documents(folder):
        stat = path.stat()
        digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _find_artifact(pattern: str) -> Path | None:
    root = output_root()
    candidates = sorted(root.rglob(pattern), key=lambda path: path.stat().st_mtime, reverse=True) if root.exists() else []
    if candidates:
        return candidates[0]
    submission = Path("submission") / pattern
    return submission.resolve() if submission.is_file() else None


def extraction_artifact(bid_id: str) -> Path | None:
    validate_bid_id(bid_id)
    return _find_artifact(f"{bid_id}_extraction.json")


def load_extraction(bid_id: str) -> dict[str, Any]:
    path = extraction_artifact(bid_id)
    if path is None:
        raise CatalogError("Validated extraction is not available")
    return json.loads(path.read_text(encoding="utf-8"))


def extraction_evidence(bid_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return original checkpoint passages for UI inspection, when retained."""

    validate_bid_id(bid_id)
    root = output_root()
    matches = sorted(
        root.rglob(f"{bid_id}_field_checkpoints.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if root.exists() else []
    for path in matches:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fields = payload.get("fields", {})
            return {
                field_name: [
                    {
                        "file": item.get("file_name"),
                        "page": item.get("page_number"),
                        "chunk_id": item.get("chunk_id"),
                        "quote": item.get("quote"),
                    }
                    for item in record.get("evidence", [])
                    if item.get("file_name") and item.get("page_number") and item.get("chunk_id")
                ]
                for field_name, record in fields.items()
            }
        except (OSError, ValueError, TypeError):
            continue
    return {}


def list_bids() -> list[dict[str, Any]]:
    root = data_root()
    if not root.exists():
        return []
    records = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if not BID_ID_PATTERN.fullmatch(folder.name):
            continue
        documents = _documents(folder)
        extraction = extraction_artifact(folder.name)
        records.append(
            {
                "bid_id": folder.name,
                "document_count": len(documents),
                "documents": document_records(folder.name),
                "index_version": index_version(folder.name),
                "extraction_status": "complete" if extraction else "not_available",
            }
        )
    return records


def historical_runs(
    limit: int = 50,
    *,
    exclude_run_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Discover preserved CLI/development runs without changing them."""

    root = output_root()
    if not root.exists():
        return []
    excluded = exclude_run_ids or set()
    by_run: dict[str, dict[str, Any]] = {}
    for path in root.rglob("run_manifest_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run_id = payload.get("run_id")
            if not run_id or run_id in excluded:
                continue
            by_run[run_id] = {
                "run_id": run_id,
                "inspection_id": run_id,
                "source": "historical",
                "operation": "extract",
                "bid_ids": [payload.get("bid_id")] if payload.get("bid_id") else [],
                "status": payload.get("run_status") or "unknown",
                "final_submission_published": bool(payload.get("final_submission_published")),
                "updated_at": path.stat().st_mtime,
            }
        except (OSError, ValueError, TypeError):
            continue
    for path in root.rglob("workflow_trace_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run_id = payload.get("run_id")
            if not run_id or run_id in excluded or run_id in by_run:
                continue
            request = payload.get("request", {})
            has_result = payload.get("qa_answer") is not None or payload.get("final_output") is not None
            by_run[run_id] = {
                "run_id": run_id,
                "inspection_id": run_id,
                "source": "historical",
                "operation": request.get("mode") or "workflow",
                "bid_ids": request.get("bid_ids") or [],
                "status": "complete" if has_result else ("failed" if payload.get("errors") else "incomplete"),
                "final_submission_published": payload.get("final_output") is not None,
                "updated_at": path.stat().st_mtime,
            }
        except (OSError, ValueError, TypeError):
            continue
    return sorted(
        by_run.values(), key=lambda item: item["updated_at"], reverse=True
    )[: max(1, min(limit, 200))]


def recent_runs(limit: int = 8) -> list[dict[str, Any]]:
    """Backward-compatible historical manifest view."""

    return historical_runs(limit)


def find_trace(run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise CatalogError("Invalid run identifier")
    root = output_root()
    matches = list(root.rglob(f"workflow_trace_{run_id}.json")) if root.exists() else []
    if not matches:
        raise CatalogError("Run trace not found")
    return matches[0]


def public_payload(value: Any) -> Any:
    """Remove internal path-bearing keys from API payloads recursively."""

    blocked = {"bid_folders", "output_dir", "checkpoint_path", "output_path", "path"}
    if isinstance(value, dict):
        return {key: public_payload(item) for key, item in value.items() if key not in blocked}
    if isinstance(value, list):
        return [public_payload(item) for item in value]
    if isinstance(value, str):
        sanitized = value
        for root, label in (
            (Path.cwd().resolve(), "<workspace>"),
            (data_root(), "<data>"),
            (output_root(), "<outputs>"),
        ):
            sanitized = sanitized.replace(str(root), label)
            sanitized = sanitized.replace(str(root).replace("\\", "/"), label)
        return sanitized
    return value
