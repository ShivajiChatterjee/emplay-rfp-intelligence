"""Deterministic ingestion/index-readiness tool agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from qdrant_client import models

from ingestion import SUPPORTED_SUFFIXES, ingest_bid
from rag.chunking import build_chunks
from rag.vector_store import COLLECTION, get_client, index_chunks


class IndexStatus(BaseModel):
    bid_id: str
    status: Literal["ready", "indexed", "partial", "error"]
    documents: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _has_indexed_bid(client, bid_id: str) -> bool:
    if not client.collection_exists(COLLECTION):
        return False
    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="bid_id",
                    match=models.MatchValue(value=bid_id),
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(points)


def ensure_bid_indexed(
    bid_dir: Path,
    output_dir: Path = Path("outputs"),
    *,
    client=None,
) -> IndexStatus:
    """Reuse the existing ingestion/chunk/index pipeline only when needed."""

    bid_dir = Path(bid_dir)
    bid_id = bid_dir.name
    if not bid_dir.is_dir():
        return IndexStatus(
            bid_id=bid_id or str(bid_dir),
            status="error",
            errors=[f"Bid folder not found: {bid_dir}"],
        )

    supported = [
        path
        for path in bid_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not supported:
        return IndexStatus(
            bid_id=bid_id,
            status="error",
            errors=["No supported PDF or HTML documents found"],
        )

    try:
        qdrant = client or get_client()
        if _has_indexed_bid(qdrant, bid_id):
            return IndexStatus(
                bid_id=bid_id,
                status="ready",
                documents=len(supported),
            )

        ingested = ingest_bid(bid_dir)
        chunks = build_chunks(ingested)
        if not chunks:
            return IndexStatus(
                bid_id=bid_id,
                status="error",
                documents=len(ingested.get("documents", [])),
                errors=["Ingestion produced no searchable chunks"],
            )

        index_chunks(qdrant, chunks)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{bid_id}_ingested.json").write_text(
            json.dumps(ingested, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / f"{bid_id}_chunks.json").write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        errors = [
            f"{item.get('file_name')}: {item.get('error')}"
            for item in ingested.get("errors", [])
        ]
        warnings = [
            warning
            for document in ingested.get("documents", [])
            for warning in document.get("warnings", [])
        ]
        return IndexStatus(
            bid_id=bid_id,
            status="partial" if errors else "indexed",
            documents=len(ingested.get("documents", [])),
            chunks=len(chunks),
            warnings=warnings,
            errors=errors,
        )
    except Exception as exc:
        return IndexStatus(
            bid_id=bid_id,
            status="error",
            errors=[f"{type(exc).__name__}: {exc}"],
        )
