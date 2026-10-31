
"""Qdrant Cloud indexing for RFP chunks.

Stores:
- Dense MiniLM embeddings
- BM25 sparse vectors
- Source and citation metadata

Supports:
- Automatic collection creation
- HNSW configuration
- Metadata payload indexes
- Incremental document-level indexing
- Replacing updated documents
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


COLLECTION = "emplay_rfp_chunks"

DENSE_MODEL = "sentence-transformers/all-minilm-l6-v2"
SPARSE_MODEL = "qdrant/bm25"

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

BATCH_SIZE = 16


# --------------------------------------------------
# 1. CONNECT TO QDRANT CLOUD
# --------------------------------------------------

def get_client() -> QdrantClient:

    load_dotenv()

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY "
            "must be configured in .env"
        )

    return QdrantClient(
        url=url,
        api_key=api_key,
        cloud_inference=True,
        timeout=90,
    )


# --------------------------------------------------
# 2. CREATE COLLECTION AND INDEXES
# --------------------------------------------------

def ensure_collection(client: QdrantClient):

    if not client.collection_exists(COLLECTION):

        client.create_collection(
            collection_name=COLLECTION,

            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                )
            },

            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },

            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )

        print(f"Created collection: {COLLECTION}")

    else:
        print(f"Using collection: {COLLECTION}")

    # The fields below support metadata filtering.
    index_fields = {
        "bid_id": models.PayloadSchemaType.KEYWORD,
        "file_name": models.PayloadSchemaType.KEYWORD,
        "doc_type": models.PayloadSchemaType.KEYWORD,
        "addendum_number": models.PayloadSchemaType.INTEGER,
        "page_number": models.PayloadSchemaType.INTEGER,
        "chunk_type": models.PayloadSchemaType.KEYWORD,
    }

    collection_info = client.get_collection(COLLECTION)

    existing_indexes = (
        collection_info.payload_schema or {}
    )

    for field, field_type in index_fields.items():

        if field in existing_indexes:
            continue

        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=field_type,
            wait=True,
        )

        print(f"Created payload index: {field}")


# --------------------------------------------------
# 3. DOCUMENT FINGERPRINTING
# --------------------------------------------------

def document_fingerprint(chunks: list[dict]) -> str:
    """Hash the indexed representation of a document."""

    serialized = json.dumps(
        chunks,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# 4. FIND EXISTING DOCUMENT CHUNKS
# --------------------------------------------------

def get_existing_points(
    client: QdrantClient,
    bid_id: str,
    file_name: str,
) -> list:

    conditions = models.Filter(
        must=[
            models.FieldCondition(
                key="bid_id",
                match=models.MatchValue(value=bid_id),
            ),
            models.FieldCondition(
                key="file_name",
                match=models.MatchValue(value=file_name),
            ),
        ]
    )

    points = []
    offset = None

    while True:

        batch, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=conditions,
            limit=128,
            offset=offset,
            with_payload=["document_hash"],
            with_vectors=False,
        )

        points.extend(batch)

        if offset is None:
            break

    return points


# --------------------------------------------------
# 5. CONVERT CHUNK TO QDRANT POINT
# --------------------------------------------------

def make_point(
    chunk: dict,
    doc_hash: str,
) -> models.PointStruct:

    text = chunk["text"]

    payload = {
        key: value
        for key, value in chunk.items()
        if key != "id"
    }

    payload["document_hash"] = doc_hash

    return models.PointStruct(
        id=chunk["id"],
        vector={
            DENSE_VECTOR: models.Document(
                text=text,
                model=DENSE_MODEL,
            ),
            SPARSE_VECTOR: models.Document(
                text=text,
                model=SPARSE_MODEL,
            ),
        },
        payload=payload,
    )


# --------------------------------------------------
# 6. INDEX ONE DOCUMENT INCREMENTALLY
# --------------------------------------------------

def index_document(
    client: QdrantClient,
    chunks: list[dict],
) -> str:

    first = chunks[0]

    bid_id = first["bid_id"]
    file_name = first["file_name"]

    doc_hash = document_fingerprint(chunks)

    existing_points = get_existing_points(
        client,
        bid_id,
        file_name,
    )

    existing_ids = {
        str(point.id)
        for point in existing_points
    }

    new_ids = {
        str(chunk["id"])
        for chunk in chunks
    }

    # If the exact same document chunks are already
    # indexed, avoid generating embeddings again.
    is_unchanged = (
        existing_ids == new_ids
        and bool(existing_points)
        and all(
            (point.payload or {}).get(
                "document_hash"
            ) == doc_hash
            for point in existing_points
        )
    )

    if is_unchanged:
        print(
            f"SKIPPED | {bid_id} | {file_name} "
            "(unchanged)"
        )
        return "skipped"

    # Upload new or updated chunks.
    for start in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[
            start:start + BATCH_SIZE
        ]

        points = [
            make_point(chunk, doc_hash)
            for chunk in batch
        ]

        client.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True,
        )

    # Remove any old chunk IDs that disappeared
    # after the source document changed.
    stale_ids = existing_ids - new_ids

    if stale_ids:

        client.delete(
            collection_name=COLLECTION,
            points_selector=models.PointIdsList(
                points=list(stale_ids)
            ),
            wait=True,
        )

    status = (
        "UPDATED"
        if existing_points
        else "ADDED"
    )

    print(
        f"{status} | {bid_id} | {file_name} "
        f"| {len(chunks)} chunks"
    )

    return status.lower()


# --------------------------------------------------
# 7. INDEX AN ENTIRE BID
# --------------------------------------------------

def index_chunks(
    client: QdrantClient,
    chunks: list[dict],
):

    if not chunks:
        print("No chunks found.")
        return

    ensure_collection(client)

    # Group by bid and source document so that
    # incremental updates never affect other bids.
    documents = defaultdict(list)

    for chunk in chunks:

        key = (
            chunk["bid_id"],
            chunk["file_name"],
        )

        documents[key].append(chunk)

    stats = {
        "added": 0,
        "updated": 0,
        "skipped": 0,
    }

    for key in sorted(documents):

        result = index_document(
            client,
            documents[key],
        )

        stats[result] += 1

    info = client.get_collection(COLLECTION)

    print("\nIndexing completed.")
    print("Documents added:", stats["added"])
    print("Documents updated:", stats["updated"])
    print("Documents skipped:", stats["skipped"])
    print("Collection:", COLLECTION)
    print("Stored points:", info.points_count)


# --------------------------------------------------
# 8. COMMAND-LINE ENTRY POINT
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Index RFP chunks in Qdrant Cloud"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to generated chunks JSON",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    chunks = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    client = get_client()

    print("Connected to Qdrant Cloud.")
    print("Chunks in input:", len(chunks))

    index_chunks(client, chunks)


if __name__ == "__main__":
    main()
