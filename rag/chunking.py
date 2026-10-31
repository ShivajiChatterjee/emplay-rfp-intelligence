
"""Section-aware and table-aware chunking of ingested RFP documents.

Reads structured JSON produced by ingestion.py.

Features:
- Heading-aware splitting
- Table-aware row grouping
- Overlap between text chunks
- Physical-page boundaries
- Source metadata for citations
- Stable chunk IDs for incremental indexing
"""

from __future__ import annotations

import argparse
import html
import json
import re
import uuid

from collections import Counter
from pathlib import Path


TARGET_CHARS = 1000
OVERLAP_CHARS = 120

# --------------------------------------------------
# 1. TEXT CLEANING
# --------------------------------------------------

def clean_text(text: str) -> str:
    """Remove non-searchable markup while preserving content."""

    text = re.sub(
        r"<!--\s*image\s*-->",
        "",
        text
    )

    text = html.unescape(text)
    text = text.replace("\u00a0", " ")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# 2. IDENTIFY HEADINGS, TABLES AND TEXT
# --------------------------------------------------

def parse_blocks(text: str):
    """Yield document blocks in their extracted reading order."""

    lines = clean_text(text).splitlines()
    index = 0

    while index < len(lines):

        line = lines[index].strip()

        if not line:
            index += 1
            continue

        # Markdown headings produced by Docling.
        if re.match(r"^#{1,6}\s+", line):
            yield "heading", line
            index += 1
            continue

        # Group consecutive Markdown table rows.
        if line.startswith("|"):

            table_lines = []

            while (
                index < len(lines)
                and lines[index].strip().startswith("|")
            ):
                table_lines.append(
                    lines[index].strip()
                )
                index += 1

            yield "table", table_lines
            continue

        # Regular paragraphs and lists.
        paragraph = []

        while index < len(lines):

            candidate = lines[index].strip()

            if (
                not candidate
                or candidate.startswith("|")
                or re.match(r"^#{1,6}\s+", candidate)
            ):
                break

            paragraph.append(candidate)
            index += 1

        if paragraph:
            yield "text", "\n".join(paragraph)


# --------------------------------------------------
# 3. LONG TEXT SPLITTING WITH OVERLAP
# --------------------------------------------------

def split_with_overlap(
    text: str,
    limit: int,
    overlap: int
) -> list[str]:
    """Split long text at whitespace boundaries."""

    if len(text) <= limit:
        return [text] if text.strip() else []

    results = []
    start = 0

    while start < len(text):

        end = min(
            start + limit,
            len(text)
        )

        # Prefer a nearby word or line boundary.
        if end < len(text):

            search_from = start + int(limit * 0.65)

            boundary = max(
                text.rfind("\n", search_from, end),
                text.rfind(" ", search_from, end)
            )

            if boundary > start:
                end = boundary

        part = text[start:end].strip()

        if part:
            results.append(part)

        if end == len(text):
            break

        # Retain context from the previous chunk.
        next_start = max(
            start + 1,
            end - overlap
        )

        # Avoid starting in the middle of a word.
        if (
            next_start > 0
            and text[next_start - 1].isalnum()
            and text[next_start].isalnum()
        ):

            match = re.search(
                r"\s+",
                text[next_start:end]
            )

            next_start += (
                match.end()
                if match
                else end - next_start
            )

        start = (
            next_start
            if next_start > start
            else end
        )

    return results


# --------------------------------------------------
# 4. TABLE-AWARE CHUNKING
# --------------------------------------------------

def split_table(
    lines: list[str],
    section: str,
    limit: int
):
    """Split table rows while retaining table-header context."""

    if not lines:
        return

    header = lines[0]

    # Remove Markdown table separator rows.
    rows = [
        row
        for row in lines[1:]
        if not re.fullmatch(
            r"[\s|:\-]+",
            row
        )
    ]

    context = (
        f"Section: {section}\n"
        if section
        else ""
    )

    prefix = (
        context
        + "Table columns: "
        + header
    )

    if not rows:
        for part in split_with_overlap(
            prefix, limit, 0
        ):
            yield part, None, None
        return

    current_rows = []
    first_row = 1

    available = max(
        200,
        limit - len(prefix) - 2
    )

    # Extremely wide headers are stored separately,
    # so table values are never discarded.
    if (
        available <= 200
        or len(prefix) >= limit - 200
    ):

        for part in split_with_overlap(
            prefix, limit, 0
        ):
            yield part, 0, 0

        prefix = context + "Table row: "

        available = max(
            200,
            limit - len(prefix) - 2
        )

    for row_index, row in enumerate(
        rows,
        start=1
    ):

        # Handle exceptionally long table rows.
        if len(row) > available:

            if current_rows:

                yield (
                    prefix
                    + "\n"
                    + "\n".join(current_rows),
                    first_row,
                    row_index - 1
                )

                current_rows = []

            for part in split_with_overlap(
                row,
                available,
                0
            ):

                yield (
                    prefix + "\n" + part,
                    row_index,
                    row_index
                )

            first_row = row_index + 1
            continue

        proposed_length = (
            len(prefix)
            + len("\n".join(current_rows))
            + len(row)
            + 2
        )

        if (
            current_rows
            and proposed_length > limit
        ):

            yield (
                prefix
                + "\n"
                + "\n".join(current_rows),
                first_row,
                row_index - 1
            )

            # Repeat the last row where it fits.
            last_row = current_rows[-1]

            can_overlap = (
                len(prefix)
                + len(last_row)
                + len(row)
                + 2
                <= limit
            )

            current_rows = (
                [last_row]
                if can_overlap
                else []
            )

            first_row = (
                row_index - 1
                if current_rows
                else row_index
            )

        if not current_rows:
            first_row = row_index

        current_rows.append(row)

    if current_rows:

        yield (
            prefix
            + "\n"
            + "\n".join(current_rows),
            first_row,
            len(rows)
        )


# --------------------------------------------------
# 5. BUILD SEARCHABLE CHUNKS
# --------------------------------------------------

def build_chunks(
    data: dict,
    max_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS
) -> list[dict]:
    """Convert ingested bid JSON into metadata-rich chunks."""

    if max_chars < 500:
        raise ValueError(
            "max_chars must be at least 500"
        )

    if not 0 <= overlap_chars < max_chars // 3:
        raise ValueError(
            "Invalid overlap_chars setting"
        )

    bid_id = data["bid_id"]
    chunks = []

    for document in data.get("documents", []):

        filename = document["file_name"]

        table_lookup = {
            table["table_id"]: table
            for table in document.get("tables", [])
        }

        for page in document.get("pages", []):

            page_number = page["page_number"]

            section_stack = []
            text_buffer = []
            page_chunks = []

            table_ids = page.get("table_ids", [])
            table_position = 0

            # ----------------------------------
            # Section context
            # ----------------------------------

            def section_name():
                return " > ".join(
                    section[1]
                    for section in section_stack[-2:]
                )

            # ----------------------------------
            # Append one searchable chunk
            # ----------------------------------

            def add_chunk(
                content,
                chunk_type,
                section,
                table_id=None,
                row_start=None,
                row_end=None
            ):

                if not content.strip():
                    return

                page_chunks.append({
                    "bid_id": bid_id,
                    "file_name": filename,
                    "doc_type": document.get("doc_type"),
                    "addendum_number": document.get(
                        "addendum_number"
                    ),
                    "document_date": document.get(
                        "document_date"
                    ),
                    "page_number": page_number,
                    "page_type": page.get("page_type"),
                    "chunk_type": chunk_type,
                    "section": section or None,
                    "table_id": table_id,
                    "row_start": row_start,
                    "row_end": row_end,
                    "text": content.strip()
                })

            # ----------------------------------
            # Flush accumulated text
            # ----------------------------------

            def flush_text():

                if not text_buffer:
                    return

                section = section_name()

                prefix = (
                    f"Section: {section}\n\n"
                    if section
                    else ""
                )

                content = "\n\n".join(
                    text_buffer
                )

                available = max(
                    250,
                    max_chars - len(prefix)
                )

                parts = split_with_overlap(
                    content,
                    available,
                    overlap_chars
                )

                for part in parts:
                    add_chunk(
                        prefix + part,
                        "text",
                        section
                    )

                text_buffer.clear()

            # ----------------------------------
            # Process page blocks
            # ----------------------------------

            for kind, value in parse_blocks(
                page.get("text", "")
            ):

                if kind == "heading":

                    flush_text()

                    match = re.match(
                        r"^(#+)\s+(.*)",
                        value
                    )

                    level = len(
                        match.group(1)
                    )

                    heading = (
                        match.group(2).strip()
                    )

                    while (
                        section_stack
                        and section_stack[-1][0] >= level
                    ):
                        section_stack.pop()

                    section_stack.append(
                        (level, heading)
                    )

                elif kind == "table":

                    flush_text()

                    table_id = (
                        table_ids[table_position]
                        if table_position < len(table_ids)
                        else None
                    )

                    table_position += 1

                    for (
                        content,
                        row_start,
                        row_end
                    ) in split_table(
                        value,
                        section_name(),
                        max_chars
                    ):

                        add_chunk(
                            content,
                            "table",
                            section_name(),
                            table_id,
                            row_start,
                            row_end
                        )

                else:
                    text_buffer.append(value)

            flush_text()

            # Preserve cover pages with headings only.
            if not page_chunks and section_stack:

                add_chunk(
                    " / ".join(
                        section[1]
                        for section in section_stack
                    ),
                    "text",
                    section_name()
                )

            # ----------------------------------
            # Structured-table fallback
            # ----------------------------------

            # Some Docling tables may exist in the
            # structured JSON but not in Markdown.
            # Include those without duplicating tables
            # already processed above.

            for table_id in table_ids[table_position:]:

                table = table_lookup.get(
                    table_id
                )

                if not table:
                    continue

                lines = [
                    "| "
                    + " | ".join(
                        str(cell)
                        for cell in row
                    )
                    + " |"
                    for row in table.get("rows", [])
                ]

                for (
                    content,
                    row_start,
                    row_end
                ) in split_table(
                    lines,
                    "",
                    max_chars
                ):

                    add_chunk(
                        content,
                        "table",
                        None,
                        table_id,
                        row_start,
                        row_end
                    )

            # ----------------------------------
            # Deterministic chunk IDs
            # ----------------------------------

            for index, item in enumerate(
                page_chunks
            ):

                item["chunk_index"] = index

                source_key = (
                    f"{bid_id}|"
                    f"{filename}|"
                    f"{page_number}|"
                    f"{index}"
                )

                item["id"] = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        source_key
                    )
                )

                chunks.append(item)

    return chunks


# --------------------------------------------------
# 6. TEST / PREVIEW COMMAND
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Inspect RFP document chunking"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to ingested JSON"
    )

    parser.add_argument(
        "--output",
        help="Optional path to save chunks as JSON"
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=3
    )

    args = parser.parse_args()

    data = json.loads(
        Path(args.input).read_text(
            encoding="utf-8"
        )
    )

    chunks = build_chunks(data)

    counts = Counter(
        chunk["chunk_type"]
        for chunk in chunks
    )

    print(
        f"Bid: {data['bid_id']} "
        f"| Total chunks: {len(chunks)} "
        f"| Text: {counts['text']} "
        f"| Table chunks: {counts['table']}"
    )

    print(
        "Longest chunk:",
        max(
            (
                len(chunk["text"])
                for chunk in chunks
            ),
            default=0
        ),
        "characters"
    )

    for chunk in chunks[:args.preview]:

        print(
            f"\n[{chunk['chunk_type']}] "
            f"{chunk['file_name']} "
            f"| Page {chunk['page_number']}"
        )

        print(
            "Section:",
            chunk["section"]
        )

        print(chunk["text"][:500])

    if args.output:

        target = Path(args.output)

        target.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        target.write_text(
            json.dumps(
                chunks,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print(f"\nSaved: {target}")


if __name__ == "__main__":
    main()
