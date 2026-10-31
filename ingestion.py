
"""Simple PDF and HTML ingestion for arbitrary RFP bid folders.

Docling handles PDF layout, tables, and OCR.
PyMuPDF handles physical-page inspection and native-text fallback.
BeautifulSoup handles HTML parsing and cleaning.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import Counter
from pathlib import Path

import pymupdf as fitz
from bs4 import BeautifulSoup, NavigableString
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm"}


# -------------------------------------------------------
# 1. TEXT CLEANING
# -------------------------------------------------------

def normalize(text: str) -> str:
    """Clean whitespace without destroying tables or paragraphs."""

    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")

    # Repair words split by line-break hyphenation.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    # Normalize horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)

    # Keep at most one blank line between paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# -------------------------------------------------------
# 2. DOCUMENT CLASSIFICATION
# -------------------------------------------------------

def classify_document(path: Path) -> tuple[str, int | None]:
    """Classify files using general document naming patterns."""

    name = re.sub(r"[_\-]+", " ", path.stem.lower())

    match = re.search(
        r"\baddend(?:um|a)\s*[-_#.]?\s*(\d+)\b",
        name
    )

    if match:
        return "addendum", int(match.group(1))

    if "affidavit" in name:
        return "affidavit", None

    if re.search(
        r"\b(spec|specs|specification|specifications)\b",
        name
    ):
        return "specs", None

    if path.suffix.lower() in {".html", ".htm"}:
        return "bid_page", None

    return "rfp", None


# -------------------------------------------------------
# 3. DOCUMENT DATE EXTRACTION
# -------------------------------------------------------

def document_date(text: str) -> str | None:
    """Extract explicit issue/publication dates, not closing dates."""

    labels = (
        r"document date|"
        r"date issued|"
        r"issued on|"
        r"publication date|"
        r"publication|"
        r"published on|"
        r"posting date|"
        r"first advertisement date/issue date"
    )

    date_pattern = (
        r"("
        r"\d{4}-\d{1,2}-\d{1,2}|"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{4}|"
        r"\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{4}|"
        r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
        r")"
    )

    pattern = (
        rf"(?im)^\s*\|?\s*(?:{labels})"
        rf"\s*\|?\s*[:\-]?\s*{date_pattern}"
    )

    match = re.search(pattern, text[:15000])

    return match.group(1).strip() if match else None


# -------------------------------------------------------
# 4. REPEATED HEADERS AND FOOTERS
# -------------------------------------------------------

def remove_repeated_margins(pages: list[dict]) -> None:
    """Remove repeated first/last lines conservatively."""

    if len(pages) < 5:
        return

    counts = Counter()

    for page in pages:
        lines = [
            line.strip()
            for line in page["text"].splitlines()
            if line.strip()
        ]

        counts.update(set(lines[:2] + lines[-2:]))

    threshold = max(4, int(len(pages) * 0.8))

    repeated = {
        line
        for line, count in counts.items()
        if count >= threshold
        and len(line) >= 8
        and not line.startswith("|")
    }

    for page in pages:
        lines = page["text"].splitlines()

        # Remove only repeated lines at the boundaries.
        while lines and lines[0].strip() in repeated:
            lines.pop(0)

        while lines and lines[-1].strip() in repeated:
            lines.pop()

        page["text"] = normalize("\n".join(lines))


# -------------------------------------------------------
# 5. DOCLING CONFIGURATION
# -------------------------------------------------------

def make_converter() -> DocumentConverter:
    """Initialize one Docling converter for all PDF files."""

    options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True
    )

    # Docling can use native PDF text where available.
    # OCR remains available for scanned or image-based content.

    if shutil.which("tesseract"):
        from docling.datamodel.pipeline_options import (
            TesseractCliOcrOptions
        )

        options.ocr_options = TesseractCliOcrOptions(
            lang=["eng"]
        )

    else:
        LOGGER.warning(
            "Tesseract not found. Using Docling's default OCR engine."
        )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )

    return converter


# -------------------------------------------------------
# 6. PDF EXTRACTION
# -------------------------------------------------------

def pdf_file(
    path: Path,
    bid_id: str,
    converter: DocumentConverter,
    page_range: tuple[int, int] | None = None
) -> dict:
    """Extract PDF text and tables while preserving physical pages."""

    kind, addendum = classify_document(path)

    with fitz.open(path) as pdf:

        page_count = pdf.page_count

        start = max(1, page_range[0]) if page_range else 1
        end = min(page_count, page_range[1]) if page_range else page_count

        if start > end:
            return {
                "bid_id": bid_id,
                "file_name": path.name,
                "doc_type": kind,
                "addendum_number": addendum,
                "document_date": None,
                "page_count": page_count,
                "pages": [],
                "tables": [],
                "warnings": ["Requested page range does not exist"]
            }

        LOGGER.info(
            "PDF: %s (pages %s-%s of %s)",
            path.name,
            start,
            end,
            page_count
        )

        converted = converter.convert(
            path,
            page_range=(start, end),
            raises_on_error=False
        )

        if converted.document is None:
            raise RuntimeError(
                f"Docling conversion failed: {converted.status}"
            )

        document = converted.document

        tables = []
        warnings = []

        # ----------------------------------------
        # Extract structured PDF tables
        # ----------------------------------------

        for index, table in enumerate(document.tables, start=1):

            try:
                df = table.export_to_dataframe(
                    doc=document
                ).fillna("")

                headers = [
                    str(column).strip()
                    for column in df.columns
                ]

                matrix = df.astype(str).values.tolist()

                # Keep meaningful column headers.
                # Do not insert generated pandas column labels.
                if headers and not all(
                    header.isdigit()
                    or header.lower().startswith("unnamed:")
                    for header in headers
                ):
                    matrix.insert(0, headers)

                page_numbers = sorted({
                    provenance.page_no
                    for provenance in table.prov
                })

                tables.append({
                    "table_id": f"table_{index}",
                    "page_numbers": page_numbers,
                    "rows": matrix
                })

            except Exception as exc:
                message = (
                    f"Table {index} could not be exported "
                    f"from {path.name}: {exc}"
                )

                LOGGER.warning(message)
                warnings.append(message)

        # ----------------------------------------
        # Extract page-by-page content
        # ----------------------------------------

        pages = []

        for page_no in range(start, end + 1):

            page = pdf[page_no - 1]

            native_text = page.get_text(
                "text",
                sort=True
            )

            image_count = len(page.get_images(full=True))

            # Docling returns layout-aware Markdown
            # including detected table structure.
            text = normalize(
                document.export_to_markdown(
                    page_no=page_no
                )
            )

            # Fallback when Docling returns no usable text.
            if not text and native_text.strip():

                text = normalize(native_text)

                warnings.append(
                    f"Page {page_no}: native-text fallback used"
                )

            if not text:
                warnings.append(
                    f"Page {page_no}: no text recovered"
                )

            # Classification is descriptive.
            # Docling internally decides which content requires OCR.
            if native_text.strip() and image_count:
                page_type = "mixed"

            elif native_text.strip():
                page_type = "native_text"

            elif image_count:
                page_type = "image_only"

            else:
                page_type = "empty_or_vector"

            page_table_ids = [
                item["table_id"]
                for item in tables
                if page_no in item["page_numbers"]
            ]

            pages.append({
                "bid_id": bid_id,
                "file_name": path.name,
                "doc_type": kind,
                "addendum_number": addendum,
                "page_number": page_no,
                "document_date": None,
                "page_type": page_type,
                "text": text,
                "table_ids": page_table_ids
            })

        # ----------------------------------------
        # Text cleaning and date metadata
        # ----------------------------------------

        remove_repeated_margins(pages)

        date = document_date(
            "\n".join(
                page["text"]
                for page in pages[:5]
            )
        )

        for page in pages:
            page["document_date"] = date

        if "success" not in str(converted.status).lower():
            warnings.append(
                f"Docling conversion status: {converted.status}"
            )

        return {
            "bid_id": bid_id,
            "file_name": path.name,
            "doc_type": kind,
            "addendum_number": addendum,
            "document_date": date,
            "page_count": page_count,
            "pages": pages,
            "tables": tables,
            "warnings": warnings
        }


# -------------------------------------------------------
# 7. HTML EXTRACTION
# -------------------------------------------------------

def html_file(path: Path, bid_id: str) -> dict:
    """Extract meaningful HTML bid content and structured tables."""

    kind, addendum = classify_document(path)

    html = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    soup = BeautifulSoup(html, "lxml")

    # Prefer semantic content containers when available.
    root = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    # Remove non-content elements.
    for tag in list(root.select(
        "script, style, noscript, template, "
        "nav, header, footer, aside, "
        "input, button, select, textarea, "
        "svg, iframe, "
        "[hidden], [aria-hidden='true']"
    )):
        if tag.parent is not None:
            tag.decompose()

    # Remove common website overlays and sharing controls.
    # Avoid broad selectors that may remove valid bid content.
    ui_selectors = (
        '[class~="hidden"], '
        '[class*="cookie-banner"], '
        '[class*="cookie-consent"], '
        '[class*="social-share"], '
        '[role="dialog"], '
        '[aria-modal="true"]'
    )

    for tag in list(root.select(ui_selectors)):
        if tag.parent is not None:
            tag.decompose()

    # ----------------------------------------
    # Extract HTML tables
    # ----------------------------------------

    tables = []

    for index, table in enumerate(
        list(root.find_all("table")),
        start=1
    ):

        rows = []

        for tr in table.find_all("tr"):

            row = [
                normalize(cell.get_text(" ", strip=True))
                for cell in tr.find_all(
                    ["th", "td"],
                    recursive=False
                )
            ]

            if row:
                rows.append(row)

        if not rows:
            continue

        table_id = f"table_{index}"

        tables.append({
            "table_id": table_id,
            "page_numbers": [1],
            "rows": rows
        })

        # Preserve tables in searchable page text too.
        markdown = "\n".join(
            " | ".join(row)
            for row in rows
        )

        table.replace_with(
            NavigableString("\n" + markdown + "\n")
        )

    # ----------------------------------------
    # Clean extracted HTML text
    # ----------------------------------------

    text = normalize(
        root.get_text("\n", strip=True)
    )

    lines = text.splitlines()

    # Remove only obvious standalone interface labels.
    ui_labels = {
        "facebook",
        "linkedin",
        "twitter",
        "see more",
        "(opens in a new window)"
    }

    cleaned_lines = [
        line
        for line in lines
        if line.strip().lower() not in ui_labels
    ]

    # Remove clusters of navigation labels only when
    # they appear together as a probable menu.
    navigation_labels = {
        "notice",
        "categories",
        "addendums",
        "documents",
        "mail",
        "email",
        "facebook",
        "linkedin",
        "twitter"
    }

    final_lines = []
    index = 0

    while index < len(cleaned_lines):

        if cleaned_lines[index].strip().lower() in navigation_labels:

            end = index

            while (
                end < len(cleaned_lines)
                and cleaned_lines[end].strip().lower()
                in navigation_labels
            ):
                end += 1

            # Three or more consecutive labels strongly
            # suggest a navigation menu.
            if end - index >= 3:
                index = end
                continue

        final_lines.append(cleaned_lines[index])
        index += 1

    text = normalize("\n".join(final_lines))

    # ----------------------------------------
    # Document metadata
    # ----------------------------------------

    date = document_date(text)

    page = {
        "bid_id": bid_id,
        "file_name": path.name,
        "doc_type": kind,
        "addendum_number": addendum,
        "page_number": 1,
        "document_date": date,
        "page_type": "html",
        "text": text,
        "table_ids": [
            table["table_id"]
            for table in tables
        ]
    }

    return {
        "bid_id": bid_id,
        "file_name": path.name,
        "doc_type": kind,
        "addendum_number": addendum,
        "document_date": date,
        "page_count": 1,
        "pages": [page],
        "tables": tables,
        "warnings": (
            []
            if text
            else ["Empty HTML content after cleaning"]
        )
    }


# -------------------------------------------------------
# 8. BID FOLDER INGESTION
# -------------------------------------------------------

def ingest_bid(
    bid_dir: Path,
    page_range: tuple[int, int] | None = None
) -> dict:
    """Ingest all supported documents in a bid folder."""

    bid_dir = Path(bid_dir)

    if not bid_dir.is_dir():
        raise ValueError(
            f"Bid folder not found: {bid_dir}"
        )

    result = {
        "bid_id": bid_dir.name,
        "documents": [],
        "errors": []
    }

    # Create the converter only when the folder
    # actually contains a PDF.
    files = [
        path
        for path in sorted(bid_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]

    converter = (
        make_converter()
        if any(path.suffix.lower() == ".pdf" for path in files)
        else None
    )

    for path in files:

        try:
            if path.suffix.lower() == ".pdf":

                document = pdf_file(
                    path,
                    bid_dir.name,
                    converter,
                    page_range
                )

            else:
                document = html_file(
                    path,
                    bid_dir.name
                )

            result["documents"].append(document)

        except Exception as exc:

            LOGGER.exception(
                "Failed to process %s",
                path.name
            )

            result["errors"].append({
                "file_name": path.name,
                "error": str(exc)
            })

    return result
