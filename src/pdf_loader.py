"""PDF validation and page-by-page text extraction with PyMuPDF.

The loader works on raw bytes (what Streamlit's uploader provides), so nothing
has to be written to disk. Page numbers are 1-based, exactly as a reader sees
them, because they are later shown to the user as citations.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import pymupdf

from src.errors import (
    EncryptedPDFError,
    InvalidPDFError,
    PDFTooLargeError,
    ScannedPDFError,
)

PDF_MAGIC = b"%PDF-"
# The PDF spec allows a little junk before the header; readers accept ~1 KB.
PDF_MAGIC_SEARCH_BYTES = 1024


@dataclass(frozen=True)
class PageText:
    """Extracted text of one page."""

    page_number: int  # 1-based
    text: str


@dataclass(frozen=True)
class LoadedPDF:
    """Result of loading one PDF.

    ``pages`` only holds pages that contain text. Pages without text and pages
    that failed to extract are reported separately so the caller can tell the
    user instead of silently dropping them.
    """

    filename: str
    file_hash: str
    page_count: int
    pages: list[PageText]
    empty_pages: list[int] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        """Human-readable notes about pages that produced no text."""
        notes: list[str] = []
        if self.empty_pages:
            notes.append(
                f"{len(self.empty_pages)} page(s) had no extractable text "
                f"(possibly images or scans): {_format_pages(self.empty_pages)}."
            )
        if self.failed_pages:
            notes.append(
                f"{len(self.failed_pages)} page(s) could not be read: "
                f"{_format_pages(self.failed_pages)}."
            )
        return notes


def _format_pages(pages: list[int], limit: int = 10) -> str:
    """Format page numbers compactly, e.g. ``1, 2, 3 ... (+7 more)``."""
    shown = ", ".join(str(p) for p in pages[:limit])
    extra = len(pages) - limit
    return f"{shown} ... (+{extra} more)" if extra > 0 else shown


def compute_file_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of the file bytes.

    The hash, not the filename, identifies a document: the same bytes always
    produce the same hash, and an edited file with the same name produces a
    different one.
    """
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    """Tidy extracted text without changing its wording.

    Deterministic by design: the same PDF must always yield the same chunks
    so that chunk IDs stay stable across re-indexing.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r" ?\n ?", "\n", text)  # trim spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()


def validate_pdf_bytes(data: bytes, filename: str, max_size_bytes: int) -> None:
    """Run cheap checks before handing the bytes to the PDF parser.

    Raises:
        InvalidPDFError: Empty file or missing ``%PDF-`` header.
        PDFTooLargeError: File exceeds the configured upload limit.
    """
    if not data:
        raise InvalidPDFError(f"'{filename}' is empty (0 bytes).")
    if len(data) > max_size_bytes:
        size_mb = len(data) / (1024 * 1024)
        limit_mb = max_size_bytes / (1024 * 1024)
        raise PDFTooLargeError(
            f"'{filename}' is {size_mb:.1f} MB, which is over the "
            f"{limit_mb:.0f} MB limit (MAX_PDF_SIZE_MB)."
        )
    if PDF_MAGIC not in data[:PDF_MAGIC_SEARCH_BYTES]:
        raise InvalidPDFError(
            f"'{filename}' does not look like a PDF file (missing %PDF header)."
        )


def load_pdf(data: bytes, filename: str, max_size_bytes: int) -> LoadedPDF:
    """Validate a PDF and extract its text page by page.

    Args:
        data: Raw file bytes.
        filename: Original filename, used for messages and citations.
        max_size_bytes: Upload size limit.

    Raises:
        PDFTooLargeError, InvalidPDFError, EncryptedPDFError, ScannedPDFError
    """
    validate_pdf_bytes(data, filename, max_size_bytes)

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated types here
        raise InvalidPDFError(
            f"'{filename}' could not be opened. The file may be corrupted."
        ) from exc

    with document:
        # ``needs_pass`` stays True only when a real password is required;
        # PDFs "encrypted" with an empty user password open transparently.
        if document.needs_pass:
            raise EncryptedPDFError(
                f"'{filename}' is password protected. Remove the password "
                "and upload it again."
            )

        page_count = document.page_count
        if page_count == 0:
            raise InvalidPDFError(f"'{filename}' contains no pages.")

        pages: list[PageText] = []
        empty_pages: list[int] = []
        failed_pages: list[int] = []

        for index in range(page_count):
            page_number = index + 1
            try:
                raw_text = document.load_page(index).get_text("text")
            except Exception:
                failed_pages.append(page_number)
                continue

            text = normalize_text(raw_text)
            if text:
                pages.append(PageText(page_number=page_number, text=text))
            else:
                empty_pages.append(page_number)

    if not pages:
        if failed_pages and not empty_pages:
            raise InvalidPDFError(
                f"None of the {page_count} page(s) in '{filename}' could be read. "
                "The file may be corrupted."
            )
        raise ScannedPDFError(
            f"'{filename}' has no extractable text. It is probably a scanned or "
            "image-only PDF. DocuMind AI needs text-based PDFs; run the file "
            "through OCR first."
        )

    return LoadedPDF(
        filename=filename,
        file_hash=compute_file_hash(data),
        page_count=page_count,
        pages=pages,
        empty_pages=empty_pages,
        failed_pages=failed_pages,
    )
