"""Tests for PDF validation and text extraction."""

from __future__ import annotations

import pytest

from src.errors import (
    EncryptedPDFError,
    InvalidPDFError,
    PDFTooLargeError,
    ScannedPDFError,
)
from src.pdf_loader import compute_file_hash, load_pdf, normalize_text
from tests.conftest import MAX_BYTES, make_encrypted_pdf, make_pdf


def test_extracts_text_page_by_page_with_one_based_numbers():
    data = make_pdf(["Alpha page content.", "Beta page content.", "Gamma page content."])

    loaded = load_pdf(data, "three.pdf", MAX_BYTES)

    assert loaded.filename == "three.pdf"
    assert loaded.page_count == 3
    assert [page.page_number for page in loaded.pages] == [1, 2, 3]
    assert "Alpha" in loaded.pages[0].text
    assert "Beta" in loaded.pages[1].text
    assert "Gamma" in loaded.pages[2].text
    assert loaded.empty_pages == []
    assert loaded.warnings == []


def test_blank_pages_keep_original_page_numbers_and_are_reported():
    data = make_pdf(["First page text.", "", "Third page text."])

    loaded = load_pdf(data, "gap.pdf", MAX_BYTES)

    assert loaded.page_count == 3
    assert [page.page_number for page in loaded.pages] == [1, 3]
    assert loaded.empty_pages == [2]
    assert len(loaded.warnings) == 1
    assert "2" in loaded.warnings[0]


def test_file_hash_is_sha256_of_bytes_and_ignores_filename():
    data = make_pdf(["Same content."])

    first = load_pdf(data, "a.pdf", MAX_BYTES)
    second = load_pdf(data, "renamed.pdf", MAX_BYTES)

    assert first.file_hash == second.file_hash == compute_file_hash(data)
    assert len(first.file_hash) == 64


def test_different_bytes_give_different_hash():
    assert compute_file_hash(make_pdf(["Version one."])) != compute_file_hash(
        make_pdf(["Version two."])
    )


def test_empty_file_is_rejected():
    with pytest.raises(InvalidPDFError, match="empty"):
        load_pdf(b"", "empty.pdf", MAX_BYTES)


def test_non_pdf_bytes_are_rejected():
    with pytest.raises(InvalidPDFError, match="does not look like a PDF"):
        load_pdf(b"This is just a text file, not a PDF.", "notes.pdf", MAX_BYTES)


def test_corrupted_pdf_is_rejected():
    # Valid header followed by garbage: passes the cheap check, fails parsing.
    with pytest.raises(InvalidPDFError):
        load_pdf(b"%PDF-1.7\n" + b"\x00garbage\xff" * 50, "broken.pdf", MAX_BYTES)


def test_oversized_pdf_is_rejected_before_parsing():
    data = make_pdf(["Small but over the tiny limit."])

    with pytest.raises(PDFTooLargeError, match="limit"):
        load_pdf(data, "big.pdf", max_size_bytes=100)


def test_encrypted_pdf_is_rejected():
    with pytest.raises(EncryptedPDFError, match="password"):
        load_pdf(make_encrypted_pdf(), "locked.pdf", MAX_BYTES)


def test_scanned_pdf_without_text_is_rejected():
    with pytest.raises(ScannedPDFError, match="no extractable text"):
        load_pdf(make_pdf(["", ""]), "scan.pdf", MAX_BYTES)


def test_normalize_text_is_deterministic_and_tidy():
    raw = "Line  one\t with   spaces \r\n\r\n\r\n\r\nLine two\x00 "

    assert normalize_text(raw) == "Line one with spaces\n\nLine two"
    assert normalize_text(raw) == normalize_text(raw)
