"""Tests for deterministic, page-aware chunking and stable chunk IDs."""

from __future__ import annotations

import pytest

from src.chunking import chunk_pages, make_chunk_id, split_text
from src.pdf_loader import PageText

WORDS = " ".join(f"word{i}" for i in range(400))  # ~2,700 characters


def test_empty_and_whitespace_text_gives_no_chunks():
    assert split_text("", 100, 10) == []
    assert split_text("   \n\t  ", 100, 10) == []


def test_short_text_is_a_single_chunk():
    assert split_text("A short sentence.", 100, 10) == ["A short sentence."]


def test_chunks_never_exceed_chunk_size():
    chunks = split_text(WORDS, 120, 30)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 120 for chunk in chunks)


def test_chunking_is_deterministic():
    assert split_text(WORDS, 120, 30) == split_text(WORDS, 120, 30)


def test_words_are_not_split_when_whitespace_is_available():
    vocabulary = set(WORDS.split())

    for chunk in split_text(WORDS, 120, 30):
        assert set(chunk.split()) <= vocabulary


def test_no_text_is_lost():
    seen: set[str] = set()
    for chunk in split_text(WORDS, 120, 30):
        seen.update(chunk.split())

    assert seen == set(WORDS.split())


def test_consecutive_chunks_overlap():
    chunks = split_text(WORDS, 120, 30)

    for previous, current in zip(chunks, chunks[1:]):
        assert previous.split()[-1] in current.split()


def test_zero_overlap_produces_disjoint_chunks():
    chunks = split_text(WORDS, 120, 0)

    assert " ".join(chunks).split() == WORDS.split()


def test_very_long_word_is_hard_cut_and_still_terminates():
    text = "x" * 1000  # no whitespace at all

    chunks = split_text(text, 100, 20)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunks[0] == "x" * 100
    assert len(chunks) < 50  # progressed, did not loop forever


@pytest.mark.parametrize("size,overlap", [(10, 9), (50, 49), (7, 0), (100, 99)])
def test_forward_progress_with_extreme_settings(size, overlap):
    text = ("ab cd " * 300).strip()

    chunks = split_text(text, size, overlap)

    assert chunks
    assert len(chunks) <= len(text)  # at least one character of progress each
    assert all(len(chunk) <= size for chunk in chunks)
    assert text.endswith(chunks[-1])


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)])
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_text("some text", size, overlap)


def test_chunk_pages_keeps_page_numbers_and_restarts_index_per_page():
    pages = [
        PageText(page_number=1, text=WORDS),
        PageText(page_number=4, text="Only one short chunk here."),
    ]

    chunks = chunk_pages(pages, 500, 50)

    page_one = [chunk for chunk in chunks if chunk.page_number == 1]
    page_four = [chunk for chunk in chunks if chunk.page_number == 4]
    assert len(page_one) > 1
    assert [chunk.chunk_index for chunk in page_one] == list(range(len(page_one)))
    assert len(page_four) == 1
    assert page_four[0].chunk_index == 0
    assert {chunk.page_number for chunk in chunks} == {1, 4}


def test_chunks_never_span_pages():
    pages = [
        PageText(page_number=1, text="apple " * 50),
        PageText(page_number=2, text="banana " * 50),
    ]

    for chunk in chunk_pages(pages, 120, 30):
        if chunk.page_number == 1:
            assert "banana" not in chunk.text
        else:
            assert "apple" not in chunk.text


def test_chunk_ids_are_stable_and_unique():
    file_hash = "a" * 64

    assert make_chunk_id(file_hash, 3, 7) == make_chunk_id(file_hash, 3, 7)
    assert make_chunk_id(file_hash, 3, 7).startswith(file_hash)

    ids = {
        make_chunk_id(h, page, index)
        for h in ("a" * 64, "b" * 64)
        for page in (1, 2, 11)
        for index in (0, 1, 10)
    }
    assert len(ids) == 2 * 3 * 3
