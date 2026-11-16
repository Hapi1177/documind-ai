"""Deterministic, page-aware, character-based chunking.

Design choices (kept simple on purpose):

* Chunk size and overlap are measured in characters. No tokenizer is needed,
  the behaviour is easy to test, and the same input always gives the same
  output.
* A chunk never spans two pages. That costs a little context at page breaks
  but means every chunk has exactly one page number, so citations are exact.
* Cuts prefer whitespace so words are not split in half. A hard cut is used
  only when a single "word" is longer than the chunk itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.pdf_loader import PageText


@dataclass(frozen=True)
class Chunk:
    """A piece of text taken from a single page."""

    text: str
    page_number: int  # 1-based page the text came from
    chunk_index: int  # 0-based position of the chunk within that page


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Reject settings that would make chunking meaningless or endless."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must not be negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})."
        )


def _find_end(text: str, start: int, chunk_size: int, chunk_overlap: int) -> int:
    """Pick where the chunk starting at ``start`` should end (exclusive).

    Prefers the last whitespace inside the window. The whitespace cut is only
    accepted if the *next* chunk would still start after ``start``; otherwise
    the hard cut at ``start + chunk_size`` is used. Because
    ``chunk_overlap < chunk_size``, the hard cut always moves forward, which
    is what guarantees the loop in ``split_text`` terminates.
    """
    hard_end = start + chunk_size
    if hard_end >= len(text):
        return len(text)
    if text[hard_end].isspace():
        return hard_end  # the window already ends exactly at a word boundary

    for position in range(hard_end - 1, start, -1):
        if text[position].isspace():
            if position - chunk_overlap > start:
                return position
            break
    return hard_end


def _find_next_start(text: str, start: int, end: int, chunk_overlap: int) -> int:
    """Pick where the next chunk starts, stepping back by the overlap.

    If stepping back lands in the middle of a word, the start is moved forward
    to the beginning of the next word so the overlap begins on a whole word.
    Moving forward can only increase progress, never reduce it.
    """
    next_start = end - chunk_overlap
    if next_start <= start:  # defensive: _find_end already prevents this
        return end
    if chunk_overlap == 0 or text[next_start - 1].isspace():
        return next_start

    position = next_start
    while position < end and not text[position].isspace():
        position += 1
    while position < end and text[position].isspace():
        position += 1
    return position if position < end else next_start


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks of at most ``chunk_size`` characters.

    Returns an empty list for empty or whitespace-only text.

    Raises:
        ValueError: If the size/overlap combination is invalid.
    """
    validate_chunk_params(chunk_size, chunk_overlap)
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = _find_end(text, start, chunk_size, chunk_overlap)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = _find_next_start(text, start, end, chunk_overlap)
    return chunks


def chunk_pages(
    pages: Iterable[PageText], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk every page separately, keeping the page number on each chunk."""
    validate_chunk_params(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for page in pages:
        for index, piece in enumerate(split_text(page.text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(text=piece, page_number=page.page_number, chunk_index=index)
            )
    return chunks


def make_chunk_id(file_hash: str, page_number: int, chunk_index: int) -> str:
    """Build a stable chunk ID from file hash, page number and chunk index.

    The same file always produces the same IDs, so re-indexing overwrites
    existing vectors (upsert) instead of creating duplicates.
    """
    return f"{file_hash}:p{page_number:05d}:c{chunk_index:04d}"
