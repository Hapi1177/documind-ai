"""Ingestion pipeline: PDF bytes -> pages -> chunks -> embeddings -> ChromaDB.

Re-indexing is idempotent. Chunk IDs are derived from the file hash, page
number and chunk index, so indexing the same bytes again can only overwrite
what is already there. A file with the same name but different bytes has a
different hash and is stored as a separate document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from src.chunking import chunk_pages, make_chunk_id
from src.config import Settings
from src.ollama_client import OllamaService
from src.pdf_loader import load_pdf
from src.vector_store import VectorStore

# Called with (message, fraction_done) so a UI can show progress.
ProgressCallback = Callable[[str, float], None]

STATUS_INDEXED = "indexed"
STATUS_REINDEXED = "reindexed"
STATUS_UNCHANGED = "unchanged"


@dataclass(frozen=True)
class IngestionResult:
    """What happened when one PDF was ingested."""

    filename: str
    file_hash: str
    page_count: int
    chunk_count: int
    status: str  # one of the STATUS_* constants
    warnings: list[str] = field(default_factory=list)


def ingest_pdf(
    data: bytes,
    filename: str,
    settings: Settings,
    ollama: OllamaService,
    store: VectorStore,
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> IngestionResult:
    """Index one PDF into the vector store.

    If the identical file is already indexed with the same chunk settings and
    filename, nothing is re-embedded and the status is ``unchanged`` (unless
    ``force`` is set). Otherwise the document is (re)indexed.

    Raises:
        PDFError, OllamaError, VectorStoreError: with user-facing messages.
    """

    def report(message: str, fraction: float) -> None:
        if progress:
            progress(message, fraction)

    report(f"Reading {filename}...", 0.0)
    loaded = load_pdf(data, filename, settings.max_pdf_size_bytes)
    chunks = chunk_pages(loaded.pages, settings.chunk_size, settings.chunk_overlap)

    existing = store.get_document(loaded.file_hash)
    is_unchanged = (
        existing is not None
        and existing.filename == filename
        and existing.chunk_size == settings.chunk_size
        and existing.chunk_overlap == settings.chunk_overlap
        and existing.chunk_count == len(chunks)
    )
    if is_unchanged and not force:
        report(f"{filename} is already indexed.", 1.0)
        return IngestionResult(
            filename=filename,
            file_hash=loaded.file_hash,
            page_count=loaded.page_count,
            chunk_count=len(chunks),
            status=STATUS_UNCHANGED,
            warnings=loaded.warnings,
        )

    report(f"Embedding {len(chunks)} chunk(s) from {filename}...", 0.1)
    embeddings = ollama.embed(
        [chunk.text for chunk in chunks],
        settings.embedding_model,
        on_batch=lambda done, total: report(
            f"Embedded {done}/{total} chunk(s) from {filename}...",
            0.1 + 0.8 * done / total,
        ),
    )

    indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ids = [
        make_chunk_id(loaded.file_hash, chunk.page_number, chunk.chunk_index)
        for chunk in chunks
    ]
    metadatas = [
        {
            "file_hash": loaded.file_hash,
            "filename": filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "page_count": loaded.page_count,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "indexed_at": indexed_at,
        }
        for chunk in chunks
    ]

    report(f"Saving {filename} to the vector database...", 0.95)
    if existing is not None:
        # Embedding already succeeded, so it is safe to drop the old chunks.
        # This removes leftovers when chunk settings changed and the new run
        # produces fewer chunks than the old one.
        store.delete_document(loaded.file_hash)
    store.upsert_chunks(ids, [chunk.text for chunk in chunks], embeddings, metadatas)

    report(f"Indexed {filename}.", 1.0)
    return IngestionResult(
        filename=filename,
        file_hash=loaded.file_hash,
        page_count=loaded.page_count,
        chunk_count=len(chunks),
        status=STATUS_REINDEXED if existing is not None else STATUS_INDEXED,
        warnings=loaded.warnings,
    )
