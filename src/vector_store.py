"""Persistent local vector storage on top of ChromaDB.

One collection holds every chunk: its text, its embedding and the metadata
needed for citations (filename, page, file hash). Embeddings are always passed
in explicitly, so ChromaDB never downloads or runs an embedding model itself.

The collection remembers which embedding model produced its vectors. Opening
it with a different model is refused, because distances between vectors from
different models are meaningless.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from src.errors import EmbeddingModelMismatchError, VectorStoreError

UPSERT_BATCH_SIZE = 500
EMBEDDING_MODEL_KEY = "embedding_model"


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit, built purely from stored data."""

    id: str
    text: str
    filename: str
    page_number: int
    chunk_index: int
    file_hash: str
    # Cosine distance from the query (0 = same direction). Lower is closer.
    # This is a ranking signal, not a probability or confidence score.
    distance: float


@dataclass(frozen=True)
class DocumentInfo:
    """Summary of one indexed PDF."""

    file_hash: str
    filename: str
    page_count: int
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    indexed_at: str


@contextmanager
def _chroma_errors(action: str) -> Iterator[None]:
    """Turn any ChromaDB failure into a ``VectorStoreError`` with context."""
    try:
        yield
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"The vector database could not {action}: {exc}") from exc


class VectorStore:
    """A single persistent ChromaDB collection."""

    def __init__(self, path: Path, collection_name: str, embedding_model: str) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        with _chroma_errors("be opened"):
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._open_collection()

    def _open_collection(self):
        """Open the collection, or create it tagged with the embedding model."""
        try:
            # embedding_function=None: we always supply our own vectors.
            collection = self._client.get_collection(
                self.collection_name, embedding_function=None
            )
        except NotFoundError:
            return self._client.create_collection(
                self.collection_name,
                embedding_function=None,
                metadata={EMBEDDING_MODEL_KEY: self.embedding_model},
                configuration={"hnsw": {"space": "cosine"}},
            )

        stored_model = (collection.metadata or {}).get(EMBEDDING_MODEL_KEY)
        if stored_model != self.embedding_model:
            raise EmbeddingModelMismatchError(
                f"The collection '{self.collection_name}' was built with the "
                f"embedding model '{stored_model}', but EMBEDDING_MODEL is now "
                f"'{self.embedding_model}'. Vectors from different models cannot "
                "be mixed. Set a new COLLECTION_NAME in .env (or delete the "
                f"folder '{self.path}') and index your documents again."
            )
        return collection

    # --- Writing ---------------------------------------------------------------

    def upsert_chunks(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Insert chunks, overwriting any that already exist with the same ID."""
        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise VectorStoreError(
                "Internal error: ids, texts, embeddings and metadata must have "
                "the same length."
            )
        with _chroma_errors("store the chunks"):
            for start in range(0, len(ids), UPSERT_BATCH_SIZE):
                end = start + UPSERT_BATCH_SIZE
                self._collection.upsert(
                    ids=list(ids[start:end]),
                    documents=list(texts[start:end]),
                    embeddings=[list(vector) for vector in embeddings[start:end]],
                    metadatas=list(metadatas[start:end]),
                )

    def delete_document(self, file_hash: str) -> int:
        """Delete every chunk of one document. Returns how many were removed."""
        with _chroma_errors("delete the document"):
            existing = self._collection.get(where={"file_hash": file_hash}, include=[])
            chunk_ids = existing["ids"]
            if chunk_ids:
                self._collection.delete(ids=chunk_ids)
            return len(chunk_ids)

    # --- Reading ---------------------------------------------------------------

    def count(self) -> int:
        """Total number of stored chunks."""
        with _chroma_errors("count the chunks"):
            return self._collection.count()

    def query(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks closest to the query vector.

        An empty collection gives an empty list.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        total = self.count()
        if total == 0:
            return []

        with _chroma_errors("run the search"):
            result = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )

        hits: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=text,
                    filename=str(metadata["filename"]),
                    page_number=int(metadata["page_number"]),
                    chunk_index=int(metadata["chunk_index"]),
                    file_hash=str(metadata["file_hash"]),
                    distance=float(distance),
                )
            )
        return hits

    def list_documents(self) -> list[DocumentInfo]:
        """Summarise the indexed documents, sorted by filename."""
        with _chroma_errors("list the documents"):
            result = self._collection.get(include=["metadatas"])

        chunk_counts: dict[str, int] = {}
        first_metadata: dict[str, dict[str, Any]] = {}
        for metadata in result["metadatas"] or []:
            file_hash = str(metadata["file_hash"])
            chunk_counts[file_hash] = chunk_counts.get(file_hash, 0) + 1
            first_metadata.setdefault(file_hash, metadata)

        documents = [
            DocumentInfo(
                file_hash=file_hash,
                filename=str(metadata["filename"]),
                page_count=int(metadata["page_count"]),
                chunk_count=chunk_counts[file_hash],
                chunk_size=int(metadata["chunk_size"]),
                chunk_overlap=int(metadata["chunk_overlap"]),
                indexed_at=str(metadata["indexed_at"]),
            )
            for file_hash, metadata in first_metadata.items()
        ]
        return sorted(documents, key=lambda doc: (doc.filename.lower(), doc.file_hash))

    def get_document(self, file_hash: str) -> DocumentInfo | None:
        """Return the summary of one document, or None if it is not indexed."""
        for document in self.list_documents():
            if document.file_hash == file_hash:
                return document
        return None
