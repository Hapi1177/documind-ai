"""Semantic retrieval: embed the question, then search ChromaDB.

The question is embedded with the same model that embedded the stored chunks.
Results are returned ranked by distance, without any similarity cut-off: a
threshold would be a guess until it has been tuned on the evaluation set.
"""

from __future__ import annotations

from src.ollama_client import OllamaService
from src.vector_store import RetrievedChunk, VectorStore


def retrieve(
    question: str,
    ollama: OllamaService,
    store: VectorStore,
    embedding_model: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Return up to ``top_k`` chunks most similar to the question.

    Returns an empty list if the question is blank or nothing is indexed.
    In the empty-database case Ollama is not called at all.

    Raises:
        OllamaError, VectorStoreError: with user-facing messages.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    question = question.strip()
    if not question or store.count() == 0:
        return []

    query_vector = ollama.embed([question], embedding_model)[0]
    return store.query(query_vector, top_k)
