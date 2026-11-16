"""Application service used by the Streamlit UI.

This is the only module the UI talks to. It wires together configuration, the
Ollama client and the vector store, and exposes the handful of actions the
app needs: check status, index a PDF, list and delete documents, and answer a
question. Keeping this out of ``app.py`` means the whole application can be
tested without starting Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from src import rag
from src.config import Settings
from src.errors import EmptyIndexError, OllamaUnavailableError
from src.ingestion import IngestionResult, ProgressCallback, ingest_pdf
from src.ollama_client import OllamaService, normalize_model_name
from src.rag import RagAnswer, Source
from src.vector_store import DocumentInfo, VectorStore


@dataclass(frozen=True)
class SystemStatus:
    """Whether Ollama and the two configured models are ready to use."""

    ollama_available: bool
    chat_model_installed: bool
    embedding_model_installed: bool
    problems: list[str] = field(default_factory=list)  # user-facing fixes

    @property
    def can_index(self) -> bool:
        return self.ollama_available and self.embedding_model_installed

    @property
    def can_chat(self) -> bool:
        return self.can_index and self.chat_model_installed


class DocumentService:
    """High-level actions for the DocuMind AI application."""

    def __init__(
        self,
        settings: Settings,
        ollama: OllamaService | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.ollama = ollama or OllamaService(settings.ollama_base_url)
        self.store = store or VectorStore(
            settings.chroma_path, settings.collection_name, settings.embedding_model
        )

    # --- Status ----------------------------------------------------------------

    def get_status(self) -> SystemStatus:
        """Check the Ollama service and both models. Never raises."""
        settings = self.settings
        try:
            installed = {normalize_model_name(name) for name in self.ollama.list_models()}
        except OllamaUnavailableError as error:
            return SystemStatus(False, False, False, problems=[str(error)])
        chat_ready = normalize_model_name(settings.chat_model) in installed
        embedding_ready = normalize_model_name(settings.embedding_model) in installed

        problems = []
        if not embedding_ready:
            problems.append(
                f"Embedding model '{settings.embedding_model}' is not installed. "
                f"Run: ollama pull {settings.embedding_model}"
            )
        if not chat_ready:
            problems.append(
                f"Chat model '{settings.chat_model}' is not installed. "
                f"Run: ollama pull {settings.chat_model}"
            )
        return SystemStatus(True, chat_ready, embedding_ready, problems=problems)

    # --- Documents -------------------------------------------------------------

    def index_pdf(
        self, data: bytes, filename: str, progress: ProgressCallback | None = None
    ) -> IngestionResult:
        """Extract, chunk, embed and store one PDF."""
        return ingest_pdf(data, filename, self.settings, self.ollama, self.store, progress)

    def list_documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    def delete_document(self, file_hash: str) -> int:
        """Remove one document. Returns the number of chunks deleted."""
        return self.store.delete_document(file_hash)

    def chunk_count(self) -> int:
        return self.store.count()

    # --- Question answering ----------------------------------------------------
    # The UI uses the three steps separately so it can show the answer while it
    # is being generated. ``ask`` runs them in one go.

    def find_sources(self, question: str) -> list[Source]:
        """Retrieve the labelled sources for a question.

        Raises:
            ValueError: If the question is blank.
            EmptyIndexError: If no document has been indexed yet.
        """
        if not question or not question.strip():
            raise ValueError("The question must not be empty.")
        self._require_documents()
        return rag.prepare_sources(question, self.ollama, self.store, self.settings)

    def _require_documents(self) -> None:
        if self.store.count() == 0:
            raise EmptyIndexError(
                "No documents are indexed yet. Upload a PDF and click "
                "'Index documents' first."
            )

    def stream_answer(self, question: str, sources: Sequence[Source]) -> Iterator[str]:
        """Yield the answer text as the local model generates it."""
        return rag.stream_answer(question, sources, self.ollama, self.settings)

    @staticmethod
    def finalize_answer(raw_answer: str, sources: Sequence[Source]) -> RagAnswer:
        """Check citations and package the answer with its real sources."""
        return rag.finalize_answer(raw_answer, sources)

    def ask(self, question: str) -> RagAnswer:
        """Answer a question from the indexed documents (non-streaming)."""
        self._require_documents()
        return rag.answer_question(question, self.ollama, self.store, self.settings)
