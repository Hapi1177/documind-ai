"""Manual end-to-end check of PDF extraction, embeddings and retrieval.

Unlike the pytest suite (which fakes Ollama), this script talks to the real
local Ollama service, so it needs Ollama running and the embedding model
pulled. It does not need the chat model.

Usage (from the project root, with the environment activated):

    python scripts/verify_retrieval.py
    python scripts/verify_retrieval.py path\\to\\file.pdf "Your question?"

Without arguments a small synthetic PDF is generated. The index is written to
a temporary folder, so your real data/chroma index is never touched.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf  # noqa: E402

from src.config import Settings, load_settings  # noqa: E402
from src.errors import DocuMindError  # noqa: E402
from src.ingestion import ingest_pdf  # noqa: E402
from src.ollama_client import OllamaService  # noqa: E402
from src.retrieval import retrieve  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

SAMPLE_PAGES = [
    "Annual leave policy. Employees receive 18 days of annual leave per "
    "calendar year. Unused leave may be carried forward for three months.",
    "Expense claims. Travel expense claims must be submitted within 30 days "
    "together with original receipts.",
    "Remote work. Staff may work remotely up to two days per week with "
    "approval from their manager.",
]
SAMPLE_QUESTION = "How long do I have to submit a travel expense claim?"


def build_sample_pdf() -> bytes:
    """Create a three-page synthetic PDF in memory."""
    document = pymupdf.open()
    for text in SAMPLE_PAGES:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


def run_check(
    service: OllamaService,
    settings: Settings,
    data: bytes,
    filename: str,
    question: str,
) -> None:
    """Index one PDF, search it, and confirm the index survives a reopen."""
    print(f"[1/5] Checking Ollama at {settings.ollama_base_url} ...")
    models = service.list_models()
    print(f"      OK - {len(models)} model(s) installed.")
    if not service.has_model(settings.embedding_model):
        print(
            f"      WARNING: '{settings.embedding_model}' is not listed. "
            f"Run: ollama pull {settings.embedding_model}"
        )

    print(f"[2/5] Extracting, chunking and embedding '{filename}' ...")
    store = VectorStore(
        settings.chroma_path, settings.collection_name, settings.embedding_model
    )
    result = ingest_pdf(data, filename, settings, service, store)
    print(
        f"      OK - {result.page_count} page(s), {result.chunk_count} chunk(s), "
        f"status: {result.status}, file hash: {result.file_hash[:12]}..."
    )
    for warning in result.warnings:
        print(f"      NOTE: {warning}")

    dimension = len(service.embed(["dimension probe"], settings.embedding_model)[0])
    print(f"[3/5] Embedding dimension reported by the model: {dimension}")

    print(f"[4/5] Retrieving top {settings.top_k} chunk(s) for: {question!r}")
    hits = retrieve(question, service, store, settings.embedding_model, settings.top_k)
    if not hits:
        raise SystemExit("      FAILED - retrieval returned no results.")
    for rank, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.text.split())[:110]
        print(
            f"      {rank}. {hit.filename} p.{hit.page_number} "
            f"(cosine distance {hit.distance:.4f}): {snippet}"
        )

    print("[5/5] Re-ingesting and reopening the database ...")
    again = ingest_pdf(data, filename, settings, service, store)
    reopened = VectorStore(
        settings.chroma_path, settings.collection_name, settings.embedding_model
    )
    if reopened.count() != result.chunk_count:
        raise SystemExit(
            f"      FAILED - expected {result.chunk_count} chunk(s) after "
            f"re-ingesting, found {reopened.count()}."
        )
    print(
        f"      OK - second ingest status: {again.status}; "
        f"{reopened.count()} chunk(s) stored, no duplicates."
    )
    print("\nAll checks passed.")


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        pdf_path = Path(argv[1])
        if not pdf_path.is_file():
            print(f"File not found: {pdf_path}")
            return 2
        data, filename = pdf_path.read_bytes(), pdf_path.name
        question = argv[2] if len(argv) > 2 else input("Question: ")
    else:
        data, filename, question = build_sample_pdf(), "sample_handbook.pdf", SAMPLE_QUESTION

    try:
        base_settings = load_settings()
        # ignore_cleanup_errors: on Windows ChromaDB may still hold file handles.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            settings = dataclasses.replace(
                base_settings,
                chroma_path=Path(temp_dir) / "chroma",
                collection_name="verify_retrieval",
            )
            service = OllamaService(settings.ollama_base_url)
            run_check(service, settings, data, filename, question)
    except DocuMindError as error:
        print(f"\nFAILED - {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
