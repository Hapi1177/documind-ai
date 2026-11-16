"""Tests for the ingestion pipeline (PDF -> chunks -> embeddings -> ChromaDB)."""

from __future__ import annotations

import dataclasses

import ollama
import pytest

from src.chunking import make_chunk_id
from src.errors import ModelNotFoundError, OllamaUnavailableError, ScannedPDFError
from src.ingestion import (
    STATUS_INDEXED,
    STATUS_REINDEXED,
    STATUS_UNCHANGED,
    ingest_pdf,
)
from src.ollama_client import OllamaService
from src.pdf_loader import compute_file_hash
from src.vector_store import VectorStore
from tests.conftest import make_pdf


@pytest.fixture
def service(fake_client, settings) -> OllamaService:
    return OllamaService(settings.ollama_base_url, client=fake_client)


@pytest.fixture
def store(settings) -> VectorStore:
    return VectorStore(settings.chroma_path, settings.collection_name, settings.embedding_model)


def stored_ids(store: VectorStore) -> set[str]:
    return set(store._collection.get(include=[])["ids"])


def test_ingest_indexes_all_pages_with_stable_ids(sample_pdf, settings, service, store):
    result = ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)

    file_hash = compute_file_hash(sample_pdf)
    assert result.status == STATUS_INDEXED
    assert result.file_hash == file_hash
    assert result.page_count == 3
    assert result.chunk_count == store.count() == 3
    assert stored_ids(store) == {make_chunk_id(file_hash, page, 0) for page in (1, 2, 3)}

    document = store.list_documents()[0]
    assert document.filename == "handbook.pdf"
    assert document.chunk_size == settings.chunk_size


def test_ingest_uses_the_configured_embedding_model(sample_pdf, settings, service, store, fake_client):
    ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)

    assert {call["model"] for call in fake_client.embed_calls} == {settings.embedding_model}


def test_same_file_twice_is_skipped_and_not_duplicated(sample_pdf, settings, service, store, fake_client):
    ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)
    ids_before = stored_ids(store)
    calls_before = len(fake_client.embed_calls)

    second = ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)

    assert second.status == STATUS_UNCHANGED
    assert store.count() == 3
    assert stored_ids(store) == ids_before
    assert len(fake_client.embed_calls) == calls_before  # nothing re-embedded


def test_forced_reindex_is_idempotent(sample_pdf, settings, service, store):
    ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)
    ids_before = stored_ids(store)

    second = ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store, force=True)

    assert second.status == STATUS_REINDEXED
    assert store.count() == 3
    assert stored_ids(store) == ids_before
    assert len(store.list_documents()) == 1


def test_same_filename_different_bytes_is_a_separate_document(settings, service, store):
    first = ingest_pdf(make_pdf(["Policy version one."]), "policy.pdf", settings, service, store)
    second = ingest_pdf(make_pdf(["Policy version two."]), "policy.pdf", settings, service, store)

    assert first.file_hash != second.file_hash
    assert second.status == STATUS_INDEXED
    assert len(store.list_documents()) == 2
    assert store.count() == 2


def test_changed_chunk_settings_reindex_without_leftover_chunks(settings, service, store):
    long_page = " ".join(f"word{i}" for i in range(300))  # ~2,000 characters
    data = make_pdf([long_page])
    small_chunks = dataclasses.replace(settings, chunk_size=200, chunk_overlap=20)
    large_chunks = dataclasses.replace(settings, chunk_size=1000, chunk_overlap=20)

    first = ingest_pdf(data, "long.pdf", small_chunks, service, store)
    second = ingest_pdf(data, "long.pdf", large_chunks, service, store)

    assert second.status == STATUS_REINDEXED
    assert second.chunk_count < first.chunk_count
    assert store.count() == second.chunk_count  # old, smaller chunks are gone
    assert store.list_documents()[0].chunk_size == 1000


def test_pages_without_text_are_reported_as_warnings(settings, service, store):
    data = make_pdf(["Real text on page one.", "", "Real text on page three."])

    result = ingest_pdf(data, "mixed.pdf", settings, service, store)

    assert result.chunk_count == 2
    assert len(result.warnings) == 1
    pages = {hit.page_number for hit in store.query(service.embed(["text"], "bge-m3")[0], 5)}
    assert pages == {1, 3}


def test_progress_callback_runs_from_start_to_finish(sample_pdf, settings, service, store):
    events: list[tuple[str, float]] = []

    ingest_pdf(
        sample_pdf, "handbook.pdf", settings, service, store,
        progress=lambda message, fraction: events.append((message, fraction)),
    )

    fractions = [fraction for _, fraction in events]
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)


def test_scanned_pdf_stores_nothing(settings, service, store, fake_client):
    with pytest.raises(ScannedPDFError):
        ingest_pdf(make_pdf([""]), "scan.pdf", settings, service, store)

    assert store.count() == 0
    assert fake_client.embed_calls == []


def test_ollama_down_stores_nothing(sample_pdf, settings, service, store, fake_client):
    fake_client.embed_error = ConnectionError("refused")

    with pytest.raises(OllamaUnavailableError):
        ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)

    assert store.count() == 0


def test_failed_reindex_keeps_the_existing_chunks(sample_pdf, settings, service, store, fake_client):
    ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)
    fake_client.embed_error = ollama.ResponseError("model not found", 404)

    with pytest.raises(ModelNotFoundError):
        ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store, force=True)

    assert store.count() == 3  # old index untouched
