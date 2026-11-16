"""Tests for semantic retrieval end to end (with fake embeddings)."""

from __future__ import annotations

import pytest

from src.errors import OllamaUnavailableError
from src.ingestion import ingest_pdf
from src.ollama_client import OllamaService
from src.retrieval import retrieve
from src.vector_store import VectorStore
from tests.conftest import make_pdf


@pytest.fixture
def service(fake_client, settings) -> OllamaService:
    return OllamaService(settings.ollama_base_url, client=fake_client)


@pytest.fixture
def store(settings) -> VectorStore:
    return VectorStore(settings.chroma_path, settings.collection_name, settings.embedding_model)


def test_empty_collection_returns_nothing_without_calling_ollama(settings, service, store, fake_client):
    assert retrieve("anything?", service, store, settings.embedding_model, 5) == []
    assert fake_client.embed_calls == []


def test_blank_question_returns_nothing(sample_pdf, settings, service, store):
    ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)

    assert retrieve("   ", service, store, settings.embedding_model, 5) == []


@pytest.mark.parametrize(
    "question,expected_page",
    [
        ("How many days of annual leave do employees receive?", 1),
        ("When must travel expense claims be submitted?", 2),
        ("How many days per week may staff work remotely?", 3),
    ],
)
def test_most_relevant_page_is_ranked_first(sample_pdf, settings, service, store, question, expected_page):
    ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)

    hits = retrieve(question, service, store, settings.embedding_model, 3)

    assert hits[0].page_number == expected_page
    assert hits[0].filename == "handbook.pdf"
    assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)


def test_results_carry_real_source_metadata(sample_pdf, settings, service, store):
    result = ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)

    hits = retrieve("annual leave", service, store, settings.embedding_model, 3)

    for hit in hits:
        assert hit.file_hash == result.file_hash
        assert hit.id.startswith(result.file_hash)
        assert 1 <= hit.page_number <= 3
        assert hit.text


def test_top_k_limits_the_number_of_results(sample_pdf, settings, service, store):
    ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)

    assert len(retrieve("leave", service, store, settings.embedding_model, 1)) == 1
    assert len(retrieve("leave", service, store, settings.embedding_model, 2)) == 2
    assert len(retrieve("leave", service, store, settings.embedding_model, 50)) == 3


def test_query_uses_the_same_embedding_model_as_ingestion(sample_pdf, settings, service, store, fake_client):
    ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)
    fake_client.embed_calls.clear()

    retrieve("annual leave", service, store, settings.embedding_model, 3)

    assert [call["model"] for call in fake_client.embed_calls] == [settings.embedding_model]
    assert fake_client.embed_calls[0]["input"] == ["annual leave"]


def test_retrieval_spans_multiple_documents(settings, service, store):
    ingest_pdf(make_pdf(["Cats are small domestic felines."]), "cats.pdf", settings, service, store)
    ingest_pdf(make_pdf(["Volcanoes erupt molten lava."]), "volcanoes.pdf", settings, service, store)

    assert retrieve("molten lava volcanoes", service, store, settings.embedding_model, 1)[0].filename == "volcanoes.pdf"
    assert retrieve("domestic felines cats", service, store, settings.embedding_model, 1)[0].filename == "cats.pdf"


def test_deleted_document_is_no_longer_retrieved(settings, service, store):
    cats = ingest_pdf(make_pdf(["Cats are small domestic felines."]), "cats.pdf", settings, service, store)
    ingest_pdf(make_pdf(["Volcanoes erupt molten lava."]), "volcanoes.pdf", settings, service, store)

    store.delete_document(cats.file_hash)

    hits = retrieve("domestic felines cats", service, store, settings.embedding_model, 5)
    assert [hit.filename for hit in hits] == ["volcanoes.pdf"]


def test_ollama_failure_is_reported(sample_pdf, settings, service, store, fake_client):
    ingest_pdf(sample_pdf, "handbook.pdf", settings, service, store)
    fake_client.embed_error = ConnectionError("refused")

    with pytest.raises(OllamaUnavailableError):
        retrieve("annual leave", service, store, settings.embedding_model, 3)


def test_invalid_top_k_is_rejected(settings, service, store):
    with pytest.raises(ValueError):
        retrieve("question", service, store, settings.embedding_model, 0)
