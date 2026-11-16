"""Tests for the service layer the Streamlit UI depends on."""

from __future__ import annotations

import pytest

from src.document_service import DocumentService
from src.errors import EmptyIndexError, InvalidPDFError, OllamaUnavailableError
from src.ingestion import STATUS_INDEXED, STATUS_UNCHANGED
from src.ollama_client import OllamaService
from src.rag import ABSTAIN_MESSAGE
from tests.conftest import FakeOllamaClient, make_pdf


def make_service(settings, client: FakeOllamaClient) -> DocumentService:
    return DocumentService(
        settings, ollama=OllamaService(settings.ollama_base_url, client=client)
    )


@pytest.fixture
def service(settings, fake_client) -> DocumentService:
    return make_service(settings, fake_client)


# --- Status -------------------------------------------------------------------


def test_status_ready_when_ollama_and_both_models_are_available(service):
    status = service.get_status()

    assert status.ollama_available
    assert status.can_index and status.can_chat
    assert status.problems == []


def test_status_when_ollama_is_down(settings, fake_client):
    fake_client.list_error = ConnectionError("refused")

    status = make_service(settings, fake_client).get_status()

    assert not status.ollama_available
    assert not status.can_index and not status.can_chat
    assert "Cannot reach Ollama" in status.problems[0]


def test_status_names_the_missing_chat_model(settings):
    status = make_service(settings, FakeOllamaClient(models=("bge-m3:latest",))).get_status()

    assert status.can_index  # embeddings still work
    assert not status.can_chat
    assert status.problems == [
        f"Chat model '{settings.chat_model}' is not installed. "
        f"Run: ollama pull {settings.chat_model}"
    ]


def test_status_names_the_missing_embedding_model(settings):
    status = make_service(settings, FakeOllamaClient(models=("qwen3.5:4b",))).get_status()

    assert not status.can_index and not status.can_chat
    assert "ollama pull bge-m3" in status.problems[0]


# --- Document management ------------------------------------------------------


def test_index_list_and_delete_documents(service, sample_pdf):
    first = service.index_pdf(sample_pdf, "handbook.pdf")
    second = service.index_pdf(make_pdf(["Cats are felines."]), "cats.pdf")

    assert first.status == second.status == STATUS_INDEXED
    assert [doc.filename for doc in service.list_documents()] == ["cats.pdf", "handbook.pdf"]
    assert service.chunk_count() == 4

    assert service.delete_document(first.file_hash) == 3
    assert [doc.filename for doc in service.list_documents()] == ["cats.pdf"]
    assert service.chunk_count() == 1


def test_indexing_the_same_pdf_twice_does_not_duplicate(service, sample_pdf):
    service.index_pdf(sample_pdf, "handbook.pdf")
    again = service.index_pdf(sample_pdf, "handbook.pdf")

    assert again.status == STATUS_UNCHANGED
    assert service.chunk_count() == 3
    assert len(service.list_documents()) == 1


def test_invalid_pdf_error_reaches_the_caller_and_stores_nothing(service):
    with pytest.raises(InvalidPDFError):
        service.index_pdf(b"not a pdf", "notes.pdf")

    assert service.chunk_count() == 0


def test_index_survives_a_restart(settings, fake_client, sample_pdf):
    make_service(settings, fake_client).index_pdf(sample_pdf, "handbook.pdf")

    restarted = make_service(settings, fake_client)

    assert restarted.chunk_count() == 3
    assert [doc.filename for doc in restarted.list_documents()] == ["handbook.pdf"]


# --- Question answering -------------------------------------------------------


def test_ask_returns_answer_with_real_sources(service, sample_pdf, fake_client):
    service.index_pdf(sample_pdf, "handbook.pdf")
    fake_client.answer = "Claims are due within 30 days [S1]."

    result = service.ask("When must travel expense claims be submitted?")

    assert result.answer == "Claims are due within 30 days [S1]."
    assert result.sources[0].filename == "handbook.pdf"
    assert result.sources[0].page_number == 2
    assert result.cited_labels == ["S1"]


def test_streaming_steps_give_the_same_result_as_ask(service, sample_pdf, fake_client):
    service.index_pdf(sample_pdf, "handbook.pdf")
    fake_client.answer = "Staff may work remotely two days per week [S1]."
    question = "How many days per week may staff work remotely?"

    sources = service.find_sources(question)
    pieces = list(service.stream_answer(question, sources))
    streamed = service.finalize_answer("".join(pieces), sources)

    assert len(pieces) > 1  # arrived in several pieces
    assert streamed == service.ask(question)


def test_asking_with_an_empty_index_raises_a_helpful_error(service, fake_client):
    with pytest.raises(EmptyIndexError, match="No documents are indexed"):
        service.ask("Anything?")
    with pytest.raises(EmptyIndexError):
        service.find_sources("Anything?")

    assert fake_client.chat_calls == []


def test_blank_question_is_rejected(service, sample_pdf):
    service.index_pdf(sample_pdf, "handbook.pdf")

    with pytest.raises(ValueError):
        service.find_sources("  ")


def test_answer_after_deleting_the_only_relevant_document(service, fake_client):
    cats = service.index_pdf(make_pdf(["Cats are small domestic felines."]), "cats.pdf")
    service.index_pdf(make_pdf(["Volcanoes erupt molten lava."]), "volcanoes.pdf")
    service.delete_document(cats.file_hash)
    fake_client.answer = ABSTAIN_MESSAGE

    result = service.ask("What are cats?")

    assert result.abstained
    assert {source.filename for source in result.sources} == {"volcanoes.pdf"}


def test_ollama_going_down_mid_session_is_reported(service, sample_pdf, fake_client):
    service.index_pdf(sample_pdf, "handbook.pdf")
    fake_client.embed_error = ConnectionError("refused")

    with pytest.raises(OllamaUnavailableError):
        service.ask("How many days of annual leave?")
