"""Tests for the Ollama wrapper. Ollama itself is always faked."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import ollama
import pytest

from src.errors import EmbeddingError, ModelNotFoundError, OllamaUnavailableError
from src.ollama_client import OllamaService, normalize_model_name
from tests.conftest import FakeOllamaClient

URL = "http://localhost:11434"


def make_service(client: FakeOllamaClient) -> OllamaService:
    return OllamaService(URL, client=client)


def test_normalize_model_name_adds_latest_tag():
    assert normalize_model_name("bge-m3") == "bge-m3:latest"
    assert normalize_model_name("qwen3.5:4b") == "qwen3.5:4b"


def test_list_models_and_has_model(fake_client):
    service = make_service(fake_client)

    assert service.is_available()
    assert service.list_models() == ["bge-m3:latest", "qwen3.5:4b"]
    assert service.has_model("bge-m3")  # implicit :latest
    assert service.has_model("qwen3.5:4b")
    assert not service.has_model("qwen3.5:2b")


def test_unreachable_service_is_reported(fake_client):
    fake_client.list_error = ConnectionError("refused")
    service = make_service(fake_client)

    assert not service.is_available()
    with pytest.raises(OllamaUnavailableError, match="Cannot reach Ollama"):
        service.list_models()


def test_embed_returns_one_vector_per_text_in_order(fake_client):
    service = make_service(fake_client)
    texts = ["annual leave policy", "expense claims", "remote work"]

    vectors = service.embed(texts, "bge-m3")

    assert len(vectors) == 3
    assert vectors[0] == service.embed(["annual leave policy"], "bge-m3")[0]
    assert vectors[0] != vectors[1]


def test_embed_sends_batches_and_reports_progress(fake_client):
    service = make_service(fake_client)
    progress: list[tuple[int, int]] = []

    vectors = service.embed(
        [f"text {i}" for i in range(5)],
        "bge-m3",
        batch_size=2,
        on_batch=lambda done, total: progress.append((done, total)),
    )

    assert len(vectors) == 5
    assert [len(call["input"]) for call in fake_client.embed_calls] == [2, 2, 1]
    assert all(call["model"] == "bge-m3" for call in fake_client.embed_calls)
    assert progress == [(2, 5), (4, 5), (5, 5)]


@pytest.mark.parametrize("dimension", [8, 64, 1024])
def test_embedding_dimension_is_inferred_not_hardcoded(dimension):
    service = make_service(FakeOllamaClient(dimension=dimension))

    assert len(service.embed(["hello world"], "any-model")[0]) == dimension


def test_embed_empty_list_makes_no_call(fake_client):
    assert make_service(fake_client).embed([], "bge-m3") == []
    assert fake_client.embed_calls == []


def test_embed_rejects_blank_text(fake_client):
    with pytest.raises(EmbeddingError, match="empty text"):
        make_service(fake_client).embed(["fine", "   "], "bge-m3")


def test_missing_model_gives_pull_instruction(fake_client):
    fake_client.embed_error = ollama.ResponseError("model not found", 404)

    with pytest.raises(ModelNotFoundError, match="ollama pull bge-m3"):
        make_service(fake_client).embed(["hello"], "bge-m3")


def test_other_ollama_errors_become_embedding_errors(fake_client):
    fake_client.embed_error = ollama.ResponseError("out of memory", 500)

    with pytest.raises(EmbeddingError, match="out of memory"):
        make_service(fake_client).embed(["hello"], "bge-m3")


@pytest.mark.parametrize(
    "error", [ConnectionError("refused"), httpx.ReadTimeout("timed out")]
)
def test_connection_problems_during_embed_are_reported(fake_client, error):
    fake_client.embed_error = error

    with pytest.raises(OllamaUnavailableError):
        make_service(fake_client).embed(["hello"], "bge-m3")


def test_wrong_number_of_embeddings_is_rejected(fake_client):
    fake_client.embed = lambda model, input: SimpleNamespace(embeddings=[[0.1, 0.2]])

    with pytest.raises(EmbeddingError, match="1 embedding"):
        make_service(fake_client).embed(["one", "two"], "bge-m3")


def test_empty_embedding_vector_is_rejected(fake_client):
    fake_client.embed = lambda model, input: SimpleNamespace(embeddings=[[]])

    with pytest.raises(EmbeddingError, match="empty embedding"):
        make_service(fake_client).embed(["one"], "bge-m3")
