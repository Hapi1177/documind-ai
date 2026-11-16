"""Tests for the ChromaDB wrapper, using a real on-disk temporary database."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.errors import EmbeddingModelMismatchError, VectorStoreError
from src.vector_store import VectorStore
from tests.conftest import fake_embedding

MODEL = "bge-m3"


def make_store(path: Path, model: str = MODEL, name: str = "test_collection") -> VectorStore:
    return VectorStore(path, name, model)


def add_document(
    store: VectorStore, file_hash: str, filename: str, texts: list[str], dimension: int = 64
) -> list[str]:
    """Store one chunk per text, each on its own page."""
    ids = [f"{file_hash}:p{page:05d}:c0000" for page in range(1, len(texts) + 1)]
    metadatas = [
        {
            "file_hash": file_hash,
            "filename": filename,
            "page_number": page,
            "chunk_index": 0,
            "page_count": len(texts),
            "chunk_size": 300,
            "chunk_overlap": 50,
            "embedding_model": MODEL,
            "indexed_at": "2026-01-01T00:00:00+00:00",
        }
        for page in range(1, len(texts) + 1)
    ]
    embeddings = [fake_embedding(text, dimension) for text in texts]
    store.upsert_chunks(ids, texts, embeddings, metadatas)
    return ids


def test_new_store_is_empty(tmp_path):
    store = make_store(tmp_path)

    assert store.count() == 0
    assert store.list_documents() == []
    assert store.query(fake_embedding("anything"), top_k=5) == []


def test_upsert_and_query_return_text_metadata_ids_and_distance(tmp_path):
    store = make_store(tmp_path)
    ids = add_document(
        store,
        "hash-a",
        "handbook.pdf",
        ["annual leave is eighteen days", "expense claims need receipts"],
    )

    hits = store.query(fake_embedding("how many days of annual leave"), top_k=2)

    assert len(hits) == 2
    best = hits[0]
    assert best.id == ids[0]
    assert best.text == "annual leave is eighteen days"
    assert best.filename == "handbook.pdf"
    assert best.page_number == 1
    assert best.chunk_index == 0
    assert best.file_hash == "hash-a"
    assert isinstance(best.distance, float)
    assert hits[0].distance <= hits[1].distance  # closest first


def test_identical_vector_has_near_zero_cosine_distance(tmp_path):
    store = make_store(tmp_path)
    add_document(store, "hash-a", "a.pdf", ["exact same words here"])

    hit = store.query(fake_embedding("exact same words here"), top_k=1)[0]

    assert hit.distance == pytest.approx(0.0, abs=1e-5)


def test_upsert_with_same_ids_does_not_duplicate(tmp_path):
    store = make_store(tmp_path)
    texts = ["first chunk", "second chunk", "third chunk"]

    add_document(store, "hash-a", "a.pdf", texts)
    add_document(store, "hash-a", "a.pdf", texts)

    assert store.count() == 3


def test_top_k_larger_than_collection_returns_everything(tmp_path):
    store = make_store(tmp_path)
    add_document(store, "hash-a", "a.pdf", ["one", "two"])

    assert len(store.query(fake_embedding("one"), top_k=10)) == 2


def test_invalid_top_k_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_store(tmp_path).query(fake_embedding("x"), top_k=0)


def test_list_documents_summarises_each_file(tmp_path):
    store = make_store(tmp_path)
    add_document(store, "hash-b", "zebra.pdf", ["z one", "z two", "z three"])
    add_document(store, "hash-a", "apple.pdf", ["a one"])

    documents = store.list_documents()

    assert [doc.filename for doc in documents] == ["apple.pdf", "zebra.pdf"]
    assert [doc.chunk_count for doc in documents] == [1, 3]
    assert documents[1].file_hash == "hash-b"
    assert documents[1].page_count == 3
    assert store.get_document("hash-a").filename == "apple.pdf"
    assert store.get_document("missing") is None


def test_same_filename_with_different_hash_is_a_separate_document(tmp_path):
    store = make_store(tmp_path)
    add_document(store, "hash-v1", "policy.pdf", ["version one text"])
    add_document(store, "hash-v2", "policy.pdf", ["version two text"])

    documents = store.list_documents()

    assert len(documents) == 2
    assert {doc.file_hash for doc in documents} == {"hash-v1", "hash-v2"}
    assert store.count() == 2


def test_delete_document_removes_only_that_document(tmp_path):
    store = make_store(tmp_path)
    add_document(store, "hash-a", "a.pdf", ["alpha one", "alpha two"])
    add_document(store, "hash-b", "b.pdf", ["beta one"])

    deleted = store.delete_document("hash-a")

    assert deleted == 2
    assert store.count() == 1
    assert [doc.file_hash for doc in store.list_documents()] == ["hash-b"]
    assert all(hit.file_hash == "hash-b" for hit in store.query(fake_embedding("alpha"), 5))


def test_delete_unknown_document_is_a_no_op(tmp_path):
    store = make_store(tmp_path)
    add_document(store, "hash-a", "a.pdf", ["alpha"])

    assert store.delete_document("does-not-exist") == 0
    assert store.count() == 1


def test_data_persists_across_store_instances(tmp_path):
    add_document(make_store(tmp_path), "hash-a", "a.pdf", ["persisted text"])

    reopened = make_store(tmp_path)

    assert reopened.count() == 1
    assert reopened.query(fake_embedding("persisted text"), 1)[0].text == "persisted text"


def test_opening_with_a_different_embedding_model_is_refused(tmp_path):
    add_document(make_store(tmp_path, model="bge-m3"), "hash-a", "a.pdf", ["text"])

    with pytest.raises(EmbeddingModelMismatchError, match="bge-m3"):
        make_store(tmp_path, model="nomic-embed-text")


def test_different_collection_name_allows_a_different_model(tmp_path):
    add_document(make_store(tmp_path, model="bge-m3"), "hash-a", "a.pdf", ["text"])

    other = make_store(tmp_path, model="nomic-embed-text", name="other_collection")

    assert other.count() == 0


@pytest.mark.parametrize("dimension", [8, 128])
def test_vector_dimension_is_taken_from_the_first_vectors(tmp_path, dimension):
    store = make_store(tmp_path)
    add_document(store, "hash-a", "a.pdf", ["some text"], dimension=dimension)

    assert len(store.query(fake_embedding("some text", dimension), 1)) == 1


def test_vector_of_wrong_dimension_gives_a_clear_error(tmp_path):
    store = make_store(tmp_path)
    add_document(store, "hash-a", "a.pdf", ["some text"], dimension=64)

    with pytest.raises(VectorStoreError):
        add_document(store, "hash-b", "b.pdf", ["other text"], dimension=32)


def test_mismatched_input_lengths_are_rejected(tmp_path):
    with pytest.raises(VectorStoreError, match="same length"):
        make_store(tmp_path).upsert_chunks(["id-1"], ["a", "b"], [[0.1]], [{}])
