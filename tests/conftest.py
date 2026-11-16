"""Shared pytest fixtures and helpers.

All PDFs are generated synthetically with PyMuPDF and Ollama is replaced by a
small fake, so the test suite needs no network, no models and no private
documents.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

# Make "import src...." work no matter where pytest is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, load_settings  # noqa: E402

MAX_BYTES = 25 * 1024 * 1024


# --- Synthetic PDF builders ---------------------------------------------------


def make_pdf(pages: list[str]) -> bytes:
    """Build a text PDF with one entry of ``pages`` per page.

    An empty string produces a page with no text layer (like a scanned page).
    """
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
        else:
            # A drawing but no text: what a scanned/image-only page looks like.
            page.draw_rect(pymupdf.Rect(100, 100, 300, 300), color=(0, 0, 0))
    data = document.tobytes()
    document.close()
    return data


def make_encrypted_pdf(text: str = "Top secret content.") -> bytes:
    """Build a PDF that needs a password to open."""
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), text)
    data = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="user-password",
        owner_pw="owner-password",
    )
    document.close()
    return data


# --- Fake Ollama --------------------------------------------------------------


def fake_embedding(text: str, dimension: int = 64) -> list[float]:
    """Deterministic bag-of-words embedding.

    Each word is hashed into one of ``dimension`` buckets, then the vector is
    L2-normalised. Texts that share words get similar vectors, which is enough
    to make retrieval tests meaningful without a real model.
    """
    vector = [0.0] * dimension
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % dimension
        vector[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class FakeOllamaClient:
    """Stands in for ``ollama.Client`` and records the calls it receives."""

    def __init__(
        self,
        models: tuple[str, ...] = ("bge-m3:latest", "qwen3.5:4b"),
        dimension: int = 64,
        answer: str = "The answer is in the document [S1].",
    ) -> None:
        self.models = list(models)
        self.dimension = dimension
        self.answer = answer
        self.embed_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.embed_error: Exception | None = None
        self.chat_error: Exception | None = None
        self.list_error: Exception | None = None

    def list(self):
        if self.list_error:
            raise self.list_error
        return SimpleNamespace(
            models=[SimpleNamespace(model=name) for name in self.models]
        )

    def embed(self, model: str, input, **kwargs):
        if self.embed_error:
            raise self.embed_error
        texts = [input] if isinstance(input, str) else list(input)
        self.embed_calls.append({"model": model, "input": texts, **kwargs})
        return SimpleNamespace(
            embeddings=[fake_embedding(text, self.dimension) for text in texts]
        )

    def chat(self, model: str, messages, **kwargs):
        if self.chat_error:
            raise self.chat_error
        self.chat_calls.append({"model": model, "messages": messages, **kwargs})
        return SimpleNamespace(message=SimpleNamespace(content=self.answer))


# --- Fixtures -----------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that keep ChromaDB inside pytest's temporary directory."""
    return load_settings(
        {
            "CHROMA_PATH": str(tmp_path / "chroma"),
            "COLLECTION_NAME": "test_collection",
            "CHUNK_SIZE": "300",
            "CHUNK_OVERLAP": "50",
            "TOP_K": "3",
        }
    )


@pytest.fixture
def fake_client() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
def sample_pdf() -> bytes:
    """Three pages on clearly different topics."""
    return make_pdf(
        [
            "Annual leave policy. Employees receive 18 days of annual leave "
            "per calendar year. Unused leave may be carried forward for three months.",
            "Expense claims. Travel expense claims must be submitted within "
            "30 days together with original receipts.",
            "Remote work. Staff may work remotely up to two days per week "
            "with approval from their manager.",
        ]
    )
