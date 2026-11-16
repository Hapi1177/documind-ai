"""Thin wrapper around the official ``ollama`` Python client.

Everything that talks to the local Ollama service lives here. The wrapper adds
three things on top of the raw client:

* clear, user-facing errors (service down, model not pulled, bad response),
* batching for embeddings,
* validation of what comes back, so bad vectors never reach the database.

The underlying client can be injected, which is how the tests replace Ollama
with a fake.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import httpx
import ollama

from src.errors import (
    EmbeddingError,
    ModelNotFoundError,
    OllamaUnavailableError,
)

# Local models can be slow to load the first time, especially on CPU.
REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_EMBED_BATCH_SIZE = 16

# Called after each embedding batch with (texts_done, texts_total).
BatchCallback = Callable[[int, int], None]


def normalize_model_name(name: str) -> str:
    """Add Ollama's implicit ``:latest`` tag so names can be compared.

    ``bge-m3`` and ``bge-m3:latest`` are the same model to Ollama.
    """
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


class OllamaService:
    """Small facade over a local Ollama server."""

    def __init__(self, base_url: str, client: Any | None = None) -> None:
        self.base_url = base_url
        self._client = client or ollama.Client(
            host=base_url, timeout=REQUEST_TIMEOUT_SECONDS
        )

    # --- Error translation ---------------------------------------------------

    def _unavailable(self) -> OllamaUnavailableError:
        return OllamaUnavailableError(
            f"Cannot reach Ollama at {self.base_url}. Start the Ollama app "
            "(or run 'ollama serve') and try again."
        )

    @staticmethod
    def _model_missing(model: str) -> ModelNotFoundError:
        return ModelNotFoundError(
            f"The model '{model}' is not installed in Ollama. "
            f"Run: ollama pull {model}"
        )

    # --- Status --------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Return the names of all locally installed models.

        Raises:
            OllamaUnavailableError: If the service cannot be reached.
        """
        try:
            response = self._client.list()
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as exc:
            raise self._unavailable() from exc
        return [model.model for model in response.models if model.model]

    def is_available(self) -> bool:
        """Return True if the Ollama service answers."""
        try:
            self.list_models()
        except OllamaUnavailableError:
            return False
        return True

    def has_model(self, model: str) -> bool:
        """Return True if ``model`` is installed locally."""
        wanted = normalize_model_name(model)
        return any(normalize_model_name(name) == wanted for name in self.list_models())

    # --- Embeddings ----------------------------------------------------------

    def embed(
        self,
        texts: Sequence[str],
        model: str,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        on_batch: BatchCallback | None = None,
    ) -> list[list[float]]:
        """Embed texts in batches and return one vector per text, in order.

        The vector size is whatever the model returns; it is checked for
        consistency but never hardcoded.

        Raises:
            OllamaUnavailableError: Service not reachable.
            ModelNotFoundError: Embedding model not pulled.
            EmbeddingError: Empty input or an unusable response.
        """
        texts = list(texts)
        if not texts:
            return []
        if any(not text or not text.strip() for text in texts):
            raise EmbeddingError("Cannot embed empty text.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch, model))
            if on_batch:
                on_batch(len(vectors), len(texts))

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise EmbeddingError(
                f"The model '{model}' returned vectors of different sizes "
                f"({sorted(dimensions)})."
            )
        return vectors

    def _embed_batch(self, batch: list[str], model: str) -> list[list[float]]:
        """Embed one batch and validate the response."""
        try:
            response = self._client.embed(model=model, input=batch)
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise self._model_missing(model) from exc
            raise EmbeddingError(
                f"Ollama could not create embeddings with '{model}': {exc.error}"
            ) from exc
        except (ConnectionError, httpx.HTTPError) as exc:
            raise self._unavailable() from exc

        embeddings = [list(vector) for vector in (response.embeddings or [])]
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings)} embedding(s) for "
                f"{len(batch)} text(s)."
            )
        if any(not vector for vector in embeddings):
            raise EmbeddingError(f"The model '{model}' returned an empty embedding.")
        return embeddings
