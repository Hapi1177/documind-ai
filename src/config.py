"""Central configuration.

All settings are read from environment variables (optionally via a ``.env``
file) in exactly one place. Every other module receives a ``Settings`` object
instead of reading the environment itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from src.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults mirror ``.env.example`` so the app also runs without a ``.env`` file.
DEFAULTS: dict[str, str] = {
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "CHAT_MODEL": "qwen3.5:4b",
    "EMBEDDING_MODEL": "bge-m3",
    "CHROMA_PATH": "data/chroma",
    "COLLECTION_NAME": "documind_local",
    "CHUNK_SIZE": "1200",
    "CHUNK_OVERLAP": "200",
    "TOP_K": "5",
    "NUM_CTX": "4096",
    "MAX_PDF_SIZE_MB": "25",
}


@dataclass(frozen=True)
class Settings:
    """Validated application settings."""

    ollama_base_url: str
    chat_model: str
    embedding_model: str
    chroma_path: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    num_ctx: int
    max_pdf_size_mb: int

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024


def _get_text(env: Mapping[str, str], name: str) -> str:
    """Return a non-empty string setting, falling back to the default."""
    value = (env.get(name) or DEFAULTS[name]).strip()
    if not value:
        raise ConfigError(f"{name} must not be empty.")
    return value


def _get_int(env: Mapping[str, str], name: str, minimum: int) -> int:
    """Return an integer setting that is at least ``minimum``."""
    raw = _get_text(env, name)
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, got '{raw}'.") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _validate_collection_name(name: str) -> str:
    """Apply ChromaDB's collection naming rules early, with a clear message."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    valid = (
        3 <= len(name) <= 63
        and set(name) <= allowed
        and name[0].isalnum()
        and name[-1].isalnum()
    )
    if not valid:
        raise ConfigError(
            "COLLECTION_NAME must be 3-63 characters, use only letters, digits, "
            f"'.', '_' or '-', and start and end with a letter or digit. Got '{name}'."
        )
    return name


def load_settings(
    env: Mapping[str, str] | None = None, *, use_dotenv: bool = True
) -> Settings:
    """Build validated settings.

    Args:
        env: Mapping to read from. Defaults to ``os.environ``. Tests pass a
            plain dict so they never depend on the developer's machine.
        use_dotenv: Load ``.env`` from the project root first (only when
            ``env`` is not supplied).

    Raises:
        ConfigError: If any value is malformed or inconsistent.
    """
    if env is None:
        if use_dotenv:
            load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ

    base_url = _get_text(env, "OLLAMA_BASE_URL").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError(
            f"OLLAMA_BASE_URL must start with http:// or https://, got '{base_url}'."
        )

    chunk_size = _get_int(env, "CHUNK_SIZE", minimum=100)
    chunk_overlap = _get_int(env, "CHUNK_OVERLAP", minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigError(
            f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than "
            f"CHUNK_SIZE ({chunk_size})."
        )

    chroma_path = Path(_get_text(env, "CHROMA_PATH"))
    if not chroma_path.is_absolute():
        chroma_path = PROJECT_ROOT / chroma_path

    return Settings(
        ollama_base_url=base_url,
        chat_model=_get_text(env, "CHAT_MODEL"),
        embedding_model=_get_text(env, "EMBEDDING_MODEL"),
        chroma_path=chroma_path,
        collection_name=_validate_collection_name(_get_text(env, "COLLECTION_NAME")),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=_get_int(env, "TOP_K", minimum=1),
        num_ctx=_get_int(env, "NUM_CTX", minimum=512),
        max_pdf_size_mb=_get_int(env, "MAX_PDF_SIZE_MB", minimum=1),
    )
