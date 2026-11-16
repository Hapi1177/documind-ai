"""Tests for settings loading and validation."""

from __future__ import annotations

import pytest

from src.config import DEFAULTS, PROJECT_ROOT, load_settings
from src.errors import ConfigError


def test_defaults_match_the_documented_values():
    settings = load_settings({})

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.chat_model == "qwen3.5:4b"
    assert settings.embedding_model == "bge-m3"
    assert settings.collection_name == "documind_local"
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200
    assert settings.top_k == 5
    assert settings.num_ctx == 4096
    assert settings.max_pdf_size_mb == 25
    assert settings.max_pdf_size_bytes == 25 * 1024 * 1024


def test_env_example_matches_code_defaults():
    documented: dict[str, str] = {}
    for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition("=")
            documented[key.strip()] = value.strip()

    assert documented == DEFAULTS


def test_relative_chroma_path_is_resolved_from_project_root():
    settings = load_settings({"CHROMA_PATH": "data/chroma"})

    assert settings.chroma_path == PROJECT_ROOT / "data" / "chroma"


def test_values_can_be_overridden():
    settings = load_settings(
        {"CHAT_MODEL": "qwen3.5:2b", "TOP_K": "8", "OLLAMA_BASE_URL": "http://127.0.0.1:11434/"}
    )

    assert settings.chat_model == "qwen3.5:2b"
    assert settings.top_k == 8
    assert settings.ollama_base_url == "http://127.0.0.1:11434"


@pytest.mark.parametrize(
    "overrides",
    [
        {"CHUNK_SIZE": "abc"},
        {"CHUNK_SIZE": "10"},
        {"CHUNK_OVERLAP": "-1"},
        {"CHUNK_SIZE": "500", "CHUNK_OVERLAP": "500"},
        {"TOP_K": "0"},
        {"NUM_CTX": "100"},
        {"MAX_PDF_SIZE_MB": "0"},
        {"OLLAMA_BASE_URL": "localhost:11434"},
        {"COLLECTION_NAME": "a"},
        {"COLLECTION_NAME": "has spaces"},
    ],
)
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ConfigError):
        load_settings(overrides)
