"""Application-specific exceptions.

Every error the user can trigger maps to one of these classes. Their messages
are written for end users, so the UI can show ``str(error)`` directly without
exposing a stack trace.
"""

from __future__ import annotations


class DocuMindError(Exception):
    """Base class for all expected, user-facing application errors."""


class ConfigError(DocuMindError):
    """A setting in ``.env`` is missing, malformed or inconsistent."""


# --- PDF errors -------------------------------------------------------------


class PDFError(DocuMindError):
    """Base class for problems with an uploaded PDF."""


class PDFTooLargeError(PDFError):
    """The upload exceeds ``MAX_PDF_SIZE_MB``."""


class InvalidPDFError(PDFError):
    """The file is empty, is not a PDF, or is too damaged to open."""


class EncryptedPDFError(PDFError):
    """The PDF is password protected."""


class ScannedPDFError(PDFError):
    """The PDF opened but contains no extractable text (likely a scan)."""


# --- Ollama errors ----------------------------------------------------------


class OllamaError(DocuMindError):
    """Base class for problems talking to the local Ollama service."""


class OllamaUnavailableError(OllamaError):
    """The Ollama service could not be reached."""


class ModelNotFoundError(OllamaError):
    """The requested model has not been pulled into Ollama."""


class EmbeddingError(OllamaError):
    """Ollama failed to produce usable embeddings."""


class GenerationError(OllamaError):
    """Ollama failed to generate a chat answer."""


# --- Vector store errors ----------------------------------------------------


class VectorStoreError(DocuMindError):
    """ChromaDB could not complete an operation."""


class EmbeddingModelMismatchError(VectorStoreError):
    """The collection was built with a different embedding model."""
