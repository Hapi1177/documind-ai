"""Smoke tests for the Streamlit UI using Streamlit's built-in test runner.

The app is run headlessly with a fake service, so no browser, Ollama or
models are needed. These tests check wiring and empty/error states; the
business logic is covered by the other test modules.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src import document_service
from src.config import PROJECT_ROOT
from src.ollama_client import OllamaService
from tests.conftest import FakeOllamaClient, make_pdf

APP_PATH = str(PROJECT_ROOT / "app.py")


@pytest.fixture
def run_app(settings, monkeypatch):
    """Return a function that starts the app wired to a fake Ollama client."""
    import streamlit as st

    def start(client: FakeOllamaClient, pdfs: dict[str, bytes] | None = None) -> AppTest:
        service = document_service.DocumentService(
            settings, ollama=OllamaService(settings.ollama_base_url, client=client)
        )
        for filename, data in (pdfs or {}).items():
            service.index_pdf(data, filename)

        # app.py builds its service with DocumentService(load_settings());
        # hand it the prepared one instead.
        monkeypatch.setattr(
            document_service, "DocumentService", lambda *args, **kwargs: service
        )
        st.cache_resource.clear()
        return AppTest.from_file(APP_PATH, default_timeout=30).run()

    yield start
    st.cache_resource.clear()


def all_text(app: AppTest) -> str:
    """Collect the visible text of the page for simple assertions."""
    elements = (
        list(app.markdown) + list(app.caption) + list(app.info) + list(app.warning)
        + list(app.error) + list(app.success) + list(app.text)
        + list(app.sidebar.markdown) + list(app.sidebar.caption)
        + list(app.sidebar.warning) + list(app.sidebar.success)
    )
    return "\n".join(str(element.value) for element in elements)


def test_empty_state_explains_what_to_do(run_app):
    app = run_app(FakeOllamaClient())

    assert not app.exception
    text = all_text(app)
    assert "No documents are indexed yet" in text
    assert "Nothing indexed yet." in text
    assert ":green[running]" in text
    assert app.chat_input[0].disabled


def test_ollama_down_is_shown_and_chat_is_disabled(run_app):
    client = FakeOllamaClient()
    client.list_error = ConnectionError("refused")

    app = run_app(client, pdfs=None)

    assert not app.exception
    text = all_text(app)
    assert ":red[not reachable]" in text
    assert "Cannot reach Ollama" in text
    assert app.chat_input[0].disabled


def test_missing_chat_model_shows_the_pull_command(run_app, settings):
    app = run_app(FakeOllamaClient(models=("bge-m3:latest",)))

    assert f"ollama pull {settings.chat_model}" in all_text(app)


def test_question_gets_an_answer_with_sources_from_metadata(run_app):
    client = FakeOllamaClient(answer="Claims are due within 30 days [S1].")
    pdf = make_pdf(["Travel expense claims must be submitted within 30 days."])
    app = run_app(client, pdfs={"handbook.pdf": pdf})

    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value("When are expense claims due?").run()

    assert not app.exception
    text = all_text(app)
    assert "When are expense claims due?" in text
    assert "Claims are due within 30 days [S1]." in text
    assert "`handbook.pdf` p. 1" in text  # citation built from stored metadata
    assert "Travel expense claims must be submitted within 30 days." in text
    assert len(app.session_state.messages) == 2


def test_generation_error_is_shown_without_a_stack_trace(run_app):
    client = FakeOllamaClient()
    pdf = make_pdf(["Travel expense claims must be submitted within 30 days."])
    app = run_app(client, pdfs={"handbook.pdf": pdf})
    client.chat_error = ConnectionError("refused")

    app.chat_input[0].set_value("When are expense claims due?").run()

    assert not app.exception
    assert any("Cannot reach Ollama" in str(error.value) for error in app.error)
    assert "Traceback" not in all_text(app)


def test_delete_and_clear_chat_buttons(run_app):
    pdf = make_pdf(["Travel expense claims must be submitted within 30 days."])
    app = run_app(FakeOllamaClient(), pdfs={"handbook.pdf": pdf})
    app.chat_input[0].set_value("When are expense claims due?").run()

    next(button for button in app.button if button.label == "Clear chat").click().run()
    assert app.session_state.messages == []

    next(button for button in app.sidebar.button if button.label == "Delete").click().run()
    assert not app.exception
    text = all_text(app)
    assert "Deleted `handbook.pdf`" in text
    assert "No documents are indexed yet" in text
