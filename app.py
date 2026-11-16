"""DocuMind AI - Streamlit user interface.

Run with:  streamlit run app.py

This file only draws the screen and reacts to clicks. All real work (PDF
parsing, embeddings, vector search, answer generation) is done by
``DocumentService`` in ``src/``.
"""

from __future__ import annotations

import logging

import streamlit as st

from src.config import PROJECT_ROOT, load_settings
from src.document_service import DocumentService, SystemStatus
from src.errors import DocuMindError
from src.ingestion import STATUS_REINDEXED, STATUS_UNCHANGED, IngestionResult
from src.rag import RagAnswer, Source

LOG_FILE = PROJECT_ROOT / "logs" / "documind.log"
UNEXPECTED_ERROR = (
    "Something unexpected went wrong. Details were written to logs/documind.log."
)

logger = logging.getLogger("documind")


# --- Setup --------------------------------------------------------------------


def configure_logging() -> None:
    """Send unexpected errors to a local log file instead of the browser."""
    if logger.handlers:
        return
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@st.cache_resource(show_spinner=False)
def get_service() -> DocumentService:
    """Create the service once and reuse it across reruns and browser tabs."""
    return DocumentService(load_settings())


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])  # chat history
    st.session_state.setdefault("notices", [])  # results of the last action
    st.session_state.setdefault("uploader_version", 0)  # bump to clear uploader


def add_notice(level: str, message: str) -> None:
    st.session_state.notices.append((level, message))


def code_span(value: str) -> str:
    """Show a filename or model name literally, without Markdown formatting."""
    return "`" + value.replace("`", "'") + "`"


def safe_markdown(text: str) -> str:
    """Stop '$' in an answer from being rendered as a LaTeX formula."""
    return text.replace("$", "\\$")


# --- Sidebar: system status ---------------------------------------------------


def render_status(service: DocumentService, status: SystemStatus) -> None:
    settings = service.settings

    def line(name: str, ok: bool, good: str, bad: str) -> str:
        state = f":green[{good}]" if ok else f":red[{bad}]"
        return f"- {name}: {state}"

    st.subheader("System status")
    st.markdown(
        "\n".join(
            [
                line("Ollama", status.ollama_available, "running", "not reachable"),
                line(
                    f"Chat model {code_span(settings.chat_model)}",
                    status.chat_model_installed,
                    "ready",
                    "not available",
                ),
                line(
                    f"Embedding model {code_span(settings.embedding_model)}",
                    status.embedding_model_installed,
                    "ready",
                    "not available",
                ),
            ]
        )
    )
    for problem in status.problems:
        st.warning(problem)
    if status.problems:
        st.button("Check again", width="stretch")


# --- Sidebar: documents -------------------------------------------------------


def describe_result(result: IngestionResult) -> tuple[str, str]:
    """Turn an ingestion result into a (level, message) notice."""
    name = code_span(result.filename)
    if result.status == STATUS_UNCHANGED:
        return "info", f"{name} is already indexed - skipped."
    verb = "Re-indexed" if result.status == STATUS_REINDEXED else "Indexed"
    return (
        "success",
        f"{verb} {name}: {result.page_count} page(s), {result.chunk_count} chunk(s).",
    )


def index_uploads(service: DocumentService, uploads: list) -> None:
    """Index every uploaded file and record one notice per file."""
    progress_bar = st.progress(0.0, text="Starting...")
    for position, upload in enumerate(uploads):

        def on_progress(message: str, fraction: float, position: int = position) -> None:
            overall = (position + fraction) / len(uploads)
            progress_bar.progress(min(overall, 1.0), text=message)

        try:
            result = service.index_pdf(upload.getvalue(), upload.name, on_progress)
        except DocuMindError as error:
            add_notice("error", str(error))
            continue
        except Exception:
            logger.exception("Unexpected error while indexing %s", upload.name)
            add_notice("error", f"{code_span(upload.name)}: {UNEXPECTED_ERROR}")
            continue

        add_notice(*describe_result(result))
        for warning in result.warnings:
            add_notice("warning", f"{code_span(result.filename)}: {warning}")
    progress_bar.empty()


def delete_document(service: DocumentService, file_hash: str, filename: str) -> None:
    """Button callback: remove one document from the index."""
    st.session_state.notices = []
    try:
        removed = service.delete_document(file_hash)
    except DocuMindError as error:
        add_notice("error", str(error))
    except Exception:
        logger.exception("Unexpected error while deleting %s", filename)
        add_notice("error", UNEXPECTED_ERROR)
    else:
        add_notice("success", f"Deleted {code_span(filename)} ({removed} chunk(s)).")


def render_upload(service: DocumentService, status: SystemStatus) -> None:
    st.subheader("Add documents")
    uploads = st.file_uploader(
        "Upload text-based PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=service.settings.max_pdf_size_mb,
        key=f"uploader_{st.session_state.uploader_version}",
    )
    clicked = st.button(
        "Index documents",
        type="primary",
        width="stretch",
        disabled=not uploads or not status.can_index,
    )
    if uploads and not status.can_index:
        st.caption("Indexing needs Ollama and the embedding model (see status above).")
    if clicked:
        st.session_state.notices = []
        index_uploads(service, uploads)
        st.session_state.uploader_version += 1  # empties the uploader
        st.rerun()

    notices_box = st.container()
    for level, message in st.session_state.notices:
        getattr(notices_box, level)(message)


def render_documents(service: DocumentService) -> int:
    """List indexed documents with delete buttons. Returns the chunk count."""
    st.subheader("Indexed documents")
    try:
        documents = service.list_documents()
        total_chunks = service.chunk_count()
    except DocuMindError as error:
        st.error(str(error))
        return 0

    left, right = st.columns(2)
    left.metric("Documents", len(documents))
    right.metric("Total chunks", total_chunks)

    if not documents:
        st.caption("Nothing indexed yet.")
    for document in documents:
        with st.container(border=True):
            st.markdown(code_span(document.filename))
            st.caption(
                f"{document.page_count} page(s) · {document.chunk_count} chunk(s) · "
                f"ID {document.file_hash[:8]}"
            )
            st.button(
                "Delete",
                key=f"delete_{document.file_hash}",
                on_click=delete_document,
                args=(service, document.file_hash, document.filename),
                width="stretch",
            )
    return total_chunks


def render_sidebar(service: DocumentService, status: SystemStatus) -> int:
    settings = service.settings
    with st.sidebar:
        st.title("DocuMind AI")
        st.caption("Local RAG chatbot. Your documents never leave this computer.")
        render_status(service, status)
        st.divider()
        render_upload(service, status)
        st.divider()
        total_chunks = render_documents(service)
        st.divider()
        with st.expander("Settings (from .env)"):
            st.markdown(
                f"- Chunk size: {settings.chunk_size} characters\n"
                f"- Chunk overlap: {settings.chunk_overlap} characters\n"
                f"- Sources per question (top-K): {settings.top_k}\n"
                f"- Context window: {settings.num_ctx} tokens\n"
                f"- Collection: {code_span(settings.collection_name)}\n"
                f"- Max upload: {settings.max_pdf_size_mb} MB per PDF"
            )
    return total_chunks


# --- Main area: conversation --------------------------------------------------


def render_sources(sources: list[Source], cited_labels: list[str], abstained: bool) -> None:
    """Show where the answer came from, using retrieved metadata only."""
    if not sources:
        return

    cited = [source for source in sources if source.label in cited_labels]
    if cited:
        references = ", ".join(
            f"[{source.label}] {code_span(source.filename)} p. {source.page_number}"
            for source in cited
        )
        st.caption(f"Cited: {references}")

    title = (
        f"Retrieved passages ({len(sources)}) - none answered the question"
        if abstained
        else f"Sources ({len(sources)} retrieved, {len(cited)} cited)"
    )
    with st.expander(title):
        for source in sources:
            with st.container(border=True):
                used = " · cited in the answer" if source.label in cited_labels else ""
                st.markdown(
                    f"**[{source.label}]** {code_span(source.filename)} · "
                    f"page {source.page_number}{used}"
                )
                # Plain text (never rendered as Markdown/HTML). PDF line breaks
                # fall mid-sentence, so whitespace is collapsed for display only.
                st.text(" ".join(source.text.split()))
                st.caption(
                    f"Cosine distance {source.distance:.3f} "
                    "(lower means closer to the question; not a confidence score)"
                )


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return
        st.markdown(safe_markdown(message["content"]))
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("cited_labels", []),
                message.get("abstained", False),
            )


def generate_answer(service: DocumentService, question: str) -> RagAnswer:
    """Retrieve sources, then show the answer while the model writes it."""
    with st.spinner("Searching your documents..."):
        sources = service.find_sources(question)

    placeholder = st.empty()
    raw_answer = ""
    with st.spinner("Writing the answer..."):
        for piece in service.stream_answer(question, sources):
            raw_answer += piece
            placeholder.markdown(safe_markdown(raw_answer) + " ▌")

    result = service.finalize_answer(raw_answer, sources)
    placeholder.markdown(safe_markdown(result.answer))
    return result


def handle_question(service: DocumentService, question: str) -> None:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        try:
            result = generate_answer(service, question)
        except DocuMindError as error:
            reply = {"role": "assistant", "content": str(error), "error": True}
            st.error(str(error))
        except Exception:
            logger.exception("Unexpected error while answering a question")
            reply = {"role": "assistant", "content": UNEXPECTED_ERROR, "error": True}
            st.error(UNEXPECTED_ERROR)
        else:
            reply = {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
                "cited_labels": result.cited_labels,
                "abstained": result.abstained,
            }
            render_sources(result.sources, result.cited_labels, result.abstained)
    st.session_state.messages.append(reply)


def render_chat(service: DocumentService, status: SystemStatus, total_chunks: int) -> None:
    header, clear = st.columns([4, 1], vertical_alignment="bottom")
    header.header("Ask your documents")
    if clear.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    if total_chunks == 0:
        st.info(
            "No documents are indexed yet. Upload one or more text-based PDFs in "
            "the sidebar and click **Index documents** to get started."
        )
    elif not status.can_chat:
        st.warning(
            "Chat is unavailable until Ollama and both models are ready. "
            "See **System status** in the sidebar."
        )

    hint = st.empty()  # first-use tip, removed as soon as a question is asked
    if total_chunks > 0 and status.can_chat and not st.session_state.messages:
        hint.caption(
            "Ask a question below. Answers are written only from your indexed "
            "PDFs, with the source file and page listed under each answer."
        )

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input(
        "Ask a question about your documents",
        disabled=total_chunks == 0 or not status.can_chat,
    )
    if question and question.strip():
        hint.empty()
        handle_question(service, question.strip())


# --- Entry point --------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="DocuMind AI", page_icon=":material/description:", layout="centered")
    configure_logging()
    init_session_state()

    try:
        service = get_service()
    except DocuMindError as error:
        st.error(f"DocuMind AI could not start: {error}")
        st.stop()
    except Exception:
        logger.exception("Unexpected error during start-up")
        st.error(f"DocuMind AI could not start. {UNEXPECTED_ERROR}")
        st.stop()

    status = service.get_status()
    total_chunks = render_sidebar(service, status)
    render_chat(service, status, total_chunks)


main()
