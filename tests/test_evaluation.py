"""Tests for the evaluation helpers (scoring logic only, no models)."""

from __future__ import annotations

import pytest

from evaluation.run_evaluation import (
    REQUIRED_COLUMNS,
    TEMPLATE,
    evaluate_row,
    is_recall_hit,
    load_questions,
    parse_pages,
    summarise,
)
from src.rag import ABSTAIN_MESSAGE, RagAnswer, Source


def make_source(label: str, filename: str, page: int) -> Source:
    return Source(label, f"id-{label}", filename, page, "text", 0.2)


SOURCES = [make_source("S1", "handbook.pdf", 4), make_source("S2", "other.pdf", 9)]


def make_row(kind: str, file: str = "handbook.pdf", pages: str = "4") -> dict[str, str]:
    return {
        "question_id": "Q1",
        "question": "A question?",
        "question_type": kind,
        "expected_answer": "",
        "expected_source_file": file if kind == "answerable" else "",
        "expected_source_pages": pages if kind == "answerable" else "",
        "notes": "",
    }


def test_template_is_valid_and_has_the_required_columns():
    rows = load_questions(TEMPLATE)

    assert REQUIRED_COLUMNS <= set(rows[0])
    assert {row["question_type"] for row in rows} == {"answerable", "unanswerable"}


def test_parse_pages():
    assert parse_pages("4") == {4}
    assert parse_pages("4; 5") == {4, 5}
    assert parse_pages("") == set()


def test_recall_hit_needs_matching_file_and_page():
    assert is_recall_hit(SOURCES, "handbook.pdf", {4})
    assert is_recall_hit(SOURCES, "handbook.pdf", {3, 4})
    assert not is_recall_hit(SOURCES, "handbook.pdf", {9})  # right page, wrong file
    assert not is_recall_hit(SOURCES, "missing.pdf", {4})
    assert not is_recall_hit([], "handbook.pdf", {4})


def test_answerable_question_answered_with_expected_page():
    answer = RagAnswer("18 days [S1].", SOURCES, ["S1"], abstained=False)

    result = evaluate_row(make_row("answerable"), answer)

    assert result["recall_hit"] == "yes"
    assert result["model_abstained"] == "no"
    assert result["abstention_correct"] == "yes"
    assert result["retrieved_sources"] == "handbook.pdf p.4; other.pdf p.9"
    assert result["cited_sources"] == "handbook.pdf p.4"
    assert result["answer_correct"] == ""  # left for the human reviewer
    assert result["citation_correct"] == ""


def test_answerable_question_with_wrong_abstention_and_missed_page():
    answer = RagAnswer(ABSTAIN_MESSAGE, SOURCES, [], abstained=True)

    result = evaluate_row(make_row("answerable", pages="7"), answer)

    assert result["recall_hit"] == "no"
    assert result["abstention_correct"] == "no"


@pytest.mark.parametrize("abstained,expected", [(True, "yes"), (False, "no")])
def test_unanswerable_question_should_be_abstained(abstained, expected):
    answer = RagAnswer("text", SOURCES, [], abstained=abstained)

    result = evaluate_row(make_row("unanswerable"), answer)

    assert result["recall_hit"] == ""  # recall is undefined without a target
    assert result["abstention_correct"] == expected


def test_summary_counts_only_what_was_measured():
    results = [
        {"question_type": "answerable", "recall_hit": "yes", "abstention_correct": "yes"},
        {"question_type": "answerable", "recall_hit": "no", "abstention_correct": "yes"},
        {"question_type": "unanswerable", "recall_hit": "", "abstention_correct": "yes"},
    ]

    summary = "\n".join(summarise(results, top_k=5))

    assert "Recall@5 (expected page retrieved): 1/2 = 50.0%" in summary
    assert "unanswerable questions: 1/1 = 100.0%" in summary
    assert "by hand" in summary


def test_summary_without_questions_of_a_type_reports_not_applicable():
    assert "n/a" in "\n".join(summarise([], top_k=5))


def test_invalid_question_type_is_rejected(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(
        "question_id,question,question_type,expected_answer,expected_source_file,expected_source_pages\n"
        "Q1,What?,maybe,,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="question_type"):
        load_questions(path)


def test_answerable_question_without_expected_source_is_rejected(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(
        "question_id,question,question_type,expected_answer,expected_source_file,expected_source_pages\n"
        "Q1,What?,answerable,Something,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_source_file"):
        load_questions(path)
