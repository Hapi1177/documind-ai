"""Run the labelled evaluation questions through the local RAG pipeline.

This needs Ollama running with both models pulled, and your PDFs already
indexed through the app. It fills in everything that can be measured
automatically and leaves the judgement calls for you:

* automatic: which sources were retrieved, whether the expected page was among
  them (Recall@K), the model's answer, and whether the model abstained;
* manual: ``answer_correct`` and ``citation_correct``, which you fill in by
  reading each answer in the results file.

Usage (from the project root):

    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --questions my_questions.csv --output my_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.document_service import DocumentService  # noqa: E402
from src.errors import DocuMindError  # noqa: E402
from src.rag import RagAnswer, Source  # noqa: E402

EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = EVALUATION_DIR / "eval_questions.csv"
TEMPLATE = EVALUATION_DIR / "eval_questions_template.csv"
RESULTS_DIR = EVALUATION_DIR / "results"

QUESTION_TYPES = {"answerable", "unanswerable"}
REQUIRED_COLUMNS = {
    "question_id",
    "question",
    "question_type",
    "expected_answer",
    "expected_source_file",
    "expected_source_pages",
}
RESULT_COLUMNS = [
    "question_id",
    "question",
    "question_type",
    "expected_answer",
    "expected_source_file",
    "expected_source_pages",
    "retrieved_sources",
    "recall_hit",
    "model_answer",
    "model_abstained",
    "abstention_correct",
    "cited_sources",
    "answer_correct",  # manual: yes / no / partial
    "citation_correct",  # manual: yes / no
    "notes",
]


def parse_pages(value: str) -> set[int]:
    """Parse '4' or '4;5' into a set of page numbers."""
    return {int(part) for part in value.replace(",", ";").split(";") if part.strip()}


def load_questions(path: Path) -> list[dict[str, str]]:
    """Read and validate the labelled questions."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing column(s): {sorted(missing)}")
        rows = [row for row in reader if (row.get("question") or "").strip()]

    for row in rows:
        kind = (row["question_type"] or "").strip().lower()
        if kind not in QUESTION_TYPES:
            raise ValueError(
                f"Question {row['question_id']}: question_type must be one of "
                f"{sorted(QUESTION_TYPES)}, got '{row['question_type']}'."
            )
        row["question_type"] = kind
        if kind == "answerable" and not (
            row["expected_source_file"].strip() and row["expected_source_pages"].strip()
        ):
            raise ValueError(
                f"Question {row['question_id']}: answerable questions need "
                "expected_source_file and expected_source_pages."
            )
    return rows


def is_recall_hit(sources: list[Source], expected_file: str, expected_pages: set[int]) -> bool:
    """True if any retrieved source is on an expected page of the expected file."""
    return any(
        source.filename == expected_file and source.page_number in expected_pages
        for source in sources
    )


def evaluate_row(row: dict[str, str], answer: RagAnswer) -> dict[str, str]:
    """Combine one labelled question with what the pipeline produced."""
    answerable = row["question_type"] == "answerable"
    recall_hit = ""
    if answerable:
        hit = is_recall_hit(
            answer.sources,
            row["expected_source_file"].strip(),
            parse_pages(row["expected_source_pages"]),
        )
        recall_hit = "yes" if hit else "no"

    cited = [source for source in answer.sources if source.label in answer.cited_labels]
    return {
        **{column: row.get(column, "") for column in RESULT_COLUMNS},
        "retrieved_sources": "; ".join(
            f"{source.filename} p.{source.page_number}" for source in answer.sources
        ),
        "recall_hit": recall_hit,
        "model_answer": answer.answer,
        "model_abstained": "yes" if answer.abstained else "no",
        # Correct = abstained exactly when the question was unanswerable.
        "abstention_correct": "yes" if answer.abstained != answerable else "no",
        "cited_sources": "; ".join(
            f"{source.filename} p.{source.page_number}" for source in cited
        ),
        "answer_correct": "",
        "citation_correct": "",
    }


def summarise(results: list[dict[str, str]], top_k: int) -> list[str]:
    """Build the automatic part of the summary. Nothing here is estimated."""
    answerable = [row for row in results if row["question_type"] == "answerable"]
    unanswerable = [row for row in results if row["question_type"] == "unanswerable"]

    def rate(rows: list[dict[str, str]], column: str) -> str:
        if not rows:
            return "n/a (no questions of this type)"
        hits = sum(1 for row in rows if row[column] == "yes")
        return f"{hits}/{len(rows)} = {hits / len(rows):.1%}"

    return [
        f"Questions evaluated: {len(results)} "
        f"({len(answerable)} answerable, {len(unanswerable)} unanswerable)",
        f"Recall@{top_k} (expected page retrieved): {rate(answerable, 'recall_hit')}",
        f"Abstained correctly on unanswerable questions: "
        f"{rate(unanswerable, 'abstention_correct')}",
        f"Answered (did not abstain) on answerable questions: "
        f"{rate(answerable, 'abstention_correct')}",
        "Answer correctness and citation correctness: fill in the "
        "'answer_correct' and 'citation_correct' columns by hand.",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.questions.is_file():
        print(
            f"Questions file not found: {args.questions}\n"
            f"Copy {TEMPLATE.name} to {DEFAULT_QUESTIONS.name} and replace the "
            "example rows with questions about your own PDFs."
        )
        return 2

    output = args.output or RESULTS_DIR / f"eval_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    try:
        questions = load_questions(args.questions)
        service = DocumentService(load_settings())
        results = []
        for number, row in enumerate(questions, start=1):
            print(f"[{number}/{len(questions)}] {row['question_id']}: {row['question']}")
            results.append(evaluate_row(row, service.ask(row["question"])))
    except (DocuMindError, ValueError) as error:
        print(f"\nFAILED - {error}")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "\n".join(summarise(results, service.settings.top_k)))
    print(f"\nResults written to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
