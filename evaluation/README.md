# Evaluating DocuMind AI

**No evaluation has been run for this project yet.** This folder contains the
tools to run one; it does not contain results, and no quality numbers are
claimed anywhere in this repository. Any figures you publish should come from
your own labelled question set.

## What is in this folder

| File | Purpose |
| --- | --- |
| `eval_questions_template.csv` | Template with the columns to fill in, plus three example rows |
| `run_evaluation.py` | Runs your questions through the real local pipeline and writes a results CSV |
| `results/` | Created on first run. Git-ignored, because answers quote your documents |

## Step 1 - Write 30 to 50 labelled questions

Copy the template and replace the `EXAMPLE` rows with questions about PDFs you
have indexed:

```powershell
Copy-Item evaluation\eval_questions_template.csv evaluation\eval_questions.csv
```

| Column | What to enter |
| --- | --- |
| `question_id` | Any unique ID, e.g. `Q01` |
| `question` | The question, worded the way a real user would ask it |
| `question_type` | `answerable` or `unanswerable` |
| `expected_answer` | The correct answer in your own words (empty for unanswerable) |
| `expected_source_file` | Exact filename as shown in the app, e.g. `handbook.pdf` |
| `expected_source_pages` | Page(s) that contain the answer, e.g. `4` or `4;5` |
| `notes` | Optional |

Tips for a useful set:

- Write the questions by reading the PDFs yourself, not by asking the chatbot.
- Mix easy lookups, questions whose wording differs from the document, and
  questions whose answer spans two pages.
- Make roughly 20-30% of the questions `unanswerable`: plausible questions on
  a topic the documents do not cover. These test abstention.
- Keep the set fixed while you tune settings, so runs are comparable.

## Step 2 - Run it

Ollama must be running with both models pulled, and the PDFs must already be
indexed through the app.

```powershell
python evaluation\run_evaluation.py
```

The script prints the automatic metrics and writes
`evaluation\results\eval_results_<timestamp>.csv`.

## Step 3 - Label the answers by hand

Open the results CSV and fill in the two manual columns for every row:

- `answer_correct`: `yes`, `partial` or `no`, compared with `expected_answer`.
- `citation_correct`: `yes` or `no`.

## The four measurements

### Recall@K (automatic, retrieval only)

For each answerable question: was a chunk from an expected page of the
expected file among the top K retrieved chunks? (K is `TOP_K`, default 5.)

    Recall@K = questions with the expected page in the top K / answerable questions

Because each question here has one labelled location, this is a per-question
hit rate. It measures the retriever alone. If Recall@K is low, the chat model
never sees the right text and no prompt change will fix that; look at chunk
size, overlap and `TOP_K` instead.

### Answer correctness (manual)

Is the answer factually right and complete compared with `expected_answer`?

    Answer correctness = answers marked "yes" / answerable questions

Report `partial` answers separately rather than counting them as correct.
This is judged by a person on purpose: automatic text-overlap scores are
unreliable for short factual answers.

### Citation correctness (manual)

Do the sources the answer cites (the `cited_sources` column) actually contain
the facts they are cited for? An answer can be right and still point at the
wrong passage, so this is scored separately from answer correctness.

    Citation correctness = answers with correct citations / answers that cite something

Note what the app guarantees and what it does not: a citation label can only
ever resolve to a chunk that really was retrieved, with its real filename and
page. Whether the model picked the *right* label is what this metric checks.

### Abstention (automatic)

- On `unanswerable` questions the assistant should reply that it could not
  find the information. The script reports how often it did.
- On `answerable` questions it should *not* abstain. The script reports that
  too, because a model that refuses everything would score perfectly on the
  first number.

Check `abstention_correct` rows marked `no` by hand: the script detects
abstention by matching the fixed "could not find this information" sentence.

## Reporting results honestly

- State the number of questions, the documents used, and the settings
  (`CHAT_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`,
  `NUM_CTX`).
- With 30-50 questions, one question is worth 2-3 percentage points. Treat
  small differences between runs as noise.
- Small local models are not fully deterministic even at low temperature;
  re-running can change individual answers.
- Do not add a similarity cut-off to retrieval until this evaluation shows a
  value that helps. The cosine distance shown in the app is a ranking signal,
  not a confidence score.
