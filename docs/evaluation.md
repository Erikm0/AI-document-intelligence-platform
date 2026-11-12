# Evaluation

DocIntel ships with a reproducible evaluation harness
(`backend/evaluation/`). It measures retrieval quality and hallucination
resistance, and its thresholds are enforced as tests in CI.

```bash
cd backend
python -m evaluation.run                               # fast: hashing embedder + lexical reranker + fake LLM
python -m evaluation.run --models real                 # bge-small + MiniLM cross-encoder
python -m evaluation.run --models real --llm ollama    # full end-to-end with your local LLM
```

Reports are written to `backend/evaluation/reports/` as Markdown and JSON.

## Dataset

The corpus (`evaluation/corpus.py`) is **fictional on purpose**: an HR handbook
(TXT), a fusion-reactor test report (PDF), a services contract (DOCX) and a
distractor contractor guide (TXT). A model cannot answer from pre-training
knowledge. Every correct answer must come from retrieval, and any answer to an
unanswerable question is by definition a hallucination. Each document is
rendered to its real file format at evaluation time, so the PDF, DOCX and TXT
parsers are part of what is measured.

The **distractor** document repeats the topics of the HR handbook and the
contract (remote work, meal expenses, incident reporting, invoices,
termination) with *conflicting* values. A retriever that matches topics but not
entities picks the wrong passage.

`evaluation/dataset.json` contains:

* **26 answerable questions**, each with the expected document, page and an
  evidence substring. They fall into three categories:
  * `factual` - direct look-ups.
  * `keyword` - exact identifiers such as "Ledgerly", "REBCO", "312 discharges".
  * `paraphrase` - vocabulary mismatch, e.g. "What happens to vacation days I
    don't use?" against text about "carried over… leave days".
* **10 unanswerable questions**:
  * near-misses that mention real entities ("Phase **III** start date", "CEO
    salary", "Northwind headcount", "dental insurance")
  * world-knowledge bait ("capital of France", "2022 World Cup")
  * off-task requests ("write a poem").

## Metrics

| Metric | Definition |
|---|---|
| Recall@k | Share of questions with a relevant chunk (right document **and** containing the evidence) in the top-k |
| MRR | Mean reciprocal rank of the first relevant chunk |
| nDCG@5 | Rank-discounted gain, binary relevance |
| page_acc | The first relevant chunk's page range contains the expected page |
| answer_rate | Answerable questions that were answered (not refused) |
| citation_accuracy | Answered questions where a **cited** source contains the evidence |
| refusal_rate / hallucination_rate | Unanswerable questions that were refused / answered anyway |

## Results

Real retrieval models (`BAAI/bge-small-en-v1.5` + `cross-encoder/ms-marco-MiniLM-L-6-v2`),
CPU, 4 documents, 11 pages, 19 chunks:

| mode | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | page_acc | latency |
|---|---|---|---|---|---|---|---|
| semantic | 0.923 | 0.962 | 1.000 | 0.950 | 0.962 | 1.000 | 33 ms |
| lexical (BM25) | 0.808 | 1.000 | 1.000 | 0.897 | 0.924 | 1.000 | 2 ms |
| hybrid (RRF) | 0.923 | 1.000 | 1.000 | 0.962 | 0.972 | 1.000 | 33 ms |
| **hybrid + rerank** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 439 ms |

| Answers & hallucination resistance | value |
|---|---|
| citation accuracy (answered) | **1.000** |
| refusal rate on unanswerable questions | **1.000** (0 hallucinations) |
| mean grounding score | 0.984 |
| answer rate on answerable questions | 0.692 ¹ |

¹ Measured with the *extractive* test LLM, which can only copy sentences that
overlap the question, so it cannot answer paraphrased questions. Every one of
its false refusals is a generation-side limitation. Thanks to the two-signal
gate, **no answerable question was blocked by the retrieval gate**. Run
`--llm ollama` for the real answer rate on your hardware and model.

Dependency-free CI stack (hashing embedder + lexical reranker): hybrid + rerank
reaches recall@1 0.923, MRR 0.962, recall@3 1.0, and also 0 hallucinations.
Here hybrid fusion (0.885 recall@1) clearly beats both single retrievers
(semantic 0.846, BM25 0.808).

## What the evaluation changed

Building the harness surfaced real design issues. Each one is now pinned by a
regression test:

1. **A single-signal evidence gate was wrong.** The first design refused when
   the cross-encoder score was below a threshold. Calibration showed the
   cross-encoder scores paraphrases near zero ("vacation" vs "annual leave"
   scored 0.00005) while scoring the unanswerable "Phase III" question 0.999
   (it is very much *on topic*). No threshold separates the two. The gate now
   accepts a chunk when **either** the cross-encoder is at least 0.1 **or** the
   dense cosine is at least 0.5. Answerable evidence always scored ≥ 0.54 on
   cosine; off-topic questions scored ≤ 0.42. The gate only removes clearly
   irrelevant material and leaves answerability to the LLM and the verifier.
2. **Smaller, section-aligned chunks help the reranker.** At 220 tokens, mixed
   chunks diluted cross-encoder scores (governing-law question: 0.57). At 128
   tokens, aligned to headings, the same question scores 0.98.
3. **Citation repair could launder hallucinations.** An adversarial test with
   the answer "Employees receive 30 days of leave [9]" showed the stripped
   invalid citation being "repaired" onto a real source. Repair now runs only
   when the model wrote *no* citations at all, and never for sentences
   containing numbers that aren't in the matched source.

## Hallucination test suite

`backend/tests/evaluation/test_hallucination.py` scripts the classic failure
modes of an LLM and asserts the pipeline catches each one:

| Scripted model output | Expected outcome |
|---|---|
| World-knowledge answer without citations | refused (`no_supporting_citations`) |
| Cites a source number that doesn't exist (`[9]`) | refused, not repaired |
| Invents a source marker (`[Source 12]`) | refused |
| Empty output / "the documents do not contain…" | refused |
| Correct claim with a wrong figure ("27 days") | answered, but `confidence: low` and `unsupported_numbers: ["27"]` |
| Off-topic question | refused **before** the LLM is called (asserted with a spy) |
| Correct, cited answer | passes with high/medium confidence |

Plus `RUN_LIVE_EVAL=1 pytest -m live` runs the unanswerable set against a real
Ollama model.
