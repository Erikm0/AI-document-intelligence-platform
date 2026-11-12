# Demo walkthrough

A 5-minute tour that shows the main capabilities. It uses the synthetic
evaluation documents, which describe fictional organisations, so you can verify
that answers come from the documents and not from the model's memory.

## 1. Start the stack

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f ollama-init   # wait for "success" (first run downloads the LLM)
```

Open <http://localhost:3000>. The sidebar health card should show **API
healthy** and the LLM model with a green dot.

> No Docker? Run the backend with `LLM_PROVIDER=fake` (see the README). The fake
> extractive LLM answers by quoting source sentences, which is enough to
> explore retrieval, citations and refusals without Ollama.

## 2. Generate and upload the demo documents

```bash
cd backend
python -c "from pathlib import Path; from evaluation.corpus import build_corpus; build_corpus(Path('demo'))"
```

This creates four files in `backend/demo/`:

* `aurora_employee_handbook.txt`
* `helios7_phase2_report.pdf`
* `meridian_master_services_agreement.docx`
* `borealis_contractor_guide.txt`

Drag them onto the **Documents** page, or upload them with curl:

```bash
for f in demo/*; do curl -s -F "file=@$f" http://localhost:8000/api/v1/documents > /dev/null; done
```

Watch each status go *Queued → Processing → Ready*. The stat cards update live
(4 documents, 11 pages, ~19 chunks). Click the **eye** icon on a document to see
its chunks, each with its page number, section path and token count.

## 3. Ask questions

Go to **Ask** and try these:

| Question | What to notice |
|---|---|
| *How much did Phase II of the Helios-7 project cost?* | Answer with a `[1]` chip. The source panel shows `helios7_phase2_report.pdf · p. 3 · Budget` at ~100% relevance |
| *And what caused the unplanned shutdowns?* | A follow-up: the query is rewritten using the conversation (see "Search query" in the diagnostics) |
| *What is the daily limit for meal expenses at Aurora Robotics?* | The distractor guide says 30 €/day for Borealis. The answer must cite the Aurora handbook (45 €) |
| *What does Ledgerly do?* | An exact rare term - BM25 finds it, dense retrieval alone may not |
| *Can I spend some weeks working from abroad?* | A paraphrase - dense retrieval bridges "abroad" ↔ "another country" |
| *When will the Helios-7 Phase III campaign start?* | **Refused**: the report only covers Phase II. The closest passages are still shown for transparency |
| *What is the capital of France?* | **Refused before the LLM is called** - nothing in the documents is relevant |

Click any `[n]` chip to highlight its passage in the **Sources** panel. Expand
**Show passage** to read the exact chunk the answer relied on. The diagnostics
block lists the standalone search query, the number of fused candidates, the
reranker, the grounding score, and latency per stage.

## 4. Scope the search

Use **Search: All documents ▾** in the chat header to restrict a conversation to
the contract only, then ask *How many days of paid annual leave do employees
get?* It is refused, because that fact lives in the handbook. Widen the scope
again and it is answered with a citation. The scope is saved with the
conversation.

## 5. Inspect the API

* <http://localhost:8000/docs> - Swagger UI for every endpoint.
* `POST /api/v1/search` with `"mode": "semantic"`, `"lexical"` or `"hybrid"`
  shows the per-retriever ranks behind a result.
* `GET /api/v1/health/ready` reports database, vector store, LLM and BM25
  index status.

## 6. Run the evaluation

```bash
cd backend
python -m evaluation.run --models real --llm ollama
```

This prints retrieval metrics per mode plus the answer and hallucination
metrics for *your* local model (see [evaluation.md](evaluation.md)).
