# DocIntel evaluation report

- **embedder**: `BAAI/bge-small-en-v1.5`
- **reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **llm**: `fake-extractive`
- **min_relevance**: `0.1`
- **questions**: `26`
- **unanswerable**: `10`

## Ingestion

- **documents**: 4
- **pages**: 11
- **chunks**: 19
- **ingest_seconds**: 1.76

## Retrieval

| mode | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | page_acc | latency_ms |
|---|---|---|---|---|---|---|---|
| semantic | 0.923 | 0.962 | 1.000 | 0.950 | 0.962 | 1.000 | 32.817 |
| lexical (BM25) | 0.808 | 1.000 | 1.000 | 0.897 | 0.924 | 1.000 | 1.862 |
| hybrid (RRF) | 0.923 | 1.000 | 1.000 | 0.962 | 0.972 | 1.000 | 32.835 |
| hybrid + rerank | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 438.731 |

## Answers & hallucination resistance

| metric | value |
|---|---|
| answer_rate (answerable) | 0.692 |
| citation_accuracy (answered) | 1.000 |
| keyword_accuracy (answered) | 0.944 |
| mean_grounding_score | 0.984 |
| refusal_rate (unanswerable) | 1.000 |
| hallucination_rate (unanswerable) | 0.000 |

## Failures

- `hr-05` (false_refusal): model_insufficient_evidence
- `hr-08` (false_refusal): model_insufficient_evidence
- `hr-09` (false_refusal): model_insufficient_evidence
- `hr-10` (false_refusal): model_insufficient_evidence
- `fx-05` (false_refusal): model_insufficient_evidence
- `fx-08` (false_refusal): model_insufficient_evidence
- `ct-06` (false_refusal): model_insufficient_evidence
- `ct-08` (false_refusal): model_insufficient_evidence
