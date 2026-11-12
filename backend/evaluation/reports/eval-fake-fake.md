# DocIntel evaluation report

- **embedder**: `hashing-512`
- **reranker**: `lexical-coverage`
- **llm**: `fake-extractive`
- **min_relevance**: `0.35`
- **questions**: `26`
- **unanswerable**: `10`

## Ingestion

- **documents**: 4
- **pages**: 11
- **chunks**: 19
- **ingest_seconds**: 1.01

## Retrieval

| mode | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | page_acc | latency_ms |
|---|---|---|---|---|---|---|---|
| semantic | 0.846 | 0.962 | 1.000 | 0.905 | 0.929 | 1.000 | 2.143 |
| lexical (BM25) | 0.808 | 1.000 | 1.000 | 0.897 | 0.924 | 1.000 | 1.492 |
| hybrid (RRF) | 0.885 | 1.000 | 1.000 | 0.936 | 0.952 | 1.000 | 2.025 |
| hybrid + rerank | 0.923 | 1.000 | 1.000 | 0.962 | 0.972 | 1.000 | 3.435 |

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
- `hr-08` (false_refusal): low_retrieval_relevance
- `hr-09` (false_refusal): low_retrieval_relevance
- `hr-10` (false_refusal): low_retrieval_relevance
- `fx-05` (false_refusal): model_insufficient_evidence
- `fx-08` (false_refusal): low_retrieval_relevance
- `ct-06` (false_refusal): model_insufficient_evidence
- `ct-08` (false_refusal): low_retrieval_relevance
