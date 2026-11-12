# API reference

Base URL: `http://localhost:8000/api/v1`. Through the Docker frontend the same
API is served at `http://localhost:3000/api/v1`.

* Interactive docs (Swagger UI): <http://localhost:8000/docs>
* ReDoc: <http://localhost:8000/redoc>
* Machine-readable spec: [`openapi.json`](openapi.json), also served at
  `/api/v1/openapi.json`

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/health/ready` | Readiness: database, vector store, LLM (503 if a critical dependency is down) |
| `GET` | `/system/info` | Active models, retrieval and guardrail configuration |
| `POST` | `/documents` | Upload a PDF/DOCX/TXT (multipart `file`) → `202 Accepted`, processed asynchronously |
| `GET` | `/documents` | List documents (`limit`, `offset`, `status`, `search`) |
| `GET` | `/documents/stats` | Corpus statistics |
| `GET` | `/documents/{id}` | Document metadata and processing status |
| `GET` | `/documents/{id}/chunks` | Chunks with page ranges and sections |
| `GET` | `/documents/{id}/file` | Download the original file |
| `POST` | `/documents/{id}/reprocess` | Re-run ingestion (e.g. after changing the chunking or embedding model) |
| `DELETE` | `/documents/{id}` | Delete the document, its chunks, vectors and file |
| `POST` | `/chat` | Ask a question (full RAG pipeline, JSON response) |
| `POST` | `/chat/stream` | Same, streamed as Server-Sent Events |
| `POST` | `/search` | Retrieval only (semantic / lexical / hybrid, optional rerank) - debugging and evaluation |
| `GET` | `/conversations` | List conversations |
| `POST` | `/conversations` | Create an (optionally scoped) conversation |
| `GET` | `/conversations/{id}` | Conversation with messages, citations and diagnostics |
| `PATCH` | `/conversations/{id}` | Rename or change the document scope |
| `DELETE` | `/conversations/{id}` | Delete a conversation |

## Upload a document

```bash
curl -F "file=@contract.pdf" http://localhost:8000/api/v1/documents
```

```json
{
  "id": "6a4602d2-2653-456f-86e8-55db4b11c2f8",
  "filename": "contract.pdf",
  "file_type": "pdf",
  "status": "pending",
  "chunk_count": 0,
  "page_count": null,
  "...": "..."
}
```

Poll `GET /documents/{id}` until `status` is `ready` (or `failed`, with
`error_message`). Validation runs **before** anything is stored:

| Status | `error.code` | When |
|---|---|---|
| 415 | `unsupported_file_type` | Extension not allowed, content doesn't match the extension (magic bytes), or the file is empty |
| 413 | `file_too_large` | Exceeds `MAX_UPLOAD_SIZE_MB` (enforced while streaming) |
| 409 | `duplicate_document` | Same SHA-256 already uploaded (`details.document_id` points to it) |

## Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What platform availability does the provider guarantee?"}'
```

Request fields:

| Field | Type | Notes |
|---|---|---|
| `question` | string (1-4000) | Required |
| `conversation_id` | uuid | Continue a conversation (enables history and follow-up rewriting) |
| `document_ids` | uuid[] | Restrict retrieval to these documents. `[]` or omitted = all ready documents. Omitted in a conversation = the conversation's saved scope |
| `top_k` | int (1-20) | Number of chunks given to the LLM (default `RETRIEVAL_TOP_K`) |

Response (trimmed):

```json
{
  "conversation_id": "21a5ba69-445b-435c-a55c-b37fbad60bae",
  "message_id": "05a00054-2d0d-49ee-937b-ce49e6cadc3e",
  "answer": "The Provider guarantees a monthly platform availability of 99.5 percent, excluding scheduled maintenance announced at least 72 hours in advance [1].",
  "refused": false,
  "refusal_reason": null,
  "confidence": "high",
  "grounding_score": 1.0,
  "citations": [
    {
      "index": 1,
      "chunk_id": "6a4602d2-...:2",
      "document_id": "6a4602d2-2653-456f-86e8-55db4b11c2f8",
      "filename": "meridian_master_services_agreement.docx",
      "page_start": 2,
      "page_end": 2,
      "section": "3. Service Levels",
      "text": "3. Service Levels\n\nThe Provider guarantees a monthly platform availability of 99.5 percent, ...",
      "relevance": 0.9991,
      "cited": true
    },
    {
      "index": 2,
      "filename": "meridian_master_services_agreement.docx",
      "page_start": 1,
      "section": "1. Term",
      "relevance": 0.0643,
      "cited": false,
      "...": "..."
    }
  ],
  "retrieval": {
    "standalone_query": "What platform availability does the provider guarantee?",
    "mode": "hybrid",
    "candidates": 19,
    "used_chunks": 1,
    "top_relevance": 0.9991,
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "timings_ms": {"embed_ms": 36.5, "semantic_ms": 6.3, "lexical_ms": 0.1, "rerank_ms": 407.4, "generate_ms": 1850.2, "total_ms": 2310.6}
  },
  "model": "llama3.2:3b",
  "latency_ms": 2310.6
}
```

* `citations[].index` matches the `[n]` markers in `answer`. `cited: false`
  marks sources that were retrieved and shown to the model but not used.
* `confidence` is `high`, `medium`, `low` or `none`. It comes from the grounding
  score, any unsupported numbers, and retrieval relevance.
* When the documents don't support an answer, the response has
  `refused: true`, `confidence: "none"`, a fixed refusal message, and one of
  these `refusal_reason` values:

| `refusal_reason` | Layer |
|---|---|
| `no_documents` | No processed documents exist |
| `no_relevant_chunks` / `low_retrieval_relevance` | Evidence gate - the LLM was not called |
| `model_insufficient_evidence` | The LLM answered `INSUFFICIENT_EVIDENCE` |
| `no_supporting_citations` | The draft answer cited nothing valid (e.g. invented sources) |
| `weak_grounding` | `STRICT_GROUNDING=true` and grounding below the medium threshold |

## Streaming (`POST /chat/stream`)

Same request body. Response is `text/event-stream`:

```text
event: sources
data: {"standalone_query": "...", "refused_early": false, "sources": [{"index": 1, "filename": "...", "page_start": 2, "relevance": 0.99, ...}]}

event: token
data: {"t": "The Provider "}

event: token
data: {"t": "guarantees "}

event: done
data: { ...full ChatResponse, after guardrails... }
```

The `done` payload is authoritative: if a guardrail rejects the streamed draft,
`done.answer` contains the refusal and the UI replaces the streamed text.
Request errors (unknown conversation or documents) return normal HTTP errors
before the stream starts. Failures mid-stream (e.g. Ollama going down) arrive as
`event: error` with `{code, message, status}`.

## Retrieval only (`POST /search`)

```bash
curl -X POST http://localhost:8000/api/v1/search -H "Content-Type: application/json" \
  -d '{"query": "REBCO magnets", "mode": "lexical", "rerank": false, "top_k": 3}'
```

Each hit includes `semantic_rank`, `lexical_rank`, the fused or reranked
`score`, and the page and section. This is useful for seeing *why* a passage
was or wasn't retrieved.

## Errors

Every error uses one envelope, and every response carries an `X-Request-ID`
header that is also written to the logs:

```json
{
  "error": {
    "code": "document_not_found",
    "message": "Document 3f2a... not found",
    "details": {"document_id": "3f2a..."}
  },
  "request_id": "9c1e0b7f4a2d4e11"
}
```

| Status | Codes |
|---|---|
| 404 | `not_found`, `document_not_found`, `conversation_not_found` |
| 409 | `duplicate_document`, `document_not_ready` |
| 413 | `file_too_large` |
| 415 | `unsupported_file_type` |
| 422 | `validation_error` (with per-field `details`), `document_parsing_failed`, `empty_document` |
| 503 | `llm_unavailable` (message says e.g. `ollama pull llama3.2:3b`), `vector_store_unavailable` |
| 500 | `internal_error` (details are logged, never leaked) |
