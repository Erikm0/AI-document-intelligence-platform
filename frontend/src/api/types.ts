// Mirrors the backend Pydantic schemas (app/schemas/*.py).

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";
export type FileType = "pdf" | "docx" | "txt";
export type Confidence = "high" | "medium" | "low" | "none";

export interface DocumentRead {
  id: string;
  filename: string;
  file_type: FileType;
  content_type: string;
  size_bytes: number;
  sha256: string;
  status: DocumentStatus;
  error_message: string | null;
  title: string | null;
  page_count: number | null;
  chunk_count: number;
  char_count: number;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  processed_at: string | null;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface DocumentStats {
  total_documents: number;
  ready_documents: number;
  processing_documents: number;
  failed_documents: number;
  total_chunks: number;
  total_pages: number;
  total_size_bytes: number;
}

export interface ChunkRead {
  id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  page_start: number | null;
  page_end: number | null;
  section: string | null;
  token_count: number;
}

export interface Citation {
  index: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_start: number | null;
  page_end: number | null;
  section: string | null;
  text: string;
  relevance: number;
  cited: boolean;
}

export interface RetrievalInfo {
  standalone_query: string;
  mode: string;
  candidates: number;
  used_chunks: number;
  top_relevance: number;
  reranker: string;
  timings_ms: Record<string, number>;
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  answer: string;
  refused: boolean;
  refusal_reason: string | null;
  confidence: Confidence;
  grounding_score: number;
  citations: Citation[];
  retrieval: RetrievalInfo;
  model: string;
  latency_ms: number;
}

export interface ChatRequest {
  question: string;
  conversation_id?: string | null;
  document_ids?: string[] | null;
  top_k?: number;
}

export interface MessageRead {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  refused: boolean;
  metadata: {
    confidence?: Confidence;
    grounding_score?: number;
    refusal_reason?: string | null;
    retrieval?: RetrievalInfo;
    model?: string;
    latency_ms?: number;
  };
  created_at: string;
}

export interface ConversationRead {
  id: string;
  title: string;
  document_ids: string[];
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationDetail extends ConversationRead {
  messages: MessageRead[];
}

export interface ComponentHealth {
  status: "ok" | "degraded" | "down";
  detail: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "down";
  version: string;
  components: Record<string, ComponentHealth>;
}

export interface SystemInfo {
  version: string;
  environment: string;
  llm_provider: string;
  llm_model: string;
  embedding_provider: string;
  embedding_model: string;
  reranker: string;
  vector_store: string;
  chunk_size_tokens: number;
  chunk_overlap_tokens: number;
  retrieval_top_k: number;
  min_relevance_score: number;
  min_semantic_score: number | null;
  max_upload_size_mb: number;
  allowed_extensions: string[];
}

export interface StreamSource {
  index: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_start: number | null;
  page_end: number | null;
  section: string | null;
  relevance: number;
}

export type StreamEvent =
  | { event: "sources"; data: { standalone_query: string; refused_early: boolean; sources: StreamSource[] } }
  | { event: "token"; data: { t: string } }
  | { event: "done"; data: ChatResponse }
  | { event: "error"; data: { code: string; message: string; status: number } };
