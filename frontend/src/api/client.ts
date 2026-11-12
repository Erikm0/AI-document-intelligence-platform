import { readSSE } from "./sse";
import type {
  ChatRequest,
  ChatResponse,
  ChunkRead,
  ConversationDetail,
  ConversationRead,
  DocumentRead,
  DocumentStats,
  HealthResponse,
  Page,
  StreamEvent,
  SystemInfo,
} from "./types";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api/v1";

/** Error with the backend's `{error: {code, message, details}}` envelope unpacked. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    const err = body?.error;
    if (err?.message) return new ApiError(err.message, response.status, err.code ?? "error", err.details);
  } catch {
    /* not JSON */
  }
  if (response.status === 502 || response.status === 503 || response.status === 504) {
    return new ApiError(
      "The DocIntel API is not reachable right now (it may still be starting up). Please try again shortly.",
      response.status,
      "api_unavailable",
    );
  }
  return new ApiError(`Request failed with HTTP ${response.status}`, response.status, "http_error");
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
    });
  } catch {
    throw new ApiError("Cannot reach the DocIntel API. Is the backend running?", 0, "network_error");
  }
  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function query(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  }
  const s = search.toString();
  return s ? `?${s}` : "";
}

export const api = {
  health: () => request<HealthResponse>("/health/ready"),
  systemInfo: () => request<SystemInfo>("/system/info"),

  listDocuments: (params: { limit?: number; offset?: number; status?: string; search?: string } = {}) =>
    request<Page<DocumentRead>>(`/documents${query({ limit: 200, ...params })}`),
  documentStats: () => request<DocumentStats>("/documents/stats"),
  getDocument: (id: string) => request<DocumentRead>(`/documents/${id}`),
  listChunks: (id: string, limit = 500) => request<Page<ChunkRead>>(`/documents/${id}/chunks${query({ limit })}`),
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  reprocessDocument: (id: string) => request<DocumentRead>(`/documents/${id}/reprocess`, { method: "POST" }),
  fileUrl: (id: string) => `${API_BASE}/documents/${id}/file`,

  listConversations: () => request<Page<ConversationRead>>("/conversations?limit=100"),
  getConversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: "DELETE" }),
  renameConversation: (id: string, title: string) =>
    request<ConversationRead>(`/conversations/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),

  chat: (body: ChatRequest) => request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(body) }),
};

/**
 * Upload with progress reporting. `fetch` cannot report upload progress, so this
 * uses XMLHttpRequest.
 */
export function uploadDocument(
  file: File,
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<DocumentRead> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/documents`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as DocumentRead);
      } else {
        const err = body?.error;
        reject(new ApiError(err?.message ?? `Upload failed (HTTP ${xhr.status})`, xhr.status, err?.code ?? "upload_failed", err?.details));
      }
    };
    xhr.onerror = () => reject(new ApiError("Network error during upload", 0, "network_error"));
    xhr.onabort = () => reject(new ApiError("Upload cancelled", 0, "aborted"));
    signal?.addEventListener("abort", () => xhr.abort());
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

/** POST /chat/stream and yield typed events. Throws ApiError for HTTP-level failures. */
export async function* streamChat(body: ChatRequest, signal?: AbortSignal): AsyncGenerator<StreamEvent> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Cannot reach the DocIntel API. Is the backend running?", 0, "network_error");
  }
  if (!response.ok || !response.body) throw await toApiError(response);
  for await (const raw of readSSE(response.body)) {
    yield { event: raw.event, data: JSON.parse(raw.data) } as StreamEvent;
  }
}
