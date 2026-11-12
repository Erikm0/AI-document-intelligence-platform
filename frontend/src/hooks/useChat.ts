import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, api, streamChat } from "../api/client";
import type { ChatResponse, Citation, Confidence, MessageRead, RetrievalInfo, StreamSource } from "../api/types";
import { keys } from "./queries";

export interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "done" | "streaming" | "error";
  citations: Citation[];
  refused: boolean;
  refusalReason?: string | null;
  confidence?: Confidence;
  groundingScore?: number;
  retrieval?: RetrievalInfo;
  latencyMs?: number;
  model?: string;
  error?: string;
  /** Evidence announced before generation starts (streaming only). */
  pendingSources?: StreamSource[];
}

let localSeq = 0;
// Globally unique so ids never collide (e.g. after a hot-module reload resets
// module state while component state survives).
const nextId = () =>
  globalThis.crypto?.randomUUID?.() ?? `local-${Date.now().toString(36)}-${++localSeq}`;

export function fromMessageRead(m: MessageRead): UiMessage {
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    status: "done",
    citations: m.citations ?? [],
    refused: m.refused,
    refusalReason: m.metadata?.refusal_reason,
    confidence: m.metadata?.confidence,
    groundingScore: m.metadata?.grounding_score,
    retrieval: m.metadata?.retrieval,
    latencyMs: m.metadata?.latency_ms,
    model: m.metadata?.model,
  };
}

function fromResponse(id: string, r: ChatResponse): UiMessage {
  return {
    id,
    role: "assistant",
    content: r.answer,
    status: "done",
    citations: r.citations,
    refused: r.refused,
    refusalReason: r.refusal_reason,
    confidence: r.confidence,
    groundingScore: r.grounding_score,
    retrieval: r.retrieval,
    latencyMs: r.latency_ms,
    model: r.model,
  };
}

interface Options {
  conversationId: string | undefined;
  onConversationCreated: (id: string, documentIds: string[]) => void;
}

export function useChat({ conversationId, onConversationCreated }: Options) {
  const qc = useQueryClient();
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [documentIds, setDocumentIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // Set when *we* created the conversation, so navigating to it does not reload.
  const adoptedRef = useRef<string | null>(null);

  useEffect(() => {
    if (!conversationId) {
      // "New chat": start empty. (The first message of a new chat streams while
      // the id is still undefined, so this effect does not fire mid-stream.)
      setMessages([]);
      setDocumentIds([]);
      adoptedRef.current = null;
      setLoadError(null);
      return;
    }
    if (adoptedRef.current === conversationId) return;
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    api
      .getConversation(conversationId)
      .then((c) => {
        if (cancelled) return;
        setMessages(c.messages.map(fromMessageRead));
        setDocumentIds(c.document_ids);
      })
      .catch((e: unknown) => !cancelled && setLoadError(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  const patch = useCallback((id: string, update: Partial<UiMessage> | ((m: UiMessage) => Partial<UiMessage>)) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...(typeof update === "function" ? update(m) : update) } : m)),
    );
  }, []);

  const send = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || streaming) return;
      const userMsg: UiMessage = { id: nextId(), role: "user", content: trimmed, status: "done", citations: [], refused: false };
      const assistantId = nextId();
      setMessages((prev) => [
        ...prev,
        userMsg,
        { id: assistantId, role: "assistant", content: "", status: "streaming", citations: [], refused: false },
      ]);
      setStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const stream = streamChat(
          { question: trimmed, conversation_id: conversationId ?? null, document_ids: documentIds },
          controller.signal,
        );
        for await (const event of stream) {
          if (event.event === "sources") {
            patch(assistantId, { pendingSources: event.data.sources });
          } else if (event.event === "token") {
            patch(assistantId, (m) => ({ content: m.content + event.data.t }));
          } else if (event.event === "done") {
            const final = event.data;
            patch(assistantId, fromResponse(assistantId, final));
            if (!conversationId) {
              adoptedRef.current = final.conversation_id;
              onConversationCreated(final.conversation_id, documentIds);
            }
            void qc.invalidateQueries({ queryKey: keys.conversations });
          } else if (event.event === "error") {
            patch(assistantId, { status: "error", error: event.data.message });
          }
        }
      } catch (e) {
        if ((e as Error).name === "AbortError") {
          patch(assistantId, (m) => ({ status: "error", error: "Generation stopped.", content: m.content }));
        } else {
          const message = e instanceof ApiError ? e.message : "Something went wrong while generating the answer.";
          patch(assistantId, { status: "error", error: message });
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [conversationId, documentIds, onConversationCreated, patch, qc, streaming],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  /** Clear the view for a new chat - also when the URL does not change. */
  const reset = useCallback(() => {
    abortRef.current?.abort();
    adoptedRef.current = null;
    setMessages([]);
    setDocumentIds([]);
    setLoadError(null);
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  return { messages, documentIds, setDocumentIds, send, stop, reset, streaming, loading, loadError };
}
