import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../api/client";
import type { DocumentRead } from "../api/types";

export const keys = {
  documents: ["documents"] as const,
  stats: ["documents", "stats"] as const,
  chunks: (id: string) => ["documents", id, "chunks"] as const,
  conversations: ["conversations"] as const,
  conversation: (id: string) => ["conversations", id] as const,
  health: ["health"] as const,
  system: ["system"] as const,
};

const isBusy = (d: DocumentRead) => d.status === "pending" || d.status === "processing";

export function useDocuments() {
  return useQuery({
    queryKey: keys.documents,
    queryFn: () => api.listDocuments(),
    // Poll while anything is still being ingested.
    refetchInterval: (query) => (query.state.data?.items.some(isBusy) ? 1500 : false),
  });
}

export function useDocumentStats(pollWhileBusy: boolean) {
  return useQuery({
    queryKey: keys.stats,
    queryFn: api.documentStats,
    refetchInterval: pollWhileBusy ? 1500 : false,
  });
}

export function useChunks(documentId: string | null) {
  return useQuery({
    queryKey: keys.chunks(documentId ?? ""),
    queryFn: () => api.listChunks(documentId!),
    enabled: Boolean(documentId),
  });
}

export function useSystemInfo() {
  return useQuery({ queryKey: keys.system, queryFn: api.systemInfo, staleTime: 5 * 60_000 });
}

export function useHealth() {
  return useQuery({ queryKey: keys.health, queryFn: api.health, refetchInterval: 20_000, retry: false });
}

export function useConversations() {
  return useQuery({ queryKey: keys.conversations, queryFn: api.listConversations });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.deleteDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.documents }),
  });
}

export function useReprocessDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.reprocessDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.documents }),
  });
}

export function useDeleteConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.deleteConversation,
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.conversations }),
  });
}
