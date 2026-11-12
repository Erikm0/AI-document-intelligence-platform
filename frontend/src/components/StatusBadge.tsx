import { AlertCircle, CheckCircle2, Clock, Loader2 } from "lucide-react";

import type { DocumentStatus } from "../api/types";

const LABELS: Record<DocumentStatus, string> = {
  pending: "Queued",
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
};

export function StatusBadge({ status, error }: { status: DocumentStatus; error?: string | null }) {
  const Icon =
    status === "ready" ? CheckCircle2 : status === "failed" ? AlertCircle : status === "processing" ? Loader2 : Clock;
  return (
    <span className={`badge ${status}`} title={error ?? undefined}>
      <Icon size={12} className={status === "processing" ? "spin" : undefined} />
      {LABELS[status]}
    </span>
  );
}

export function FileIcon({ type }: { type: string }) {
  return <span className={`file-icon ${type}`}>{type.toUpperCase()}</span>;
}
