import { X } from "lucide-react";
import { useEffect } from "react";

import type { DocumentRead } from "../../api/types";
import { useChunks } from "../../hooks/queries";
import { pageLabel } from "../../lib/format";

export function ChunkDrawer({ document, onClose }: { document: DocumentRead; onClose: () => void }) {
  const chunks = useChunks(document.id);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const meta = document.metadata as Record<string, unknown>;
  return (
    <>
      <div className="overlay" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label={`Chunks of ${document.filename}`}>
        <div className="drawer-header">
          <div>
            <h2>{document.title ?? document.filename}</h2>
            <div className="muted" style={{ fontSize: 12 }}>
              {document.filename} · {document.chunk_count} chunks
              {document.page_count != null && ` · ${document.page_count} pages`}
              {typeof meta.avg_chunk_tokens === "number" && ` · ~${meta.avg_chunk_tokens} tokens/chunk`}
              {typeof meta.page_numbering === "string" && ` · page numbers: ${meta.page_numbering.replace("_", " ")}`}
            </div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="drawer-body">
          {chunks.isLoading && <div className="muted">Loading chunks...</div>}
          {chunks.error && <div className="error-text">{chunks.error.message}</div>}
          {chunks.data?.items.map((chunk) => (
            <div key={chunk.id} className="card chunk">
              <div className="chunk-meta">
                <span className="badge">#{chunk.chunk_index + 1}</span>
                {chunk.page_start != null && <span className="badge">{pageLabel(chunk.page_start, chunk.page_end)}</span>}
                {chunk.section && <span className="badge" title={chunk.section}>{chunk.section.slice(0, 48)}</span>}
                <span className="badge">{chunk.token_count} tokens</span>
              </div>
              <div className="chunk-text">{chunk.text}</div>
            </div>
          ))}
        </div>
      </aside>
    </>
  );
}
