import { Blocks, Download, Eye, FileText, HardDrive, Layers, RotateCcw, Search, Trash2 } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";

import { api } from "../api/client";
import type { DocumentRead } from "../api/types";
import { ChunkDrawer } from "../components/documents/ChunkDrawer";
import { UploadDropzone } from "../components/documents/UploadDropzone";
import { FileIcon, StatusBadge } from "../components/StatusBadge";
import { useDeleteDocument, useDocumentStats, useDocuments, useReprocessDocument, useSystemInfo } from "../hooks/queries";
import { formatBytes, relativeTime } from "../lib/format";

function Stat({ icon, label, value }: { icon: ReactNode; label: string; value: string | number }) {
  return (
    <div className="card stat">
      <div className="stat-label">
        {icon}
        {label}
      </div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

export function DocumentsPage() {
  const documents = useDocuments();
  const busy = documents.data?.items.some((d) => d.status === "pending" || d.status === "processing") ?? false;
  const stats = useDocumentStats(busy);
  const info = useSystemInfo();
  const remove = useDeleteDocument();
  const reprocess = useReprocessDocument();
  const [filter, setFilter] = useState("");
  const [inspecting, setInspecting] = useState<DocumentRead | null>(null);

  const rows = useMemo(() => {
    const items = documents.data?.items ?? [];
    const q = filter.trim().toLowerCase();
    return q ? items.filter((d) => `${d.filename} ${d.title ?? ""}`.toLowerCase().includes(q)) : items;
  }, [documents.data, filter]);

  const confirmDelete = (doc: DocumentRead) => {
    if (window.confirm(`Delete "${doc.filename}"? Its chunks and vectors will be removed.`)) remove.mutate(doc.id);
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Documents</h1>
          <p>Upload PDF, DOCX and TXT files. Each file is parsed, cleaned, chunked with page provenance and indexed for hybrid search.</p>
        </div>
      </div>

      <div className="stats">
        <Stat icon={<FileText size={14} />} label="Documents" value={stats.data?.total_documents ?? "–"} />
        <Stat icon={<Layers size={14} />} label="Ready" value={stats.data ? `${stats.data.ready_documents}${stats.data.processing_documents ? ` (+${stats.data.processing_documents})` : ""}` : "–"} />
        <Stat icon={<Blocks size={14} />} label="Indexed chunks" value={stats.data?.total_chunks ?? "–"} />
        <Stat icon={<FileText size={14} />} label="Pages" value={stats.data?.total_pages ?? "–"} />
        <Stat icon={<HardDrive size={14} />} label="Storage" value={stats.data ? formatBytes(stats.data.total_size_bytes) : "–"} />
      </div>

      <UploadDropzone allowedExtensions={info.data?.allowed_extensions} maxSizeMb={info.data?.max_upload_size_mb} />

      <div className="card">
        <div className="toolbar">
          <strong>Library</strong>
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Search size={15} className="muted" />
            <input
              className="search-input"
              placeholder="Filter by name..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              aria-label="Filter documents"
            />
          </label>
        </div>

        {documents.isLoading ? (
          <div className="empty">Loading documents...</div>
        ) : documents.error ? (
          <div className="empty error-text">{documents.error.message}</div>
        ) : rows.length === 0 ? (
          <div className="empty">
            <FileText size={28} />
            <h3>{filter ? "No matching documents" : "No documents yet"}</h3>
            <span>{filter ? "Try a different filter." : "Upload a file above to start asking questions about it."}</span>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="docs">
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Status</th>
                  <th>Pages</th>
                  <th>Chunks</th>
                  <th>Size</th>
                  <th>Uploaded</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((doc) => (
                  <tr key={doc.id}>
                    <td>
                      <div className="doc-name">
                        <FileIcon type={doc.file_type} />
                        <div style={{ minWidth: 0 }}>
                          <div className="title">{doc.title && doc.title !== doc.filename ? doc.title : doc.filename}</div>
                          <div className="sub">
                            {doc.title && doc.title !== doc.filename ? doc.filename : doc.content_type}
                          </div>
                          {doc.status === "failed" && doc.error_message && (
                            <div className="sub error-text">{doc.error_message}</div>
                          )}
                        </div>
                      </div>
                    </td>
                    <td>
                      <StatusBadge status={doc.status} error={doc.error_message} />
                    </td>
                    <td>{doc.page_count ?? "–"}</td>
                    <td>{doc.status === "ready" ? doc.chunk_count : "–"}</td>
                    <td>{formatBytes(doc.size_bytes)}</td>
                    <td title={new Date(doc.created_at).toLocaleString()}>{relativeTime(doc.created_at)}</td>
                    <td>
                      <div className="actions">
                        <button
                          className="icon-btn"
                          title="Inspect chunks"
                          aria-label={`Inspect chunks of ${doc.filename}`}
                          disabled={doc.status !== "ready"}
                          onClick={() => setInspecting(doc)}
                        >
                          <Eye size={16} />
                        </button>
                        <a className="icon-btn" title="Download original" href={api.fileUrl(doc.id)} aria-label={`Download ${doc.filename}`}>
                          <Download size={16} />
                        </a>
                        {doc.status === "failed" && (
                          <button
                            className="icon-btn"
                            title="Reprocess"
                            aria-label={`Reprocess ${doc.filename}`}
                            onClick={() => reprocess.mutate(doc.id)}
                          >
                            <RotateCcw size={16} />
                          </button>
                        )}
                        <button
                          className="icon-btn danger"
                          title="Delete"
                          aria-label={`Delete ${doc.filename}`}
                          onClick={() => confirmDelete(doc)}
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {inspecting && <ChunkDrawer document={inspecting} onClose={() => setInspecting(null)} />}
    </div>
  );
}
