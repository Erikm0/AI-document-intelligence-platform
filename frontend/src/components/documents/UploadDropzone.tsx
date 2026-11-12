import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, FileUp, Loader2, X, XCircle } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { ApiError, uploadDocument } from "../../api/client";
import { keys } from "../../hooks/queries";
import { formatBytes } from "../../lib/format";
import { validateFile } from "../../lib/validation";

interface UploadState {
  id: number;
  name: string;
  size: number;
  progress: number;
  status: "uploading" | "done" | "error";
  error?: string;
}

interface Props {
  allowedExtensions?: string[];
  maxSizeMb?: number;
}

let uploadSeq = 0;

export function UploadDropzone({ allowedExtensions, maxSizeMb }: Props) {
  const qc = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploads, setUploads] = useState<UploadState[]>([]);

  const update = (id: number, patch: Partial<UploadState>) =>
    setUploads((prev) => prev.map((u) => (u.id === id ? { ...u, ...patch } : u)));

  const handleFiles = useCallback(
    (files: FileList | File[]) => {
      for (const file of Array.from(files)) {
        const id = ++uploadSeq;
        const invalid = validateFile(file, allowedExtensions, maxSizeMb);
        setUploads((prev) => [
          { id, name: file.name, size: file.size, progress: 0, status: invalid ? "error" : "uploading", error: invalid ?? undefined },
          ...prev,
        ]);
        if (invalid) continue;
        uploadDocument(file, (p) => update(id, { progress: p }))
          .then(() => {
            update(id, { status: "done", progress: 1 });
            void qc.invalidateQueries({ queryKey: keys.documents });
            setTimeout(() => setUploads((prev) => prev.filter((u) => u.id !== id)), 4000);
          })
          .catch((e: unknown) => {
            const message =
              e instanceof ApiError && e.code === "duplicate_document"
                ? `${e.message} - skipped`
                : e instanceof Error
                  ? e.message
                  : "Upload failed";
            update(id, { status: "error", error: message });
          });
      }
    },
    [allowedExtensions, maxSizeMb, qc],
  );

  return (
    <>
      <div
        className={`dropzone${dragging ? " active" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload documents"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <div className="dropzone-icon">
          <FileUp size={22} />
        </div>
        <div>
          <strong>Drop files here or click to upload</strong>
          <span className="muted">
            {(allowedExtensions ?? [".pdf", ".docx", ".txt"]).join(", ").toUpperCase().replaceAll(".", "")} · up to{" "}
            {maxSizeMb ?? 25} MB each · parsed, chunked and embedded locally
          </span>
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          hidden
          data-testid="file-input"
          accept={(allowedExtensions ?? [".pdf", ".docx", ".txt"]).join(",")}
          onChange={(e) => {
            if (e.target.files) handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {uploads.length > 0 && (
        <div className="uploads">
          {uploads.map((u) => (
            <div key={u.id} className="card upload-item">
              {u.status === "uploading" ? (
                <Loader2 size={18} className="spin muted" />
              ) : u.status === "done" ? (
                <CheckCircle2 size={18} color="var(--success)" />
              ) : (
                <XCircle size={18} color="var(--danger)" />
              )}
              <div style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 600 }}>
                  {u.name} <span className="muted">· {formatBytes(u.size)}</span>
                </div>
                {u.status === "error" ? (
                  <div className="error-text">{u.error}</div>
                ) : u.status === "done" ? (
                  <div className="muted">Uploaded - processing in the background</div>
                ) : (
                  <div className="progress">
                    <div style={{ width: `${Math.round(u.progress * 100)}%` }} />
                  </div>
                )}
              </div>
              <button
                className="icon-btn"
                aria-label="Dismiss"
                onClick={() => setUploads((prev) => prev.filter((x) => x.id !== u.id))}
              >
                <X size={16} />
              </button>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
