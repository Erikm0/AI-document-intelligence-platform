import { ChevronDown, Files } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { DocumentRead } from "../../api/types";
import { FileIcon } from "../StatusBadge";

interface Props {
  documents: DocumentRead[];
  selected: string[];
  onChange: (ids: string[]) => void;
  disabled?: boolean;
}

export function ScopePicker({ documents, selected, onChange, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const ready = documents.filter((d) => d.status === "ready");

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const label =
    selected.length === 0
      ? `All documents (${ready.length})`
      : selected.length === 1
        ? (ready.find((d) => d.id === selected[0])?.filename ?? "1 document")
        : `${selected.length} documents`;

  const toggle = (id: string) =>
    onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);

  return (
    <div className="scope" ref={ref}>
      <button
        className="btn btn-sm"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <Files size={14} />
        <span style={{ maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis" }}>Search: {label}</span>
        <ChevronDown size={14} />
      </button>
      {open && (
        <div className="scope-menu" role="listbox" aria-multiselectable>
          <label className="scope-option">
            <input type="checkbox" checked={selected.length === 0} onChange={() => onChange([])} />
            <span className="name">
              <strong>All documents</strong>
            </span>
          </label>
          <div className="scope-divider" />
          {ready.length === 0 && <div className="muted" style={{ padding: 10 }}>No processed documents yet.</div>}
          {ready.map((doc) => (
            <label key={doc.id} className="scope-option">
              <input type="checkbox" checked={selected.includes(doc.id)} onChange={() => toggle(doc.id)} />
              <FileIcon type={doc.file_type} />
              <span className="name" title={doc.title ?? doc.filename}>
                {doc.filename}
                <div className="muted" style={{ fontSize: 11 }}>
                  {doc.page_count != null ? `${doc.page_count} pages · ` : ""}
                  {doc.chunk_count} chunks
                </div>
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
