import { BookMarked, ChevronDown, ChevronUp, Download } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api } from "../../api/client";
import type { Citation, StreamSource } from "../../api/types";
import type { UiMessage } from "../../hooks/useChat";
import { formatMs, pageLabel, percent } from "../../lib/format";

interface SourceCardProps {
  source: Citation | (StreamSource & { text?: string; cited?: boolean });
  active: boolean;
  onActivate: () => void;
}

function SourceCard({ source, active, onActivate }: SourceCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (active) {
      setExpanded(true);
      ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [active]);

  const location = [pageLabel(source.page_start, source.page_end), source.section].filter(Boolean).join(" · ");
  return (
    <div
      ref={ref}
      className={`source${active ? " active" : ""}${source.cited === false ? " uncited" : ""}`}
      onClick={onActivate}
      data-testid={`source-${source.index}`}
    >
      <div className="source-head">
        <span className="source-index">{source.index}</span>
        <span className="source-file" title={source.filename}>
          {source.filename}
        </span>
        <a
          className="icon-btn"
          href={api.fileUrl(source.document_id)}
          title="Open original document"
          onClick={(e) => e.stopPropagation()}
        >
          <Download size={14} />
        </a>
      </div>
      {location && <div className="source-loc">{location}</div>}
      {source.text && (
        <>
          <div className={`source-text${expanded ? "" : " clamped"}`}>{source.text}</div>
          <button
            className="btn btn-ghost btn-sm"
            style={{ paddingLeft: 0 }}
            onClick={(e) => {
              e.stopPropagation();
              setExpanded((v) => !v);
            }}
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {expanded ? "Show less" : "Show passage"}
          </button>
        </>
      )}
      <div className="relevance" title="Cross-encoder relevance of this passage to the question">
        relevance
        <div className="bar">
          <div style={{ width: percent(Math.max(0.02, source.relevance)) }} />
        </div>
        {percent(source.relevance)}
      </div>
    </div>
  );
}

interface Props {
  message: UiMessage | null;
  activeCitation: number | null;
  onActivate: (index: number) => void;
}

export function SourcePanel({ message, activeCitation, onActivate }: Props) {
  const citations = message?.citations ?? [];
  const cited = citations.filter((c) => c.cited);
  const other = citations.filter((c) => !c.cited);
  const pending = !citations.length ? (message?.pendingSources ?? []) : [];

  return (
    <aside className="sources" aria-label="Sources">
      <div className="sources-header">
        <h3>
          <BookMarked size={16} /> Sources
        </h3>
        <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>
          {message ? "Evidence behind the selected answer" : "Select an answer to see its evidence"}
        </div>
      </div>
      <div className="sources-body">
        {!message && <div className="muted">No answer selected.</div>}

        {pending.length > 0 && (
          <>
            <div className="section-label">Retrieved - generating answer...</div>
            {pending.map((s) => (
              <SourceCard key={s.chunk_id} source={s} active={false} onActivate={() => onActivate(s.index)} />
            ))}
          </>
        )}

        {cited.length > 0 && (
          <>
            <div className="section-label">Cited in the answer</div>
            {cited.map((c) => (
              <SourceCard key={c.chunk_id} source={c} active={activeCitation === c.index} onActivate={() => onActivate(c.index)} />
            ))}
          </>
        )}

        {other.length > 0 && (
          <>
            <div className="section-label">{message?.refused ? "Closest passages found" : "Also retrieved"}</div>
            {other.map((c) => (
              <SourceCard key={c.chunk_id} source={c} active={activeCitation === c.index} onActivate={() => onActivate(c.index)} />
            ))}
          </>
        )}

        {message && message.status === "done" && citations.length === 0 && (
          <div className="muted">No passages were relevant enough to show.</div>
        )}

        {message?.retrieval && (
          <>
            <div className="section-label">Retrieval diagnostics</div>
            <dl className="diagnostics">
              <dt>Search query</dt>
              <dd>{message.retrieval.standalone_query}</dd>
              <dt>Candidates</dt>
              <dd>
                {message.retrieval.candidates} fused (dense + BM25) → {message.retrieval.reranker}
              </dd>
              <dt>Top relevance</dt>
              <dd>{percent(message.retrieval.top_relevance)}</dd>
              {message.groundingScore != null && !message.refused && (
                <>
                  <dt>Grounding</dt>
                  <dd>{percent(message.groundingScore)} of answer terms found in cited sources</dd>
                </>
              )}
              <dt>Latency</dt>
              <dd>
                {formatMs(message.retrieval.timings_ms.total_ms)}
                {message.retrieval.timings_ms.generate_ms != null &&
                  ` (LLM ${formatMs(message.retrieval.timings_ms.generate_ms)}, rerank ${formatMs(message.retrieval.timings_ms.rerank_ms)})`}
              </dd>
            </dl>
          </>
        )}
      </div>
    </aside>
  );
}
