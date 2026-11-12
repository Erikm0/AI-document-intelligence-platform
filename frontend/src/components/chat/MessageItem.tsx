import { AlertTriangle, Bot, ShieldAlert } from "lucide-react";

import type { UiMessage } from "../../hooks/useChat";
import { formatMs, percent } from "../../lib/format";
import { AnswerText } from "./AnswerText";

const REFUSAL_REASONS: Record<string, string> = {
  no_documents: "No processed documents yet",
  no_relevant_chunks: "Nothing relevant was retrieved",
  low_retrieval_relevance: "Retrieved passages were not relevant enough",
  model_insufficient_evidence: "The model found the passages insufficient",
  no_supporting_citations: "The draft answer was not supported by any source",
  weak_grounding: "The draft answer was weakly grounded",
};

interface Props {
  message: UiMessage;
  selected: boolean;
  activeCitation: number | null;
  onSelect: () => void;
  onCite: (index: number) => void;
}

export function MessageItem({ message, selected, activeCitation, onSelect, onCite }: Props) {
  if (message.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  const streaming = message.status === "streaming";
  const valid = new Set(
    message.citations.length
      ? message.citations.map((c) => c.index)
      : (message.pendingSources ?? []).map((s) => s.index),
  );
  const citedCount = message.citations.filter((c) => c.cited).length;

  return (
    <div className="msg assistant">
      <div className="avatar">
        <Bot size={16} />
      </div>
      <div className="bubble">
        <div
          className={`answer${message.refused ? " refused" : ""}${selected ? " selected" : ""}`}
          onClick={onSelect}
          role="article"
        >
          {message.refused && (
            <div className="refusal-head">
              <ShieldAlert size={16} /> Not enough evidence
            </div>
          )}
          {streaming && !message.content ? (
            <div className="typing" aria-label="Searching documents">
              <span />
              <span />
              <span />
            </div>
          ) : (
            <div className={streaming ? "cursor" : undefined}>
              <AnswerText
                text={message.content}
                validIndices={streaming ? undefined : valid}
                activeIndex={selected ? activeCitation : null}
                onCite={(i) => {
                  onSelect();
                  onCite(i);
                }}
              />
            </div>
          )}
          {message.status === "error" && (
            <div className="banner danger" style={{ marginTop: 10 }}>
              <AlertTriangle size={16} /> {message.error}
            </div>
          )}
        </div>
        {message.status === "done" && (
          <div className="msg-meta">
            {message.confidence && (
              <span className={`badge ${message.confidence}`} title="Lexical grounding of the answer in its cited sources">
                {message.refused ? "refused" : `${message.confidence} confidence`}
                {!message.refused && message.groundingScore != null && ` · ${percent(message.groundingScore)} grounded`}
              </span>
            )}
            {message.refused && message.refusalReason && (
              <span>{REFUSAL_REASONS[message.refusalReason] ?? message.refusalReason}</span>
            )}
            {citedCount > 0 && (
              <span>
                {citedCount} source{citedCount === 1 ? "" : "s"} cited
              </span>
            )}
            {message.latencyMs != null && <span>{formatMs(message.latencyMs)}</span>}
            {message.model && <span>{message.model}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
