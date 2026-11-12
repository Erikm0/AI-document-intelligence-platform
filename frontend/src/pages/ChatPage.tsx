import { FileUp, ScanSearch } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { Composer } from "../components/chat/Composer";
import { ConversationSidebar } from "../components/chat/ConversationSidebar";
import { MessageItem } from "../components/chat/MessageItem";
import { ScopePicker } from "../components/chat/ScopePicker";
import { SourcePanel } from "../components/chat/SourcePanel";
import { useConversations, useDocuments } from "../hooks/queries";
import { useChat } from "../hooks/useChat";

const SUGGESTIONS = [
  "Summarise the key obligations and deadlines in these documents.",
  "What numbers or figures are reported, and where?",
  "List any risks or incidents that are mentioned.",
  "What does the document say about termination or cancellation?",
];

export function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const documents = useDocuments();
  const conversations = useConversations();
  const readyCount = documents.data?.items.filter((d) => d.status === "ready").length ?? 0;

  const onConversationCreated = useCallback(
    (id: string) => navigate(`/chat/${id}`, { replace: true }),
    [navigate],
  );
  const { messages, documentIds, setDocumentIds, send, stop, reset, streaming, loading, loadError } = useChat({
    conversationId,
    onConversationCreated,
  });

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Follow the newest assistant message unless the user picked another one.
  const lastAssistant = useMemo(() => [...messages].reverse().find((m) => m.role === "assistant") ?? null, [messages]);
  useEffect(() => {
    setSelectedId(lastAssistant?.id ?? null);
    setActiveCitation(null);
  }, [lastAssistant?.id]);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const selected = messages.find((m) => m.id === selectedId) ?? lastAssistant;
  const title = conversations.data?.items.find((c) => c.id === conversationId)?.title ?? "New conversation";

  return (
    <div className="chat">
      <ConversationSidebar
        activeId={conversationId}
        onSelect={(id) => navigate(`/chat/${id}`)}
        onNew={() => {
          reset();
          navigate("/chat");
        }}
      />

      <section className="chat-main">
        <header className="chat-header">
          <h2>{title}</h2>
          <ScopePicker
            documents={documents.data?.items ?? []}
            selected={documentIds}
            onChange={setDocumentIds}
            disabled={streaming}
          />
        </header>

        <div className="messages" ref={scrollRef}>
          <div className="messages-inner">
            {loading && <div className="muted">Loading conversation...</div>}
            {loadError && <div className="banner danger">{loadError}</div>}
            {!loading && messages.length === 0 && (
              <div className="empty" style={{ paddingTop: 72 }}>
                <div className="brand-mark" style={{ width: 48, height: 48, borderRadius: 14 }}>
                  <ScanSearch size={24} />
                </div>
                <h3>Ask your documents anything</h3>
                {readyCount === 0 ? (
                  <>
                    <span>Upload a PDF, DOCX or TXT file to get started.</span>
                    <Link to="/documents" className="btn btn-primary" style={{ marginTop: 8 }}>
                      <FileUp size={16} /> Upload documents
                    </Link>
                  </>
                ) : (
                  <>
                    <span>
                      Searching {documentIds.length ? `${documentIds.length} selected` : `all ${readyCount}`} document
                      {(documentIds.length || readyCount) === 1 ? "" : "s"}. Every answer cites the passages it relies on -
                      and says so when the documents don't contain the answer.
                    </span>
                    <div className="suggestions">
                      {SUGGESTIONS.map((s) => (
                        <button key={s} className="suggestion" onClick={() => void send(s)}>
                          {s}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
            {messages.map((m) => (
              <MessageItem
                key={m.id}
                message={m}
                selected={m.id === selected?.id}
                activeCitation={activeCitation}
                onSelect={() => setSelectedId(m.id)}
                onCite={setActiveCitation}
              />
            ))}
          </div>
        </div>

        <Composer onSend={(q) => void send(q)} onStop={stop} streaming={streaming} disabled={loading} />
      </section>

      <SourcePanel
        message={selected?.role === "assistant" ? selected : null}
        activeCitation={activeCitation}
        onActivate={setActiveCitation}
      />
    </div>
  );
}
