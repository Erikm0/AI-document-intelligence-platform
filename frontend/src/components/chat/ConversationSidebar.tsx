import { MessageSquare, Plus, Trash2 } from "lucide-react";

import { useConversations, useDeleteConversation } from "../../hooks/queries";
import { relativeTime } from "../../lib/format";

interface Props {
  activeId: string | undefined;
  onSelect: (id: string) => void;
  onNew: () => void;
}

export function ConversationSidebar({ activeId, onSelect, onNew }: Props) {
  const conversations = useConversations();
  const remove = useDeleteConversation();

  return (
    <nav className="conversations" aria-label="Conversations">
      <div className="conversations-header">
        <button className="btn btn-primary" style={{ width: "100%" }} onClick={onNew}>
          <Plus size={16} /> New chat
        </button>
      </div>
      <div className="conversation-list">
        {conversations.isLoading && <div className="muted" style={{ padding: 10 }}>Loading...</div>}
        {conversations.data?.items.length === 0 && (
          <div className="muted" style={{ padding: 10, fontSize: 13 }}>
            Your conversations will appear here.
          </div>
        )}
        {conversations.data?.items.map((c) => (
          <div
            key={c.id}
            className={`conversation${c.id === activeId ? " active" : ""}`}
            onClick={() => onSelect(c.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && onSelect(c.id)}
          >
            <MessageSquare size={15} style={{ flex: "none" }} />
            <div className="label">
              <div title={c.title}>{c.title}</div>
              <small>
                {relativeTime(c.updated_at)} · {Math.ceil(c.message_count / 2)} question{c.message_count === 2 ? "" : "s"}
              </small>
            </div>
            <button
              className="icon-btn danger"
              aria-label={`Delete conversation ${c.title}`}
              onClick={(e) => {
                e.stopPropagation();
                if (window.confirm("Delete this conversation?")) {
                  remove.mutate(c.id, { onSuccess: () => c.id === activeId && onNew() });
                }
              }}
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
    </nav>
  );
}
