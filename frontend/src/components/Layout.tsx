import { BookOpenText, FileStack, MessagesSquare, ScanSearch } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

import { useHealth, useSystemInfo } from "../hooks/queries";

function HealthCard() {
  const health = useHealth();
  const info = useSystemInfo();
  const llm = health.data?.components.llm;
  const apiStatus = health.isError ? "down" : (health.data?.status ?? "degraded");
  return (
    <div className="health-card" aria-live="polite">
      <div className="health-row" title={health.error?.message}>
        <span className={`dot ${apiStatus}`} />
        API {health.isError ? "unreachable" : apiStatus === "ok" ? "healthy" : apiStatus}
      </div>
      <div className="health-row" title={llm?.detail ?? undefined}>
        <span className={`dot ${llm?.status ?? (health.isError ? "down" : "")}`} />
        LLM {info.data?.llm_model ?? "-"}
      </div>
      {info.data && (
        <div className="health-row muted" title={`Reranker: ${info.data.reranker}`}>
          <span className="dot ok" />
          {info.data.embedding_model.split("/").pop()}
        </div>
      )}
    </div>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ScanSearch size={17} />
          </div>
          <span>
            DocIntel
            <small>Document Intelligence</small>
          </span>
        </div>
        <NavLink to="/chat" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
          <MessagesSquare size={18} />
          <span>Ask</span>
        </NavLink>
        <NavLink to="/documents" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
          <FileStack size={18} />
          <span>Documents</span>
        </NavLink>
        <div className="sidebar-footer">
          <a className="nav-link" href="/docs" target="_blank" rel="noreferrer">
            <BookOpenText size={18} />
            <span>API docs</span>
          </a>
          <HealthCard />
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
