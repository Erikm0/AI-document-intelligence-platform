import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";

import type { Citation } from "../api/types";
import { AnswerText } from "../components/chat/AnswerText";
import { MessageItem } from "../components/chat/MessageItem";
import { ScopePicker } from "../components/chat/ScopePicker";
import { SourcePanel } from "../components/chat/SourcePanel";
import { UploadDropzone } from "../components/documents/UploadDropzone";
import type { UiMessage } from "../hooks/useChat";

const citation = (index: number, cited = true): Citation => ({
  index,
  chunk_id: `doc:${index}`,
  document_id: "11111111-1111-1111-1111-111111111111",
  filename: `report-${index}.pdf`,
  page_start: index + 1,
  page_end: index + 1,
  section: "Results",
  text: `Passage number ${index}`,
  relevance: 0.9,
  cited,
});

const answer: UiMessage = {
  id: "a1",
  role: "assistant",
  content: "Ice was found in 14 craters [1].",
  status: "done",
  citations: [citation(1), citation(2, false)],
  refused: false,
  confidence: "high",
  groundingScore: 0.92,
  latencyMs: 1200,
  model: "llama3.2:3b",
};

function withQuery(ui: ReactNode) {
  return <QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>;
}

describe("AnswerText", () => {
  it("renders citation chips and reports clicks", async () => {
    const onCite = vi.fn();
    render(<AnswerText text="Fact one [1]. Bad ref [7]." validIndices={new Set([1])} onCite={onCite} />);
    await userEvent.click(screen.getByRole("button", { name: "Show source 1" }));
    expect(onCite).toHaveBeenCalledWith(1);
    const invalid = screen.getByRole("button", { name: "Show source 7" });
    expect(invalid).toHaveClass("invalid");
    await userEvent.click(invalid);
    expect(onCite).toHaveBeenCalledTimes(1);
  });
});

describe("MessageItem", () => {
  it("shows confidence, grounding and cited-source count", () => {
    render(<MessageItem message={answer} selected activeCitation={null} onSelect={() => {}} onCite={() => {}} />);
    expect(screen.getByText(/high confidence/)).toHaveTextContent("92% grounded");
    expect(screen.getByText("1 source cited")).toBeInTheDocument();
  });

  it("renders refusals distinctly with the reason", () => {
    const refused: UiMessage = {
      ...answer,
      content: "I couldn't find enough evidence in the selected documents.",
      refused: true,
      refusalReason: "low_retrieval_relevance",
      confidence: "none",
      citations: [],
    };
    render(<MessageItem message={refused} selected={false} activeCitation={null} onSelect={() => {}} onCite={() => {}} />);
    expect(screen.getByText("Not enough evidence")).toBeInTheDocument();
    expect(screen.getByText("Retrieved passages were not relevant enough")).toBeInTheDocument();
  });

  it("shows a typing indicator while waiting for the first token", () => {
    render(
      <MessageItem
        message={{ ...answer, content: "", status: "streaming", citations: [] }}
        selected={false}
        activeCitation={null}
        onSelect={() => {}}
        onCite={() => {}}
      />,
    );
    expect(screen.getByLabelText("Searching documents")).toBeInTheDocument();
  });
});

describe("SourcePanel", () => {
  it("groups cited and retrieved-only sources with page references", () => {
    render(<SourcePanel message={answer} activeCitation={1} onActivate={() => {}} />);
    expect(screen.getByText("Cited in the answer")).toBeInTheDocument();
    expect(screen.getByText("Also retrieved")).toBeInTheDocument();
    expect(screen.getByTestId("source-1")).toHaveClass("active");
    expect(screen.getByText("p. 2 · Results")).toBeInTheDocument();
  });
});

describe("ScopePicker", () => {
  it("lets the user restrict the search to specific documents", async () => {
    const onChange = vi.fn();
    const docs = [
      { id: "d1", filename: "a.pdf", file_type: "pdf", status: "ready", page_count: 3, chunk_count: 5, title: null },
      { id: "d2", filename: "b.txt", file_type: "txt", status: "processing", page_count: null, chunk_count: 0, title: null },
    ] as never[];
    render(<ScopePicker documents={docs} selected={[]} onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: /All documents \(1\)/ }));
    expect(screen.queryByText("b.txt")).not.toBeInTheDocument(); // not ready -> not selectable
    await userEvent.click(screen.getByRole("checkbox", { name: /a\.pdf/ }));
    expect(onChange).toHaveBeenCalledWith(["d1"]);
  });
});

describe("UploadDropzone", () => {
  it("rejects unsupported files before uploading", async () => {
    render(withQuery(<UploadDropzone />));
    const input = screen.getByTestId("file-input");
    await userEvent.upload(input, new File(["x"], "photo.png", { type: "image/png" }), { applyAccept: false });
    expect(await screen.findByText(/Unsupported file type/)).toBeInTheDocument();
  });
});
