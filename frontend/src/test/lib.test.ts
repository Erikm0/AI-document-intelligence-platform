import { SSEParser } from "../api/sse";
import { citedIndices, parseAnswer, parseInline } from "../lib/citations";
import { formatBytes, pageLabel, relativeTime } from "../lib/format";
import { extensionOf, validateFile } from "../lib/validation";

describe("SSEParser", () => {
  it("parses complete events and buffers partial ones", () => {
    const parser = new SSEParser();
    expect(parser.push('event: token\ndata: {"t":"Hel')).toEqual([]);
    expect(parser.push('lo"}\n\nevent: done\ndata: {}\n\n')).toEqual([
      { event: "token", data: '{"t":"Hello"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("handles CRLF, comments, multi-line data and a trailing event on flush", () => {
    const parser = new SSEParser();
    expect(parser.push(": keep-alive\r\n\r\n")).toEqual([]);
    expect(parser.push("data: line1\ndata: line2\n\n")).toEqual([{ event: "message", data: "line1\nline2" }]);
    parser.push("event: error\ndata: {\"code\":\"x\"}");
    expect(parser.flush()).toEqual([{ event: "error", data: '{"code":"x"}' }]);
  });
});

describe("citations", () => {
  it("splits text, bold, code and citation markers", () => {
    expect(parseInline("Leave is **26 days** [1][2], see `policy` [3, 4].")).toEqual([
      { kind: "text", text: "Leave is " },
      { kind: "bold", text: "26 days" },
      { kind: "text", text: " " },
      { kind: "cite", index: 1 },
      { kind: "cite", index: 2 },
      { kind: "text", text: ", see " },
      { kind: "code", text: "policy" },
      { kind: "text", text: " " },
      { kind: "cite", index: 3 },
      { kind: "cite", index: 4 },
      { kind: "text", text: "." },
    ]);
  });

  it("groups paragraphs and lists", () => {
    const blocks = parseAnswer("Intro line\ncontinues [1].\n\n- first [2]\n- second\n1. one\n2. two");
    expect(blocks.map((b) => b.kind)).toEqual(["paragraph", "list", "list"]);
    expect(blocks[1]).toMatchObject({ kind: "list", ordered: false });
    expect(blocks[2]).toMatchObject({ kind: "list", ordered: true });
    expect(citedIndices("A [2]. B [1][2].\n- C [3]")).toEqual([2, 1, 3]);
  });
});

describe("format", () => {
  it("formats bytes", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(25 * 1024 * 1024)).toBe("25 MB");
  });

  it("formats page labels", () => {
    expect(pageLabel(null, null)).toBe("");
    expect(pageLabel(3, 3)).toBe("p. 3");
    expect(pageLabel(3, 5)).toBe("pp. 3–5");
  });

  it("formats relative times", () => {
    const now = new Date("2026-01-10T12:00:00Z");
    expect(relativeTime("2026-01-10T11:59:50Z", now)).toBe("just now");
    expect(relativeTime("2026-01-10T11:30:00Z", now)).toBe("30 min ago");
    expect(relativeTime("2026-01-10T09:00:00Z", now)).toBe("3 h ago");
  });
});

describe("validateFile", () => {
  it("accepts supported files", () => {
    expect(validateFile({ name: "Report.PDF", size: 1000 })).toBeNull();
    expect(extensionOf("archive.tar.gz")).toBe(".gz");
  });

  it("rejects unsupported, empty and oversized files", () => {
    expect(validateFile({ name: "image.png", size: 10 })).toMatch(/Unsupported/);
    expect(validateFile({ name: "noext", size: 10 })).toMatch(/Unsupported/);
    expect(validateFile({ name: "a.txt", size: 0 })).toMatch(/empty/);
    expect(validateFile({ name: "a.txt", size: 30 * 1024 * 1024 }, [".txt"], 25)).toMatch(/25 MB/);
  });
});
