/**
 * Splits an answer into renderable segments: plain text, **bold**, `code` and
 * citation markers ([1], [2][3], [1, 2]). Lines are grouped into paragraphs and
 * bullet / numbered lists so model output renders cleanly without a full
 * Markdown dependency.
 */

export type InlineSegment =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "code"; text: string }
  | { kind: "cite"; index: number };

export type AnswerBlock =
  | { kind: "paragraph"; segments: InlineSegment[] }
  | { kind: "list"; ordered: boolean; items: InlineSegment[][] };

const INLINE_RE = /\*\*([^*]+)\*\*|`([^`]+)`|\[(\d+(?:\s*,\s*\d+)*)\]/g;
const BULLET_RE = /^\s*(?:[-*•])\s+(.*)$/;
const ORDERED_RE = /^\s*\d+[.)]\s+(.*)$/;

export function parseInline(text: string): InlineSegment[] {
  const segments: InlineSegment[] = [];
  let last = 0;
  for (const match of text.matchAll(INLINE_RE)) {
    const start = match.index ?? 0;
    if (start > last) segments.push({ kind: "text", text: text.slice(last, start) });
    if (match[1] !== undefined) segments.push({ kind: "bold", text: match[1] });
    else if (match[2] !== undefined) segments.push({ kind: "code", text: match[2] });
    else if (match[3] !== undefined) {
      for (const n of match[3].split(",")) segments.push({ kind: "cite", index: Number(n.trim()) });
    }
    last = start + match[0].length;
  }
  if (last < text.length) segments.push({ kind: "text", text: text.slice(last) });
  return segments;
}

export function parseAnswer(answer: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ kind: "paragraph", segments: parseInline(paragraph.join(" ")) });
      paragraph = [];
    }
  };

  for (const line of answer.split("\n")) {
    const bullet = BULLET_RE.exec(line);
    const ordered = bullet ? null : ORDERED_RE.exec(line);
    const item = bullet?.[1] ?? ordered?.[1];
    if (item !== undefined) {
      flushParagraph();
      const isOrdered = Boolean(ordered);
      const prev = blocks[blocks.length - 1];
      if (prev && prev.kind === "list" && prev.ordered === isOrdered) prev.items.push(parseInline(item));
      else blocks.push({ kind: "list", ordered: isOrdered, items: [parseInline(item)] });
    } else if (!line.trim()) {
      flushParagraph();
    } else {
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  return blocks;
}

export function citedIndices(answer: string): number[] {
  const seen = new Set<number>();
  for (const block of parseAnswer(answer)) {
    const lists = block.kind === "paragraph" ? [block.segments] : block.items;
    for (const segs of lists) for (const s of segs) if (s.kind === "cite") seen.add(s.index);
  }
  return [...seen];
}
