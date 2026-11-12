import { Fragment } from "react";

import { type InlineSegment, parseAnswer } from "../../lib/citations";

interface Props {
  text: string;
  /** Source numbers that exist; markers outside this set render as invalid. */
  validIndices?: Set<number>;
  activeIndex?: number | null;
  onCite?: (index: number) => void;
}

function Inline({ segments, validIndices, activeIndex, onCite }: Omit<Props, "text"> & { segments: InlineSegment[] }) {
  return (
    <>
      {segments.map((s, i) => {
        switch (s.kind) {
          case "bold":
            return <strong key={i}>{s.text}</strong>;
          case "code":
            return <code key={i}>{s.text}</code>;
          case "cite": {
            const valid = !validIndices || validIndices.has(s.index);
            return (
              <button
                key={i}
                type="button"
                className={`cite${activeIndex === s.index ? " active" : ""}${valid ? "" : " invalid"}`}
                onClick={() => valid && onCite?.(s.index)}
                aria-label={`Show source ${s.index}`}
                title={valid ? `Source ${s.index}` : "Unknown source"}
              >
                {s.index}
              </button>
            );
          }
          default:
            return <Fragment key={i}>{s.text}</Fragment>;
        }
      })}
    </>
  );
}

export function AnswerText({ text, ...rest }: Props) {
  return (
    <>
      {parseAnswer(text).map((block, i) =>
        block.kind === "paragraph" ? (
          <p key={i}>
            <Inline segments={block.segments} {...rest} />
          </p>
        ) : block.ordered ? (
          <ol key={i}>
            {block.items.map((item, j) => (
              <li key={j}>
                <Inline segments={item} {...rest} />
              </li>
            ))}
          </ol>
        ) : (
          <ul key={i}>
            {block.items.map((item, j) => (
              <li key={j}>
                <Inline segments={item} {...rest} />
              </li>
            ))}
          </ul>
        ),
      )}
    </>
  );
}
