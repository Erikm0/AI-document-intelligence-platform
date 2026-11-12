import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export function Composer({ onSend, onStop, streaming, disabled, placeholder }: Props) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [value]);

  const submit = () => {
    if (!value.trim() || streaming || disabled) return;
    onSend(value);
    setValue("");
  };

  return (
    <div className="composer">
      <div className="composer-box">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          placeholder={placeholder ?? "Ask a question about your documents..."}
          aria-label="Question"
          maxLength={4000}
          disabled={disabled}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
        />
        {streaming ? (
          <button className="btn" onClick={onStop} aria-label="Stop generating">
            <Square size={14} /> Stop
          </button>
        ) : (
          <button className="btn btn-primary" onClick={submit} disabled={!value.trim() || disabled} aria-label="Send">
            <ArrowUp size={16} />
          </button>
        )}
      </div>
      <div className="composer-hint">
        Answers use only your documents and cite their sources. Enter to send · Shift+Enter for a new line
      </div>
    </div>
  );
}
