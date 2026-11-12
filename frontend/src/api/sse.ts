/**
 * Incremental Server-Sent Events parser.
 *
 * EventSource only supports GET, but the chat stream is a POST with a JSON body,
 * so we read `fetch()`'s body stream and parse the `event:` / `data:` framing
 * ourselves. Chunks can split anywhere - including inside a multi-byte UTF-8
 * character - so the decoder runs in streaming mode and we buffer until "\n\n".
 */
export interface RawSSEEvent {
  event: string;
  data: string;
}

export class SSEParser {
  private buffer = "";

  push(chunk: string): RawSSEEvent[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const events: RawSSEEvent[] = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const parsed = parseBlock(block);
      if (parsed) events.push(parsed);
      boundary = this.buffer.indexOf("\n\n");
    }
    return events;
  }

  flush(): RawSSEEvent[] {
    const rest = this.buffer.trim();
    this.buffer = "";
    const parsed = rest ? parseBlock(rest) : null;
    return parsed ? [parsed] : [];
  }
}

function parseBlock(block: string): RawSSEEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { event, data: data.join("\n") } : null;
}

export async function* readSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<RawSSEEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }
    yield* parser.push(decoder.decode());
    yield* parser.flush();
  } finally {
    reader.releaseLock();
  }
}
