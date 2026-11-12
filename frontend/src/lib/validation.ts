export const DEFAULT_EXTENSIONS = [".pdf", ".docx", ".txt"];
export const DEFAULT_MAX_MB = 25;

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot).toLowerCase() : "";
}

/** Client-side pre-check; the server re-validates everything (incl. magic bytes). */
export function validateFile(
  file: Pick<File, "name" | "size">,
  allowed: string[] = DEFAULT_EXTENSIONS,
  maxMb: number = DEFAULT_MAX_MB,
): string | null {
  const ext = extensionOf(file.name);
  if (!allowed.includes(ext)) {
    return `Unsupported file type${ext ? ` "${ext}"` : ""}. Allowed: ${allowed.join(", ")}`;
  }
  if (file.size === 0) return "The file is empty";
  if (file.size > maxMb * 1024 * 1024) return `File is larger than ${maxMb} MB`;
  return null;
}
