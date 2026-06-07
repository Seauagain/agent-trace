/** Minimal JSONL writers for serialized samples. */

import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

/** Render rows as a JSONL string (one compact JSON object per line). */
export function toJsonl(rows: unknown[]): string {
  return rows.map((row) => JSON.stringify(row)).join("\n") + (rows.length > 0 ? "\n" : "");
}

/** Write rows to a JSONL file, creating parent directories as needed. */
export async function writeJsonl(path: string, rows: unknown[]): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, toJsonl(rows), "utf-8");
}

/** Append rows to a JSONL file (useful for streaming many sessions). */
export async function appendJsonl(path: string, rows: unknown[]): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await appendFile(path, toJsonl(rows), "utf-8");
}
