/**
 * Asynchronous on-disk persistence for captured completion records.
 *
 * Writes one JSON file per completion under:
 *   <saveDir>/task_<taskId>/sessions/<sessionId>/completions/<NNNN>-<id>.json
 *
 * Writes happen off the request hot path on a background drain loop. If the
 * bounded queue is full, completions are dropped (with a warning). The
 * in-memory copy in SessionStore stays authoritative.
 */

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

type Dict = Record<string, unknown>;

const TRUNCATED_MARKER = "__truncated";

function approxByteSize(value: unknown): number {
  if (value == null) return 0;
  if (typeof value === "string") return Buffer.byteLength(value, "utf-8");
  if (typeof value === "number" || typeof value === "boolean") return 16;
  if (Array.isArray(value)) return value.reduce((acc: number, v) => acc + approxByteSize(v), 0);
  if (typeof value === "object") {
    let total = 0;
    for (const [k, v] of Object.entries(value)) total += approxByteSize(k) + approxByteSize(v);
    return total;
  }
  try {
    return Buffer.byteLength(JSON.stringify(value), "utf-8");
  } catch {
    return 0;
  }
}

function truncateValue(value: unknown, maxBytes: number): unknown {
  if (approxByteSize(value) <= maxBytes) return value;
  if (typeof value === "string") {
    const buf = Buffer.from(value, "utf-8").subarray(0, maxBytes);
    return buf.toString("utf-8") + "\u2026";
  }
  if (Array.isArray(value)) {
    const out: unknown[] = [];
    let running = 0;
    for (const item of value) {
      const pieceSize = approxByteSize(item);
      if (running + pieceSize > maxBytes) {
        out.push({ [TRUNCATED_MARKER]: true });
        break;
      }
      out.push(item);
      running += pieceSize;
    }
    return out;
  }
  if (value && typeof value === "object") {
    const out: Dict = {};
    let running = 0;
    const keys = Object.keys(value);
    let i = 0;
    for (const [key, item] of Object.entries(value)) {
      const pieceSize = approxByteSize(item);
      if (running + pieceSize > maxBytes) {
        out[TRUNCATED_MARKER] = true;
        out["_truncated_keys_omitted"] = keys.slice(i);
        break;
      }
      out[key] = item;
      running += pieceSize;
      i++;
    }
    return out;
  }
  return value;
}

interface WriteItem {
  taskId: string;
  sessionId: string;
  sequence: number;
  completionId: string;
  payload: Dict;
}

export interface CompletionWriterOptions {
  saveDir: string | null;
  maxFieldBytes?: number;
  queueSize?: number;
  enabled?: boolean;
}

export class CompletionWriter {
  readonly saveDir: string | null;
  readonly maxFieldBytes: number;
  readonly queueSize: number;
  readonly enabled: boolean;

  private readonly queue: WriteItem[] = [];
  private readonly sequences = new Map<string, number>();
  private dropCount = 0;
  private draining = false;
  private closed = false;
  private idle: Promise<void> = Promise.resolve();
  private idleResolve: (() => void) | null = null;

  constructor(opts: CompletionWriterOptions) {
    this.saveDir = opts.saveDir;
    this.maxFieldBytes = opts.maxFieldBytes ?? 1 * 1024 * 1024;
    this.queueSize = opts.queueSize ?? 1024;
    this.enabled = (opts.enabled ?? true) && this.saveDir !== null;
  }

  /** Non-blocking enqueue. Returns false if dropped/disabled. */
  enqueue(args: {
    taskId: string | null;
    sessionId: string;
    completionId: string;
    record: Dict;
  }): boolean {
    if (!this.enabled) return false;
    // Standalone captures (passthrough mode) have no rollout task id; bucket
    // them under "default" rather than dropping them off disk.
    const taskId = args.taskId && args.taskId.trim() ? args.taskId : "default";
    if (this.queue.length >= this.queueSize) {
      this.dropCount += 1;
      if (this.dropCount === 1 || this.dropCount % 100 === 0) {
        console.warn(
          `CompletionWriter queue full (${this.dropCount} drops); persistence dropped for session ${args.sessionId}`,
        );
      }
      return false;
    }
    const sequence = (this.sequences.get(args.sessionId) ?? 0) + 1;
    this.sequences.set(args.sessionId, sequence);
    this.queue.push({
      taskId,
      sessionId: args.sessionId,
      sequence,
      completionId: args.completionId,
      payload: this.truncateRecord(args.record),
    });
    void this.drain();
    return true;
  }

  /** Resolve once the queue is fully flushed to disk. */
  async flush(): Promise<void> {
    await this.idle;
  }

  async close(): Promise<void> {
    this.closed = true;
    await this.idle;
  }

  private truncateRecord(record: Dict): Dict {
    const out: Dict = {};
    for (const [key, value] of Object.entries(record)) {
      out[key] = truncateValue(value, this.maxFieldBytes);
    }
    return out;
  }

  private pathFor(item: WriteItem): string | null {
    if (this.saveDir === null) return null;
    const seq = String(item.sequence).padStart(4, "0");
    return join(
      this.saveDir,
      `task_${item.taskId}`,
      "sessions",
      item.sessionId,
      "completions",
      `${seq}-${item.completionId}.json`,
    );
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    this.idle = new Promise<void>((resolve) => {
      this.idleResolve = resolve;
    });
    try {
      while (this.queue.length > 0) {
        const item = this.queue.shift()!;
        try {
          await this.writeToDisk(item);
        } catch (err) {
          console.error(
            `CompletionWriter failed to write ${item.sessionId}/${item.completionId}: ${String(err)}`,
          );
        }
      }
    } finally {
      this.draining = false;
      this.idleResolve?.();
      this.idleResolve = null;
    }
  }

  private async writeToDisk(item: WriteItem): Promise<void> {
    const path = this.pathFor(item);
    if (path === null) return;
    await mkdir(join(path, ".."), { recursive: true });
    const payload = { ...item.payload, __written_at: new Date().toISOString() };
    await writeFile(path, JSON.stringify(payload), "utf-8");
  }
}
