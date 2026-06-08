/**
 * Schema-less protobuf spelunking.
 *
 * We don't have Cursor's `.proto` files, but their agent messages embed the
 * conversation as JSON strings inside length-delimited protobuf fields. Rather
 * than reverse a brittle schema, we walk the wire format generically, collect
 * every length-delimited field, and keep the ones that parse as JSON objects.
 * Everything is bounded (depth + node budget) so a hostile/garbage body can't
 * blow up.
 */

const DEFAULT_MAX_DEPTH = 6;
const DEFAULT_NODE_BUDGET = 50000;

interface VarintRead {
  value: number;
  next: number;
}

function readVarint(buf: Buffer, off: number): VarintRead | null {
  let shift = 0;
  let value = 0;
  let pos = off;
  while (pos < buf.length) {
    const byte = buf[pos]!;
    value += (byte & 0x7f) * 2 ** shift;
    pos++;
    if ((byte & 0x80) === 0) return { value, next: pos };
    shift += 7;
    if (shift > 63) return null;
  }
  return null;
}

function looksLikeJson(buf: Buffer): boolean {
  for (let i = 0; i < buf.length; i++) {
    const c = buf[i]!;
    if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d) continue;
    return c === 0x7b || c === 0x5b; // '{' or '['
  }
  return false;
}

function tryParseJson(buf: Buffer): unknown {
  if (!looksLikeJson(buf)) return undefined;
  try {
    return JSON.parse(buf.toString("utf-8"));
  } catch {
    return undefined;
  }
}

export interface ScanOptions {
  maxDepth?: number;
  nodeBudget?: number;
  /** Keep parsed JSON values for which this returns true. Default: any object. */
  keep?: (value: unknown) => boolean;
}

/**
 * Collect embedded JSON values from a protobuf-ish byte buffer. Returns parsed
 * values (deduped by serialized form) that satisfy `keep`.
 */
export function collectJsonValues(buf: Buffer, opts: ScanOptions = {}): unknown[] {
  const maxDepth = opts.maxDepth ?? DEFAULT_MAX_DEPTH;
  const keep = opts.keep ?? ((v) => typeof v === "object" && v !== null);
  let budget = opts.nodeBudget ?? DEFAULT_NODE_BUDGET;
  const out: unknown[] = [];
  const seen = new Set<string>();

  const walk = (b: Buffer, depth: number): void => {
    if (depth > maxDepth || budget <= 0) return;
    let off = 0;
    while (off < b.length) {
      if (--budget <= 0) return;
      const key = readVarint(b, off);
      if (key === null) return;
      const wireType = key.value & 0x07;
      off = key.next;
      if (wireType === 0) {
        const v = readVarint(b, off);
        if (v === null) return;
        off = v.next;
      } else if (wireType === 1) {
        off += 8;
      } else if (wireType === 5) {
        off += 4;
      } else if (wireType === 2) {
        const len = readVarint(b, off);
        if (len === null) return;
        const start = len.next;
        const end = start + len.value;
        if (end > b.length) return;
        const payload = b.subarray(start, end);
        off = end;
        const parsed = tryParseJson(payload);
        if (parsed !== undefined) {
          if (keep(parsed)) {
            const sig = JSON.stringify(parsed);
            if (!seen.has(sig)) {
              seen.add(sig);
              out.push(parsed);
            }
          }
        } else if (payload.length >= 2) {
          walk(payload, depth + 1);
        }
      } else {
        return; // groups / unknown wire types: bail this level
      }
    }
  };

  walk(buf, 0);
  return out;
}

/** Convenience: collect JSON *objects* that carry a string `role` field. */
export function collectRoleMessages(buf: Buffer): Record<string, unknown>[] {
  return collectJsonValues(buf, {
    keep: (v) =>
      typeof v === "object" &&
      v !== null &&
      !Array.isArray(v) &&
      typeof (v as Record<string, unknown>)["role"] === "string",
  }) as Record<string, unknown>[];
}
