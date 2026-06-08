/**
 * Connect-RPC / gRPC message framing.
 *
 * Cursor's agent endpoint speaks Connect-RPC over HTTP/2 with protobuf bodies.
 * Both unary and streaming bodies are a sequence of length-prefixed frames:
 *
 *     [1 byte flags][4 bytes big-endian length][length bytes payload]
 *
 * `flags & 1` marks a compressed payload (gzip per the `grpc-encoding` /
 * `connect-content-encoding` header); Connect's trailing "end-of-stream" frame
 * sets `flags & 2` and carries JSON metadata rather than a protobuf message.
 */

import { brotliDecompressSync, gunzipSync, inflateSync } from "node:zlib";

export interface ConnectFrame {
  flags: number;
  /** Raw (still-compressed if `compressed`) payload bytes. */
  payload: Buffer;
  compressed: boolean;
  /** End-of-stream trailer frame (Connect streaming). */
  endStream: boolean;
}

/** Split a Connect/gRPC body into its frames. Partial trailing bytes are ignored. */
export function deframe(buf: Buffer): ConnectFrame[] {
  const frames: ConnectFrame[] = [];
  let off = 0;
  while (off + 5 <= buf.length) {
    const flags = buf[off]!;
    const len = buf.readUInt32BE(off + 1);
    const start = off + 5;
    const end = start + len;
    if (end > buf.length) break;
    frames.push({
      flags,
      payload: buf.subarray(start, end),
      compressed: (flags & 0x01) !== 0,
      endStream: (flags & 0x02) !== 0,
    });
    off = end;
  }
  return frames;
}

/** Best-effort decompression of a frame payload given its flags + encoding hint. */
export function decompressFrame(frame: ConnectFrame, encoding?: string | null): Buffer {
  if (!frame.compressed) return frame.payload;
  const enc = (encoding ?? "").toLowerCase();
  const attempts: Array<(b: Buffer) => Buffer> = [];
  if (enc.includes("br")) attempts.push(brotliDecompressSync);
  else if (enc.includes("deflate")) attempts.push(inflateSync);
  else attempts.push(gunzipSync);
  // Fall back across codecs when the header lies or is absent.
  attempts.push(gunzipSync, brotliDecompressSync, inflateSync);
  for (const fn of attempts) {
    try {
      return fn(frame.payload);
    } catch {
      /* try next */
    }
  }
  return frame.payload;
}

/** Deframe + decompress every non-trailer frame into plaintext payloads. */
export function decodeMessageFrames(buf: Buffer, encoding?: string | null): Buffer[] {
  return deframe(buf)
    .filter((f) => !f.endStream)
    .map((f) => decompressFrame(f, encoding));
}
