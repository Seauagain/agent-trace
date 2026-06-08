/**
 * Connect-RPC / gRPC framing: deframe length-prefixed messages, decompress
 * gzip frames, skip the end-of-stream trailer, and recover embedded JSON.
 */

import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";

import { deframe, decodeMessageFrames, decompressFrame } from "../src/capture/forward/connect.js";

function frame(payload: Buffer, flags = 0): Buffer {
  const head = Buffer.alloc(5);
  head[0] = flags;
  head.writeUInt32BE(payload.length, 1);
  return Buffer.concat([head, payload]);
}

describe("connect framing", () => {
  it("splits a body into frames with flags + payload", () => {
    const body = Buffer.concat([frame(Buffer.from("hello")), frame(Buffer.from("world"))]);
    const frames = deframe(body);
    expect(frames).toHaveLength(2);
    expect(frames[0]!.payload.toString()).toBe("hello");
    expect(frames[1]!.payload.toString()).toBe("world");
    expect(frames[0]!.compressed).toBe(false);
  });

  it("ignores a truncated trailing frame", () => {
    const body = Buffer.concat([frame(Buffer.from("ok")), Buffer.from([0, 0, 0, 0, 9, 1, 2])]);
    expect(deframe(body)).toHaveLength(1);
  });

  it("decompresses a gzip-flagged frame", () => {
    const payload = gzipSync(Buffer.from("compressed-payload"));
    const f = deframe(frame(payload, 0x01))[0]!;
    expect(f.compressed).toBe(true);
    expect(decompressFrame(f, "gzip").toString()).toBe("compressed-payload");
  });

  it("skips the end-of-stream trailer frame", () => {
    const body = Buffer.concat([
      frame(Buffer.from("msg")),
      frame(Buffer.from('{"metadata":{}}'), 0x02),
    ]);
    const msgs = decodeMessageFrames(body);
    expect(msgs).toHaveLength(1);
    expect(msgs[0]!.toString()).toBe("msg");
  });
});
