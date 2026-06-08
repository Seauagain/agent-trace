/**
 * Cursor decoder: recover the AI-SDK conversation embedded as JSON inside
 * Connect-RPC protobuf frames, reshape it into an OpenAI-chat (request,
 * response) pair, and confirm that pair flows through the builder into a real
 * SFT sample (the whole point of capturing Cursor for training).
 */

import { describe, expect, it } from "vitest";

import { cursorDecoder } from "../src/capture/forward/decoders/cursor.js";
import { decodeExchange, defaultDecoders } from "../src/capture/forward/decoders/index.js";
import type { HttpExchange } from "../src/capture/forward/decoders/types.js";
import { SessionStore } from "../src/capture/sessionStore.js";
import { defaultBuilderRegistry } from "../src/trajectory/registry.js";
import { trajectoryToSFTSamples } from "../src/serialize/toSFTSample.js";

type Dict = Record<string, unknown>;

function varint(n: number): Buffer {
  const bytes: number[] = [];
  let v = n;
  while (v > 0x7f) {
    bytes.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  bytes.push(v);
  return Buffer.from(bytes);
}

/** Encode a length-delimited (wiretype 2) field carrying `bytes`. */
function field2(fieldNum: number, bytes: Buffer): Buffer {
  const key = (fieldNum << 3) | 2;
  return Buffer.concat([varint(key), varint(bytes.length), bytes]);
}

function frame(payload: Buffer): Buffer {
  const head = Buffer.alloc(5);
  head.writeUInt32BE(payload.length, 1);
  return Buffer.concat([head, payload]);
}

/** Build a Connect frame embedding each message as a JSON protobuf string field. */
function cursorBody(messages: Dict[]): Buffer {
  const fields = messages.map((m) => field2(1, Buffer.from(JSON.stringify(m), "utf-8")));
  return frame(Buffer.concat(fields));
}

const MESSAGES: Dict[] = [
  { role: "system", content: "You are Cursor's agent." },
  { role: "user", content: "Refactor the auth module." },
  {
    role: "assistant",
    content: [
      { type: "redacted-reasoning", data: "opaque" },
      { type: "text", text: "Done — extracted a helper." },
    ],
    providerOptions: { cursor: { modelName: "composer-2.5-fast" } },
  },
];

function exchange(messages: Dict[]): HttpExchange {
  return {
    host: "api2.cursor.sh",
    method: "POST",
    path: "/aiserver.v1.AgentService/Run",
    reqHeaders: { "content-type": "application/connect+proto" },
    reqBody: Buffer.alloc(0),
    status: 200,
    resHeaders: { "content-type": "application/connect+proto" },
    resBody: cursorBody(messages),
  };
}

describe("cursor decoder", () => {
  it("matches cursor protobuf traffic and splits the final assistant turn", () => {
    const ex = exchange(MESSAGES);
    expect(cursorDecoder.matches(ex)).toBe(true);

    const decoded = cursorDecoder.decode(ex)!;
    expect(decoded).not.toBeNull();
    expect(decoded.model).toBe("composer-2.5-fast");

    const reqMessages = decoded.request["messages"] as Dict[];
    expect(reqMessages.map((m) => m["role"])).toEqual(["system", "user"]);

    const choice = (decoded.response["choices"] as Dict[])[0]!;
    const message = choice["message"] as Dict;
    expect(message["content"]).toBe("Done — extracted a helper.");
    expect(message["reasoning_content"]).toContain("[redacted-reasoning]");
  });

  it("collapses a streamed/duplicated trailing assistant run into one completion", () => {
    const streamed: Dict[] = [
      { role: "user", content: "ping" },
      { role: "assistant", content: [{ type: "text", text: "PO" }] },
      { role: "assistant", content: [{ type: "text", text: "PONG" }] },
    ];
    const decoded = cursorDecoder.decode(exchange(streamed))!;
    const reqMessages = decoded.request["messages"] as Dict[];
    expect(reqMessages.map((m) => m["role"])).toEqual(["user"]);
    const message = (decoded.response["choices"] as Dict[])[0]!["message"] as Dict;
    expect(message["content"]).toBe("PONG");
  });

  it("ignores non-cursor hosts and bodies without role messages", () => {
    const notCursor = { ...exchange(MESSAGES), host: "registry.npmjs.org" };
    expect(cursorDecoder.matches(notCursor)).toBe(false);
    const empty = { ...exchange([]), resBody: Buffer.from("not protobuf json") };
    expect(cursorDecoder.matches(empty)).toBe(false);
  });

  it("is selected by the default registry over the JSON decoder", () => {
    const decoded = decodeExchange(exchange(MESSAGES), defaultDecoders());
    expect(decoded?.apiType).toBe("cursor_agent");
  });

  it("produces a real SFT sample through the build pipeline", async () => {
    const decoded = decodeExchange(exchange(MESSAGES), defaultDecoders())!;
    const store = new SessionStore();
    store.saveMessage("forward", decoded.request, decoded.response, {
      modelUsed: decoded.model,
      modelRequested: decoded.model,
      apiType: decoded.apiType,
    });

    const session = store.loadCompletionSession("forward");
    const builder = defaultBuilderRegistry().create({ strategy: "prefix_merging", config: {} });
    const trajectory = await builder.build(session);
    const samples = trajectoryToSFTSamples(trajectory);

    expect(samples.length).toBeGreaterThan(0);
    const serialized = JSON.stringify(samples);
    expect(serialized).toContain("Refactor the auth module.");
    expect(serialized).toContain("Done — extracted a helper.");
  });
});
