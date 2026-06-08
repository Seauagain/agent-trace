/**
 * JSON-wire decoder for the forward proxy: OpenAI-chat passes through, Anthropic
 * Messages is normalized to OpenAI-chat (incl. gzip + SSE handling), reusing the
 * same normalization as the base-URL passthrough path.
 */

import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";

import { jsonDecoder } from "../src/capture/forward/decoders/json.js";
import type { HttpExchange } from "../src/capture/forward/decoders/types.js";

type Dict = Record<string, unknown>;

function ex(over: Partial<HttpExchange>): HttpExchange {
  return {
    host: "api.openai.com",
    method: "POST",
    path: "/v1/chat/completions",
    reqHeaders: { "content-type": "application/json" },
    reqBody: Buffer.alloc(0),
    status: 200,
    resHeaders: { "content-type": "application/json" },
    resBody: Buffer.alloc(0),
    ...over,
  };
}

describe("json forward decoder", () => {
  it("captures OpenAI chat and decompresses a gzip response", () => {
    const req: Dict = { model: "gpt-4o", messages: [{ role: "user", content: "hi" }] };
    const res: Dict = {
      id: "c1",
      object: "chat.completion",
      model: "gpt-4o-2024",
      choices: [{ index: 0, message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
    };
    const exchange = ex({
      reqBody: Buffer.from(JSON.stringify(req)),
      resHeaders: { "content-type": "application/json", "content-encoding": "gzip" },
      resBody: gzipSync(Buffer.from(JSON.stringify(res))),
    });

    expect(jsonDecoder.matches(exchange)).toBe(true);
    const decoded = jsonDecoder.decode(exchange)!;
    expect(decoded.apiType).toBe("openai_chat");
    expect(decoded.model).toBe("gpt-4o-2024");
    const msg = (decoded.response["choices"] as Dict[])[0]!["message"] as Dict;
    expect(msg["content"]).toBe("hello");
  });

  it("normalizes an Anthropic Messages response to OpenAI-chat", () => {
    const req: Dict = {
      model: "claude-sonnet-4",
      system: "be terse",
      messages: [{ role: "user", content: "hi" }],
    };
    const res: Dict = {
      id: "m1",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4",
      content: [{ type: "text", text: "yo" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 3, output_tokens: 1 },
    };
    const exchange = ex({
      host: "api.anthropic.com",
      path: "/v1/messages",
      reqHeaders: { "content-type": "application/json", "anthropic-version": "2023-06-01" },
      reqBody: Buffer.from(JSON.stringify(req)),
      resBody: Buffer.from(JSON.stringify(res)),
    });

    const decoded = jsonDecoder.decode(exchange)!;
    expect(decoded.apiType).toBe("anthropic");
    const msg = (decoded.response["choices"] as Dict[])[0]!["message"] as Dict;
    expect(msg["content"]).toBe("yo");
    expect(decoded.response["usage"]).toMatchObject({ prompt_tokens: 3, completion_tokens: 1 });
    expect(Array.isArray(decoded.request["messages"])).toBe(true);
  });

  it("ignores non-chat JSON bodies", () => {
    expect(jsonDecoder.matches(ex({ reqBody: Buffer.from('{"ping":true}') }))).toBe(false);
  });
});
