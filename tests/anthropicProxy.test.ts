/**
 * End-to-end Anthropic capture: an agent (e.g. Claude Code) points its
 * ANTHROPIC_BASE_URL at the proxy and hits /v1/messages. The proxy transforms
 * to OpenAI chat for the backend, captures, and transforms the response back
 * into an Anthropic message. Finalize still yields a trainable trace.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createServer, type Server } from "node:http";

import { CaptureProxy } from "../src/capture/server.js";

type Dict = Record<string, unknown>;

const BACKEND_RESPONSE: Dict = {
  id: "cmpl-1",
  object: "chat.completion",
  created: 0,
  model: "stub",
  prompt_token_ids: [1, 2, 3, 4],
  choices: [
    {
      index: 0,
      token_ids: [10, 11, 12],
      message: { role: "assistant", content: "Hello there" },
      finish_reason: "stop",
      logprobs: {
        content: [
          { token: "a", logprob: -0.1 },
          { token: "b", logprob: -0.2 },
          { token: "c", logprob: -0.3 },
        ],
      },
    },
  ],
  usage: { prompt_tokens: 4, completion_tokens: 3 },
};

let backend: Server;
let proxy: CaptureProxy;
let proxyPort: number;

beforeAll(async () => {
  backend = createServer((req, res) => {
    if (req.method === "POST" && req.url === "/v1/chat/completions") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(BACKEND_RESPONSE));
      return;
    }
    res.writeHead(404).end();
  });
  const backendPort = await new Promise<number>((resolve) => {
    backend.listen(0, "127.0.0.1", () => {
      const addr = backend.address();
      resolve(typeof addr === "object" && addr ? addr.port : 0);
    });
  });

  proxy = new CaptureProxy({
    inferenceBaseUrl: `http://127.0.0.1:${backendPort}`,
    modelServed: "stub-model",
    engine: "vllm",
    logger: { info: () => {}, warn: () => {}, error: () => {} },
  });
  proxyPort = await proxy.listen(0, "127.0.0.1");
});

afterAll(async () => {
  await proxy.close();
  await new Promise<void>((resolve) => backend.close(() => resolve()));
});

describe("capture proxy (Anthropic Messages)", () => {
  it("captures /v1/messages and returns an Anthropic message shape", async () => {
    const sid = "sess-anthropic";
    const resp = await fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": sid,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-3-5-sonnet",
        max_tokens: 256,
        system: "be brief",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(resp.status).toBe(200);
    const body = (await resp.json()) as Dict;
    expect(body["type"]).toBe("message");
    expect(body["role"]).toBe("assistant");
    expect(body["model"]).toBe("claude-3-5-sonnet");
    const content = body["content"] as Dict[];
    expect(content[0]).toEqual({ type: "text", text: "Hello there" });

    // The captured completion builds a single trainable trace.
    const finalize = await fetch(
      `http://127.0.0.1:${proxyPort}/sessions/${sid}/finalize?builder=per_request&format=rl`,
      { method: "POST" },
    );
    const result = (await finalize.json()) as Dict;
    expect(result["status"]).toBe("COMPLETED");
    const samples = result["samples"] as Dict[];
    expect(samples).toHaveLength(1);
    expect(samples[0]!["token_ids"]).toEqual([1, 2, 3, 4, 10, 11, 12]);
    expect(samples[0]!["prompt_len"]).toBe(4);
    expect(samples[0]!["loss_mask"]).toEqual([1, 1, 1]);
  });
});
