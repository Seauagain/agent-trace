/**
 * End-to-end capture proxy test: a stub vLLM-style backend, a multi-turn
 * capture through the proxy, finalize with prefix_merging, and assert the
 * SFT/RL JSONL invariants (the same merged stream the unit test pins, but now
 * driven through the real HTTP proxy + engine normalization).
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createServer, type Server } from "node:http";

import { CaptureProxy } from "../src/capture/server.js";

type Dict = Record<string, unknown>;
const EOT = 99;

// Canned vLLM-native responses forming a valid append-only 2-turn chain.
function vllmResponse(promptIds: number[], responseIds: number[], logprobs: number[]): Dict {
  return {
    id: "cmpl-x",
    object: "chat.completion",
    created: 0,
    model: "stub",
    prompt_token_ids: promptIds,
    choices: [
      {
        index: 0,
        token_ids: responseIds,
        message: { role: "assistant", content: "A" },
        finish_reason: "stop",
        logprobs: {
          content: responseIds.map((tid, i) => ({ token: `t${tid}`, logprob: logprobs[i], bytes: [] })),
        },
      },
    ],
    usage: { prompt_tokens: promptIds.length, completion_tokens: responseIds.length },
  };
}

const RESPONSES: Dict[] = [
  vllmResponse([1, 2, 3], [10, 11, EOT], [-0.1, -0.2, -0.3]),
  vllmResponse([1, 2, 3, 10, 11, EOT, 50, 51], [20, 21, EOT], [-0.5, -0.6, -0.7]),
];

let backend: Server;
let backendPort: number;
let proxy: CaptureProxy;
let proxyPort: number;
let callIndex = 0;

beforeAll(async () => {
  backend = createServer((req, res) => {
    if (req.method === "POST" && req.url === "/v1/chat/completions") {
      const body = RESPONSES[Math.min(callIndex, RESPONSES.length - 1)]!;
      callIndex++;
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(body));
      return;
    }
    res.writeHead(404).end();
  });
  backendPort = await new Promise<number>((resolve) => {
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

async function chat(sessionId: string, messages: Dict[], stream = false): Promise<Response> {
  return fetch(`http://127.0.0.1:${proxyPort}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${sessionId}` },
    body: JSON.stringify({ model: "gpt-4o", messages, stream }),
  });
}

describe("capture proxy (OpenAI Chat)", () => {
  it("captures a multi-turn session and finalizes to an RL sample", async () => {
    const sid = "sess-rl";
    const r1 = await chat(sid, [{ role: "user", content: "Q1" }]);
    expect(r1.status).toBe(200);
    const j1 = (await r1.json()) as Dict;
    // Response is transformed back to the requested model name.
    expect(j1["model"]).toBe("gpt-4o");

    await chat(sid, [
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
      { role: "tool", content: "result" },
    ]);

    const listed = await (
      await fetch(`http://127.0.0.1:${proxyPort}/sessions/${sid}/completions`)
    ).json();
    expect((listed as Dict)["completions"]).toHaveLength(2);

    const finalize = await fetch(
      `http://127.0.0.1:${proxyPort}/sessions/${sid}/finalize?builder=prefix_merging&format=rl&end_of_turn_token_id=${EOT}`,
      { method: "POST" },
    );
    const result = (await finalize.json()) as Dict;
    expect(result["status"]).toBe("COMPLETED");
    expect(result["trace_count"]).toBe(1);
    const samples = result["samples"] as Dict[];
    expect(samples).toHaveLength(1);
    const sample = samples[0]!;
    expect(sample["token_ids"]).toEqual([1, 2, 3, 10, 11, EOT, 50, 51, 20, 21, EOT]);
    expect(sample["prompt_len"]).toBe(3);
    expect(sample["response_len"]).toBe(8);
    expect(sample["loss_mask"]).toEqual([1, 1, 1, 0, 0, 1, 1, 1]);
    expect(sample["logprobs"]).toEqual([-0.1, -0.2, -0.3, 0.0, 0.0, -0.5, -0.6, -0.7]);
  });

  it("serves synthetic SSE when the agent requests streaming", async () => {
    callIndex = 0; // reset canned responses
    const resp = await chat("sess-stream", [{ role: "user", content: "Q1" }], true);
    expect(resp.headers.get("content-type")).toContain("text/event-stream");
    const text = await resp.text();
    expect(text).toContain("data: ");
    expect(text.trimEnd().endsWith("data: [DONE]")).toBe(true);
  });

  it("rejects unsupported API families in this build", async () => {
    const resp = await fetch(`http://127.0.0.1:${proxyPort}/v1beta/models/gemini:generateContent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: "hi" }] }] }),
    });
    expect(resp.status).toBe(400);
    const j = (await resp.json()) as Dict;
    expect(String((j["error"] as string) ?? "")).toContain("google");
  });
});
