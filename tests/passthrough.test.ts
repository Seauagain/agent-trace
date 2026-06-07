/**
 * Passthrough capture proxy: a stub "real API" upstream (OpenAI + Anthropic),
 * requests forwarded verbatim (original body + auth header), responses streamed
 * back byte-for-byte, and the full trace captured + normalized for SFT/RL.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createServer, type IncomingMessage, type Server } from "node:http";

import { CaptureProxy } from "../src/capture/server.js";
import { buildTraceFromCompletion } from "../src/trajectory/recordUtils.js";
import { toSFTSample } from "../src/serialize/toSFTSample.js";
import type { CompletionRecord } from "../src/trajectory/models.js";

type Dict = Record<string, unknown>;

interface Received {
  url: string;
  headers: Record<string, string | string[] | undefined>;
  body: Dict;
}

const received: Received[] = [];

const OPENAI_JSON: Dict = {
  id: "c2",
  object: "chat.completion",
  created: 2,
  model: "gpt-4o-2024",
  choices: [
    { index: 0, message: { role: "assistant", content: "Answer" }, finish_reason: "stop" },
  ],
  usage: { prompt_tokens: 3, completion_tokens: 1 },
};

const OPENAI_SSE = [
  `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}`,
  `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}`,
  `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
  `data: [DONE]`,
  "",
].join("\n\n");

const ANTHROPIC_JSON: Dict = {
  id: "msg_2",
  type: "message",
  role: "assistant",
  model: "claude-3-5",
  content: [
    { type: "text", text: "Let me check" },
    { type: "tool_use", id: "toolu_1", name: "get_weather", input: { city: "SF" } },
  ],
  stop_reason: "tool_use",
  usage: { input_tokens: 10, output_tokens: 8 },
};

const ANTHROPIC_SSE = [
  `event: message_start`,
  `data: {"type":"message_start","message":{"id":"msg_1","model":"claude-3","role":"assistant","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}`,
  ``,
  `event: content_block_start`,
  `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
  ``,
  `event: content_block_delta`,
  `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
  ``,
  `event: content_block_delta`,
  `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}`,
  ``,
  `event: content_block_stop`,
  `data: {"type":"content_block_stop","index":0}`,
  ``,
  `event: message_delta`,
  `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
  ``,
  `event: message_stop`,
  `data: {"type":"message_stop"}`,
  ``,
].join("\n");

// A *non-redacting* upstream (i.e. what the official Anthropic API returns when
// extended thinking is on): the thinking block carries the real CoT text + a
// verification signature. Public relays often strip this; these fixtures prove
// agent-trace loses nothing when the upstream actually sends it.
const THINK_A = "Let me factor 9973. It's odd, not divisible by 3, 7, 11, 13... ";
const THINK_B = "checking primes up to 99: none divide it, so 9973 is prime.";
const THINK_SIG = "ErcBSGVsbG9TaWduYXR1cmVCbG9iRm9yVmVyaWZpY2F0aW9u";
const THINK_FULL = THINK_A + THINK_B;

const ANTHROPIC_THINKING_JSON: Dict = {
  id: "msg_think",
  type: "message",
  role: "assistant",
  model: "claude-thinking",
  content: [
    { type: "thinking", thinking: THINK_FULL, signature: THINK_SIG },
    { type: "text", text: "9973 is prime." },
  ],
  stop_reason: "end_turn",
  usage: {
    input_tokens: 12,
    output_tokens: 40,
    output_tokens_details: { thinking_tokens: 30 },
  },
};

const ANTHROPIC_THINKING_SSE = [
  `event: message_start`,
  `data: {"type":"message_start","message":{"id":"msg_think_s","model":"claude-thinking","role":"assistant","content":[],"usage":{"input_tokens":12,"output_tokens":0}}}`,
  ``,
  `event: content_block_start`,
  `data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
  ``,
  `event: content_block_delta`,
  `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":${JSON.stringify(THINK_A)}}}`,
  ``,
  `event: content_block_delta`,
  `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":${JSON.stringify(THINK_B)}}}`,
  ``,
  `event: content_block_delta`,
  `data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":${JSON.stringify(THINK_SIG)}}}`,
  ``,
  `event: content_block_stop`,
  `data: {"type":"content_block_stop","index":0}`,
  ``,
  `event: content_block_start`,
  `data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
  ``,
  `event: content_block_delta`,
  `data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"9973 is prime."}}`,
  ``,
  `event: content_block_stop`,
  `data: {"type":"content_block_stop","index":1}`,
  ``,
  `event: message_delta`,
  `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":40}}`,
  ``,
  `event: message_stop`,
  `data: {"type":"message_stop"}`,
  ``,
].join("\n");

let upstream: Server;
let upstreamPort: number;
let proxy: CaptureProxy;
let proxyPort: number;

async function readBody(req: IncomingMessage): Promise<Dict> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const text = Buffer.concat(chunks).toString("utf-8");
  return text ? (JSON.parse(text) as Dict) : {};
}

beforeAll(async () => {
  upstream = createServer((req, res) => {
    void (async () => {
      const body = await readBody(req);
      received.push({ url: req.url ?? "", headers: req.headers, body });
      const stream = body["stream"] === true;
      const url = req.url ?? "";
      if (url.includes("/v1/messages")) {
        const wantsThinking =
          typeof body["model"] === "string" && (body["model"] as string).includes("thinking");
        if (wantsThinking) {
          if (stream) {
            res.writeHead(200, { "Content-Type": "text/event-stream" });
            res.end(ANTHROPIC_THINKING_SSE);
          } else {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify(ANTHROPIC_THINKING_JSON));
          }
          return;
        }
        if (stream) {
          res.writeHead(200, { "Content-Type": "text/event-stream" });
          res.end(ANTHROPIC_SSE);
        } else {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(ANTHROPIC_JSON));
        }
        return;
      }
      if (url.includes("/v1/chat/completions")) {
        if (stream) {
          res.writeHead(200, { "Content-Type": "text/event-stream" });
          res.end(OPENAI_SSE);
        } else {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(OPENAI_JSON));
        }
        return;
      }
      res.writeHead(404).end();
    })();
  });
  upstreamPort = await new Promise<number>((resolve) => {
    upstream.listen(0, "127.0.0.1", () => {
      const addr = upstream.address();
      resolve(typeof addr === "object" && addr ? addr.port : 0);
    });
  });

  proxy = new CaptureProxy({
    mode: "passthrough",
    upstreams: { default: `http://127.0.0.1:${upstreamPort}` },
    logger: { info: () => {}, warn: () => {}, error: () => {} },
  });
  proxyPort = await proxy.listen(0, "127.0.0.1");
});

afterAll(async () => {
  await proxy.close();
  await new Promise<void>((resolve) => upstream.close(() => resolve()));
});

function completionsOf(sid: string): Promise<Dict> {
  return fetch(`http://127.0.0.1:${proxyPort}/sessions/${sid}/completions`).then(
    (r) => r.json() as Promise<Dict>,
  );
}

describe("passthrough capture proxy", () => {
  it("forwards OpenAI chat verbatim (real key upstream) and captures the trace", async () => {
    received.length = 0;
    const sid = "sess-oai";
    const resp = await fetch(`http://127.0.0.1:${proxyPort}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer REAL-OPENAI-KEY",
        "x-session-id": sid,
      },
      body: JSON.stringify({ model: "gpt-4o", messages: [{ role: "user", content: "Q1" }] }),
    });
    expect(resp.status).toBe(200);
    // Client receives the upstream response untouched.
    const j = (await resp.json()) as Dict;
    expect(j["id"]).toBe("c2");
    expect(j["model"]).toBe("gpt-4o-2024");

    // Upstream saw the original body + the real bearer key (not a session id).
    const last = received.at(-1)!;
    expect(last.url).toContain("/v1/chat/completions");
    expect(last.headers["authorization"]).toBe("Bearer REAL-OPENAI-KEY");
    expect((last.body["model"] as string)).toBe("gpt-4o");

    // Capture is grouped under the session id, normalized to OpenAI chat.
    const listed = await completionsOf(sid);
    const completions = listed["completions"] as Dict[];
    expect(completions).toHaveLength(1);
    const rec = completions[0]!;
    const reqMsgs = (rec["request"] as Dict)["messages"] as Dict[];
    expect(reqMsgs.at(-1)!["content"]).toBe("Q1");
    const msg = ((rec["response"] as Dict)["choices"] as Dict[])[0]!["message"] as Dict;
    expect(msg["content"]).toBe("Answer");
  });

  it("captures Anthropic tool_use, normalizing it to OpenAI tool_calls", async () => {
    received.length = 0;
    const sid = "sess-ant";
    const resp = await fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": "REAL-ANTHROPIC-KEY",
        "anthropic-version": "2023-06-01",
        "x-session-id": sid,
      },
      body: JSON.stringify({
        model: "claude-3-5-sonnet",
        max_tokens: 1024,
        system: "You are helpful.",
        messages: [{ role: "user", content: "weather in SF?" }],
      }),
    });
    expect(resp.status).toBe(200);
    const j = (await resp.json()) as Dict;
    // Verbatim Anthropic body to the client.
    expect(j["id"]).toBe("msg_2");
    expect(Array.isArray(j["content"])).toBe(true);

    // Upstream received the real anthropic key + version header.
    const last = received.at(-1)!;
    expect(last.url).toContain("/v1/messages");
    expect(last.headers["x-api-key"]).toBe("REAL-ANTHROPIC-KEY");
    expect(last.headers["anthropic-version"]).toBe("2023-06-01");

    // Normalized capture: assistant text + an OpenAI-style tool call.
    const completions = (await completionsOf(sid))["completions"] as Dict[];
    expect(completions).toHaveLength(1);
    const rec = completions[0]!;
    const msg = ((rec["response"] as Dict)["choices"] as Dict[])[0]!["message"] as Dict;
    expect(msg["content"]).toBe("Let me check");
    const toolCalls = msg["tool_calls"] as Dict[];
    expect(toolCalls).toHaveLength(1);
    expect((toolCalls[0]!["function"] as Dict)["name"]).toBe("get_weather");
    expect(JSON.parse(String((toolCalls[0]!["function"] as Dict)["arguments"]))).toEqual({
      city: "SF",
    });
    // The native Anthropic request is preserved losslessly.
    expect((rec["original_request"] as Dict)["system"]).toBe("You are helpful.");
  });

  it("streams OpenAI SSE through and reconstructs the captured content", async () => {
    received.length = 0;
    const sid = "sess-oai-stream";
    const resp = await fetch(`http://127.0.0.1:${proxyPort}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-session-id": sid },
      body: JSON.stringify({
        model: "gpt-4o",
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(resp.headers.get("content-type")).toContain("text/event-stream");
    const text = await resp.text();
    expect(text).toContain("data: [DONE]");

    const completions = (await completionsOf(sid))["completions"] as Dict[];
    expect(completions).toHaveLength(1);
    const msg = ((completions[0]!["response"] as Dict)["choices"] as Dict[])[0]!["message"] as Dict;
    expect(msg["content"]).toBe("Hello");
  });

  it("streams Anthropic SSE through and reconstructs the captured content", async () => {
    received.length = 0;
    const sid = "sess-ant-stream";
    const resp = await fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-session-id": sid,
      },
      body: JSON.stringify({
        model: "claude-3-5-sonnet",
        max_tokens: 1024,
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(resp.headers.get("content-type")).toContain("text/event-stream");
    const text = await resp.text();
    expect(text).toContain("message_start");

    const completions = (await completionsOf(sid))["completions"] as Dict[];
    expect(completions).toHaveLength(1);
    const msg = ((completions[0]!["response"] as Dict)["choices"] as Dict[])[0]!["message"] as Dict;
    expect(msg["content"]).toBe("Hi there");
  });

  it("preserves Claude thinking CoT (non-streaming): raw_response -> reasoning_content -> SFT", async () => {
    received.length = 0;
    const sid = "sess-think";
    const resp = await fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": "REAL-ANTHROPIC-KEY",
        "anthropic-version": "2023-06-01",
        "x-session-id": sid,
      },
      body: JSON.stringify({
        model: "claude-thinking",
        max_tokens: 2048,
        thinking: { type: "enabled", budget_tokens: 1024 },
        messages: [{ role: "user", content: "Is 9973 prime?" }],
      }),
    });
    expect(resp.status).toBe(200);

    const rec = ((await completionsOf(sid))["completions"] as Dict[])[0]!;

    // (1) raw_response keeps the native thinking block verbatim: full CoT + signature.
    const raw = (rec["metadata"] as Dict)["raw_response"] as Dict;
    const think = (raw["content"] as Dict[]).find((b) => b["type"] === "thinking")!;
    expect(think["thinking"]).toBe(THINK_FULL);
    expect(think["signature"]).toBe(THINK_SIG);

    // (2) the normalized response exposes the CoT as reasoning_content (text stays separate).
    const msg = ((rec["response"] as Dict)["choices"] as Dict[])[0]!["message"] as Dict;
    expect(msg["reasoning_content"]).toBe(THINK_FULL);
    expect(msg["content"]).toBe("9973 is prime.");

    // (3) the CoT rides all the way into the SFT sample's assistant message.
    const trace = buildTraceFromCompletion(rec as unknown as CompletionRecord);
    const sft = toSFTSample(trace);
    const asst = sft.messages.find((m) => m["role"] === "assistant")!;
    expect(asst["reasoning_content"]).toBe(THINK_FULL);
  });

  it("reassembles streamed thinking_delta + signature_delta into the captured CoT", async () => {
    received.length = 0;
    const sid = "sess-think-stream";
    const resp = await fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-session-id": sid,
      },
      body: JSON.stringify({
        model: "claude-thinking",
        max_tokens: 2048,
        stream: true,
        thinking: { type: "enabled", budget_tokens: 1024 },
        messages: [{ role: "user", content: "Is 9973 prime?" }],
      }),
    });
    expect(resp.headers.get("content-type")).toContain("text/event-stream");
    // The client still receives the native thinking stream untouched.
    const text = await resp.text();
    expect(text).toContain("thinking_delta");
    expect(text).toContain("signature_delta");

    const rec = ((await completionsOf(sid))["completions"] as Dict[])[0]!;
    const raw = (rec["metadata"] as Dict)["raw_response"] as Dict;
    const think = (raw["content"] as Dict[]).find((b) => b["type"] === "thinking")!;
    // Reassembled from two thinking_delta chunks + a signature_delta.
    expect(think["thinking"]).toBe(THINK_FULL);
    expect(think["signature"]).toBe(THINK_SIG);

    const msg = ((rec["response"] as Dict)["choices"] as Dict[])[0]!["message"] as Dict;
    expect(msg["reasoning_content"]).toBe(THINK_FULL);
    expect(msg["content"]).toBe("9973 is prime.");
  });
});
