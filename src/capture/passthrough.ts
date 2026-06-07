/**
 * Capture-only passthrough: forward an agent's request verbatim to the API it
 * already uses (Anthropic / OpenAI, public or a relay) and capture the full
 * trace on the wire, without a self-hosted inference server.
 *
 * This is what makes agent-trace a universal, runtime-agnostic capture plugin:
 * any harness that lets you redirect its base URL is captured, with no payload
 * rewriting. The forwarded bytes are untouched (lossless to the client); a
 * normalized OpenAI-chat copy is derived purely for SFT/RL export.
 *
 * Frontier APIs don't return token ids / logprobs, so captures are
 * message-level (text / tool calls / results / reasoning). Token-level
 * on-policy RL still requires the inference-mode proxy + a model you serve.
 */

import { APIType } from "./detection.js";
import {
  cleanSessionId,
  generateSessionId,
  type SessionRegistry,
} from "./sessionId.js";
import type { BaseTransformer } from "./transform/base.js";

type Dict = Record<string, unknown>;

function isDict(v: unknown): v is Dict {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function lowerHeaders(headers: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) out[k.toLowerCase()] = v;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream selection + header / session handling
// ---------------------------------------------------------------------------

/** Per-API-family upstream base URLs. `default` routes every family to one. */
export interface PassthroughUpstreams {
  anthropic?: string;
  openai?: string;
  default?: string;
}

const PUBLIC_UPSTREAMS = {
  anthropic: "https://api.anthropic.com",
  openai: "https://api.openai.com",
} as const;

/** Resolve the upstream base URL for an API family (public defaults, no trailing slash). */
export function resolveUpstreamBase(apiType: APIType, upstreams: PassthroughUpstreams): string {
  const base =
    apiType === APIType.ANTHROPIC
      ? (upstreams.anthropic ?? upstreams.default ?? PUBLIC_UPSTREAMS.anthropic)
      : (upstreams.openai ?? upstreams.default ?? PUBLIC_UPSTREAMS.openai);
  return base.replace(/\/+$/, "");
}

const HOP_BY_HOP = new Set([
  "host",
  "content-length",
  "connection",
  "transfer-encoding",
  "accept-encoding",
  "content-encoding",
]);

/** Copy request headers for forwarding, dropping hop-by-hop ones (keeps auth). */
export function filterForwardHeaders(headers: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) {
    if (!HOP_BY_HOP.has(k.toLowerCase())) out[k] = v;
  }
  return out;
}

/**
 * Resolve the capture session id. Unlike the inference proxy, the bearer token
 * here is the user's *real* API key (forwarded upstream), so it must NOT be used
 * as a session id. We honor an explicit x-session-id / ?session_id, else fall
 * back to the proxy's default session (one session per `capture` run).
 */
export function resolveCaptureSessionId(
  registry: SessionRegistry,
  headers: Record<string, string>,
  querySessionId: string | null | undefined,
  fallback: string,
): string {
  const lower = lowerHeaders(headers);
  const safe = (v: string | null | undefined): string | null => {
    try {
      return cleanSessionId(v);
    } catch {
      return null;
    }
  };
  const id =
    safe(lower["x-session-id"]) ??
    safe(lower["x_session_id"]) ??
    safe(querySessionId) ??
    safe(fallback) ??
    generateSessionId();
  if (registry.get(id) === undefined) registry.register(id);
  else registry.updateActivity(id);
  return id;
}

/** Headers to echo back to the client for a streamed (SSE) passthrough response. */
export function passthroughStreamHeaders(upstream: Response): Record<string, string> {
  return {
    "Content-Type": upstream.headers.get("content-type") ?? "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  };
}

/** Headers to echo back for a non-streamed passthrough response. */
export function passthroughJsonHeaders(upstream: Response): Record<string, string> {
  return { "Content-Type": upstream.headers.get("content-type") ?? "application/json" };
}

// ---------------------------------------------------------------------------
// Request / response normalization (native -> OpenAI chat, for export only)
// ---------------------------------------------------------------------------

/** Normalize the captured request into OpenAI-chat shape via the family transformer. */
export function normalizeCapturedRequest(transformer: BaseTransformer, body: Dict): Dict {
  return transformer.transformRequest(structuredClone(body));
}

/** Normalize an upstream native response into an OpenAI-chat completion. */
export function normalizeCapturedResponse(apiType: APIType, native: Dict): Dict {
  if (apiType === APIType.ANTHROPIC) return anthropicResponseToOpenAi(native);
  return native; // OpenAI chat is already canonical
}

function anthropicUsageToOpenAi(usage: Dict): Dict {
  const prompt = Number(usage["input_tokens"] ?? 0);
  const completion = Number(usage["output_tokens"] ?? 0);
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
  };
}

function anthropicStopToFinishReason(stop: unknown): string {
  if (stop === "tool_use") return "tool_calls";
  if (stop === "max_tokens") return "length";
  return "stop";
}

/** Convert an Anthropic Messages response object into an OpenAI chat completion. */
export function anthropicResponseToOpenAi(msg: Dict): Dict {
  const content = Array.isArray(msg["content"]) ? msg["content"] : [];
  const textParts: string[] = [];
  const toolCalls: Dict[] = [];
  let reasoning = "";

  for (const block of content) {
    if (!isDict(block)) continue;
    const type = block["type"];
    if (type === "text" && typeof block["text"] === "string") {
      textParts.push(block["text"]);
    } else if (type === "thinking" && typeof block["thinking"] === "string") {
      reasoning += block["thinking"];
    } else if (type === "tool_use") {
      toolCalls.push({
        id: String(block["id"] ?? `call_${toolCalls.length}`),
        type: "function",
        function: {
          name: String(block["name"] ?? ""),
          arguments: JSON.stringify(block["input"] ?? {}),
        },
      });
    }
  }

  const message: Dict = {
    role: "assistant",
    content: textParts.length > 0 ? textParts.join("") : toolCalls.length > 0 ? null : "",
  };
  if (reasoning.length > 0) message["reasoning_content"] = reasoning;
  if (toolCalls.length > 0) message["tool_calls"] = toolCalls;

  const out: Dict = {
    id: msg["id"],
    object: "chat.completion",
    model: msg["model"],
    choices: [
      { index: 0, message, finish_reason: anthropicStopToFinishReason(msg["stop_reason"]) },
    ],
  };
  if (isDict(msg["usage"])) out["usage"] = anthropicUsageToOpenAi(msg["usage"]);
  return out;
}

// ---------------------------------------------------------------------------
// SSE assembly (streamed responses -> a single native response object)
// ---------------------------------------------------------------------------

interface SseEvent {
  event?: string;
  data: string;
}

/** Parse an SSE text buffer into discrete events (handles multi-line data). */
export function parseSse(text: string): SseEvent[] {
  const events: SseEvent[] = [];
  let event: string | undefined;
  let data: string[] = [];
  const flush = (): void => {
    if (data.length > 0 || event !== undefined) events.push({ event, data: data.join("\n") });
    event = undefined;
    data = [];
  };
  for (const line of text.split(/\r?\n/)) {
    if (line === "") {
      flush();
      continue;
    }
    if (line.startsWith(":")) continue;
    const idx = line.indexOf(":");
    const field = idx === -1 ? line : line.slice(0, idx);
    let value = idx === -1 ? "" : line.slice(idx + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  flush();
  return events;
}

interface ToolCallAcc {
  id?: string;
  type: string;
  name: string;
  args: string;
}

/** Reconstruct a non-streaming OpenAI chat completion from its SSE stream. */
export function assembleOpenAiChatStream(sseText: string): Dict {
  let id: unknown;
  let model: unknown;
  let created: unknown;
  let content = "";
  let reasoning = "";
  let finishReason: string | null = null;
  let usage: Dict | undefined;
  const toolCalls = new Map<number, ToolCallAcc>();

  for (const ev of parseSse(sseText)) {
    if (ev.data === "[DONE]" || ev.data === "") continue;
    let chunk: Dict;
    try {
      chunk = JSON.parse(ev.data) as Dict;
    } catch {
      continue;
    }
    if (id === undefined && chunk["id"] !== undefined) id = chunk["id"];
    if (model === undefined && chunk["model"] !== undefined) model = chunk["model"];
    if (created === undefined && chunk["created"] !== undefined) created = chunk["created"];
    if (isDict(chunk["usage"])) usage = chunk["usage"];

    const choices = chunk["choices"];
    if (!Array.isArray(choices) || choices.length === 0) continue;
    const choice = choices[0];
    if (!isDict(choice)) continue;
    if (choice["finish_reason"] != null) finishReason = String(choice["finish_reason"]);
    const delta = choice["delta"];
    if (!isDict(delta)) continue;
    if (typeof delta["content"] === "string") content += delta["content"];
    if (typeof delta["reasoning_content"] === "string") reasoning += delta["reasoning_content"];
    const deltaToolCalls = delta["tool_calls"];
    if (Array.isArray(deltaToolCalls)) {
      for (const tc of deltaToolCalls) {
        if (!isDict(tc)) continue;
        const index = typeof tc["index"] === "number" ? tc["index"] : 0;
        const acc = toolCalls.get(index) ?? { type: "function", name: "", args: "" };
        if (typeof tc["id"] === "string") acc.id = tc["id"];
        if (typeof tc["type"] === "string") acc.type = tc["type"];
        const fn = tc["function"];
        if (isDict(fn)) {
          if (typeof fn["name"] === "string") acc.name += fn["name"];
          if (typeof fn["arguments"] === "string") acc.args += fn["arguments"];
        }
        toolCalls.set(index, acc);
      }
    }
  }

  const message: Dict = {
    role: "assistant",
    content: content.length > 0 ? content : toolCalls.size > 0 ? null : "",
  };
  if (reasoning.length > 0) message["reasoning_content"] = reasoning;
  if (toolCalls.size > 0) {
    message["tool_calls"] = [...toolCalls.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, acc]) => ({
        id: acc.id ?? "",
        type: acc.type,
        function: { name: acc.name, arguments: acc.args },
      }));
  }

  const out: Dict = {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [{ index: 0, message, finish_reason: finishReason ?? "stop" }],
  };
  if (usage) out["usage"] = usage;
  return out;
}

/** Reconstruct a non-streaming Anthropic Messages response from its SSE stream. */
export function assembleAnthropicStream(sseText: string): Dict {
  const message: Dict = { type: "message", role: "assistant", content: [] };
  const blocks = new Map<number, Dict>();
  const argBufs = new Map<number, string>();
  const order: number[] = [];
  let stopReason: string | null = null;
  let usage: Dict | undefined;

  for (const ev of parseSse(sseText)) {
    if (ev.data === "") continue;
    let data: Dict;
    try {
      data = JSON.parse(ev.data) as Dict;
    } catch {
      continue;
    }
    const type = (data["type"] ?? ev.event) as string | undefined;

    if (type === "message_start" && isDict(data["message"])) {
      const m = data["message"];
      if (m["id"] !== undefined) message["id"] = m["id"];
      if (m["model"] !== undefined) message["model"] = m["model"];
      if (m["role"] !== undefined) message["role"] = m["role"];
      if (isDict(m["usage"])) usage = { ...m["usage"] };
    } else if (type === "content_block_start") {
      const index = Number(data["index"] ?? 0);
      const block: Dict = isDict(data["content_block"])
        ? structuredClone(data["content_block"])
        : {};
      if (block["type"] === "text" && typeof block["text"] !== "string") block["text"] = "";
      if (block["type"] === "thinking" && typeof block["thinking"] !== "string") {
        block["thinking"] = "";
      }
      if (block["type"] === "tool_use") argBufs.set(index, "");
      blocks.set(index, block);
      order.push(index);
    } else if (type === "content_block_delta") {
      const index = Number(data["index"] ?? 0);
      const block = blocks.get(index);
      const delta = data["delta"];
      if (!block || !isDict(delta)) continue;
      const deltaType = delta["type"];
      if (deltaType === "text_delta" && typeof delta["text"] === "string") {
        block["text"] = (typeof block["text"] === "string" ? block["text"] : "") + delta["text"];
      } else if (deltaType === "thinking_delta" && typeof delta["thinking"] === "string") {
        block["thinking"] =
          (typeof block["thinking"] === "string" ? block["thinking"] : "") + delta["thinking"];
      } else if (deltaType === "input_json_delta" && typeof delta["partial_json"] === "string") {
        argBufs.set(index, (argBufs.get(index) ?? "") + delta["partial_json"]);
      } else if (deltaType === "signature_delta" && typeof delta["signature"] === "string") {
        block["signature"] = delta["signature"];
      }
    } else if (type === "message_delta") {
      const delta = data["delta"];
      if (isDict(delta) && delta["stop_reason"] != null) stopReason = String(delta["stop_reason"]);
      if (isDict(data["usage"])) usage = { ...(usage ?? {}), ...data["usage"] };
    }
  }

  const content: Dict[] = [];
  for (const index of order) {
    const block = blocks.get(index);
    if (!block) continue;
    if (block["type"] === "tool_use") {
      const buf = argBufs.get(index) ?? "";
      if (buf.length > 0) {
        try {
          block["input"] = JSON.parse(buf);
        } catch {
          block["input"] = block["input"] ?? {};
        }
      } else if (block["input"] === undefined) {
        block["input"] = {};
      }
    }
    content.push(block);
  }
  message["content"] = content;
  if (stopReason !== null) message["stop_reason"] = stopReason;
  if (usage) message["usage"] = usage;
  return message;
}
