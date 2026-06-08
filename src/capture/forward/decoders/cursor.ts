/**
 * Cursor agent decoder.
 *
 * Cursor's CLI/IDE talk to `*.cursor.sh` over Connect-RPC + protobuf (HTTP/2),
 * not the OpenAI/Anthropic JSON wire, so they ignore `*_BASE_URL` and are only
 * capturable via TLS interception. The agent request/response frames embed the
 * conversation as AI-SDK JSON messages inside protobuf string fields, which we
 * recover schema-lessly (see protoScan) and reshape into an OpenAI-chat pair.
 *
 * Limitations (inherent to the wire, not the decoder): reasoning is delivered
 * as opaque `redacted-reasoning` blobs (no token-level CoT), and exact sampling
 * params aren't present — so this yields SFT-grade message traces, not the
 * token-id/logprob signal needed for on-policy RL.
 */

import { decodeMessageFrames } from "../connect.js";
import { collectRoleMessages } from "../protoScan.js";
import {
  type DecodedCapture,
  type Dict,
  headerValue,
  type HttpExchange,
  type WireDecoder,
} from "./types.js";

function isProtoLike(ex: HttpExchange): boolean {
  const ct = (
    headerValue(ex.resHeaders, "content-type") ??
    headerValue(ex.reqHeaders, "content-type") ??
    ""
  ).toLowerCase();
  return (
    ct.includes("proto") ||
    ct.includes("grpc") ||
    ct.includes("connect") ||
    ct.includes("application/octet-stream")
  );
}

function pickModel(messages: Dict[]): string | undefined {
  for (const m of messages) {
    const direct = m["model"];
    if (typeof direct === "string" && direct) return direct;
    const po = m["providerOptions"] ?? m["providerMetadata"];
    if (po && typeof po === "object") {
      for (const v of Object.values(po as Dict)) {
        if (v && typeof v === "object") {
          const name = (v as Dict)["modelName"] ?? (v as Dict)["model"];
          if (typeof name === "string" && name) return name;
        }
      }
    }
  }
  return undefined;
}

interface NormalizedMessage {
  role: string;
  content: string | null;
  reasoning?: string;
  tool_calls?: Dict[];
  tool_call_id?: string;
}

/** Map one AI-SDK message (string or content-part array) to OpenAI-chat shape. */
function normalizeMessage(m: Dict): NormalizedMessage {
  const role = String(m["role"] ?? "user");
  const content = m["content"];

  if (typeof content === "string") {
    return { role, content };
  }
  if (!Array.isArray(content)) {
    return { role, content: content == null ? "" : JSON.stringify(content) };
  }

  const texts: string[] = [];
  const reasoning: string[] = [];
  const toolCalls: Dict[] = [];
  let toolCallId: string | undefined;
  let toolResult: unknown;

  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as Dict;
    switch (p["type"]) {
      case "text":
        if (typeof p["text"] === "string") texts.push(p["text"]);
        break;
      case "reasoning":
        if (typeof p["text"] === "string") reasoning.push(p["text"]);
        break;
      case "redacted-reasoning":
        reasoning.push("[redacted-reasoning]");
        break;
      case "tool-call":
        toolCalls.push({
          id: String(p["toolCallId"] ?? `call_${toolCalls.length}`),
          type: "function",
          function: {
            name: String(p["toolName"] ?? ""),
            arguments: JSON.stringify(p["args"] ?? p["input"] ?? {}),
          },
        });
        break;
      case "tool-result":
        toolCallId = typeof p["toolCallId"] === "string" ? p["toolCallId"] : toolCallId;
        toolResult = p["result"] ?? p["output"];
        break;
      default:
        break;
    }
  }

  const out: NormalizedMessage = {
    role,
    content:
      role === "tool"
        ? typeof toolResult === "string"
          ? toolResult
          : JSON.stringify(toolResult ?? null)
        : texts.length > 0
          ? texts.join("")
          : toolCalls.length > 0
            ? null
            : "",
  };
  if (reasoning.length > 0) out.reasoning = reasoning.join("\n");
  if (toolCalls.length > 0) out.tool_calls = toolCalls;
  if (toolCallId) out.tool_call_id = toolCallId;
  return out;
}

function toOpenAiMessage(n: NormalizedMessage): Dict {
  const msg: Dict = { role: n.role, content: n.content };
  if (n.reasoning) msg["reasoning_content"] = n.reasoning;
  if (n.tool_calls) msg["tool_calls"] = n.tool_calls;
  if (n.tool_call_id) msg["tool_call_id"] = n.tool_call_id;
  return msg;
}

/** Recover the conversation JSON messages from a (possibly framed) protobuf body. */
function messagesFromBody(body: Buffer, encoding?: string | null): Dict[] {
  if (body.length === 0) return [];
  // Prefer Connect/gRPC framing; fall back to scanning the raw bytes.
  const frames = decodeMessageFrames(body, encoding);
  const found: Dict[] = [];
  for (const f of frames) found.push(...collectRoleMessages(f));
  if (found.length === 0) found.push(...collectRoleMessages(body));
  return found;
}

export const cursorDecoder: WireDecoder = {
  name: "cursor_agent",

  matches(ex: HttpExchange): boolean {
    if (!ex.host.toLowerCase().includes("cursor.sh")) return false;
    if (!isProtoLike(ex)) return false;
    // Cheap structural probe: does either body carry role-tagged JSON?
    return (
      messagesFromBody(ex.reqBody, headerValue(ex.reqHeaders, "grpc-encoding")).length > 0 ||
      messagesFromBody(ex.resBody, headerValue(ex.resHeaders, "grpc-encoding")).length > 0
    );
  },

  decode(ex: HttpExchange): DecodedCapture | null {
    const reqMsgs = messagesFromBody(ex.reqBody, headerValue(ex.reqHeaders, "grpc-encoding"));
    const resMsgs = messagesFromBody(ex.resBody, headerValue(ex.resHeaders, "grpc-encoding"));
    // The response stream echoes the full transcript incl. the assistant turn;
    // prefer it, falling back to the request's message list.
    const transcript = resMsgs.length >= reqMsgs.length ? resMsgs : reqMsgs;
    if (transcript.length === 0) return null;

    const normalized = transcript.map(normalizeMessage);
    // The stream echoes the assistant turn as partial+final messages; collapse
    // the trailing run of assistant messages into a single completion (longest).
    let runStart = normalized.length;
    while (runStart - 1 >= 0 && normalized[runStart - 1]!.role === "assistant") runStart--;
    if (runStart === normalized.length) return null; // no assistant turn

    const assistant = normalized
      .slice(runStart)
      .reduce((a, b) => ((b.content?.length ?? 0) >= (a.content?.length ?? 0) ? b : a));
    const priorMessages = normalized.slice(0, runStart).map(toOpenAiMessage);
    const model = pickModel(transcript) ?? "cursor-unknown";

    const request: Dict = { model, messages: priorMessages };
    const assistantMsg = toOpenAiMessage(assistant);
    const response: Dict = {
      object: "chat.completion",
      model,
      choices: [
        {
          index: 0,
          message: assistantMsg,
          finish_reason: assistant.tool_calls ? "tool_calls" : "stop",
        },
      ],
    };

    return {
      request,
      response,
      model,
      apiType: "cursor_agent",
      metadata: {
        wire: "connect-rpc+protobuf",
        host: ex.host,
        path: ex.path,
        message_count: normalized.length,
      },
    };
  },
};
