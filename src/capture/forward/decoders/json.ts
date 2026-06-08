/**
 * JSON-wire decoder for the MITM forward proxy.
 *
 * Covers clients that DO speak OpenAI-chat / Anthropic JSON but reach the API
 * over a pinned/hard-coded host (ignoring `*_BASE_URL`), so they're only
 * capturable via TLS interception. Reuses the same normalization as the
 * base-URL passthrough path, including SSE assembly.
 */

import { brotliDecompressSync, gunzipSync, inflateSync } from "node:zlib";

import { APIType, detect, extractModel } from "../../detection.js";
import {
  anthropicResponseToOpenAi,
  assembleAnthropicStream,
  assembleOpenAiChatStream,
  normalizeCapturedRequest,
} from "../../passthrough.js";
import { TransformManager } from "../../transform/index.js";
import {
  type DecodedCapture,
  type Dict,
  headerValue,
  type HttpExchange,
  type WireDecoder,
} from "./types.js";

const transforms = new TransformManager();

function decompress(body: Buffer, encoding?: string): Buffer {
  if (body.length === 0) return body;
  const enc = (encoding ?? "").toLowerCase();
  try {
    if (enc.includes("br")) return brotliDecompressSync(body);
    if (enc.includes("gzip")) return gunzipSync(body);
    if (enc.includes("deflate")) return inflateSync(body);
  } catch {
    /* fall through to raw */
  }
  return body;
}

function parseJson(buf: Buffer): Dict | null {
  try {
    const v = JSON.parse(buf.toString("utf-8"));
    return typeof v === "object" && v !== null ? (v as Dict) : null;
  } catch {
    return null;
  }
}

function looksLikeChat(body: Dict): boolean {
  return Array.isArray(body["messages"]) || Array.isArray(body["contents"]);
}

export const jsonDecoder: WireDecoder = {
  name: "json_chat",

  matches(ex: HttpExchange): boolean {
    const ct = (headerValue(ex.reqHeaders, "content-type") ?? "").toLowerCase();
    if (!ct.includes("json")) return false;
    const req = parseJson(ex.reqBody);
    return req !== null && looksLikeChat(req);
  },

  decode(ex: HttpExchange): DecodedCapture | null {
    const reqBody = parseJson(ex.reqBody);
    if (reqBody === null) return null;

    const apiType = detect(ex.path, ex.reqHeaders, reqBody);
    if (apiType !== APIType.OPENAI_CHAT && apiType !== APIType.ANTHROPIC) return null;

    const resCt = (headerValue(ex.resHeaders, "content-type") ?? "").toLowerCase();
    const resBytes = decompress(ex.resBody, headerValue(ex.resHeaders, "content-encoding"));

    let native: Dict | null;
    if (resCt.includes("event-stream")) {
      const text = resBytes.toString("utf-8");
      native =
        apiType === APIType.ANTHROPIC
          ? assembleAnthropicStream(text)
          : assembleOpenAiChatStream(text);
    } else {
      native = parseJson(resBytes);
    }
    if (native === null) return null;

    const transformer = transforms.get(apiType);
    const request = normalizeCapturedRequest(transformer, reqBody);
    const response =
      apiType === APIType.ANTHROPIC ? anthropicResponseToOpenAi(native) : native;
    const model =
      (response["model"] as string | undefined) ?? extractModel(apiType, reqBody) ?? "unknown";

    return {
      request,
      response,
      model,
      apiType,
      metadata: { wire: "json", host: ex.host, path: ex.path },
    };
  },
};
