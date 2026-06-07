/**
 * The transparent capture proxy.
 *
 * An agent runtime points its OPENAI_BASE_URL (and bearer token = session id)
 * at this server. Every POST is detected, transformed to OpenAI chat, forwarded
 * to the inference backend with training-signal params, captured as a
 * completion record, then transformed back into the shape the agent expects
 * (with synthetic SSE when the agent asked to stream). A `finalize` endpoint
 * builds the trajectory and emits SFT/RL JSONL.
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { join } from "node:path";

import {
  type BuilderRegistry,
  defaultBuilderRegistry,
} from "../trajectory/registry.js";
import { trajectoryToRLSamples } from "../serialize/toRLSample.js";
import { trajectoryToSFTSamples } from "../serialize/toSFTSample.js";
import { toJsonl } from "../serialize/writeJsonl.js";
import { mkdir, writeFile } from "node:fs/promises";

import { APIType, detect, extractModel } from "./detection.js";
import { getEngine } from "./engine.js";
import {
  InferenceClient,
  UpstreamError,
  UpstreamHTTPError,
  UpstreamTimeoutError,
} from "./inferenceClient.js";
import { CompletionWriter } from "./completionWriter.js";
import { SessionStore } from "./sessionStore.js";
import {
  cleanSessionId,
  InvalidSessionIdError,
  resolveSessionId,
  SessionRegistry,
  type SessionInfo,
} from "./sessionId.js";
import { TransformManager } from "./transform/index.js";
import type { BaseTransformer } from "./transform/base.js";
import {
  assembleAnthropicStream,
  assembleOpenAiChatStream,
  filterForwardHeaders,
  normalizeCapturedRequest,
  normalizeCapturedResponse,
  type PassthroughUpstreams,
  passthroughJsonHeaders,
  passthroughStreamHeaders,
  resolveCaptureSessionId,
  resolveUpstreamBase,
} from "./passthrough.js";

type Dict = Record<string, unknown>;

/**
 * - `inference`: forward to a self-hosted OpenAI-compatible server (vLLM/SGLang),
 *   injecting token-id/logprob params — token-level capture for on-policy RL.
 * - `passthrough`: forward verbatim to the API the agent already uses
 *   (Anthropic/OpenAI, public or relay), capturing the full trace message-level.
 *   The universal, no-backend plugin path.
 */
export type ProxyMode = "inference" | "passthrough";

export interface ProxyConfig {
  /** Capture mode. Default: "inference". */
  mode?: ProxyMode;
  /** [inference] OpenAI-compatible inference server base URL. */
  inferenceBaseUrl?: string;
  /** [inference] Model name forwarded to the backend (overrides the agent's model). */
  modelServed?: string;
  /** [inference] Backend that emits token ids + logprobs. Default: vllm. */
  engine?: "sglang" | "vllm";
  /** [passthrough] Per-API-family upstream base URLs (default: public endpoints). */
  upstreams?: PassthroughUpstreams;
  /** [passthrough] Default session id when the request carries none. Default: "capture". */
  defaultSessionId?: string;
  /** When set, persist one JSON file per captured completion under this dir. */
  saveDir?: string | null;
  persistence?: { enabled?: boolean; maxFieldBytes?: number; queueSize?: number };
  /** Default builder strategy used by /finalize. Default: prefix_merging. */
  defaultBuilder?: string;
  /** Optional logger; defaults to console. */
  logger?: Pick<Console, "info" | "warn" | "error">;
}

export class CaptureProxy {
  readonly config: Required<Pick<ProxyConfig, "engine" | "defaultBuilder">> & ProxyConfig;
  readonly mode: ProxyMode;
  /** Inference client (inference mode only; null in passthrough mode). */
  readonly inference: InferenceClient | null;
  readonly upstreams: PassthroughUpstreams;
  readonly defaultSessionId: string;
  readonly store: SessionStore;
  readonly registry: SessionRegistry;
  readonly transforms: TransformManager;
  readonly builders: BuilderRegistry;
  readonly completionWriter: CompletionWriter;
  private readonly log: Pick<Console, "info" | "warn" | "error">;
  private server: Server | null = null;

  constructor(config: ProxyConfig) {
    this.config = {
      ...config,
      engine: config.engine ?? "vllm",
      defaultBuilder: config.defaultBuilder ?? "prefix_merging",
    };
    this.mode = config.mode ?? "inference";
    this.log = config.logger ?? console;
    this.upstreams = config.upstreams ?? {};
    this.defaultSessionId = config.defaultSessionId ?? "capture";
    if (this.mode === "inference") {
      if (!config.inferenceBaseUrl || !config.modelServed) {
        throw new Error("inference mode requires inferenceBaseUrl and modelServed");
      }
      this.inference = new InferenceClient(config.inferenceBaseUrl, getEngine(this.config.engine));
    } else {
      this.inference = null;
    }
    this.completionWriter = new CompletionWriter({
      saveDir: config.saveDir ?? null,
      maxFieldBytes: config.persistence?.maxFieldBytes,
      queueSize: config.persistence?.queueSize,
      enabled: (config.persistence?.enabled ?? true) && Boolean(config.saveDir),
    });
    this.store = new SessionStore({ completionWriter: this.completionWriter });
    this.registry = new SessionRegistry();
    this.transforms = new TransformManager();
    this.builders = defaultBuilderRegistry();
  }

  /** Build a Node http.Server (does not start listening). */
  createHttpServer(): Server {
    return createServer((req, res) => {
      this.handle(req, res).catch((err) => {
        this.log.error(`unhandled error: ${String(err)}`);
        if (!res.headersSent) sendJson(res, 500, { error: String(err) });
        else res.end();
      });
    });
  }

  /** Start listening. Resolves with the bound port. */
  async listen(port = 0, host = "127.0.0.1"): Promise<number> {
    this.server = this.createHttpServer();
    const server = this.server;
    await new Promise<void>((resolve, reject) => {
      const onError = (err: Error): void => {
        server.removeListener("listening", onListening);
        reject(err);
      };
      const onListening = (): void => {
        server.removeListener("error", onError);
        resolve();
      };
      server.once("error", onError);
      server.once("listening", onListening);
      server.listen(port, host);
    });
    const addr = server.address();
    return typeof addr === "object" && addr ? addr.port : port;
  }

  async close(): Promise<void> {
    if (this.server) await new Promise<void>((resolve) => this.server!.close(() => resolve()));
    await this.completionWriter.close();
    this.store.close();
  }

  // ---------------------------------------------------------------------

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname;
    const method = req.method ?? "GET";

    if (method === "GET" && (path === "/" || path === "")) {
      return sendJson(res, 200, { status: "ok", service: "agent-trace-proxy" });
    }
    if (method === "GET" && path === "/health") return this.handleHealth(res);
    if (method === "GET" && path === "/v1/models") return this.handleListModels(res);
    if (this.inference) {
      if (method === "GET" && path === "/admin/inference/status") {
        return sendJson(res, 200, this.inference.generationStatus());
      }
      if (method === "POST" && path === "/admin/inference/pause") {
        const status = await this.inference.pauseGeneration();
        return sendJson(res, 200, status);
      }
      if (method === "POST" && path === "/admin/inference/resume") {
        return sendJson(res, 200, this.inference.resumeGeneration());
      }
    }

    const completionsMatch = /^\/sessions\/([^/]+)\/completions$/.exec(path);
    if (method === "GET" && completionsMatch) {
      return this.handleListCompletions(res, decodeURIComponent(completionsMatch[1]!));
    }
    const finalizeMatch = /^\/sessions\/([^/]+)\/finalize$/.exec(path);
    if (method === "POST" && finalizeMatch) {
      return this.handleFinalize(req, res, decodeURIComponent(finalizeMatch[1]!), url);
    }
    const sessionMatch = /^\/sessions\/([^/]+)$/.exec(path);
    if (method === "DELETE" && sessionMatch) {
      return this.handleDelete(res, decodeURIComponent(sessionMatch[1]!));
    }

    if (method === "POST") return this.handleProxy(req, res, url);

    return sendJson(res, 404, { error: "Not found" });
  }

  private async handleHealth(res: ServerResponse): Promise<void> {
    if (!this.inference) {
      return sendJson(res, 200, {
        status: "ok",
        mode: this.mode,
        upstreams: this.upstreams,
        active_sessions: this.registry.list().length,
      });
    }
    let upstream: Dict;
    try {
      upstream = await this.inference.health();
    } catch (err) {
      upstream = { status: "error", error: String(err) };
    }
    sendJson(res, 200, {
      status: "ok",
      mode: this.mode,
      engine: this.config.engine,
      model_served: this.config.modelServed,
      inference: upstream,
      active_sessions: this.registry.list().length,
    });
  }

  private async handleListModels(res: ServerResponse): Promise<void> {
    if (!this.inference) {
      return sendJson(res, 200, { object: "list", data: [] });
    }
    try {
      sendJson(res, 200, await this.inference.listModels());
    } catch (err) {
      sendJson(res, 502, { error: String(err) });
    }
  }

  private handleListCompletions(res: ServerResponse, rawId: string): void {
    let safe: string | null;
    try {
      safe = cleanSessionId(rawId);
    } catch (err) {
      return sendJson(res, 400, { error: String((err as Error).message) });
    }
    if (safe === null) return sendJson(res, 400, { error: "Session id required" });
    sendJson(res, 200, { session_id: safe, completions: this.store.getCompletions(safe) });
  }

  private handleDelete(res: ServerResponse, rawId: string): void {
    let safe: string | null;
    try {
      safe = cleanSessionId(rawId);
    } catch (err) {
      return sendJson(res, 400, { error: String((err as Error).message) });
    }
    if (safe === null) return sendJson(res, 400, { error: "Session id required" });
    const deleted = this.store.deleteSession(safe);
    this.registry.remove(safe);
    sendJson(res, 200, { session_id: safe, deleted: true, messages_deleted: deleted });
  }

  /** Build the trajectory for a session and emit SFT/RL samples. */
  private async handleFinalize(
    req: IncomingMessage,
    res: ServerResponse,
    rawId: string,
    url: URL,
  ): Promise<void> {
    let safe: string | null;
    try {
      safe = cleanSessionId(rawId);
    } catch (err) {
      return sendJson(res, 400, { error: String((err as Error).message) });
    }
    if (safe === null) return sendJson(res, 400, { error: "Session id required" });

    const q = url.searchParams;
    const builderName = q.get("builder") ?? this.config.defaultBuilder;
    const format = (q.get("format") ?? "rl").toLowerCase();
    const includeTokens = q.get("include_tokens") === "true";
    const eot = q.get("end_of_turn_token_id");

    const session = this.store.loadCompletionSession(safe);
    let builder;
    try {
      builder = this.builders.create({
        strategy: builderName,
        config: eot !== null ? { end_of_turn_token_id: Number(eot) } : {},
      });
    } catch (err) {
      return sendJson(res, 400, { error: String((err as Error).message) });
    }
    const trajectory = await builder.build(session);

    let samples: unknown[];
    if (format === "sft") samples = trajectoryToSFTSamples(trajectory, { includeTokens });
    else if (format === "trajectory") samples = [trajectory];
    else samples = trajectoryToRLSamples(trajectory);

    let writtenTo: string | null = null;
    if (q.get("save") === "true" && this.config.saveDir) {
      const taskId = (session.task_id as string | null) ?? "default";
      writtenTo = join(this.config.saveDir, `task_${taskId}`, `${safe}.${format}.jsonl`);
      await mkdir(join(writtenTo, ".."), { recursive: true });
      await writeFile(writtenTo, toJsonl(samples), "utf-8");
    }

    sendJson(res, 200, {
      session_id: safe,
      builder: builderName,
      format,
      status: trajectory.status,
      trace_count: trajectory.traces.length,
      samples,
      written_to: writtenTo,
    });
  }

  private async handleProxy(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
    let body: Dict;
    try {
      body = await readJsonBody(req);
    } catch {
      return sendJson(res, 400, { error: "Invalid JSON body" });
    }

    if (this.mode === "passthrough") return this.handlePassthrough(res, req, url, body);

    const headers = headerMap(req);
    const path = url.pathname;
    const apiType = detect(path, headers, body);

    if (!this.transforms.supports(apiType)) {
      return sendJson(res, 400, {
        error: `API family ${apiType} is not supported in this build (OpenAI Chat only).`,
      });
    }

    let sessionId: string;
    try {
      sessionId = resolveSessionId(this.registry, headers, body, {
        querySessionId: url.searchParams.get("session_id") ?? url.searchParams.get("key"),
      });
    } catch (err) {
      if (err instanceof InvalidSessionIdError) return sendJson(res, 400, { error: err.message });
      throw err;
    }

    const originalModel = extractModel(apiType, body);
    const transformer = this.transforms.get(apiType);
    const sessionInfo = this.registry.get(sessionId);

    this.log.info(`<- POST ${path} | api=${apiType} model=${originalModel} session=${sessionId}`);

    const transformedBody: Dict = { ...body, _at_model_served: this.config.modelServed };
    const openaiRequest = transformer.transformRequest(transformedBody);
    openaiRequest["model"] = this.config.modelServed;
    const isStreaming = Boolean(openaiRequest["stream"]);

    if (isStreaming) {
      return this.handleStreaming(res, apiType, transformer, openaiRequest, body, sessionId, {
        originalModel,
        sessionInfo,
      });
    }
    return this.handleNonStreaming(res, apiType, transformer, openaiRequest, body, sessionId, {
      originalModel,
      sessionInfo,
    });
  }

  // --- passthrough mode ---------------------------------------------------

  /**
   * Forward the request verbatim to the API the agent already uses and capture
   * the full trace, without rewriting the payload or needing a self-hosted
   * backend. Streamed responses are piped byte-for-byte to the client while a
   * copy is reconstructed and normalized for export.
   */
  private async handlePassthrough(
    res: ServerResponse,
    req: IncomingMessage,
    url: URL,
    body: Dict,
  ): Promise<void> {
    const headers = headerMap(req);
    const path = url.pathname;
    const apiType = detect(path, headers, body);

    if (apiType !== APIType.OPENAI_CHAT && apiType !== APIType.ANTHROPIC) {
      return sendJson(res, 400, {
        error:
          `passthrough captures OpenAI Chat (/v1/chat/completions) and Anthropic ` +
          `(/v1/messages); got ${apiType}.`,
      });
    }

    const sessionId = resolveCaptureSessionId(
      this.registry,
      headers,
      url.searchParams.get("session_id") ?? url.searchParams.get("key"),
      this.defaultSessionId,
    );
    const sessionInfo = this.registry.get(sessionId);
    const upstreamBase = resolveUpstreamBase(apiType, this.upstreams);
    const target = `${upstreamBase}${path}${url.search}`;
    const isStreaming = Boolean(body["stream"]);

    this.log.info(
      `<- POST ${path} | api=${apiType} session=${sessionId} -> ${upstreamBase} stream=${isStreaming}`,
    );

    let upstream: Response;
    try {
      upstream = await fetch(target, {
        method: "POST",
        headers: { ...filterForwardHeaders(headers), "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (err) {
      return sendJson(
        res,
        502,
        buildErrorBody(apiType, `Upstream request failed: ${String(err)}`, null),
      );
    }

    const transformer = this.transforms.get(apiType);
    const normalizedRequest = normalizeCapturedRequest(transformer, body);
    const originalModel = extractModel(apiType, body);
    const upstreamCt = upstream.headers.get("content-type") ?? "";

    if (isStreaming && upstream.body && upstreamCt.includes("event-stream")) {
      res.writeHead(upstream.status, passthroughStreamHeaders(upstream));
      const decoder = new TextDecoder();
      const reader = upstream.body.getReader();
      let buf = "";
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            res.write(Buffer.from(value));
            buf += decoder.decode(value, { stream: true });
          }
        }
      } finally {
        res.end();
      }
      buf += decoder.decode();
      if (upstream.ok) {
        const native =
          apiType === APIType.ANTHROPIC
            ? assembleAnthropicStream(buf)
            : assembleOpenAiChatStream(buf);
        this.capturePassthrough(sessionId, normalizedRequest, native, body, apiType, {
          originalModel,
          sessionInfo,
        });
      }
      return;
    }

    const text = await upstream.text();
    res.writeHead(upstream.status, passthroughJsonHeaders(upstream));
    res.end(text);
    if (!upstream.ok) return;
    let native: Dict;
    try {
      native = JSON.parse(text) as Dict;
    } catch {
      return;
    }
    this.capturePassthrough(sessionId, normalizedRequest, native, body, apiType, {
      originalModel,
      sessionInfo,
    });
  }

  private capturePassthrough(
    sessionId: string,
    normalizedRequest: Dict,
    nativeResponse: Dict,
    originalRequest: Dict,
    apiType: APIType,
    ctx: { originalModel: string; sessionInfo: SessionInfo | undefined },
  ): void {
    const normalizedResponse = normalizeCapturedResponse(apiType, nativeResponse);
    const modelUsed =
      (normalizedResponse["model"] as string | undefined) ?? ctx.originalModel ?? "unknown";
    const metadata = completionMetadata(ctx.sessionInfo);
    metadata["capture_mode"] = "passthrough";
    metadata["raw_response"] = nativeResponse;
    this.store.saveMessage(sessionId, normalizedRequest, normalizedResponse, {
      originalRequest,
      modelRequested: ctx.originalModel,
      modelUsed,
      apiType,
      taskId: ctx.sessionInfo?.taskId ?? null,
      createdAt: ctx.sessionInfo?.createdAt ?? null,
      metadata,
    });
  }

  private async handleNonStreaming(
    res: ServerResponse,
    apiType: APIType,
    transformer: BaseTransformer,
    openaiRequest: Dict,
    originalRequest: Dict,
    sessionId: string,
    ctx: { originalModel: string; sessionInfo: SessionInfo | undefined },
  ): Promise<void> {
    let response: Dict;
    try {
      response = await this.inference!.completion(openaiRequest);
    } catch (err) {
      if (err instanceof UpstreamError) return this.sendUpstreamError(res, apiType, err);
      throw err;
    }

    this.capture(sessionId, openaiRequest, response, originalRequest, apiType, ctx);
    sendJson(res, 200, transformer.transformResponse(response, originalRequest));
  }

  private async handleStreaming(
    res: ServerResponse,
    apiType: APIType,
    transformer: BaseTransformer,
    openaiRequest: Dict,
    originalRequest: Dict,
    sessionId: string,
    ctx: { originalModel: string; sessionInfo: SessionInfo | undefined },
  ): Promise<void> {
    const nonStreamRequest: Dict = { ...openaiRequest };
    delete nonStreamRequest["stream_options"];
    nonStreamRequest["stream"] = false;

    let response: Dict;
    try {
      response = await this.inference!.completion(nonStreamRequest);
    } catch (err) {
      if (err instanceof UpstreamError) return this.sendUpstreamError(res, apiType, err);
      throw err;
    }

    this.capture(sessionId, openaiRequest, response, originalRequest, apiType, ctx);

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });

    const syntheticChunk = responseToStreamChunk(response);
    const streamState = transformer.createStreamState(originalRequest);
    try {
      if (streamState !== null) {
        for (const ev of streamState.processChunk(syntheticChunk, true)) {
          res.write(formatOpenAiSse(ev));
        }
        for (const ev of streamState.finalize()) res.write(formatOpenAiSse(ev));
      } else {
        const out = transformer.transformStreamChunk(syntheticChunk, originalRequest, true);
        const chunks = Array.isArray(out) ? out : [out];
        for (const ev of chunks) res.write(formatOpenAiSse(ev));
      }
      if (apiType === APIType.OPENAI_CHAT) res.write("data: [DONE]\n\n");
    } finally {
      res.end();
    }
  }

  private capture(
    sessionId: string,
    openaiRequest: Dict,
    response: Dict,
    originalRequest: Dict,
    apiType: APIType,
    ctx: { originalModel: string; sessionInfo: SessionInfo | undefined },
  ): void {
    this.store.saveMessage(sessionId, openaiRequest, response, {
      originalRequest,
      modelRequested: ctx.originalModel,
      modelUsed: openaiRequest["model"] as string,
      apiType,
      taskId: ctx.sessionInfo?.taskId ?? null,
      createdAt: ctx.sessionInfo?.createdAt ?? null,
      metadata: completionMetadata(ctx.sessionInfo),
    });
  }

  private sendUpstreamError(res: ServerResponse, apiType: APIType, err: UpstreamError): void {
    let status = 502;
    let upstreamBody: Dict | string | null = null;
    if (err instanceof UpstreamHTTPError) {
      status = err.statusCode;
      upstreamBody = err.body;
    } else if (err instanceof UpstreamTimeoutError) {
      status = 504;
    }
    sendJson(res, status, buildErrorBody(apiType, err.message, upstreamBody));
  }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function completionMetadata(info: SessionInfo | undefined): Dict {
  const metadata: Dict = { ...(info?.metadata ?? {}) };
  if (info) {
    if (metadata["session_id"] === undefined) metadata["session_id"] = info.sessionId;
    if (info.taskId != null && metadata["task_id"] === undefined) metadata["task_id"] = info.taskId;
  }
  return metadata;
}

function buildErrorBody(apiType: APIType, message: string, upstreamBody: Dict | string | null): Dict {
  if (apiType === APIType.ANTHROPIC) {
    if (upstreamBody && typeof upstreamBody === "object" && upstreamBody["type"] === "error") {
      return upstreamBody;
    }
    return { type: "error", error: { type: "api_error", message } };
  }
  if (upstreamBody && typeof upstreamBody === "object" && "error" in upstreamBody) {
    return upstreamBody;
  }
  return { error: { message, type: "upstream_error" } };
}

/** Convert a non-streaming chat completion into one delta chunk for SSE replay. */
function responseToStreamChunk(response: Dict): Dict {
  const choices = (response["choices"] as Dict[] | undefined) ?? [{}];
  const choice = choices[0] ?? {};
  const message = (choice["message"] as Dict | undefined) ?? {};

  const toolCallsDelta: Dict[] = [];
  const toolCalls = (message["tool_calls"] as Dict[] | undefined) ?? [];
  toolCalls.forEach((tc, i) => {
    const func = (tc["function"] as Dict | undefined) ?? {};
    toolCallsDelta.push({
      index: i,
      id: tc["id"],
      type: tc["type"] ?? "function",
      function: { name: func["name"] ?? "", arguments: func["arguments"] ?? "" },
    });
  });

  const delta: Dict = { role: "assistant" };
  if (message["content"] != null) delta["content"] = message["content"];
  if (message["reasoning_content"] != null) delta["reasoning_content"] = message["reasoning_content"];
  if (toolCallsDelta.length > 0) delta["tool_calls"] = toolCallsDelta;

  return {
    id: response["id"],
    object: "chat.completion.chunk",
    created: response["created"],
    model: response["model"],
    choices: [{ index: 0, delta, finish_reason: choice["finish_reason"] }],
    usage: response["usage"],
  };
}

function formatOpenAiSse(chunk: Dict): string {
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

function headerMap(req: IncomingMessage): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(req.headers)) {
    if (typeof v === "string") out[k] = v;
    else if (Array.isArray(v)) out[k] = v.join(", ");
  }
  return out;
}

async function readJsonBody(req: IncomingMessage): Promise<Dict> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const text = Buffer.concat(chunks).toString("utf-8");
  if (!text) return {};
  return JSON.parse(text) as Dict;
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(payload);
}
