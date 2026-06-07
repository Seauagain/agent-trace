/**
 * HTTP client for forwarding requests to an OpenAI-compatible inference server
 * The InferenceEngine strategy injects backend
 * request params and canonicalizes responses; this client is backend-agnostic.
 *
 * Generation can be paused/resumed: a training bridge pauses new generation
 * while it syncs weights, lets in-flight calls drain, then resumes.
 */

import type { InferenceEngine } from "./engine.js";

type Dict = Record<string, unknown>;

export class UpstreamError extends Error {}
export class UpstreamHTTPError extends UpstreamError {
  constructor(
    readonly statusCode: number,
    readonly body: Dict | string | null = null,
  ) {
    super(UpstreamHTTPError.buildMessage(statusCode, body));
  }

  private static buildMessage(statusCode: number, body: Dict | string | null): string {
    if (body && typeof body === "object") {
      const error = body["error"];
      if (error && typeof error === "object") {
        const message = (error as Dict)["message"];
        if (typeof message === "string" && message) return message;
      }
      const message = body["message"];
      if (typeof message === "string" && message) return message;
    }
    if (typeof body === "string" && body) return body;
    return `Upstream request failed with status ${statusCode}`;
  }
}
export class UpstreamTimeoutError extends UpstreamError {}
export class UpstreamTransportError extends UpstreamError {}

const LIVENESS_TIMEOUT_MS = 900_000;

export class InferenceClient {
  readonly baseUrl: string;
  readonly engine: InferenceEngine;

  private generationPaused = false;
  private inflightGenerations = 0;
  private readonly waiters: Array<() => void> = [];

  constructor(baseUrl: string, engine: InferenceEngine) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.engine = engine;
  }

  /** Non-streaming chat completion. Returns the full JSON response. */
  async completion(request: Dict): Promise<Dict> {
    await this.acquireGenerationSlot();
    let requestCopy: Dict = structuredClone(request);
    delete requestCopy["stream"];
    requestCopy["stream"] = false;
    requestCopy = this.engine.prepareRequest(requestCopy);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), LIVENESS_TIMEOUT_MS);
    let resp: Response;
    try {
      resp = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestCopy),
        signal: controller.signal,
      });
    } catch (err) {
      throw InferenceClient.translateTransportError(err);
    } finally {
      clearTimeout(timer);
      this.releaseGenerationSlot();
    }

    await this.raiseForStatus(resp);
    return this.engine.normalizeResponse((await resp.json()) as Dict);
  }

  async listModels(): Promise<Dict> {
    const resp = await this.safeGet("/v1/models");
    return (await resp.json()) as Dict;
  }

  async health(): Promise<Dict> {
    const resp = await this.safeGet("/health");
    const text = (await resp.text()).trim();
    if (!text) return { status: "ok" };
    try {
      return JSON.parse(text) as Dict;
    } catch {
      return { status: "ok", body: text };
    }
  }

  // --- pause / resume -----------------------------------------------------

  async pauseGeneration(timeoutMs = 300_000): Promise<Dict> {
    this.generationPaused = true;
    const start = Date.now();
    while (this.inflightGenerations > 0) {
      if (Date.now() - start > timeoutMs) {
        throw new UpstreamTimeoutError("Timed out waiting for inference requests to drain");
      }
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    return this.generationStatus();
  }

  resumeGeneration(): Dict {
    this.generationPaused = false;
    this.wakeWaiters();
    return this.generationStatus();
  }

  generationStatus(): Dict {
    return {
      paused: this.generationPaused,
      inflight: this.inflightGenerations,
      base_url: this.baseUrl,
      engine: this.engine.name,
    };
  }

  // --- internals ----------------------------------------------------------

  private async acquireGenerationSlot(): Promise<void> {
    while (this.generationPaused) {
      await new Promise<void>((resolve) => this.waiters.push(resolve));
    }
    this.inflightGenerations += 1;
  }

  private releaseGenerationSlot(): void {
    this.inflightGenerations -= 1;
  }

  private wakeWaiters(): void {
    while (this.waiters.length > 0) this.waiters.shift()!();
  }

  private async safeGet(path: string): Promise<Response> {
    let resp: Response;
    try {
      resp = await fetch(`${this.baseUrl}${path}`, { method: "GET" });
    } catch (err) {
      throw InferenceClient.translateTransportError(err);
    }
    await this.raiseForStatus(resp);
    return resp;
  }

  private async raiseForStatus(resp: Response): Promise<void> {
    if (resp.ok) return;
    const text = (await resp.text()).trim();
    let body: Dict | string | null = null;
    if (text) {
      try {
        body = JSON.parse(text) as Dict;
      } catch {
        body = text;
      }
    }
    throw new UpstreamHTTPError(resp.status, body);
  }

  private static translateTransportError(err: unknown): UpstreamError {
    if (err instanceof Error && err.name === "AbortError") {
      return new UpstreamTimeoutError("Upstream request timed out");
    }
    return new UpstreamTransportError(`Upstream request failed: ${String(err)}`);
  }
}
