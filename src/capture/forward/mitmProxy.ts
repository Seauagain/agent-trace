/**
 * MITM forward proxy.
 *
 * `exec`/`capture` only redirect `*_BASE_URL`, so they can't see clients that
 * pin their endpoint or speak a non-JSON wire (Cursor's protobuf agent API).
 * This proxy fills that gap: the agent points `HTTPS_PROXY` here and trusts our
 * local CA (`NODE_EXTRA_CA_CERTS`); we terminate TLS for an allowlist of hosts,
 * tee the decrypted exchange to a decoder registry, and forward everything else
 * untouched. Decoded turns land in the same SessionStore as passthrough, so
 * `agent-trace build` produces the same SFT/RL output.
 */

import { connect as http2Connect, createSecureServer, type Http2Server } from "node:http2";
import {
  createServer as createHttpServer,
  type IncomingMessage,
  request as httpRequest,
  type Server as HttpServer,
  type ServerResponse,
} from "node:http";
import { request as httpsRequest } from "node:https";
import { connect as netConnect, type Socket } from "node:net";
import { createSecureContext, type TLSSocket } from "node:tls";

import { CompletionWriter } from "../completionWriter.js";
import { SessionStore } from "../sessionStore.js";
import { MitmCA } from "./ca.js";
import {
  decodeExchange,
  defaultDecoders,
  type HttpExchange,
  type WireDecoder,
} from "./decoders/index.js";

type Headers = Record<string, string | string[] | undefined>;

const DEFAULT_MITM_HOSTS = ["cursor.sh", "anthropic.com", "openai.com"];

// Headers that must not be copied across the proxy hop / between h1 and h2.
const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-connection",
  "transfer-encoding",
  "upgrade",
  "host",
  "te",
  "trailer",
]);

export interface ForwardProxyConfig {
  /** Host substrings to intercept. Default: cursor/anthropic/openai. */
  mitmHosts?: string[];
  /** Intercept every host (blind-tunnel nothing). */
  mitmAll?: boolean;
  /** Directory holding the persistent root CA. */
  caDir?: string;
  /** Persist one JSON file per captured completion under this dir. */
  saveDir?: string | null;
  /** Session id for everything captured in this run. Default: "forward". */
  defaultSessionId?: string;
  defaultBuilder?: string;
  decoders?: WireDecoder[];
  logger?: Pick<Console, "info" | "warn" | "error">;
}

export class ForwardProxy {
  readonly ca: MitmCA;
  readonly store: SessionStore;
  readonly completionWriter: CompletionWriter;
  readonly sessionId: string;
  private readonly decoders: WireDecoder[];
  private readonly mitmHosts: string[];
  private readonly mitmAll: boolean;
  private readonly log: Pick<Console, "info" | "warn" | "error">;
  private http: HttpServer | null = null;
  private secure: Http2Server | null = null;

  constructor(config: ForwardProxyConfig = {}) {
    this.log = config.logger ?? console;
    this.ca = MitmCA.loadOrCreate(config.caDir);
    this.mitmHosts = config.mitmHosts ?? DEFAULT_MITM_HOSTS;
    this.mitmAll = config.mitmAll ?? false;
    this.decoders = config.decoders ?? defaultDecoders();
    this.sessionId = config.defaultSessionId ?? "forward";
    this.completionWriter = new CompletionWriter({
      saveDir: config.saveDir ?? null,
      enabled: Boolean(config.saveDir),
    });
    this.store = new SessionStore({ completionWriter: this.completionWriter });
  }

  /** Absolute path to the root CA cert (for NODE_EXTRA_CA_CERTS). */
  get caCertPath(): string {
    return this.ca.certPath;
  }

  private shouldMitm(host: string): boolean {
    if (this.mitmAll) return true;
    const h = host.toLowerCase();
    return this.mitmHosts.some((p) => h.includes(p.toLowerCase()));
  }

  /** Start the proxy. Resolves with the bound port. */
  async listen(port = 0, host = "127.0.0.1"): Promise<number> {
    this.secure = createSecureServer({
      allowHTTP1: true,
      ALPNProtocols: ["h2", "http/1.1"],
      settings: { enableConnectProtocol: true },
      ...this.ca.leafFor("localhost"),
      SNICallback: (servername, cb) => {
        try {
          cb(null, createSecureContext(this.ca.leafFor(servername)));
        } catch (err) {
          cb(err as Error);
        }
      },
    });
    this.secure.on("request", (req, res) => {
      void this.handleIntercepted(
        req as unknown as IncomingMessage,
        res as unknown as ServerResponse,
      );
    });
    this.secure.on("sessionError", (err) => this.log.warn(`tls session error: ${err.message}`));
    this.secure.on("clientError", () => {
      /* ignore */
    });

    this.http = createHttpServer((req, res) => {
      // Plain-HTTP forward-proxy request (absolute-form URL). Rare for LLM APIs.
      void this.handlePlainHttp(req, res);
    });
    this.http.on("connect", (req, socket, head) => this.handleConnect(req, socket as Socket, head));

    const server = this.http;
    await new Promise<void>((resolve, reject) => {
      const onErr = (e: Error): void => {
        server.removeListener("listening", onOk);
        reject(e);
      };
      const onOk = (): void => {
        server.removeListener("error", onErr);
        resolve();
      };
      server.once("error", onErr);
      server.once("listening", onOk);
      server.listen(port, host);
    });
    const addr = server.address();
    return typeof addr === "object" && addr ? addr.port : port;
  }

  async close(): Promise<void> {
    if (this.secure) await new Promise<void>((r) => this.secure!.close(() => r()));
    if (this.http) await new Promise<void>((r) => this.http!.close(() => r()));
    await this.completionWriter.close();
    this.store.close();
  }

  // --- CONNECT handling --------------------------------------------------

  private handleConnect(req: IncomingMessage, clientSocket: Socket, head: Buffer): void {
    const [rawHost, rawPort] = (req.url ?? "").split(":");
    const host = rawHost ?? "";
    const port = Number(rawPort ?? "443");

    if (this.shouldMitm(host)) {
      clientSocket.write("HTTP/1.1 200 Connection Established\r\n\r\n");
      if (head && head.length) clientSocket.unshift(head);
      // Hand the raw socket to the TLS-terminating server; SNI picks the cert.
      this.secure!.emit("connection", clientSocket);
      clientSocket.on("error", () => clientSocket.destroy());
      return;
    }

    // Blind tunnel: forward bytes without decrypting.
    const upstream = netConnect(port, host, () => {
      clientSocket.write("HTTP/1.1 200 Connection Established\r\n\r\n");
      if (head && head.length) upstream.write(head);
      upstream.pipe(clientSocket);
      clientSocket.pipe(upstream);
    });
    upstream.on("error", () => clientSocket.destroy());
    clientSocket.on("error", () => upstream.destroy());
  }

  // --- intercepted (decrypted) request handling --------------------------

  private resolveHost(req: IncomingMessage): string {
    const sock = req.socket as TLSSocket | undefined;
    const sni = sock && typeof sock.servername === "string" ? sock.servername : "";
    if (sni) return sni;
    const auth = (req.headers[":authority"] as string | undefined) ?? req.headers.host ?? "";
    return String(auth).split(":")[0] ?? "";
  }

  private async handleIntercepted(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const host = this.resolveHost(req);
    const isH2 = req.httpVersionMajor >= 2;
    const method = req.method ?? "GET";
    const path = req.url ?? "/";
    // Capture headers now; body is teed while it streams (do NOT buffer-then-
    // forward: Cursor's agent RPC is long-lived, so the request stream may not
    // "end" until the response is flowing — buffering deadlocks it).
    const reqHeaders = flattenHeaders(req.headers);
    if (isH2) await this.forwardH2(host, method, path, req, res, reqHeaders);
    else await this.forwardH1(host, method, path, req, res, reqHeaders);
  }

  private forwardH2(
    host: string,
    method: string,
    path: string,
    req: IncomingMessage,
    res: ServerResponse,
    reqHeaders: Record<string, string>,
  ): Promise<void> {
    return new Promise<void>((resolve) => {
      const client = http2Connect(`https://${host}`, { servername: host });
      const reqChunks: Buffer[] = [];
      const resChunks: Buffer[] = [];
      let settled = false;
      const finish = (): void => {
        if (settled) return;
        settled = true;
        resolve();
      };
      client.on("error", (e) => {
        this.log.warn(`h2 upstream ${host}${path}: ${e.message}`);
        if (!res.headersSent) {
          try {
            res.writeHead(502);
          } catch {
            /* already gone */
          }
        }
        try {
          res.end();
        } catch {
          /* already gone */
        }
        finish();
      });

      const outHeaders: Record<string, string> = {
        ":method": method,
        ":path": path,
        ":authority": host,
        ":scheme": "https",
        ...forwardableHeaders(req.headers),
      };
      const upstream = client.request(outHeaders);
      upstream.on("error", () => finish());
      req.on("data", (d: Buffer) => reqChunks.push(d));
      req.on("error", () => {});
      req.pipe(upstream);

      upstream.on("response", (headers) => {
        const status = Number(headers[":status"] ?? 502);
        try {
          res.writeHead(status, responseHeaders(headers, false));
        } catch {
          /* client may have gone */
        }
        upstream.on("data", (d: Buffer) => resChunks.push(d));
        upstream.pipe(res);
        upstream.on("end", () => {
          this.captureExchange({
            host,
            method,
            path,
            reqHeaders,
            reqBody: Buffer.concat(reqChunks),
            status,
            resHeaders: flattenHeaders(headers),
            resBody: Buffer.concat(resChunks),
          });
          client.close();
          finish();
        });
      });
    });
  }

  private forwardH1(
    host: string,
    method: string,
    path: string,
    req: IncomingMessage,
    res: ServerResponse,
    reqHeaders: Record<string, string>,
  ): Promise<void> {
    return new Promise<void>((resolve) => {
      const reqChunks: Buffer[] = [];
      const resChunks: Buffer[] = [];
      let settled = false;
      const finish = (): void => {
        if (settled) return;
        settled = true;
        resolve();
      };
      const upstream = httpsRequest(
        {
          host,
          port: 443,
          method,
          path,
          servername: host,
          headers: { ...forwardableHeaders(req.headers), host },
        },
        (up) => {
          const status = up.statusCode ?? 502;
          try {
            res.writeHead(status, responseHeaders(up.headers, true));
          } catch {
            /* client may have gone */
          }
          up.on("data", (d: Buffer) => resChunks.push(d));
          up.pipe(res);
          up.on("end", () => {
            this.captureExchange({
              host,
              method,
              path,
              reqHeaders,
              reqBody: Buffer.concat(reqChunks),
              status,
              resHeaders: flattenHeaders(up.headers),
              resBody: Buffer.concat(resChunks),
            });
            finish();
          });
        },
      );
      upstream.on("error", (e) => {
        this.log.warn(`h1 upstream ${host}${path}: ${e.message}`);
        if (!res.headersSent) {
          try {
            res.writeHead(502);
          } catch {
            /* already gone */
          }
        }
        try {
          res.end();
        } catch {
          /* already gone */
        }
        finish();
      });
      req.on("data", (d: Buffer) => reqChunks.push(d));
      req.on("error", () => {});
      req.pipe(upstream);
    });
  }

  private async handlePlainHttp(req: IncomingMessage, res: ServerResponse): Promise<void> {
    let target: URL;
    try {
      target = new URL(req.url ?? "");
    } catch {
      res.statusCode = 400;
      res.end("agent-trace forward proxy: expected absolute-form request URL");
      return;
    }
    const reqBody = await readBody(req).catch(() => Buffer.alloc(0));
    await new Promise<void>((resolve) => {
      const upstream = httpRequest(
        {
          host: target.hostname,
          port: target.port || 80,
          method: req.method,
          path: target.pathname + target.search,
          headers: { ...forwardableHeaders(req.headers), host: target.host },
        },
        (up) => {
          const status = up.statusCode ?? 502;
          res.writeHead(status, responseHeaders(up.headers, true));
          const chunks: Buffer[] = [];
          up.on("data", (c: Buffer) => {
            chunks.push(c);
            res.write(c);
          });
          up.on("end", () => {
            res.end();
            this.captureExchange({
              host: target.hostname,
              method: req.method ?? "GET",
              path: target.pathname + target.search,
              reqHeaders: flattenHeaders(req.headers),
              reqBody,
              status,
              resHeaders: flattenHeaders(up.headers),
              resBody: Buffer.concat(chunks),
            });
            resolve();
          });
        },
      );
      upstream.on("error", () => {
        if (!res.headersSent) res.statusCode = 502;
        res.end();
        resolve();
      });
      upstream.end(reqBody);
    });
  }

  private captureExchange(ex: HttpExchange): void {
    const decoded = decodeExchange(ex, this.decoders);
    if (!decoded) return;
    this.log.info(`captured ${decoded.apiType} turn from ${ex.host}${ex.path} (model=${decoded.model})`);
    this.store.saveMessage(this.sessionId, decoded.request, decoded.response, {
      originalRequest: { wire_host: ex.host, wire_path: ex.path },
      modelRequested: decoded.model,
      modelUsed: decoded.model,
      apiType: decoded.apiType,
      taskId: null,
      metadata: { capture_mode: "forward_mitm", ...(decoded.metadata ?? {}) },
    });
  }
}

// ---------------------------------------------------------------------------
// header helpers
// ---------------------------------------------------------------------------

function flattenHeaders(headers: Headers): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) {
    if (k.startsWith(":")) continue;
    if (typeof v === "string") out[k] = v;
    else if (Array.isArray(v)) out[k] = v.join(", ");
  }
  return out;
}

/** Request headers safe to forward upstream (drop pseudo + hop-by-hop). */
function forwardableHeaders(headers: Headers): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) {
    const lower = k.toLowerCase();
    if (lower.startsWith(":")) continue;
    if (HOP_BY_HOP.has(lower)) continue;
    if (lower === "content-length") continue; // body length set by the client lib
    if (typeof v === "string") out[k] = v;
    else if (Array.isArray(v)) out[k] = v.join(", ");
  }
  return out;
}

/** Response headers safe to send to the client (drop pseudo + hop-by-hop). */
function responseHeaders(headers: Headers, includeContentLength: boolean): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) {
    const lower = k.toLowerCase();
    if (lower.startsWith(":")) continue;
    if (HOP_BY_HOP.has(lower)) continue;
    if (!includeContentLength && lower === "content-length") continue;
    if (typeof v === "string") out[k] = v;
    else if (Array.isArray(v)) out[k] = v.join(", ");
  }
  return out;
}

async function readBody(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks);
}
