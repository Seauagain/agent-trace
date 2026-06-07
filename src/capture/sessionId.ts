/**
 * Session id validation + a lightweight session registry and request resolver.
 *
 * The proxy injects the session id as the API key when launching an agent, so a
 * request's bearer token (or x-session-id header) identifies its session.
 */

import { randomUUID } from "node:crypto";

const SESSION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;

export class InvalidSessionIdError extends Error {}

/** Normalize and validate an external session id. Returns null for empty input. */
export function cleanSessionId(value: string | null | undefined): string | null {
  if (value == null) return null;
  const normalized = value.trim();
  if (!normalized) return null;
  if (!SESSION_ID_PATTERN.test(normalized)) {
    throw new InvalidSessionIdError(
      "Session IDs may only contain letters, numbers, '.', '_' or '-'.",
    );
  }
  return normalized;
}

/** Generate a new storage-safe session id. */
export function generateSessionId(): string {
  return randomUUID();
}

export interface SessionInfo {
  sessionId: string;
  createdAt: string;
  lastActivity: string;
  taskId: string | null;
  metadata: Record<string, unknown>;
}

export class SessionRegistry {
  private readonly sessions = new Map<string, SessionInfo>();

  register(
    sessionId: string,
    opts: { taskId?: string | null; metadata?: Record<string, unknown> } = {},
  ): SessionInfo {
    const id = cleanSessionId(sessionId) ?? generateSessionId();
    const now = new Date().toISOString();
    const existing = this.sessions.get(id);
    if (existing) {
      existing.lastActivity = now;
      if (opts.taskId != null) existing.taskId = opts.taskId;
      if (opts.metadata) existing.metadata = { ...existing.metadata, ...opts.metadata };
      return existing;
    }
    const info: SessionInfo = {
      sessionId: id,
      createdAt: now,
      lastActivity: now,
      taskId: opts.taskId ?? null,
      metadata: { ...(opts.metadata ?? {}) },
    };
    this.sessions.set(id, info);
    return info;
  }

  get(sessionId: string): SessionInfo | undefined {
    return this.sessions.get(sessionId);
  }

  updateActivity(sessionId: string): void {
    const info = this.sessions.get(sessionId);
    if (info) info.lastActivity = new Date().toISOString();
  }

  remove(sessionId: string): void {
    this.sessions.delete(sessionId);
  }

  list(): SessionInfo[] {
    return [...this.sessions.values()];
  }
}

/** Extract API key from Authorization, X-Api-Key, or x-goog-api-key headers. */
export function extractApiKey(headers: Record<string, string>): string | null {
  const lower: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) lower[k.toLowerCase()] = v;

  const auth = lower["authorization"] ?? "";
  if (auth.toLowerCase().startsWith("bearer ")) {
    const token = auth.slice(7).trim();
    return token || null;
  }
  const xApiKey = lower["x-api-key"] ?? lower["x_api_key"];
  if (xApiKey) return xApiKey.trim() || null;

  const googKey = lower["x-goog-api-key"];
  if (googKey) return googKey.trim() || null;

  return null;
}

/** Resolve session id from explicit ids or auth, else create a new one. */
export function resolveSessionId(
  registry: SessionRegistry,
  headers: Record<string, string>,
  body: Record<string, unknown>,
  opts: { querySessionId?: string | null } = {},
): string {
  const lower: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) lower[k.toLowerCase()] = v;

  const explicit =
    cleanSessionId(lower["x-session-id"]) ??
    cleanSessionId(lower["x_session_id"]) ??
    cleanSessionId(lower["proxy-x-session-id"]) ??
    cleanSessionId(lower["proxy_x_session_id"]) ??
    cleanSessionId(opts.querySessionId) ??
    cleanSessionId(body["_proxy_session_id"] as string | undefined);

  if (explicit) {
    if (registry.get(explicit) === undefined) registry.register(explicit);
    else registry.updateActivity(explicit);
    return explicit;
  }

  // The proxy injects the session id as the agent's API key, so a bearer token
  // (or x-api-key) directly identifies the session. If it isn't a storage-safe
  // id we fall back to a generated one rather than rejecting the request.
  const apiKey = extractApiKey(headers);
  if (apiKey) {
    let safe: string | null = null;
    try {
      safe = cleanSessionId(apiKey);
    } catch {
      safe = null;
    }
    if (safe) {
      if (registry.get(safe) === undefined) registry.register(safe);
      else registry.updateActivity(safe);
      return safe;
    }
  }

  return registry.register(generateSessionId()).sessionId;
}
