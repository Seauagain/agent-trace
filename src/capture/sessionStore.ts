/**
 * In-memory storage for captured completion records. The authoritative copy lives here; the optional
 * CompletionWriter persists a JSON file per completion off the hot path.
 */

import { randomUUID } from "node:crypto";

import {
  type CompletionRecord,
  CompletionRecordSchema,
  type CompletionSession,
  parseCompletionSession,
} from "../trajectory/models.js";
import type { CompletionWriter } from "./completionWriter.js";

type Dict = Record<string, unknown>;

interface SessionState {
  sessionId: string;
  createdAt: string | null;
  taskId: string | null;
  modelRequested: string | null;
  modelUsed: string | null;
  apiType: string | null;
  metadata: Dict;
  completions: CompletionRecord[];
}

function mergeField<T>(existing: T | null, incoming: T | null | undefined): T | null {
  if (incoming == null || incoming === "" || incoming === "unknown") return existing;
  if (existing == null || existing === "" || existing === "unknown") return incoming as T;
  return existing;
}

export class SessionStore {
  private readonly sessions = new Map<string, SessionState>();
  private readonly completionWriter: CompletionWriter | null;

  constructor(opts: { completionWriter?: CompletionWriter | null } = {}) {
    this.completionWriter = opts.completionWriter ?? null;
  }

  close(): void {
    this.sessions.clear();
  }

  getCompletions(sessionId: string): CompletionRecord[] {
    return this.sessions.get(sessionId)?.completions ?? [];
  }

  ensureSession(
    sessionId: string,
    args: {
      modelRequested?: string | null;
      modelUsed?: string | null;
      apiType?: string | null;
      taskId?: string | null;
      createdAt?: string | null;
      metadata?: Dict | null;
    } = {},
  ): void {
    const state = this.getOrCreate(sessionId, args.createdAt ?? null);
    this.mergeMetadata(state, args);
  }

  /** Append one captured completion record to the in-memory session. */
  saveMessage(
    sessionId: string,
    request: Dict,
    response: Dict,
    args: {
      originalRequest?: Dict | null;
      modelRequested?: string | null;
      modelUsed?: string | null;
      apiType?: string | null;
      taskId?: string | null;
      createdAt?: string | null;
      metadata?: Dict | null;
    } = {},
  ): string {
    const effectiveModelUsed = args.modelUsed ?? (request["model"] as string) ?? "unknown";
    const record = CompletionRecordSchema.parse({
      completion_id: `msg_${randomUUID().replace(/-/g, "").slice(0, 12)}`,
      timestamp: new Date().toISOString(),
      request,
      original_request: args.originalRequest ?? {},
      response,
      metadata: { ...(args.metadata ?? {}) },
    });

    const state = this.getOrCreate(sessionId, args.createdAt ?? null);
    this.mergeMetadata(state, { ...args, modelUsed: effectiveModelUsed });
    state.completions.push(record);
    const effectiveTaskId = state.taskId;

    if (this.completionWriter !== null) {
      this.completionWriter.enqueue({
        taskId: effectiveTaskId,
        sessionId,
        completionId: record.completion_id,
        record: {
          completion_id: record.completion_id,
          timestamp: record.timestamp,
          session_id: sessionId,
          task_id: effectiveTaskId,
          api_type: args.apiType ?? null,
          model_requested: args.modelRequested ?? null,
          model_used: effectiveModelUsed,
          original_request: args.originalRequest ?? {},
          transformed_request: request,
          response,
          metadata: { ...(args.metadata ?? {}) },
        },
      });
    }
    return record.completion_id;
  }

  getSessionMetadata(sessionId: string): Dict | null {
    const state = this.sessions.get(sessionId);
    if (state === undefined) return null;
    return this.metadataPayload(state);
  }

  /** Load the typed completion session from in-memory state (for a builder). */
  loadCompletionSession(sessionId: string): CompletionSession {
    const state = this.sessions.get(sessionId);
    if (state === undefined) {
      return parseCompletionSession({
        session_id: sessionId,
        completion_count: 0,
        completions: [],
      });
    }
    return parseCompletionSession({
      ...this.metadataPayload(state),
      completions: state.completions,
    });
  }

  deleteSession(sessionId: string): number {
    const state = this.sessions.get(sessionId);
    if (state === undefined) return 0;
    this.sessions.delete(sessionId);
    return state.completions.length;
  }

  private getOrCreate(sessionId: string, createdAt: string | null): SessionState {
    let state = this.sessions.get(sessionId);
    if (state === undefined) {
      state = {
        sessionId,
        createdAt: createdAt ?? new Date().toISOString(),
        taskId: null,
        modelRequested: null,
        modelUsed: null,
        apiType: null,
        metadata: {},
        completions: [],
      };
      this.sessions.set(sessionId, state);
      return state;
    }
    if (state.createdAt === null) state.createdAt = createdAt ?? new Date().toISOString();
    return state;
  }

  private mergeMetadata(
    state: SessionState,
    args: {
      taskId?: string | null;
      modelRequested?: string | null;
      modelUsed?: string | null;
      apiType?: string | null;
      metadata?: Dict | null;
    },
  ): void {
    state.taskId = mergeField(state.taskId, args.taskId);
    state.modelRequested = mergeField(state.modelRequested, args.modelRequested);
    state.modelUsed = mergeField(state.modelUsed, args.modelUsed);
    state.apiType = mergeField(state.apiType, args.apiType);
    if (args.metadata) Object.assign(state.metadata, args.metadata);
  }

  private metadataPayload(state: SessionState): Dict {
    return {
      session_id: state.sessionId,
      created_at: state.createdAt,
      completion_count: state.completions.length,
      task_id: state.taskId,
      model_requested: state.modelRequested,
      model_used: state.modelUsed,
      api_type: state.apiType,
      metadata: { ...state.metadata },
    };
  }
}
