/**
 * Shared completion-session and trajectory schemas.
 *
 * Data models as Zod schemas (originally pydantic). These are the
 * data shapes that flow through the pipeline: the proxy captures
 * `CompletionRecord`s into a `CompletionSession`; a builder turns that into a
 * `Trajectory` of trainable `Trace`s.
 */

import { z } from "zod";

const Json = z.any();

// ---------------------------------------------------------------------------
// Strategy / evaluator specs
// ---------------------------------------------------------------------------

/** Identifies a builder strategy with optional per-request config. */
export const StrategySpecSchema = z.object({
  strategy: z.string(),
  config: z.record(Json).default({}),
});
export type StrategySpec = z.infer<typeof StrategySpecSchema>;

/** Evaluator output — a caller may merge rewards into the trajectory. */
export const EvalResultSchema = z.object({
  outcome_reward: z.number().nullable().default(null),
  trace_rewards: z.array(z.number().nullable()).nullable().default(null),
  metadata: z.record(Json).default({}),
});
export type EvalResult = z.infer<typeof EvalResultSchema>;

// ---------------------------------------------------------------------------
// Completion and trajectory models
// ---------------------------------------------------------------------------

/** One normalized upstream completion payload. */
export const CompletionRecordSchema = z
  .object({
    completion_id: z.string(),
    timestamp: z.string().nullable().optional(),
    request: z.record(Json).default({}),
    original_request: z.record(Json).default({}),
    response: z.record(Json).default({}),
    metadata: z.record(Json).default({}),
  })
  .passthrough();
export type CompletionRecord = z.infer<typeof CompletionRecordSchema>;

const CompletionSessionShape = z
  .object({
    session_id: z.string(),
    created_at: z.string().nullable().optional(),
    completion_count: z.number().int().default(0),
    task_id: z.string().nullable().optional(),
    model_requested: z.string().nullable().optional(),
    model_used: z.string().nullable().optional(),
    api_type: z.string().nullable().optional(),
    metadata: z.record(Json).default({}),
    completions: z.array(CompletionRecordSchema).default([]),
  })
  .passthrough();

/**
 * Builder-facing session payload. Completions are auto-sorted by
 * (timestamp, completion_id) so builders see them in deterministic order —
 * matching the pydantic `_sort_completions` model validator.
 */
export const CompletionSessionSchema = CompletionSessionShape.transform((s) => ({
  ...s,
  completions: [...s.completions].sort((a, b) => {
    const ka = String(a.timestamp ?? "");
    const kb = String(b.timestamp ?? "");
    if (ka < kb) return -1;
    if (ka > kb) return 1;
    return a.completion_id < b.completion_id ? -1 : a.completion_id > b.completion_id ? 1 : 0;
  }),
}));
export type CompletionSession = z.infer<typeof CompletionSessionSchema>;

const VALID_STATUS = ["COMPLETED", "TIMEOUT", "ERROR"] as const;
export type TrajectoryStatus = (typeof VALID_STATUS)[number];

/** One reconstructed completion interaction (a trainable example). */
export const TraceSchema = z
  .object({
    prompt_ids: z.array(z.number().int()).default([]),
    response_ids: z.array(z.number().int()).default([]),
    loss_mask: z.array(z.number().int()).default([]),
    prompt_messages: z.array(z.record(Json)).default([]),
    response_messages: z.array(z.record(Json)).default([]),
    tools: z.array(z.record(Json)).nullable().default(null),
    finish_reason: z.string().nullable().default(null),
    response_logprobs: z.array(z.number()).nullable().default(null),
    reward: z.number().nullable().default(null),
    metadata: z.record(Json).default({}),
  })
  .superRefine((trace, ctx) => {
    for (const m of trace.loss_mask) {
      if (m !== 0 && m !== 1) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: "loss_mask values must be 0 or 1" });
        break;
      }
    }
    if (trace.loss_mask.length > 0 && trace.loss_mask.length !== trace.response_ids.length) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "loss_mask length must match response_ids length",
      });
    }
    if (
      trace.response_logprobs !== null &&
      trace.response_logprobs.length !== trace.response_ids.length
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "response_logprobs length must match response_ids length",
      });
    }
  });
export type Trace = z.infer<typeof TraceSchema>;

/** Structured trajectory reconstructed from session completion records. */
export const TrajectorySchema = z.object({
  status: z.enum(VALID_STATUS),
  metadata: z.record(Json).default({}),
  traces: z.array(TraceSchema).default([]),
  error: z.string().nullable().default(null),
});
export type Trajectory = z.infer<typeof TrajectorySchema>;

// ---------------------------------------------------------------------------
// Convenience constructors (validate on construction, like pydantic)
// ---------------------------------------------------------------------------

/** Build and validate a `Trace`, applying defaults. Throws on invalid shape. */
export function createTrace(input: Partial<Trace>): Trace {
  return TraceSchema.parse(input);
}

/** Build and validate a `Trajectory`. */
export function createTrajectory(input: {
  status: TrajectoryStatus;
  metadata?: Record<string, unknown>;
  traces?: Trace[];
  error?: string | null;
}): Trajectory {
  return TrajectorySchema.parse(input);
}

/** Parse + sort an untyped object into a `CompletionSession`. */
export function parseCompletionSession(input: unknown): CompletionSession {
  return CompletionSessionSchema.parse(input);
}
