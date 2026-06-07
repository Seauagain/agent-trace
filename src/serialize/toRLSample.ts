/**
 * Serialize trajectory traces into RL-trainable samples.
 *
 * The token-level shape mirrors what an RL trainer (e.g. Slime/GRPO) consumes:
 * a flat token stream with a prompt/response split, a per-response-token loss
 * mask, the rollout logprobs, and a scalar reward. This is the same information
 * `slime_bridge` reads out of a `Trace`.
 */

import type { Trace, Trajectory } from "../trajectory/models.js";

export interface RLSample {
  /** prompt_ids + response_ids, the full token stream. */
  token_ids: number[];
  /** Length of the non-trainable prompt prefix. */
  prompt_len: number;
  /** Length of the trainable response suffix (== loss_mask / logprobs length). */
  response_len: number;
  /** Per-response-token 0/1 mask (1 = train on this token). */
  loss_mask: number[];
  /** Per-response-token rollout logprobs, aligned with the response suffix. */
  logprobs: number[] | null;
  /** Scalar reward attached by an evaluator (null if not yet scored). */
  reward: number | null;
  finish_reason: string | null;
  metadata: Record<string, unknown>;
}

export interface RLSampleOptions {
  /** Extra fields merged into every emitted sample (e.g. group_id for GRPO). */
  extra?: Record<string, unknown>;
}

/** Convert one trace into an RL sample. */
export function toRLSample(trace: Trace, options: RLSampleOptions = {}): RLSample {
  const responseLen = trace.response_ids.length;
  const lossMask = trace.loss_mask.length > 0 ? trace.loss_mask : trace.response_ids.map(() => 1);
  return {
    token_ids: [...trace.prompt_ids, ...trace.response_ids],
    prompt_len: trace.prompt_ids.length,
    response_len: responseLen,
    loss_mask: lossMask,
    logprobs: trace.response_logprobs,
    reward: trace.reward,
    finish_reason: trace.finish_reason,
    metadata: { ...trace.metadata, ...(options.extra ?? {}) },
  };
}

/** Convert all traces of a trajectory into RL samples (one per trace). */
export function trajectoryToRLSamples(
  trajectory: Trajectory,
  options: RLSampleOptions = {},
): RLSample[] {
  const extra = {
    ...(options.extra ?? {}),
    trajectory_status: trajectory.status,
  };
  return trajectory.traces.map((trace) => toRLSample(trace, { extra }));
}
