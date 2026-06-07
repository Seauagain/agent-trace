/** Base interface for trajectory builders. */

import type { CompletionSession, Trajectory } from "../models.js";

/**
 * Strategy plugin that converts a completion session into a trajectory.
 * Builders are constructed per-request with a config object.
 */
export interface TrajectoryBuilder {
  build(session: CompletionSession): Promise<Trajectory>;
}

/** Factory signature: a registry constructs one builder per request from config. */
export type BuilderFactory = (config?: Record<string, unknown>) => TrajectoryBuilder;

const SCHEDULER_KEYS = ["group_id", "policy_version", "rollout_step"] as const;

/** Lift the scheduler-tracking keys to the top level of trajectory metadata. */
export function topLevelSchedulerMetadata(
  metadata: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of SCHEDULER_KEYS) {
    if (key in metadata) out[key] = metadata[key];
  }
  return out;
}
