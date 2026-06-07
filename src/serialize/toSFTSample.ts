/**
 * Serialize trajectory traces into SFT-trainable samples.
 *
 * Two complementary shapes:
 *  - message-level: `{ messages, tools }` in OpenAI chat form, directly
 *    consumable by chat-template SFT trainers (HF TRL, axolotl, ...).
 *  - token-level (opt-in): `{ input_ids, labels }` where labels are -100 on the
 *    prompt and on any loss_mask=0 position, so only sampled assistant tokens
 *    contribute to the loss. This requires captured token ids.
 */

import type { Trace, Trajectory } from "../trajectory/models.js";

/** HF convention: ignore index excluded from the loss. */
export const IGNORE_INDEX = -100;

export interface SFTSample {
  messages: Record<string, unknown>[];
  tools: Record<string, unknown>[] | null;
  input_ids?: number[];
  labels?: number[];
  metadata: Record<string, unknown>;
}

export interface SFTSampleOptions {
  /** Also emit token-level `input_ids`/`labels` (requires captured token ids). */
  includeTokens?: boolean;
  /** Extra fields merged into every emitted sample. */
  extra?: Record<string, unknown>;
}

/** Convert one trace into an SFT sample. */
export function toSFTSample(trace: Trace, options: SFTSampleOptions = {}): SFTSample {
  const sample: SFTSample = {
    messages: [
      ...trace.prompt_messages.map((m) => structuredClone(m)),
      ...trace.response_messages.map((m) => structuredClone(m)),
    ],
    tools: trace.tools ? trace.tools.map((t) => structuredClone(t)) : null,
    metadata: { ...trace.metadata, ...(options.extra ?? {}) },
  };

  if (options.includeTokens) {
    const inputIds = [...trace.prompt_ids, ...trace.response_ids];
    const lossMask = trace.loss_mask.length > 0 ? trace.loss_mask : trace.response_ids.map(() => 1);
    const labels: number[] = [
      ...trace.prompt_ids.map(() => IGNORE_INDEX),
      ...trace.response_ids.map((tokenId, i) => (lossMask[i] === 1 ? tokenId : IGNORE_INDEX)),
    ];
    sample.input_ids = inputIds;
    sample.labels = labels;
  }

  return sample;
}

/** Convert all traces of a trajectory into SFT samples (one per trace). */
export function trajectoryToSFTSamples(
  trajectory: Trajectory,
  options: SFTSampleOptions = {},
): SFTSample[] {
  return trajectory.traces.map((trace) => toSFTSample(trace, options));
}
