/**
 * Prefix-merging trajectory builder.
 *
 * Reconstructs a single token-level training trace out of the many independent
 * LLM completions an agent emits during one rollout. A harness drives the agent
 * and each turn hits the proxy as a separate completion request; this builder
 * stitches those completions back into the
 *   prompt + response_1 + interstitial + response_2 + ...
 * stream an RL trainer needs, without introducing tokenization drift.
 *
 * Two stages:
 *  1. Grouping — route each completion to the chain it append-extends, tested
 *     purely on tokens: a completion joins the chain whose last prompt is a
 *     prefix of it. Robust to BPE re-tokenization because it compares only
 *     server-tokenized prompts.
 *  2. Finalization — walk each chain and build a merged token stream. Assistant
 *     bodies come from the raw sampled response_ids (real logprobs, never
 *     decode->re-encode). Interstitials come from the next completion's
 *     canonical prompt tail, split at the first end-of-turn token. Interstitial
 *     slots get synthesized logprobs and loss_mask=0; sampled slots keep real
 *     logprobs and loss_mask=1.
 */

import { buildTraceFromCompletion } from "../recordUtils.js";
import {
  type CompletionRecord,
  type CompletionSession,
  createTrace,
  createTrajectory,
  type Trace,
  type Trajectory,
} from "../models.js";
import { type TrajectoryBuilder, topLevelSchedulerMetadata } from "./base.js";

type Dict = Record<string, unknown>;

// finish_reasons where the model emitted the natural end-of-turn token itself.
const NATURAL_STOP_REASONS = new Set(["stop", "tool_calls", "stop_sequence"]);

interface ReconstructionStats {
  chains_total: number;
  chains_reconstructed_full: number;
  chains_reconstructed_truncated: number;
  completions_total: number;
  completions_merged: number;
}

function prefixEquals(arr: number[], prefix: number[]): boolean {
  if (prefix.length > arr.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    if (arr[i] !== prefix[i]) return false;
  }
  return true;
}

export interface PrefixMergingConfig {
  /**
   * Explicit end-of-turn (EOT) token id used to locate the canonical-tail split
   * between the prior assistant body and the interstitial. When undefined, the
   * builder auto-detects it from the last token of the first completion with a
   * natural stop reason (e.g. `<|im_end|>` on Qwen / ChatML).
   */
  end_of_turn_token_id?: number | null;
}

export class PrefixMergingBuilder implements TrajectoryBuilder {
  private readonly configuredEotId: number | null;

  constructor(config: PrefixMergingConfig = {}) {
    this.configuredEotId = config.end_of_turn_token_id ?? null;
  }

  async build(session: CompletionSession): Promise<Trajectory> {
    if (session.completions.length === 0) {
      return createTrajectory({
        status: "ERROR",
        metadata: {
          builder: "prefix_merging",
          session_id: session.session_id,
          task_metadata: { ...session.metadata },
          record_count: 0,
          ...topLevelSchedulerMetadata(session.metadata),
        },
        traces: [],
        error: "no completions",
      });
    }

    const chains: CompletionRecord[][] = [];
    const chainTips: number[][] = []; // last completion's prompt_ids, per chain

    for (const completion of session.completions) {
      const promptIds = buildTraceFromCompletion(completion).prompt_ids;
      let chainIdx = this.findExtendableChain(promptIds, chainTips);
      if (chainIdx === null) {
        chainIdx = chains.length;
        chains.push([]);
        chainTips.push([]);
      }
      chains[chainIdx]!.push(completion);
      chainTips[chainIdx] = promptIds;
    }

    const stats: ReconstructionStats = {
      chains_total: chains.length,
      chains_reconstructed_full: 0,
      chains_reconstructed_truncated: 0,
      completions_total: session.completions.length,
      completions_merged: 0,
    };
    const finalTraces = chains.map((chain) => this.finalizeChain(chain, stats));

    return createTrajectory({
      status: "COMPLETED",
      metadata: {
        builder: "prefix_merging",
        session_id: session.session_id,
        task_id: session.task_id ?? null,
        api_type: session.api_type ?? null,
        model_requested: session.model_requested ?? null,
        model_used: session.model_used ?? null,
        record_count: session.completions.length,
        task_metadata: { ...session.metadata },
        trace_count: chains.length,
        reconstruction_stats: stats,
        ...topLevelSchedulerMetadata(session.metadata),
      },
      traces: finalTraces,
    });
  }

  // ------------------------------------------------------------------
  // Chain finalization
  // ------------------------------------------------------------------

  private finalizeChain(chain: CompletionRecord[], stats: ReconstructionStats): Trace {
    const firstTrace = buildTraceFromCompletion(chain[0]!);
    const eotId = this.resolveEotId(chain);

    const promptIds = [...firstTrace.prompt_ids];
    const streamIds: number[] = [...promptIds];
    const responseSlots: (number | null)[] = [];
    const lossMask: number[] = [];
    const responseMessages: Dict[] = [];

    let prevPromptIds: number[] = [...firstTrace.prompt_ids];
    let prevRawResponse: number[] = [...firstTrace.response_ids];

    // Running count of messages consumed = prompt_messages + all response_messages emitted.
    let msgAcc = firstTrace.prompt_messages.length;

    appendResponseTokens(firstTrace, streamIds, responseSlots, lossMask);
    for (const m of firstTrace.response_messages) responseMessages.push(structuredClone(m));
    msgAcc += firstTrace.response_messages.length;
    let kept = 1;

    for (let i = 1; i < chain.length; i++) {
      const ciTrace = buildTraceFromCompletion(chain[i]!);
      const ciPromptIds = [...ciTrace.prompt_ids];

      // Canonical-vs-canonical prefix check: both sides are server-side
      // tokenizations of the same message prefix.
      if (ciPromptIds.length < prevPromptIds.length || !prefixEquals(ciPromptIds, prevPromptIds)) {
        break;
      }

      const canonicalTail = ciPromptIds.slice(prevPromptIds.length);
      const interstitial = sliceInterstitial(canonicalTail, prevRawResponse, eotId);
      if (interstitial === null) {
        break;
      }

      if (interstitial.length > 0) {
        streamIds.push(...interstitial);
        for (let j = 0; j < interstitial.length; j++) {
          responseSlots.push(null);
          lossMask.push(0);
        }
      }

      // Message-level interstitial bookkeeping.
      if (ciTrace.prompt_messages.length > msgAcc) {
        const interstitialMsgs = ciTrace.prompt_messages.slice(msgAcc);
        for (const m of interstitialMsgs) responseMessages.push(structuredClone(m));
        msgAcc += interstitialMsgs.length;
      }

      appendResponseTokens(ciTrace, streamIds, responseSlots, lossMask);
      for (const m of ciTrace.response_messages) responseMessages.push(structuredClone(m));
      msgAcc += ciTrace.response_messages.length;

      prevPromptIds = ciPromptIds;
      prevRawResponse = [...ciTrace.response_ids];
      kept += 1;
    }

    stats.completions_merged += kept;
    if (kept === chain.length) {
      stats.chains_reconstructed_full += 1;
    } else {
      stats.chains_reconstructed_truncated += 1;
    }

    const responseIds = streamIds.slice(promptIds.length);
    const responseLogprobs = finalizeLogprobs(responseSlots);
    const lastKeptTrace = buildTraceFromCompletion(chain[kept - 1]!);

    return createTrace({
      prompt_ids: promptIds,
      response_ids: responseIds,
      loss_mask: lossMask,
      prompt_messages: firstTrace.prompt_messages.map((m) => structuredClone(m)),
      response_messages: responseMessages,
      tools: firstTrace.tools ? firstTrace.tools.map((t) => structuredClone(t)) : null,
      finish_reason: lastKeptTrace.finish_reason,
      response_logprobs: responseLogprobs,
      metadata: chainMetadata(chain.slice(0, kept)),
    });
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  private resolveEotId(chain: CompletionRecord[]): number | null {
    if (this.configuredEotId !== null) return this.configuredEotId;
    for (const completion of chain) {
      const trace = buildTraceFromCompletion(completion);
      if (
        trace.finish_reason !== null &&
        NATURAL_STOP_REASONS.has(trace.finish_reason) &&
        trace.response_ids.length > 0
      ) {
        return trace.response_ids[trace.response_ids.length - 1]!;
      }
    }
    return null;
  }

  private findExtendableChain(promptIds: number[], chainTips: number[][]): number | null {
    let bestIdx: number | null = null;
    let bestLen = -1;
    for (let idx = 0; idx < chainTips.length; idx++) {
      const tip = chainTips[idx]!;
      const n = tip.length;
      if (n > bestLen && n > 0 && n <= promptIds.length && prefixEquals(promptIds, tip)) {
        bestIdx = idx;
        bestLen = n;
      }
    }
    return bestIdx;
  }
}

/**
 * Extract the canonical interstitial from the next completion's prompt tail.
 * The first occurrence of eotId marks the end of the prev assistant body;
 * everything after is interstitial. If prevRawResponse already ends with eotId,
 * skip it to avoid duplication; otherwise include it so the stream still closes
 * the assistant turn. Returns null if eotId is unknown or not present.
 */
function sliceInterstitial(
  canonicalTail: number[],
  prevRawResponse: number[],
  eotId: number | null,
): number[] | null {
  if (eotId === null) return null;
  const k = canonicalTail.indexOf(eotId);
  if (k === -1) return null;
  if (prevRawResponse.length > 0 && prevRawResponse[prevRawResponse.length - 1] === eotId) {
    return canonicalTail.slice(k + 1);
  }
  return canonicalTail.slice(k);
}

/** Append a completion's response_ids and parallel logprob slots. */
function appendResponseTokens(
  trace: Trace,
  streamIds: number[],
  responseSlots: (number | null)[],
  lossMask: number[],
): void {
  const responseIds = [...trace.response_ids];
  streamIds.push(...responseIds);
  const traceLossMask = trace.loss_mask.length > 0 ? [...trace.loss_mask] : responseIds.map(() => 1);
  if (traceLossMask.length !== responseIds.length) {
    throw new Error("trace loss_mask length must match response_ids length");
  }
  lossMask.push(...traceLossMask);
  const logprobs = trace.response_logprobs ?? [];
  for (let pos = 0; pos < responseIds.length; pos++) {
    const value = pos < logprobs.length ? logprobs[pos] : null;
    responseSlots.push(typeof value === "number" ? value : null);
  }
}

/** Interstitial slots (tool results, chat glue) get 0.0; loss_mask=0 ignores them. */
function finalizeLogprobs(slots: (number | null)[]): number[] | null {
  if (!slots.some((slot) => slot !== null)) return null;
  return slots.map((slot) => (slot !== null ? slot : 0.0));
}

function chainMetadata(chain: CompletionRecord[]): Dict {
  const completionMetadata = chain.map((completion) => ({ ...completion.metadata }));
  const merged: Dict = completionMetadata.length > 0 ? { ...completionMetadata[0] } : {};
  merged["completion_metadata"] = completionMetadata;
  return merged;
}
