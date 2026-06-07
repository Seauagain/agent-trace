/**
 * agent-trace — capture agent LLM trajectories and serialize them for SFT/RL.
 *
 * Library surface: the trajectory schemas + builders (turn captured completion
 * records into trainable traces) and the serializers (turn traces into SFT/RL
 * JSONL). The transparent capture proxy is exported separately from
 * `agent-trace/proxy`.
 */

export * from "./trajectory/index.js";
export * from "./serialize/index.js";
