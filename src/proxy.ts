/**
 * agent-trace/proxy — the transparent capture proxy.
 *
 * Point any agent runtime's OPENAI_BASE_URL at a running proxy (with the bearer
 * token set to a session id) and every LLM call is captured. Call
 * POST /sessions/:id/finalize to build the trajectory and get SFT/RL JSONL.
 */

export * from "./capture/index.js";
