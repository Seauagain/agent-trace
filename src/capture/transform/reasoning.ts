/**
 * Reasoning round-trip helpers.
 *
 * SGLang/vLLM split chain-of-thought into the assistant message's
 * `reasoning_content`. Signatures only need to round-trip opaquely through the
 * harness, so we use deterministic synthetic tokens — no real cryptography.
 */

import { createHash } from "node:crypto";

type Dict = Record<string, unknown>;

/** Deterministic synthetic signature for an Anthropic thinking block. */
export function makeSignature(reasoningText: string): string {
  if (!reasoningText) return "";
  const digest = createHash("sha256").update(reasoningText, "utf-8").digest();
  return "sg_at_" + digest.toString("base64url");
}

/** Extract reasoning_content from Anthropic assistant content blocks. */
export function extractReasoningFromAnthropicContent(content: unknown): string {
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (block && typeof block === "object" && (block as Dict)["type"] === "thinking") {
      const text = (block as Dict)["thinking"];
      if (typeof text === "string" && text) parts.push(text);
    }
  }
  return parts.join("\n");
}
