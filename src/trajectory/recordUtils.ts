/**
 * Helpers for converting completion records into trajectory traces.
 *
 * The canonical
 * captured response shape (SGLang's patched output, which vLLM is normalized
 * to) is:
 *   - prompt token ids:   choice.input_token_ids (or response.prompt_token_ids)
 *   - response token ids: choice.token_ids       (or logprobs.content[].token_id)
 *   - per-token logprobs: choice.logprobs.content[] with {token, token_id, logprob}
 */

import { type CompletionRecord, createTrace, type Trace } from "./models.js";

type Dict = Record<string, unknown>;

function isDict(v: unknown): v is Dict {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function extractResponseIds(response: Dict, choice: Dict): number[] {
  const tokenIds = choice["token_ids"] ?? response["token_ids"];
  if (Array.isArray(tokenIds)) {
    return tokenIds.map((t) => Number(t));
  }

  const logprobs = choice["logprobs"];
  if (isDict(logprobs)) {
    const content = logprobs["content"];
    if (Array.isArray(content)) {
      const extracted: number[] = [];
      for (const item of content) {
        if (isDict(item) && item["token_id"] != null) {
          extracted.push(Number(item["token_id"]));
        }
      }
      if (extracted.length > 0) return extracted;
    }
  }
  return [];
}

/**
 * Sampled-token logprob per position, aligned 1:1 with response_ids. The token
 * id is intentionally dropped — it is already in response_ids at the same index.
 */
function extractResponseLogprobs(choice: Dict): number[] | null {
  const logprobs = choice["logprobs"];
  if (isDict(logprobs)) {
    const content = logprobs["content"];
    if (Array.isArray(content)) {
      return content.map((item) =>
        isDict(item) && typeof item["logprob"] === "number" ? (item["logprob"] as number) : 0.0,
      );
    }
  }
  return null;
}

function extractPromptMessages(request: Dict): Dict[] {
  const messages = request["messages"];
  if (!Array.isArray(messages)) return [];
  return messages.filter(isDict).map((m) => structuredClone(m));
}

function extractTools(request: Dict): Dict[] | null {
  const tools = request["tools"];
  if (!Array.isArray(tools) || tools.length === 0) return null;
  const extracted = tools.filter(isDict).map((t) => structuredClone(t));
  return extracted.length > 0 ? extracted : null;
}

/** Normalize one stored completion record into a trajectory trace. */
export function buildTraceFromCompletion(completion: CompletionRecord): Trace {
  const request = isDict(completion.request) ? completion.request : {};
  const response = isDict(completion.response) ? completion.response : {};
  const choices = response["choices"];
  const firstChoice: Dict =
    Array.isArray(choices) && choices.length > 0 && isDict(choices[0]) ? (choices[0] as Dict) : {};

  const promptIdsRaw = firstChoice["input_token_ids"] ?? response["prompt_token_ids"];
  const responseMessage = firstChoice["message"];
  const finishReason = firstChoice["finish_reason"];

  const responseIds = extractResponseIds(response, firstChoice);

  return createTrace({
    prompt_ids: Array.isArray(promptIdsRaw) ? promptIdsRaw.map((t) => Number(t)) : [],
    response_ids: responseIds,
    loss_mask: responseIds.map(() => 1),
    prompt_messages: extractPromptMessages(request),
    response_messages: isDict(responseMessage) ? [structuredClone(responseMessage)] : [],
    tools: extractTools(request),
    finish_reason: finishReason != null ? String(finishReason) : null,
    response_logprobs: extractResponseLogprobs(firstChoice),
    metadata: structuredClone(completion.metadata ?? {}),
  });
}
