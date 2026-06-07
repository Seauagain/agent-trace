/**
 * Inference backend strategies.
 *
 * The proxy speaks the OpenAI Chat Completions API to a local inference server.
 * Backends differ only in (1) the request params that make them emit token ids
 * + per-token logprobs we need for training, and (2) the response shape of
 * those fields. The base encodes the canonical contract (request `logprobs`,
 * pass-through response); a backend overrides only what it does differently.
 *
 * Canonical shape (SGLang's patched output):
 *   - prompt token ids:   choice.input_token_ids (or response.prompt_token_ids)
 *   - response token ids: choice.token_ids       (or logprobs.content[].token_id)
 *   - per-token logprobs: choice.logprobs.content[] with {token, token_id, logprob}
 */

type Dict = Record<string, unknown>;

function isDict(v: unknown): v is Dict {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export interface InferenceEngine {
  readonly name: string;
  /** Inject the request params this backend needs to emit training signals. */
  prepareRequest(request: Dict): Dict;
  /** Canonicalize the backend's response (in place) and return it. */
  normalizeResponse(response: Dict): Dict;
}

/** Canonical backend: emits the training shape with no per-request adaptation. */
export class SGLangEngine implements InferenceEngine {
  readonly name = "sglang";

  prepareRequest(request: Dict): Dict {
    request["logprobs"] = true;
    return request;
  }

  normalizeResponse(response: Dict): Dict {
    return response;
  }
}

/**
 * vLLM via its native OpenAI-compatible server. `return_token_ids` makes vLLM
 * emit response.prompt_token_ids and choice.token_ids. `top_logprobs` must be
 * set (not null) for vLLM to populate logprobs.content[]; 0 returns just the
 * sampled token's logprob, which is all training needs.
 */
export class VLLMEngine implements InferenceEngine {
  readonly name = "vllm";

  prepareRequest(request: Dict): Dict {
    request["logprobs"] = true;
    request["return_token_ids"] = true;
    if (request["top_logprobs"] === undefined) request["top_logprobs"] = 0;
    // vLLM reads input reasoning from `reasoning`, not `reasoning_content`.
    const messages = request["messages"];
    if (Array.isArray(messages)) {
      for (const message of messages) {
        if (isDict(message) && message["reasoning_content"] != null) {
          message["reasoning"] = message["reasoning_content"];
          delete message["reasoning_content"];
        }
      }
    }
    return request;
  }

  normalizeResponse(response: Dict): Dict {
    const choices = response["choices"];
    if (!Array.isArray(choices)) return response;
    for (const choice of choices) {
      if (!isDict(choice)) continue;
      VLLMEngine.canonicalizeReasoning(choice["message"]);
      VLLMEngine.stampTokenIdsOntoLogprobs(choice);
    }
    return response;
  }

  private static canonicalizeReasoning(message: unknown): void {
    if (!isDict(message)) return;
    if (message["reasoning_content"] == null && message["reasoning"] != null) {
      message["reasoning_content"] = message["reasoning"];
      delete message["reasoning"];
    }
  }

  private static stampTokenIdsOntoLogprobs(choice: Dict): void {
    const tokenIds = choice["token_ids"];
    const logprobs = choice["logprobs"];
    if (!Array.isArray(tokenIds) || !isDict(logprobs)) return;
    const content = logprobs["content"];
    if (!Array.isArray(content) || content.length !== tokenIds.length) return;
    for (let i = 0; i < content.length; i++) {
      const entry = content[i];
      if (isDict(entry) && entry["token_id"] === undefined) {
        entry["token_id"] = tokenIds[i];
      }
    }
  }
}

const ENGINES: Record<string, () => InferenceEngine> = {
  sglang: () => new SGLangEngine(),
  vllm: () => new VLLMEngine(),
};

/** Return the inference engine strategy for `name` (`sglang` | `vllm`). */
export function getEngine(name: string): InferenceEngine {
  const factory = ENGINES[name];
  if (factory === undefined) {
    const supported = Object.keys(ENGINES).sort().join(", ");
    throw new Error(`Unknown inference engine ${JSON.stringify(name)}; supported: ${supported}`);
  }
  return factory();
}
