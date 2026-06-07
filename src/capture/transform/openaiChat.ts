/** OpenAI Chat Completions transformer — near-passthrough (port of openai_chat.py). */

import { BaseTransformer } from "./base.js";

type Dict = Record<string, unknown>;

export class OpenAIChatTransformer extends BaseTransformer {
  override transformRequest(body: Dict): Dict {
    const result = { ...body };
    if (!("max_tokens" in result) && "max_completion_tokens" in result) {
      result["max_tokens"] = result["max_completion_tokens"];
    }
    return this.normalizeRequest(result, body["_at_model_served"] as string | undefined);
  }

  override transformResponse(response: Dict, originalRequest: Dict): Dict {
    const result = { ...response };
    if ("model" in originalRequest) result["model"] = originalRequest["model"];
    return result;
  }

  override transformStreamChunk(chunk: Dict, originalRequest: Dict, _isFirst: boolean): Dict {
    const result = { ...chunk };
    if ("model" in originalRequest) result["model"] = originalRequest["model"];
    return result;
  }
}
