/** APIType -> transformer registry. */

import { APIType } from "../detection.js";
import type { BaseTransformer } from "./base.js";
import { OpenAIChatTransformer } from "./openaiChat.js";
import { AnthropicTransformer } from "./anthropic.js";

export class TransformManager {
  private readonly transformers: Partial<Record<APIType, BaseTransformer>>;

  constructor() {
    this.transformers = {
      [APIType.OPENAI_CHAT]: new OpenAIChatTransformer(),
      [APIType.ANTHROPIC]: new AnthropicTransformer(),
    };
  }

  /** Return the transformer for an API type, falling back to OpenAI Chat. */
  get(apiType: APIType): BaseTransformer {
    const transformer = this.transformers[apiType];
    if (transformer === undefined) {
      throw new Error(
        `No transformer registered for API type ${JSON.stringify(apiType)}. ` +
          `Supported in this build: ${Object.keys(this.transformers).join(", ")}.`,
      );
    }
    return transformer;
  }

  supports(apiType: APIType): boolean {
    return this.transformers[apiType] !== undefined;
  }
}

export { BaseTransformer } from "./base.js";
export { OpenAIChatTransformer } from "./openaiChat.js";
export { AnthropicTransformer, AnthropicStreamState } from "./anthropic.js";
