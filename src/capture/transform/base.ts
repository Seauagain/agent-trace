/**
 * Base transformer interface.
 *
 * A transformer converts a request from its native API into OpenAI Chat
 * Completions (for the inference backend), and converts the response back.
 * The canonical internal format is OpenAI Chat Completions. Training-signal
 * params (logprobs / token ids) are added later by the inference engine.
 */

type Dict = Record<string, unknown>;

export interface StreamState {
  processChunk(chunk: Dict, isFirst: boolean): Dict[];
  finalize(): Dict[];
}

export abstract class BaseTransformer {
  abstract transformRequest(body: Dict): Dict;
  abstract transformResponse(response: Dict, originalRequest: Dict): Dict;
  abstract transformStreamChunk(chunk: Dict, originalRequest: Dict, isFirst: boolean): Dict | Dict[];

  isStreamingRequest(body: Dict): boolean {
    return Boolean(body["stream"]);
  }

  createStreamState(_originalRequest: Dict): StreamState | null {
    return null;
  }

  protected static isQwen35Model(modelName: string | null | undefined): boolean {
    if (!modelName) return false;
    return modelName.toLowerCase().includes("qwen3.5");
  }

  protected static contentToText(content: unknown): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      const parts: string[] = [];
      for (const block of content) {
        if (typeof block === "string") parts.push(block);
        else if (block && typeof block === "object") {
          const text = (block as Dict)["text"];
          if (typeof text === "string") parts.push(text);
        }
      }
      return parts.join("\n");
    }
    return content ? String(content) : "";
  }

  /** Rename 'developer' role to 'system' and merge all system messages into one. */
  protected static mergeDeveloperRole(request: Dict): Dict {
    const messages = request["messages"];
    if (!Array.isArray(messages)) return request;

    const normalized = messages.map((msg) =>
      msg && typeof msg === "object" && (msg as Dict)["role"] === "developer"
        ? { ...(msg as Dict), role: "system" }
        : msg,
    );

    const systemParts: string[] = [];
    const nonSystem: unknown[] = [];
    for (const msg of normalized) {
      if (msg && typeof msg === "object" && (msg as Dict)["role"] === "system") {
        const text = BaseTransformer.contentToText((msg as Dict)["content"] ?? "");
        if (text) systemParts.push(text);
      } else {
        nonSystem.push(msg);
      }
    }

    if (systemParts.length > 0) {
      request["messages"] = [{ role: "system", content: systemParts.join("\n\n") }, ...nonSystem];
    } else {
      request["messages"] = nonSystem;
    }
    return request;
  }

  /** Drop internal keys, merge system roles, apply per-model template fixes. */
  protected normalizeRequest(request: Dict, modelName?: string | null): Dict {
    delete request["_at_model_served"];
    request = BaseTransformer.mergeDeveloperRole(request);

    if (BaseTransformer.isQwen35Model(modelName)) {
      const kwargs = { ...((request["chat_template_kwargs"] as Dict) ?? {}) };
      if (kwargs["enable_thinking"] === undefined) kwargs["enable_thinking"] = false;
      request["chat_template_kwargs"] = kwargs;
    }
    return request;
  }
}
