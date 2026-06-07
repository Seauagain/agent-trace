/**
 * Anthropic Messages API transformer.
 * Transforms between Anthropic Messages and OpenAI Chat Completions, including
 * tool_use/tool_result, thinking blocks, and stateful SSE reconstruction.
 */

import { randomUUID } from "node:crypto";

import { BaseTransformer, type StreamState } from "./base.js";
import {
  anthropicContentToOpenaiChat,
  openaiChatContentToAnthropicBlocks,
} from "./images.js";
import { extractReasoningFromAnthropicContent, makeSignature } from "./reasoning.js";

type Dict = Record<string, unknown>;

function isDict(v: unknown): v is Dict {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function hex(): string {
  return randomUUID().replace(/-/g, "");
}

// Claude Code SDK leaks a per-request billing header as the first system line;
// the hash changes every turn and breaks prefix_merging, so strip it.
const CLAUDE_CODE_BILLING_HEADER_RE = /^\s*x-anthropic-billing-header:[^\n]*\n?/i;

const FINISH_TO_STOP_REASON: Record<string, string> = {
  stop: "end_turn",
  length: "max_tokens",
  tool_calls: "tool_use",
  content_filter: "refusal",
  stop_sequence: "stop_sequence",
};

interface ToolCallState {
  id: string;
  name: string;
  anthropicIndex: number | null;
  bufferedArguments: string;
  started: boolean;
}

export class AnthropicStreamState implements StreamState {
  private readonly messageId = `msg_${hex()}`;
  private nextBlockIndex = 0;
  private textBlockIndex: number | null = null;
  private textBlockStarted = false;
  private thinkingBlockIndex: number | null = null;
  private thinkingBlockStarted = false;
  private thinkingBuffer = "";
  private readonly toolCalls = new Map<number, ToolCallState>();
  private stopReason = "end_turn";
  private outputTokens = 0;
  private anyBlockStarted = false;
  private completed = false;

  constructor(
    private readonly model: string,
    private readonly finishToStopReason: Record<string, string>,
  ) {}

  processChunk(chunk: Dict, isFirst: boolean): Dict[] {
    const events: Dict[] = [];

    if (isFirst) {
      events.push({
        type: "message_start",
        message: {
          id: this.messageId,
          type: "message",
          role: "assistant",
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      });
    }

    const usage = (chunk["usage"] as Dict | undefined) ?? {};
    if (Object.keys(usage).length > 0) {
      this.outputTokens = (usage["completion_tokens"] as number | undefined) ?? this.outputTokens;
    }

    const choices = (chunk["choices"] as Dict[] | undefined) ?? [];
    if (choices.length === 0) return events;

    const choice = choices[0]!;
    const delta = (choice["delta"] as Dict | undefined) ?? {};
    const finishReason = choice["finish_reason"];
    if (finishReason) {
      this.stopReason = this.finishToStopReason[finishReason as string] ?? "end_turn";
    }

    const reasoning = delta["reasoning_content"];
    if (reasoning) {
      if (!this.thinkingBlockStarted) events.push(this.openThinkingBlock());
      events.push({
        type: "content_block_delta",
        index: this.thinkingBlockIndex,
        delta: { type: "thinking_delta", thinking: reasoning },
      });
      this.thinkingBuffer += String(reasoning);
    }

    const content = delta["content"];
    if (content) {
      const thinkingStop = this.closeThinkingBlock();
      if (thinkingStop) events.push(...thinkingStop);
      if (!this.textBlockStarted) events.push(this.openTextBlock());
      events.push({
        type: "content_block_delta",
        index: this.textBlockIndex,
        delta: { type: "text_delta", text: content },
      });
    }

    let toolCallDeltas = delta["tool_calls"] ?? [];
    if (!Array.isArray(toolCallDeltas)) toolCallDeltas = [toolCallDeltas];
    for (const toolCallDelta of toolCallDeltas as unknown[]) {
      if (isDict(toolCallDelta)) events.push(...this.processToolCall(toolCallDelta));
    }

    return events;
  }

  finalize(): Dict[] {
    if (this.completed) return [];
    const events: Dict[] = [];

    const thinkingStop = this.closeThinkingBlock();
    if (thinkingStop) events.push(...thinkingStop);

    const textStop = this.closeTextBlock();
    if (textStop) events.push(textStop);

    for (const toolIndex of [...this.toolCalls.keys()].sort((a, b) => a - b)) {
      const toolState = this.toolCalls.get(toolIndex)!;
      if (toolState.started && toolState.anthropicIndex !== null) {
        events.push({ type: "content_block_stop", index: toolState.anthropicIndex });
      }
    }

    if (!this.anyBlockStarted) {
      const emptyIndex = this.nextBlockIndex;
      events.push({
        type: "content_block_start",
        index: emptyIndex,
        content_block: { type: "text", text: "" },
      });
      events.push({ type: "content_block_stop", index: emptyIndex });
    }

    events.push({
      type: "message_delta",
      delta: { stop_reason: this.stopReason, stop_sequence: null },
      usage: { output_tokens: this.outputTokens },
    });
    events.push({ type: "message_stop" });

    this.completed = true;
    return events;
  }

  private openTextBlock(): Dict {
    this.textBlockStarted = true;
    this.textBlockIndex = this.nextBlockIndex;
    this.nextBlockIndex += 1;
    this.anyBlockStarted = true;
    return {
      type: "content_block_start",
      index: this.textBlockIndex,
      content_block: { type: "text", text: "" },
    };
  }

  private closeTextBlock(): Dict | null {
    if (!this.textBlockStarted || this.textBlockIndex === null) return null;
    const event = { type: "content_block_stop", index: this.textBlockIndex };
    this.textBlockStarted = false;
    this.textBlockIndex = null;
    return event;
  }

  private openThinkingBlock(): Dict {
    this.thinkingBlockStarted = true;
    this.thinkingBlockIndex = this.nextBlockIndex;
    this.nextBlockIndex += 1;
    this.anyBlockStarted = true;
    return {
      type: "content_block_start",
      index: this.thinkingBlockIndex,
      content_block: { type: "thinking", thinking: "", signature: "" },
    };
  }

  private closeThinkingBlock(): Dict[] | null {
    if (!this.thinkingBlockStarted || this.thinkingBlockIndex === null) return null;
    const idx = this.thinkingBlockIndex;
    const events: Dict[] = [
      {
        type: "content_block_delta",
        index: idx,
        delta: { type: "signature_delta", signature: makeSignature(this.thinkingBuffer) },
      },
      { type: "content_block_stop", index: idx },
    ];
    this.thinkingBlockStarted = false;
    this.thinkingBlockIndex = null;
    return events;
  }

  private processToolCall(toolCallDelta: Dict): Dict[] {
    const events: Dict[] = [];

    let toolIndex = toolCallDelta["index"];
    if (typeof toolIndex !== "number") toolIndex = 0;

    let toolState = this.toolCalls.get(toolIndex as number);
    if (toolState === undefined) {
      toolState = {
        id: (toolCallDelta["id"] as string | undefined) ?? `toolu_${hex().slice(0, 24)}`,
        name: "",
        anthropicIndex: null,
        bufferedArguments: "",
        started: false,
      };
      this.toolCalls.set(toolIndex as number, toolState);
    } else if (toolCallDelta["id"]) {
      toolState.id = toolCallDelta["id"] as string;
    }

    const func = (toolCallDelta["function"] as Dict | undefined) ?? {};
    const name = func["name"];
    if (typeof name === "string" && name) toolState.name += name;

    const args = func["arguments"];
    let argsStr = "";
    if (typeof args === "string" && args) argsStr = args;
    else if (args !== null && args !== undefined && args !== "") argsStr = JSON.stringify(args);

    if (argsStr) toolState.bufferedArguments += argsStr;

    if (toolState.name && !toolState.started) {
      const thinkingStop = this.closeThinkingBlock();
      if (thinkingStop) events.push(...thinkingStop);
      const textStop = this.closeTextBlock();
      if (textStop) events.push(textStop);

      toolState.started = true;
      toolState.anthropicIndex = this.nextBlockIndex;
      this.nextBlockIndex += 1;
      this.anyBlockStarted = true;

      events.push({
        type: "content_block_start",
        index: toolState.anthropicIndex,
        content_block: { type: "tool_use", id: toolState.id, name: toolState.name, input: {} },
      });

      if (toolState.bufferedArguments) {
        events.push({
          type: "content_block_delta",
          index: toolState.anthropicIndex,
          delta: { type: "input_json_delta", partial_json: toolState.bufferedArguments },
        });
        toolState.bufferedArguments = "";
      }
    } else if (toolState.started && argsStr && toolState.anthropicIndex !== null) {
      events.push({
        type: "content_block_delta",
        index: toolState.anthropicIndex,
        delta: { type: "input_json_delta", partial_json: argsStr },
      });
    }

    return events;
  }
}

export class AnthropicTransformer extends BaseTransformer {
  override transformRequest(body: Dict): Dict {
    const messages: Dict[] = [];

    const system = body["system"];
    if (system) {
      let systemContent = this.flattenContent(system);
      systemContent = systemContent.replace(CLAUDE_CODE_BILLING_HEADER_RE, "");
      if (systemContent) messages.push({ role: "system", content: systemContent });
    }

    for (const msg of (body["messages"] as unknown[] | undefined) ?? []) {
      if (!isDict(msg)) continue;
      const transformed = this.transformMessage(msg);
      if (transformed) {
        if (Array.isArray(transformed)) messages.push(...transformed);
        else messages.push(transformed);
      }
    }

    const result: Dict = {
      messages,
      max_tokens: body["max_tokens"] ?? 4096,
    };
    if ("model" in body) result["model"] = body["model"];
    if ("temperature" in body) result["temperature"] = body["temperature"];
    if ("top_p" in body) result["top_p"] = body["top_p"];
    if ("top_k" in body) result["top_k"] = body["top_k"];
    if ("stop_sequences" in body) result["stop"] = body["stop_sequences"];
    if (body["stream"]) result["stream"] = true;

    const thinkingCfg = body["thinking"];
    if (
      isDict(thinkingCfg) &&
      (thinkingCfg["type"] === "enabled" || thinkingCfg["type"] === "adaptive")
    ) {
      const kwargs = { ...((result["chat_template_kwargs"] as Dict | undefined) ?? {}) };
      kwargs["enable_thinking"] = true;
      result["chat_template_kwargs"] = kwargs;
    }

    if ("tools" in body) {
      const tools = this.transformToolsToOpenai(body["tools"] as unknown[]);
      if (tools.length > 0) {
        result["tools"] = tools;
        result["tool_choice"] = this.transformToolChoiceToOpenai(
          body["tool_choice"] ?? { type: "auto" },
        );
      }
    }

    return this.normalizeRequest(result, body["_at_model_served"] as string | undefined);
  }

  override transformResponse(response: Dict, originalRequest: Dict): Dict {
    const choices = (response["choices"] as Dict[] | undefined) ?? [];
    if (choices.length === 0) return AnthropicTransformer.errorResponse("No choices in response");

    const choice = choices[0]!;
    const message = (choice["message"] as Dict | undefined) ?? {};

    const content: Dict[] = [];
    const reasoning = message["reasoning_content"];
    if (typeof reasoning === "string" && reasoning) {
      content.push({ type: "thinking", thinking: reasoning, signature: makeSignature(reasoning) });
    }

    const text = message["content"];
    if (text || (Array.isArray(text) && text.length > 0)) {
      content.push(...openaiChatContentToAnthropicBlocks(text));
    }

    for (const toolCall of (message["tool_calls"] as Dict[] | undefined) ?? []) {
      const func = (toolCall["function"] as Dict | undefined) ?? {};
      content.push({
        type: "tool_use",
        id: (toolCall["id"] as string | undefined) ?? `toolu_${hex().slice(0, 24)}`,
        name: (func["name"] as string | undefined) ?? "",
        input: AnthropicTransformer.parseJsonSafe((func["arguments"] as string | undefined) ?? "{}"),
      });
    }

    const finishReason = (choice["finish_reason"] as string | undefined) ?? "stop";
    const stopReason = FINISH_TO_STOP_REASON[finishReason] ?? "end_turn";
    const usage = (response["usage"] as Dict | undefined) ?? {};

    if (content.length === 0) content.push({ type: "text", text: "" });

    return {
      id: `msg_${(response["id"] as string | undefined) ?? hex()}`,
      type: "message",
      role: "assistant",
      content,
      model: (originalRequest["model"] as string | undefined) ?? "claude-3",
      stop_reason: stopReason,
      stop_sequence: null,
      usage: AnthropicTransformer.usageToAnthropic(usage),
    };
  }

  override createStreamState(originalRequest: Dict): AnthropicStreamState {
    return new AnthropicStreamState(
      (originalRequest["model"] as string | undefined) ?? "claude-3",
      FINISH_TO_STOP_REASON,
    );
  }

  override transformStreamChunk(chunk: Dict, originalRequest: Dict, isFirst: boolean): Dict[] {
    const state = this.createStreamState(originalRequest);
    const events = state.processChunk(chunk, isFirst);
    const choices = (chunk["choices"] as Dict[] | undefined) ?? [];
    if (choices.length > 0 && choices[0]!["finish_reason"]) events.push(...state.finalize());
    return events;
  }

  private transformMessage(msg: Dict): Dict | Dict[] | null {
    const role = (msg["role"] as string | undefined) ?? "user";
    const content = msg["content"] ?? "";

    if (typeof content === "string") return { role, content };
    if (!Array.isArray(content)) return { role, content: String(content) };

    const toolResults = content.filter((c) => isDict(c) && c["type"] === "tool_result") as Dict[];
    const toolUses = content.filter((c) => isDict(c) && c["type"] === "tool_use") as Dict[];
    const textBlocks = content.filter((c) => isDict(c) && c["type"] === "text") as Dict[];

    const messages: Dict[] = [];

    let reasoningText = "";
    if (role === "assistant") reasoningText = extractReasoningFromAnthropicContent(content);

    if (role === "assistant" && toolUses.length > 0) {
      const toolCalls: Dict[] = [];
      const textParts: string[] = [];
      for (const block of content) {
        if (!isDict(block)) continue;
        if (block["type"] === "text") {
          textParts.push((block["text"] as string | undefined) ?? "");
        } else if (block["type"] === "tool_use") {
          toolCalls.push({
            id: (block["id"] as string | undefined) ?? `call_${hex().slice(0, 24)}`,
            type: "function",
            function: {
              name: (block["name"] as string | undefined) ?? "",
              arguments: JSON.stringify(block["input"] ?? {}),
            },
          });
        }
      }
      const msgDict: Dict = {
        role: "assistant",
        content: textParts.length > 0 ? textParts.join("\n") : null,
      };
      if (reasoningText) msgDict["reasoning_content"] = reasoningText;
      if (toolCalls.length > 0) msgDict["tool_calls"] = toolCalls;
      return msgDict;
    }

    if (role === "user" && toolResults.length > 0) {
      for (const tr of toolResults) {
        const toolContent = tr["content"] ?? "";
        const convertedContent = anthropicContentToOpenaiChat(toolContent);
        let textContent = this.flattenContent(convertedContent);
        if (tr["is_error"]) {
          textContent = textContent ? `[Tool Error] ${textContent}` : "[Tool Error]";
        }
        messages.push({
          role: "tool",
          tool_call_id: (tr["tool_use_id"] as string | undefined) ?? "",
          content: textContent,
        });
        const imageParts = AnthropicTransformer.imageParts(convertedContent);
        if (imageParts.length > 0) messages.push({ role: "user", content: imageParts });
      }

      const textParts = textBlocks
        .map((b) => (b["text"] as string | undefined) ?? "")
        .filter((t) => t);
      if (textParts.length > 0) messages.push({ role: "user", content: textParts.join("\n") });
      return messages.length > 0 ? messages : null;
    }

    const result: Dict = { role, content: anthropicContentToOpenaiChat(content) };
    if (role === "assistant" && reasoningText) result["reasoning_content"] = reasoningText;
    return result;
  }

  private flattenContent(content: unknown): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      const parts: string[] = [];
      for (const block of content) {
        if (typeof block === "string") parts.push(block);
        else if (isDict(block)) {
          if (block["type"] === "text") parts.push((block["text"] as string | undefined) ?? "");
          else if (block["type"] === "tool_result")
            parts.push(this.flattenContent(block["content"] ?? ""));
        }
      }
      return parts.join("\n");
    }
    return content ? String(content) : "";
  }

  private static imageParts(content: unknown): Dict[] {
    if (!Array.isArray(content)) return [];
    return content.filter((part) => isDict(part) && part["type"] === "image_url") as Dict[];
  }

  private transformToolsToOpenai(tools: unknown[]): Dict[] {
    const result: Dict[] = [];
    for (const tool of tools) {
      if (!isDict(tool)) continue;
      const toolType = tool["type"];
      if (
        toolType &&
        toolType !== "custom" &&
        toolType !== "function" &&
        !("input_schema" in tool)
      ) {
        continue;
      }
      const name = tool["name"];
      if (typeof name !== "string" || !name) continue;
      result.push({
        type: "function",
        function: {
          name,
          description: (tool["description"] as string | undefined) ?? "",
          parameters: tool["input_schema"] ?? {},
        },
      });
    }
    return result;
  }

  private transformToolChoiceToOpenai(toolChoice: unknown): unknown {
    if (isDict(toolChoice)) {
      const tcType = toolChoice["type"];
      if (tcType === "auto") return "auto";
      if (tcType === "any") return "required";
      if (tcType === "none") return "none";
      if (tcType === "tool") {
        return { type: "function", function: { name: (toolChoice["name"] as string) ?? "" } };
      }
    }
    return "auto";
  }

  private static parseJsonSafe(s: string): Dict {
    try {
      const parsed = JSON.parse(s);
      return isDict(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }

  private static usageToAnthropic(usage: Dict): Dict {
    const promptTokens = (usage["prompt_tokens"] as number | undefined) ?? 0;
    const completionTokens = (usage["completion_tokens"] as number | undefined) ?? 0;
    const cacheRead = AnthropicTransformer.cachedPromptTokens(usage);
    const inputTokens = cacheRead ? Math.max(promptTokens - cacheRead, 0) : promptTokens;

    const result: Dict = { input_tokens: inputTokens, output_tokens: completionTokens };
    if (cacheRead) result["cache_read_input_tokens"] = cacheRead;
    const cacheCreation = usage["cache_creation_input_tokens"];
    if (typeof cacheCreation === "number" && cacheCreation) {
      result["cache_creation_input_tokens"] = cacheCreation;
    }
    return result;
  }

  private static cachedPromptTokens(usage: Dict): number {
    const details = usage["prompt_tokens_details"];
    if (isDict(details)) {
      const cached = details["cached_tokens"];
      if (typeof cached === "number") return cached;
    }
    const cached = usage["cached_tokens"];
    return typeof cached === "number" ? cached : 0;
  }

  private static errorResponse(message: string): Dict {
    return { type: "error", error: { type: "api_error", message } };
  }
}
