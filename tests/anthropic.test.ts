import { describe, expect, it } from "vitest";

import { AnthropicTransformer } from "../src/capture/transform/anthropic.js";

type Dict = Record<string, unknown>;

const t = new AnthropicTransformer();

describe("AnthropicTransformer.transformRequest", () => {
  it("maps system + user content and tools to OpenAI chat", () => {
    const out = t.transformRequest({
      model: "claude-3-5-sonnet",
      system: "You are helpful",
      max_tokens: 1024,
      messages: [{ role: "user", content: "hi" }],
      tools: [
        { name: "lookup", description: "look up", input_schema: { type: "object" } },
        { type: "web_search_20250101", name: "ws" }, // server tool, dropped
      ],
    });
    const messages = out["messages"] as Dict[];
    expect(messages[0]).toEqual({ role: "system", content: "You are helpful" });
    expect(messages[1]).toEqual({ role: "user", content: "hi" });
    expect(out["max_tokens"]).toBe(1024);
    const tools = out["tools"] as Dict[];
    expect(tools).toHaveLength(1);
    expect((tools[0]!["function"] as Dict)["name"]).toBe("lookup");
    expect(out["tool_choice"]).toBe("auto");
  });

  it("converts assistant tool_use and user tool_result blocks", () => {
    const out = t.transformRequest({
      model: "claude-3",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "text", text: "let me check" },
            { type: "tool_use", id: "toolu_1", name: "lookup", input: { q: "x" } },
          ],
        },
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "42" }],
        },
      ],
    });
    const messages = out["messages"] as Dict[];
    const assistant = messages[0]!;
    expect(assistant["role"]).toBe("assistant");
    const toolCalls = assistant["tool_calls"] as Dict[];
    expect((toolCalls[0]!["function"] as Dict)["arguments"]).toBe(JSON.stringify({ q: "x" }));
    const toolMsg = messages[1]!;
    expect(toolMsg).toEqual({ role: "tool", tool_call_id: "toolu_1", content: "42" });
  });

  it("strips the Claude Code billing header from the system prompt", () => {
    const out = t.transformRequest({
      model: "claude-3",
      system: "x-anthropic-billing-header: cch=abc123;\nReal instructions",
      messages: [{ role: "user", content: "hi" }],
    });
    const messages = out["messages"] as Dict[];
    expect(messages[0]).toEqual({ role: "system", content: "Real instructions" });
  });
});

describe("AnthropicTransformer.transformResponse", () => {
  it("renders an OpenAI chat response as an Anthropic message with tool_use", () => {
    const out = t.transformResponse(
      {
        id: "abc",
        choices: [
          {
            message: {
              role: "assistant",
              content: "done",
              tool_calls: [
                { id: "call_1", function: { name: "lookup", arguments: '{"q":"x"}' } },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      },
      { model: "claude-3-5-sonnet" },
    );
    expect(out["type"]).toBe("message");
    expect(out["model"]).toBe("claude-3-5-sonnet");
    expect(out["stop_reason"]).toBe("tool_use");
    const content = out["content"] as Dict[];
    expect(content[0]).toEqual({ type: "text", text: "done" });
    expect(content[1]).toEqual({
      type: "tool_use",
      id: "call_1",
      name: "lookup",
      input: { q: "x" },
    });
    expect(out["usage"]).toEqual({ input_tokens: 10, output_tokens: 5 });
  });
});

describe("AnthropicStreamState", () => {
  it("emits a well-formed SSE block sequence for a single synthetic chunk", () => {
    const state = t.createStreamState({ model: "claude-3" });
    const chunk = {
      choices: [
        { delta: { role: "assistant", content: "hello" }, finish_reason: "stop" },
      ],
      usage: { completion_tokens: 1 },
    };
    const events = [...state.processChunk(chunk, true), ...state.finalize()];
    const types = events.map((e) => e["type"]);
    expect(types[0]).toBe("message_start");
    expect(types).toContain("content_block_start");
    expect(types).toContain("content_block_delta");
    expect(types).toContain("content_block_stop");
    expect(types).toContain("message_delta");
    expect(types[types.length - 1]).toBe("message_stop");
  });
});
