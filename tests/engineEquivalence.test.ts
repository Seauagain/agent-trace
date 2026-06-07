/**
 * Both backends must produce the same trajectory from the same generation.
 * Engine vs. trajectory-builder equivalence.
 * (the slime-adapter logprob test is omitted — slime_bridge is out of scope).
 */

import { describe, expect, it } from "vitest";

import { VLLMEngine } from "../src/capture/engine.js";
import { PerRequestBuilder } from "../src/trajectory/builders/perRequest.js";
import { PrefixMergingBuilder } from "../src/trajectory/builders/prefixMerging.js";
import { buildTraceFromCompletion } from "../src/trajectory/recordUtils.js";
import {
  type CompletionRecord,
  CompletionRecordSchema,
  parseCompletionSession,
  type Trace,
} from "../src/trajectory/models.js";

const EOT = 99; // synthetic end-of-turn token id

interface RecordArgs {
  promptIds: number[];
  responseIds: number[];
  logprobs: number[];
  content: string;
  reasoning: string | null;
  finishReason: string;
  promptMessages: Record<string, unknown>[];
  responseMessage: Record<string, unknown>;
}

function sglangRecord(completionId: string, a: RecordArgs): CompletionRecord {
  const message: Record<string, unknown> = {
    role: "assistant",
    content: a.content,
    ...a.responseMessage,
  };
  if (a.reasoning !== null) message["reasoning_content"] = a.reasoning;
  return CompletionRecordSchema.parse({
    completion_id: completionId,
    request: { messages: a.promptMessages },
    response: {
      choices: [
        {
          input_token_ids: [...a.promptIds],
          message,
          finish_reason: a.finishReason,
          logprobs: {
            content: a.responseIds.map((tid, i) => ({
              token: `t${tid}`,
              token_id: tid,
              logprob: a.logprobs[i],
              bytes: [],
            })),
          },
        },
      ],
    },
  });
}

function vllmRecord(completionId: string, a: RecordArgs): CompletionRecord {
  const message: Record<string, unknown> = {
    role: "assistant",
    content: a.content,
    ...a.responseMessage,
  };
  if (a.reasoning !== null) message["reasoning"] = a.reasoning; // vLLM names it `reasoning`
  let response: Record<string, unknown> = {
    prompt_token_ids: [...a.promptIds],
    choices: [
      {
        token_ids: [...a.responseIds],
        message,
        finish_reason: a.finishReason,
        logprobs: {
          content: a.responseIds.map((tid, i) => ({
            token: `t${tid}`,
            logprob: a.logprobs[i],
            bytes: [],
          })),
        },
      },
    ],
  };
  response = new VLLMEngine().normalizeResponse(response);
  return CompletionRecordSchema.parse({
    completion_id: completionId,
    request: { messages: a.promptMessages },
    response,
  });
}

function assertTracesEqual(x: Trace, y: Trace): void {
  expect(x.prompt_ids).toEqual(y.prompt_ids);
  expect(x.response_ids).toEqual(y.response_ids);
  expect(x.loss_mask).toEqual(y.loss_mask);
  expect(x.finish_reason).toEqual(y.finish_reason);
  expect(x.response_messages).toEqual(y.response_messages);
  expect(x.response_logprobs).toEqual(y.response_logprobs);
}

describe("engine trajectory equivalence", () => {
  it("single-turn trace is identical across engines", () => {
    const common: RecordArgs = {
      promptIds: [1, 2, 3],
      responseIds: [10, 11, 12, 13],
      logprobs: [-0.1, -0.2, -0.3, -0.4],
      content: "4",
      reasoning: "thinking",
      finishReason: "stop",
      promptMessages: [{ role: "user", content: "2+2?" }],
      responseMessage: {},
    };
    const sg = buildTraceFromCompletion(sglangRecord("c1", common));
    const vllm = buildTraceFromCompletion(vllmRecord("c1", common));

    assertTracesEqual(sg, vllm);
    expect(vllm.prompt_ids).toEqual([1, 2, 3]);
    expect(vllm.response_ids).toEqual([10, 11, 12, 13]);
    expect(vllm.loss_mask).toEqual([1, 1, 1, 1]);
    expect(vllm.response_messages[0]!["reasoning_content"]).toBe("thinking");
    expect(vllm.response_logprobs).toEqual([-0.1, -0.2, -0.3, -0.4]);
  });

  it("per_request builder is identical across engines", async () => {
    const common: RecordArgs = {
      promptIds: [1, 2, 3],
      responseIds: [10, 11, 12, 13],
      logprobs: [-0.1, -0.2, -0.3, -0.4],
      content: "4",
      reasoning: null,
      finishReason: "stop",
      promptMessages: [{ role: "user", content: "2+2?" }],
      responseMessage: {},
    };
    const sgTraj = await new PerRequestBuilder().build(
      parseCompletionSession({ session_id: "s", completions: [sglangRecord("c1", common)] }),
    );
    const vllmTraj = await new PerRequestBuilder().build(
      parseCompletionSession({ session_id: "s", completions: [vllmRecord("c1", common)] }),
    );
    assertTracesEqual(sgTraj.traces[0]!, vllmTraj.traces[0]!);
  });

  it("prefix_merging chain is identical across engines", async () => {
    const twoTurnChain = (recordFn: (id: string, a: RecordArgs) => CompletionRecord) => {
      const q1 = { role: "user", content: "Q1" };
      const a1 = { role: "assistant", content: "A1" };
      const tool = { role: "tool", content: "result" };
      const c1 = recordFn("c1", {
        promptIds: [1, 2, 3],
        responseIds: [10, 11, EOT],
        logprobs: [-0.1, -0.2, -0.3],
        content: "A1",
        reasoning: null,
        finishReason: "stop",
        promptMessages: [q1],
        responseMessage: {},
      });
      // canonical_tail = [10, 11, EOT, 50, 51] -> interstitial after EOT = [50, 51]
      const c2 = recordFn("c2", {
        promptIds: [1, 2, 3, 10, 11, EOT, 50, 51],
        responseIds: [20, 21, EOT],
        logprobs: [-0.5, -0.6, -0.7],
        content: "A2",
        reasoning: null,
        finishReason: "stop",
        promptMessages: [q1, a1, tool],
        responseMessage: {},
      });
      return [c1, c2];
    };

    const builder = new PrefixMergingBuilder({ end_of_turn_token_id: EOT });
    const sg = await builder.build(
      parseCompletionSession({ session_id: "s", completions: twoTurnChain(sglangRecord) }),
    );
    const vllm = await builder.build(
      parseCompletionSession({ session_id: "s", completions: twoTurnChain(vllmRecord) }),
    );

    expect(sg.traces.length).toBe(1);
    expect(vllm.traces.length).toBe(1);
    assertTracesEqual(sg.traces[0]!, vllm.traces[0]!);
    expect(vllm.traces[0]!.response_ids).toEqual([10, 11, EOT, 50, 51, 20, 21, EOT]);
    expect(vllm.traces[0]!.loss_mask).toEqual([1, 1, 1, 0, 0, 1, 1, 1]);
  });
});
