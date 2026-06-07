import { describe, expect, it } from "vitest";

import { PerRequestBuilder } from "../src/trajectory/builders/perRequest.js";
import { parseCompletionSession } from "../src/trajectory/models.js";

describe("PerRequestBuilder", () => {
  it("returns ERROR for an empty session", async () => {
    const session = parseCompletionSession({
      session_id: "session-1",
      metadata: { group_id: "g1", policy_version: 7 },
    });

    const trajectory = await new PerRequestBuilder().build(session);

    expect(trajectory.status).toBe("ERROR");
    expect(trajectory.error).toBe("no completions");
    expect(trajectory.metadata["builder"]).toBe("per_request");
    expect(trajectory.metadata["group_id"]).toBe("g1");
    expect(trajectory.metadata["policy_version"]).toBe(7);
    expect(trajectory.traces).toEqual([]);
  });

  it("emits one trace per completion", async () => {
    const session = parseCompletionSession({
      session_id: "session-1",
      task_id: "task-1",
      model_requested: "requested",
      model_used: "served",
      api_type: "openai_chat",
      metadata: { rollout_step: 3 },
      completions: [
        {
          completion_id: "completion-1",
          request: {
            messages: [{ role: "user", content: "Say hi" }],
            tools: [{ type: "function", function: { name: "lookup" } }],
          },
          response: {
            choices: [
              {
                input_token_ids: [1, 2],
                token_ids: [3, 4],
                message: { role: "assistant", content: "Hi" },
                finish_reason: "stop",
                logprobs: {
                  content: [
                    { token_id: 3, logprob: -0.1 },
                    { token_id: 4, logprob: -0.2 },
                  ],
                },
              },
            ],
          },
          metadata: { completion_metadata: true },
        },
      ],
    });

    const trajectory = await new PerRequestBuilder().build(session);

    expect(trajectory.status).toBe("COMPLETED");
    expect(trajectory.metadata["record_count"]).toBe(1);
    expect(trajectory.metadata["trace_count"]).toBe(1);
    expect(trajectory.metadata["rollout_step"]).toBe(3);
    const trace = trajectory.traces[0]!;
    expect(trace.prompt_ids).toEqual([1, 2]);
    expect(trace.response_ids).toEqual([3, 4]);
    expect(trace.loss_mask).toEqual([1, 1]);
    expect(trace.prompt_messages).toEqual([{ role: "user", content: "Say hi" }]);
    expect(trace.response_messages).toEqual([{ role: "assistant", content: "Hi" }]);
    expect(trace.tools).toEqual([{ type: "function", function: { name: "lookup" } }]);
    expect(trace.response_logprobs).toEqual([-0.1, -0.2]);
  });
});
