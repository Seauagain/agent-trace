import { describe, expect, it } from "vitest";

import { createTrace, createTrajectory } from "../src/trajectory/models.js";
import { toRLSample, trajectoryToRLSamples } from "../src/serialize/toRLSample.js";
import { IGNORE_INDEX, toSFTSample } from "../src/serialize/toSFTSample.js";
import { toJsonl } from "../src/serialize/writeJsonl.js";

const trace = createTrace({
  prompt_ids: [1, 2, 3],
  response_ids: [10, 11, 99, 50, 20, 99],
  loss_mask: [1, 1, 1, 0, 1, 1],
  prompt_messages: [{ role: "user", content: "Q" }],
  response_messages: [{ role: "assistant", content: "A" }],
  tools: [{ type: "function", function: { name: "lookup" } }],
  finish_reason: "stop",
  response_logprobs: [-0.1, -0.2, -0.3, 0.0, -0.5, -0.6],
  reward: 1.0,
});

describe("toRLSample", () => {
  it("flattens prompt+response with prompt_len split and reward", () => {
    const sample = toRLSample(trace);
    expect(sample.token_ids).toEqual([1, 2, 3, 10, 11, 99, 50, 20, 99]);
    expect(sample.prompt_len).toBe(3);
    expect(sample.response_len).toBe(6);
    expect(sample.loss_mask).toEqual([1, 1, 1, 0, 1, 1]);
    expect(sample.logprobs).toEqual([-0.1, -0.2, -0.3, 0.0, -0.5, -0.6]);
    expect(sample.reward).toBe(1.0);
    expect(sample.finish_reason).toBe("stop");
  });

  it("stamps trajectory status onto samples", () => {
    const traj = createTrajectory({ status: "COMPLETED", traces: [trace] });
    const samples = trajectoryToRLSamples(traj);
    expect(samples).toHaveLength(1);
    expect(samples[0]!.metadata["trajectory_status"]).toBe("COMPLETED");
  });
});

describe("toSFTSample", () => {
  it("emits chat messages and tools", () => {
    const sample = toSFTSample(trace);
    expect(sample.messages).toEqual([
      { role: "user", content: "Q" },
      { role: "assistant", content: "A" },
    ]);
    expect(sample.tools).toEqual([{ type: "function", function: { name: "lookup" } }]);
    expect(sample.input_ids).toBeUndefined();
  });

  it("emits token-level labels with -100 on prompt and masked positions", () => {
    const sample = toSFTSample(trace, { includeTokens: true });
    expect(sample.input_ids).toEqual([1, 2, 3, 10, 11, 99, 50, 20, 99]);
    // prompt (3x -100), then response with position index 3 (token 50) masked.
    expect(sample.labels).toEqual([
      IGNORE_INDEX,
      IGNORE_INDEX,
      IGNORE_INDEX,
      10,
      11,
      99,
      IGNORE_INDEX,
      20,
      99,
    ]);
  });
});

describe("toJsonl", () => {
  it("renders one compact object per line with trailing newline", () => {
    const out = toJsonl([{ a: 1 }, { b: 2 }]);
    expect(out).toBe('{"a":1}\n{"b":2}\n');
    expect(toJsonl([])).toBe("");
  });
});
