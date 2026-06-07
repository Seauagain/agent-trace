import { describe, expect, it } from "vitest";

import { parseCompletionSession, createTrajectory } from "../src/trajectory/models.js";
import type { CompletionSession, Trajectory } from "../src/trajectory/models.js";
import type { TrajectoryBuilder } from "../src/trajectory/builders/base.js";
import { BuilderRegistry, defaultBuilderRegistry } from "../src/trajectory/registry.js";
import { PerRequestBuilder } from "../src/trajectory/builders/perRequest.js";

class DummyBuilder implements TrajectoryBuilder {
  constructor(private readonly status: "COMPLETED" | "TIMEOUT" | "ERROR" = "COMPLETED") {}
  async build(session: CompletionSession): Promise<Trajectory> {
    return createTrajectory({ status: this.status, metadata: { session_id: session.session_id } });
  }
}

describe("BuilderRegistry", () => {
  it("creates fresh instances with config", async () => {
    const registry = new BuilderRegistry();
    registry.register("dummy", (config) => new DummyBuilder(config?.["status"] as never));

    const builder = registry.create({ strategy: "dummy", config: { status: "TIMEOUT" } });
    const trajectory = await builder.build(parseCompletionSession({ session_id: "session-1" }));

    expect(trajectory.status).toBe("TIMEOUT");
    expect(trajectory.metadata["session_id"]).toBe("session-1");
  });

  it("rejects non-function factories", () => {
    const registry = new BuilderRegistry();
    expect(() => registry.register("bad", {} as never)).toThrow(/function/);
  });

  it("exposes the built-in builders", () => {
    const registry = defaultBuilderRegistry();
    expect(registry.list()).toEqual(["per_request", "prefix_merging"]);
    expect(registry.create({ strategy: "per_request", config: {} })).toBeInstanceOf(
      PerRequestBuilder,
    );
  });

  it("raises a clear error for unknown strategies", () => {
    const registry = new BuilderRegistry();
    expect(() => registry.create({ strategy: "missing", config: {} })).toThrow(/Unknown strategy/);
  });
});
