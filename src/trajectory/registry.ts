/**
 * Strategy registry for trajectory builders.
 *
 * Unlike the Python version, there is no `"module:ClassName"` dynamic-import
 * escape hatch — in JS you simply `register(name, factory)` with any factory,
 * or construct the builder yourself. A fresh builder is created per request
 * from the spec config.
 */

import type { StrategySpec } from "./models.js";
import type { BuilderFactory, TrajectoryBuilder } from "./builders/base.js";
import { PerRequestBuilder } from "./builders/perRequest.js";
import { PrefixMergingBuilder } from "./builders/prefixMerging.js";

export class BuilderRegistry {
  private readonly factories = new Map<string, BuilderFactory>();

  /** Register a builder factory under a stable name. */
  register(name: string, factory: BuilderFactory): void {
    if (typeof factory !== "function") {
      throw new TypeError(`factory must be a function, got ${typeof factory}`);
    }
    this.factories.set(name, factory);
  }

  /** Instantiate a fresh builder from a spec (config passed to the factory). */
  create(spec: StrategySpec): TrajectoryBuilder {
    const factory = this.factories.get(spec.strategy);
    if (factory === undefined) {
      throw new Error(`Unknown strategy: ${JSON.stringify(spec.strategy)}`);
    }
    return factory(spec.config);
  }

  list(): string[] {
    return [...this.factories.keys()].sort();
  }
}

/** Pre-populated registry with the built-in builders. */
export function defaultBuilderRegistry(): BuilderRegistry {
  const registry = new BuilderRegistry();
  registry.register("per_request", () => new PerRequestBuilder());
  registry.register(
    "prefix_merging",
    (config) =>
      new PrefixMergingBuilder({
        end_of_turn_token_id: (config?.["end_of_turn_token_id"] as number | null | undefined) ?? null,
      }),
  );
  return registry;
}
