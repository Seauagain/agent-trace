/**
 * `exec` trace-routing resolution: the child's canonical base URLs are pointed
 * at the proxy, the parent shell vars are never read as the override source,
 * and tracing is opt-in per provider via *_BASE_URL_TRACE_PROXY.
 */

import { describe, expect, it } from "vitest";

import {
  childEnvOverrides,
  needsLocalProxy,
  planTraceRoutes,
  upstreamsFromRoutes,
  type Env,
} from "../src/capture/execEnv.js";

const PROXY = "http://127.0.0.1:8787";

describe("planTraceRoutes", () => {
  it("spawns a local proxy and forwards to the real ANTHROPIC_BASE_URL", () => {
    const env: Env = { ANTHROPIC_BASE_URL: "https://api.gpugeek.com" };
    const routes = planTraceRoutes(env, PROXY);

    const ant = routes.find((r) => r.provider === "anthropic")!;
    expect(ant.external).toBe(false);
    expect(ant.childBaseUrl).toBe(PROXY); // no /v1 suffix for Anthropic
    expect(ant.upstream).toBe("https://api.gpugeek.com");

    // The child is pointed at the proxy; the real upstream is what we forward to.
    expect(childEnvOverrides(routes)["ANTHROPIC_BASE_URL"]).toBe(PROXY);
    expect(upstreamsFromRoutes(routes).anthropic).toBe("https://api.gpugeek.com");
    expect(needsLocalProxy(routes)).toBe(true);
  });

  it("appends /v1 for OpenAI and prefers *_UPSTREAM_BASE_URL over the canonical var", () => {
    const env: Env = {
      OPENAI_BASE_URL: "http://127.0.0.1:8787/v1", // e.g. a stale value; must not win
      OPENAI_UPSTREAM_BASE_URL: "https://api.openai.com",
    };
    const routes = planTraceRoutes(env, PROXY);
    const oai = routes.find((r) => r.provider === "openai")!;
    expect(oai.childBaseUrl).toBe("http://127.0.0.1:8787/v1");
    expect(oai.upstream).toBe("https://api.openai.com");
  });

  it("uses an external trace proxy when *_BASE_URL_TRACE_PROXY is set (no spawn, no upstream)", () => {
    const env: Env = {
      ANTHROPIC_BASE_URL: "https://api.gpugeek.com",
      ANTHROPIC_BASE_URL_TRACE_PROXY: "https://trace.example.com/",
    };
    const routes = planTraceRoutes(env, PROXY);
    const ant = routes.find((r) => r.provider === "anthropic")!;
    expect(ant.external).toBe(true);
    expect(ant.childBaseUrl).toBe("https://trace.example.com"); // trailing slash stripped
    expect(ant.upstream).toBeUndefined();
    expect(upstreamsFromRoutes(routes).anthropic).toBeUndefined();
  });

  it("only spawns when at least one provider is non-external", () => {
    const env: Env = {
      ANTHROPIC_BASE_URL_TRACE_PROXY: "https://t1.example.com",
      OPENAI_BASE_URL_TRACE_PROXY: "https://t2.example.com",
    };
    const routes = planTraceRoutes(env, PROXY);
    expect(needsLocalProxy(routes)).toBe(false);
    expect(childEnvOverrides(routes)).toEqual({
      ANTHROPIC_BASE_URL: "https://t1.example.com",
      OPENAI_BASE_URL: "https://t2.example.com",
    });
  });

  it("falls back to public default (undefined upstream) when no base url is set", () => {
    const routes = planTraceRoutes({}, PROXY);
    const ant = routes.find((r) => r.provider === "anthropic")!;
    expect(ant.external).toBe(false);
    expect(ant.upstream).toBeUndefined(); // proxy uses its public default
    expect(needsLocalProxy(routes)).toBe(true);
  });
});
