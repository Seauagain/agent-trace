/**
 * Non-destructive trace routing for `agent-trace exec -- <agent command>`.
 *
 * The agent only ever reads the *canonical* provider base URL
 * (`ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`), so capturing it requires the agent
 * to talk to the proxy. Rather than make the user overwrite their real
 * `ANTHROPIC_BASE_URL` (destructive, leaks into every other tool, easy to forget
 * to undo), `exec` rewires only the *child* process's environment and leaves the
 * parent shell untouched. Tracing is opt-in via a dedicated variable:
 *
 *   - `ANTHROPIC_BASE_URL_TRACE_PROXY` / `OPENAI_BASE_URL_TRACE_PROXY`
 *       Address of an already-running trace proxy. When set, the child's
 *       canonical base URL is pointed here and no proxy is spawned.
 *   - Otherwise `exec` starts an in-process capture proxy, forwards it to the
 *       *real* upstream (`*_UPSTREAM_BASE_URL` else the canonical `*_BASE_URL`),
 *       and points the child at the spawned proxy.
 *
 * Either way the canonical `ANTHROPIC_BASE_URL` in the user's shell keeps its
 * original meaning ("the real backend"); the proxy hop lives in its own var.
 */

export type Env = Record<string, string | undefined>;

export interface ProviderRoute {
  provider: "anthropic" | "openai";
  /** Canonical env var to override in the child (e.g. ANTHROPIC_BASE_URL). */
  baseVar: string;
  /** Value the child's baseVar is set to (the trace proxy it should call). */
  childBaseUrl: string;
  /**
   * Real upstream the *spawned* proxy must forward to. `undefined` => let the
   * proxy use its public default. Omitted entirely when `external` is true.
   */
  upstream?: string;
  /** True when a `*_BASE_URL_TRACE_PROXY` was supplied (use it, don't spawn). */
  external: boolean;
}

interface ProviderSpec {
  provider: "anthropic" | "openai";
  baseVar: string;
  proxyVar: string;
  upstreamVar: string;
  /** Suffix appended to the spawned-proxy base for this provider's SDK. */
  childSuffix: string;
}

const PROVIDERS: ProviderSpec[] = [
  {
    provider: "anthropic",
    baseVar: "ANTHROPIC_BASE_URL",
    proxyVar: "ANTHROPIC_BASE_URL_TRACE_PROXY",
    upstreamVar: "ANTHROPIC_UPSTREAM_BASE_URL",
    childSuffix: "",
  },
  {
    provider: "openai",
    baseVar: "OPENAI_BASE_URL",
    proxyVar: "OPENAI_BASE_URL_TRACE_PROXY",
    upstreamVar: "OPENAI_UPSTREAM_BASE_URL",
    childSuffix: "/v1",
  },
];

function clean(v: string | undefined): string | undefined {
  const t = v?.trim();
  return t ? t : undefined;
}

function stripTrailingSlashes(url: string): string {
  return url.replace(/\/+$/, "");
}

/**
 * Resolve, per provider, where the child should send requests and (for the
 * spawned-proxy path) which real upstream the proxy forwards to.
 *
 * @param env              parent environment snapshot (never mutated)
 * @param localProxyBaseUrl base URL of the proxy `exec` would spawn (e.g. http://127.0.0.1:8787)
 */
export function planTraceRoutes(env: Env, localProxyBaseUrl: string): ProviderRoute[] {
  const local = stripTrailingSlashes(localProxyBaseUrl);
  return PROVIDERS.map((spec) => {
    const external = clean(env[spec.proxyVar]);
    if (external) {
      return {
        provider: spec.provider,
        baseVar: spec.baseVar,
        childBaseUrl: stripTrailingSlashes(external),
        external: true,
      };
    }
    return {
      provider: spec.provider,
      baseVar: spec.baseVar,
      childBaseUrl: local + spec.childSuffix,
      upstream: clean(env[spec.upstreamVar]) ?? clean(env[spec.baseVar]),
      external: false,
    };
  });
}

/** Does any provider require us to spawn a local proxy (i.e. not external)? */
export function needsLocalProxy(routes: ProviderRoute[]): boolean {
  return routes.some((r) => !r.external);
}

/** Upstreams config (for the spawned CaptureProxy) derived from the routes. */
export function upstreamsFromRoutes(
  routes: ProviderRoute[],
): { anthropic?: string; openai?: string } {
  const out: { anthropic?: string; openai?: string } = {};
  for (const r of routes) {
    if (r.external || r.upstream === undefined) continue;
    out[r.provider] = r.upstream;
  }
  return out;
}

/** Child-process env overrides (canonical base vars pointed at the proxy). */
export function childEnvOverrides(routes: ProviderRoute[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const r of routes) out[r.baseVar] = r.childBaseUrl;
  return out;
}
