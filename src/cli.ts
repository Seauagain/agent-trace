#!/usr/bin/env node
/**
 * agent-trace CLI.
 *
 *   agent-trace capture [--port 8787] [--anthropic-base-url URL] [--openai-base-url URL]
 *                      [--upstream-base-url URL] [--session-id ID] [--save-dir DIR]
 *   agent-trace serve   --inference-base-url URL --model NAME [--engine vllm] [--port 8080] [--save-dir DIR]
 *   agent-trace build   <records-dir> [--builder prefix_merging] [--format rl|sft|trajectory] [--out FILE]
 *
 * `capture` is the universal, no-backend plugin path: a transparent proxy that
 * forwards each request verbatim to the API the agent already uses (Anthropic /
 * OpenAI, public or a relay) and captures the full trace on the wire. `serve`
 * is the self-hosted-model path that also captures token ids + logprobs for
 * token-level RL. `build` reconstructs trajectories from persisted records.
 */

import { spawn } from "node:child_process";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

import { CaptureProxy } from "./capture/server.js";
import { ForwardProxy } from "./capture/forward/mitmProxy.js";
import { MitmCA } from "./capture/forward/ca.js";
import {
  childEnvOverrides,
  needsLocalProxy,
  planTraceRoutes,
  upstreamsFromRoutes,
} from "./capture/execEnv.js";
import {
  blockMarkers,
  removeBlock,
  renderWrapperBlock,
  type ShellKind,
  upsertBlock,
} from "./install/shellWrapper.js";
import { defaultBuilderRegistry } from "./trajectory/registry.js";
import { parseCompletionSession } from "./trajectory/models.js";
import { trajectoryToRLSamples } from "./serialize/toRLSample.js";
import { trajectoryToSFTSamples } from "./serialize/toSFTSample.js";
import { writeJsonl, toJsonl } from "./serialize/writeJsonl.js";

type Flags = Record<string, string | boolean>;

function parseArgs(argv: string[]): { positionals: string[]; flags: Flags } {
  const positionals: string[] = [];
  const flags: Flags = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith("--")) {
        flags[key] = true;
      } else {
        flags[key] = next;
        i++;
      }
    } else {
      positionals.push(arg);
    }
  }
  return { positionals, flags };
}

function flagStr(flags: Flags, key: string, fallback?: string): string | undefined {
  const v = flags[key];
  return typeof v === "string" ? v : fallback;
}

function isTruthyEnv(v: string | undefined): boolean {
  return v !== undefined && v !== "" && v !== "0" && v.toLowerCase() !== "false";
}

/**
 * Whether `exec` may write to the terminal. Off by default so the wrapper never
 * intrudes on an agent's TUI (Claude Code etc.); opt in with --verbose or
 * AGENT_TRACE_VERBOSE=1. When quiet, even the capture proxy's per-request logs
 * are suppressed (they'd otherwise corrupt the TUI on stdout).
 */
function execVerbose(flags: Flags): boolean {
  return flags["verbose"] === true || isTruthyEnv(process.env["AGENT_TRACE_VERBOSE"]);
}

const SILENT_LOGGER: Pick<Console, "info" | "warn" | "error"> = {
  info: () => {},
  warn: () => {},
  error: () => {},
};

/** Logger for exec-spawned proxies: silent unless verbose, and always to stderr. */
function execLogger(verbose: boolean): Pick<Console, "info" | "warn" | "error"> {
  if (!verbose) return SILENT_LOGGER;
  const toErr = (...args: unknown[]): void => console.error(...args);
  return { info: toErr, warn: toErr, error: toErr };
}

async function serve(flags: Flags): Promise<void> {
  const inferenceBaseUrl = flagStr(flags, "inference-base-url") ?? process.env["INFERENCE_BASE_URL"];
  const modelServed = flagStr(flags, "model") ?? process.env["MODEL_SERVED"];
  if (!inferenceBaseUrl || !modelServed) {
    console.error(
      "serve requires --inference-base-url and --model (or INFERENCE_BASE_URL / MODEL_SERVED env).",
    );
    process.exit(1);
  }
  const engine = (flagStr(flags, "engine") ?? "vllm") as "vllm" | "sglang";
  const port = Number(flagStr(flags, "port") ?? process.env["PORT"] ?? "8080");
  const host = flagStr(flags, "host") ?? "127.0.0.1";
  const saveDir = flagStr(flags, "save-dir") ?? null;

  const proxy = new CaptureProxy({
    inferenceBaseUrl,
    modelServed,
    engine,
    saveDir,
    defaultBuilder: flagStr(flags, "builder") ?? "prefix_merging",
  });
  const bound = await proxy.listen(port, host);
  console.info(`agent-trace proxy listening on http://${host}:${bound}`);
  console.info(`  -> forwarding to ${inferenceBaseUrl} (engine=${engine}, model=${modelServed})`);
  console.info(`  point your agent's OPENAI_BASE_URL at http://${host}:${bound}/v1`);
  if (saveDir) console.info(`  persisting completions under ${saveDir}`);

  const shutdown = () => {
    void proxy.close().then(() => process.exit(0));
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

async function capture(flags: Flags): Promise<void> {
  const port = Number(flagStr(flags, "port") ?? process.env["PORT"] ?? "8787");
  const host = flagStr(flags, "host") ?? "127.0.0.1";
  const saveDir = flagStr(flags, "save-dir") ?? null;
  const sessionId = flagStr(flags, "session-id") ?? "capture";

  const upstreamDefault =
    flagStr(flags, "upstream-base-url") ?? process.env["AGENT_TRACE_UPSTREAM_BASE_URL"];
  // Fall back to the canonical provider base URLs so the proxy "just knows"
  // your real backend (e.g. cc-dp's ANTHROPIC_BASE_URL) without retyping it.
  const anthropic =
    flagStr(flags, "anthropic-base-url") ??
    process.env["ANTHROPIC_UPSTREAM_BASE_URL"] ??
    process.env["ANTHROPIC_BASE_URL"];
  const openai =
    flagStr(flags, "openai-base-url") ??
    process.env["OPENAI_UPSTREAM_BASE_URL"] ??
    process.env["OPENAI_BASE_URL"];

  const proxy = new CaptureProxy({
    mode: "passthrough",
    upstreams: {
      ...(anthropic ? { anthropic } : {}),
      ...(openai ? { openai } : {}),
      ...(upstreamDefault ? { default: upstreamDefault } : {}),
    },
    defaultSessionId: sessionId,
    saveDir,
    defaultBuilder: flagStr(flags, "builder") ?? "prefix_merging",
  });
  const bound = await proxy.listen(port, host);
  const anthropicTarget = anthropic ?? upstreamDefault ?? "https://api.anthropic.com";
  const openaiTarget = openai ?? upstreamDefault ?? "https://api.openai.com";

  console.info(`agent-trace capture proxy listening on http://${host}:${bound}`);
  console.info(`  Anthropic (/v1/messages)        -> ${anthropicTarget}`);
  console.info(`  OpenAI    (/v1/chat/completions) -> ${openaiTarget}`);
  console.info(`  Claude Code: ANTHROPIC_BASE_URL=http://${host}:${bound}`);
  console.info(`  OpenAI SDK : OPENAI_BASE_URL=http://${host}:${bound}/v1`);
  if (saveDir) console.info(`  persisting completions under ${saveDir}`);
  else console.info(`  (no --save-dir: captures kept in memory; finalize via /sessions/<id>/finalize)`);
  console.info(`  session id for this run: ${sessionId}`);

  const shutdown = () => {
    void proxy.close().then(() => process.exit(0));
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

/**
 * `exec` runs an agent command with its provider base URL transparently routed
 * through a capture proxy, WITHOUT mutating the parent shell's canonical
 * ANTHROPIC_BASE_URL / OPENAI_BASE_URL (only the child's env is overridden).
 */
async function exec(childArgv: string[], flags: Flags): Promise<void> {
  if (childArgv.length === 0) {
    console.error("exec requires a command after `--`, e.g. agent-trace exec -- claude");
    process.exit(1);
  }
  // MITM mode captures clients that ignore *_BASE_URL (e.g. cursor-agent) by
  // routing HTTPS_PROXY through a TLS-intercepting proxy + a trusted local CA.
  if (flags["mitm"] === true) return execMitm(childArgv, flags);
  // Default port 0 = ephemeral, so concurrent interactive sessions never collide.
  const requestedPort = Number(flagStr(flags, "port") ?? process.env["AGENT_TRACE_PORT"] ?? "0");
  const host = flagStr(flags, "host") ?? "127.0.0.1";
  const saveDir = flagStr(flags, "save-dir") ?? process.env["AGENT_TRACE_SAVE_DIR"] ?? null;
  const sessionId = flagStr(flags, "session-id") ?? "capture";
  const verbose = execVerbose(flags);

  // Externals (and thus whether we must spawn a proxy) don't depend on the port.
  const probe = planTraceRoutes(process.env, `http://${host}:0`);
  let proxy: CaptureProxy | null = null;
  let routes = probe;
  if (needsLocalProxy(probe)) {
    proxy = new CaptureProxy({
      mode: "passthrough",
      upstreams: upstreamsFromRoutes(probe),
      defaultSessionId: sessionId,
      saveDir,
      defaultBuilder: flagStr(flags, "builder") ?? "prefix_merging",
      logger: execLogger(verbose),
    });
    let bound: number;
    try {
      bound = await proxy.listen(requestedPort, host);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`agent-trace exec: could not start trace proxy: ${msg}`);
      console.error(`  (use --port to pick a free port, or set *_BASE_URL_TRACE_PROXY)`);
      process.exit(1);
    }
    routes = planTraceRoutes(process.env, `http://${host}:${bound}`);
  }
  const localProxyBaseUrl = routes.find((r) => !r.external)?.childBaseUrl ?? `http://${host}:0`;

  // Quiet by default: the wrapper must not scribble over the agent's TUI.
  if (verbose) {
    console.error(`agent-trace exec: tracing \`${childArgv.join(" ")}\``);
    for (const r of routes) {
      if (r.external) console.error(`  ${r.baseVar} -> ${r.childBaseUrl}  (external trace proxy)`);
      else
        console.error(
          `  ${r.baseVar} -> ${r.childBaseUrl}  (proxy -> ${r.upstream ?? "public default"})`,
        );
    }
    if (proxy && saveDir) console.error(`  saving completions under ${saveDir}`);
    else if (proxy)
      console.error(
        `  (no --save-dir: in-memory; GET ${localProxyBaseUrl}/sessions/${sessionId}/completions)`,
      );
  }

  const overrides = childEnvOverrides(routes);
  const child = spawn(childArgv[0]!, childArgv.slice(1), {
    stdio: "inherit",
    env: { ...process.env, ...overrides },
  });

  const relay = (sig: NodeJS.Signals) => () => {
    if (!child.killed) child.kill(sig);
  };
  const onInt = relay("SIGINT");
  const onTerm = relay("SIGTERM");
  process.on("SIGINT", onInt);
  process.on("SIGTERM", onTerm);

  const code: number = await new Promise((resolve) => {
    child.on("exit", (c, signal) => resolve(signal ? 1 : (c ?? 0)));
    child.on("error", (err) => {
      console.error(`agent-trace exec: failed to start \`${childArgv[0]}\`: ${err.message}`);
      resolve(127);
    });
  });

  process.off("SIGINT", onInt);
  process.off("SIGTERM", onTerm);
  if (proxy) await proxy.close();
  process.exit(code);
}

function mitmHostsFromFlags(flags: Flags): string[] | undefined {
  const raw = flagStr(flags, "mitm-host");
  if (!raw) return undefined;
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Run an agent command behind the TLS-intercepting forward proxy. */
async function execMitm(childArgv: string[], flags: Flags): Promise<void> {
  const requestedPort = Number(flagStr(flags, "port") ?? process.env["AGENT_TRACE_PORT"] ?? "0");
  const host = flagStr(flags, "host") ?? "127.0.0.1";
  const saveDir = flagStr(flags, "save-dir") ?? process.env["AGENT_TRACE_SAVE_DIR"] ?? null;
  const sessionId = flagStr(flags, "session-id") ?? "forward";
  const verbose = execVerbose(flags);

  const proxy = new ForwardProxy({
    saveDir,
    defaultSessionId: sessionId,
    mitmHosts: mitmHostsFromFlags(flags),
    mitmAll: flags["mitm-all"] === true,
    caDir: flagStr(flags, "ca-dir"),
    logger: execLogger(verbose),
  });
  let bound: number;
  try {
    bound = await proxy.listen(requestedPort, host);
  } catch (err) {
    console.error(`agent-trace exec --mitm: could not start forward proxy: ${String(err)}`);
    process.exit(1);
  }
  const proxyUrl = `http://${host}:${bound}`;

  // Quiet by default so the agent's TUI stays clean.
  if (verbose) {
    console.error(`agent-trace exec --mitm: tracing \`${childArgv.join(" ")}\` via TLS interception`);
    console.error(`  HTTPS_PROXY=${proxyUrl}  NODE_EXTRA_CA_CERTS=${proxy.caCertPath}`);
    console.error(
      `  intercepting: ${flags["mitm-all"] === true ? "all hosts" : (mitmHostsFromFlags(flags) ?? ["cursor.sh", "anthropic.com", "openai.com"]).join(", ")}`,
    );
    if (saveDir) console.error(`  saving completions under ${saveDir}`);
    else
      console.error(
        `  (no --save-dir: in-memory; GET ${proxyUrl}/sessions/${sessionId}/completions is not exposed in forward mode)`,
      );
  }

  const overrides: Record<string, string> = {
    HTTP_PROXY: proxyUrl,
    HTTPS_PROXY: proxyUrl,
    http_proxy: proxyUrl,
    https_proxy: proxyUrl,
    ALL_PROXY: proxyUrl,
    NODE_EXTRA_CA_CERTS: proxy.caCertPath,
  };
  const child = spawn(childArgv[0]!, childArgv.slice(1), {
    stdio: "inherit",
    env: { ...process.env, ...overrides },
  });

  const relay = (sig: NodeJS.Signals) => () => {
    if (!child.killed) child.kill(sig);
  };
  const onInt = relay("SIGINT");
  const onTerm = relay("SIGTERM");
  process.on("SIGINT", onInt);
  process.on("SIGTERM", onTerm);

  const code: number = await new Promise((resolve) => {
    child.on("exit", (c, signal) => resolve(signal ? 1 : (c ?? 0)));
    child.on("error", (err) => {
      console.error(`agent-trace exec --mitm: failed to start \`${childArgv[0]}\`: ${err.message}`);
      resolve(127);
    });
  });

  process.off("SIGINT", onInt);
  process.off("SIGTERM", onTerm);
  await proxy.close();
  process.exit(code);
}

/** Run the TLS-intercepting forward proxy standalone (set HTTPS_PROXY yourself). */
async function forward(flags: Flags): Promise<void> {
  const port = Number(flagStr(flags, "port") ?? process.env["PORT"] ?? "8788");
  const host = flagStr(flags, "host") ?? "127.0.0.1";
  const saveDir = flagStr(flags, "save-dir") ?? null;
  const sessionId = flagStr(flags, "session-id") ?? "forward";

  const proxy = new ForwardProxy({
    saveDir,
    defaultSessionId: sessionId,
    mitmHosts: mitmHostsFromFlags(flags),
    mitmAll: flags["mitm-all"] === true,
    caDir: flagStr(flags, "ca-dir"),
  });
  let bound: number;
  try {
    bound = await proxy.listen(port, host);
  } catch (err) {
    console.error(`agent-trace forward: could not start forward proxy: ${String(err)}`);
    process.exit(1);
  }
  const proxyUrl = `http://${host}:${bound}`;
  console.info(`agent-trace forward proxy (TLS MITM) listening on ${proxyUrl}`);
  console.info(`  point your agent at it, in its own shell:`);
  console.info(`    export HTTPS_PROXY=${proxyUrl}`);
  console.info(`    export NODE_EXTRA_CA_CERTS=${proxy.caCertPath}`);
  console.info(
    `  intercepting: ${flags["mitm-all"] === true ? "all hosts" : (mitmHostsFromFlags(flags) ?? ["cursor.sh", "anthropic.com", "openai.com"]).join(", ")}`,
  );
  if (saveDir) console.info(`  persisting completions under ${saveDir}`);
  else console.info(`  (no --save-dir: captures kept in memory only)`);
  console.info(`  session id for this run: ${sessionId}`);

  const shutdown = (): void => {
    void proxy.close().then(() => process.exit(0));
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

/** Ensure the local root CA exists and print its path (for trust setup). */
function ca(flags: Flags): void {
  const mitmCa = MitmCA.loadOrCreate(flagStr(flags, "ca-dir"));
  if (flags["print"] === true) {
    process.stdout.write(`${mitmCa.caCertPem}\n`);
    return;
  }
  console.info(`agent-trace root CA: ${mitmCa.certPath}`);
  console.info(`  Node clients:  export NODE_EXTRA_CA_CERTS=${mitmCa.certPath}`);
  console.info(`  print the PEM: agent-trace ca --print`);
}

function detectShell(flags: Flags): ShellKind {
  const explicit = flagStr(flags, "shell");
  if (explicit === "zsh" || explicit === "bash") return explicit;
  return (process.env["SHELL"] ?? "").toLowerCase().includes("zsh") ? "zsh" : "bash";
}

function defaultRcPath(shell: ShellKind): string {
  return join(homedir(), shell === "zsh" ? ".zshrc" : ".bashrc");
}

/**
 * Install a transparent shell wrapper so interactive `claude` (etc.) auto-routes
 * through the capture proxy and persists to a default dir — without touching the
 * user's canonical ANTHROPIC_BASE_URL.
 */
async function install(flags: Flags): Promise<void> {
  const shell = detectShell(flags);
  const rcPath = flagStr(flags, "rc") ?? defaultRcPath(shell);
  const command = flagStr(flags, "command") ?? "claude";
  const saveDir = flagStr(flags, "save-dir") ?? "$HOME/.agent-trace/captures";
  const cliName = flagStr(flags, "cli-name") ?? "agent-trace";
  const block = renderWrapperBlock({ command, saveDir, shell, cliName });

  if (flags["print"] === true) {
    process.stdout.write(`${block}\n`);
    return;
  }

  let rc = "";
  try {
    rc = await readFile(rcPath, "utf-8");
  } catch {
    rc = "";
  }
  const markers = blockMarkers(command);
  const { content, replaced } = upsertBlock(rc, block, markers);
  await writeFile(rcPath, content, "utf-8");

  console.info(
    `agent-trace: ${replaced ? "updated" : "installed"} '${command}' auto-capture wrapper in ${rcPath}`,
  );
  console.info(`  default save dir: ${saveDir}`);
  console.info(`  activate now:  source ${rcPath}   (or open a new shell)`);
  console.info(
    `  uninstall:     agent-trace uninstall${command === "claude" ? "" : ` --command ${command}`}`,
  );
}

/** Remove a previously installed wrapper block. */
async function uninstall(flags: Flags): Promise<void> {
  const shell = detectShell(flags);
  const rcPath = flagStr(flags, "rc") ?? defaultRcPath(shell);
  const command = flagStr(flags, "command") ?? "claude";

  let rc: string;
  try {
    rc = await readFile(rcPath, "utf-8");
  } catch {
    console.error(`agent-trace: ${rcPath} not found.`);
    process.exit(1);
  }
  const markers = blockMarkers(command);
  if (!rc.includes(markers.begin)) {
    console.info(`agent-trace: no '${command}' wrapper found in ${rcPath}.`);
    return;
  }
  const content = `${removeBlock(rc, markers).replace(/\s*$/, "")}\n`;
  await writeFile(rcPath, content, "utf-8");
  console.info(`agent-trace: removed '${command}' auto-capture wrapper from ${rcPath}.`);
}

interface PersistedRecord {
  completion_id?: string;
  timestamp?: string | null;
  session_id?: string;
  task_id?: string | null;
  api_type?: string | null;
  model_requested?: string | null;
  model_used?: string | null;
  original_request?: Record<string, unknown>;
  transformed_request?: Record<string, unknown>;
  request?: Record<string, unknown>;
  response?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

async function findJsonFiles(dir: string): Promise<string[]> {
  const out: string[] = [];
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await findJsonFiles(full)));
    else if (entry.isFile() && entry.name.endsWith(".json")) out.push(full);
  }
  return out;
}

async function build(positionals: string[], flags: Flags): Promise<void> {
  const dir = positionals[0];
  if (!dir) {
    console.error("build requires a <records-dir> positional argument.");
    process.exit(1);
  }
  const builderName = flagStr(flags, "builder") ?? "prefix_merging";
  const format = (flagStr(flags, "format") ?? "rl").toLowerCase();
  const includeTokens = flags["include-tokens"] === true;
  const eot = flagStr(flags, "end-of-turn-token-id");
  const out = flagStr(flags, "out");

  const files = await findJsonFiles(dir);
  // Group persisted completion records by session id.
  const bySession = new Map<string, { taskId: string | null; records: PersistedRecord[] }>();
  for (const file of files) {
    let rec: PersistedRecord;
    try {
      rec = JSON.parse(await readFile(file, "utf-8")) as PersistedRecord;
    } catch {
      continue;
    }
    if (!rec.completion_id || !(rec.response || rec.transformed_request || rec.request)) continue;
    const sid = rec.session_id ?? "unknown";
    const group = bySession.get(sid) ?? { taskId: rec.task_id ?? null, records: [] };
    group.records.push(rec);
    bySession.set(sid, group);
  }

  if (bySession.size === 0) {
    console.error(`No completion records found under ${dir}.`);
    process.exit(1);
  }

  const registry = defaultBuilderRegistry();
  const allSamples: unknown[] = [];
  for (const [sessionId, group] of bySession) {
    const session = parseCompletionSession({
      session_id: sessionId,
      task_id: group.taskId,
      completions: group.records.map((rec) => ({
        completion_id: rec.completion_id,
        timestamp: rec.timestamp ?? null,
        request: rec.transformed_request ?? rec.request ?? {},
        original_request: rec.original_request ?? {},
        response: rec.response ?? {},
        metadata: rec.metadata ?? {},
      })),
    });
    const builder = registry.create({
      strategy: builderName,
      config: eot !== undefined ? { end_of_turn_token_id: Number(eot) } : {},
    });
    const trajectory = await builder.build(session);
    if (format === "sft") allSamples.push(...trajectoryToSFTSamples(trajectory, { includeTokens }));
    else if (format === "trajectory") allSamples.push(trajectory);
    else allSamples.push(...trajectoryToRLSamples(trajectory));
  }

  if (out) {
    await writeJsonl(out, allSamples);
    console.info(
      `Wrote ${allSamples.length} ${format} sample(s) from ${bySession.size} session(s) to ${out}`,
    );
  } else {
    process.stdout.write(toJsonl(allSamples));
  }
}

async function main(): Promise<void> {
  const [, , command, ...rest] = process.argv;

  // `exec` is special: everything after the first standalone `--` is the agent
  // command (with its own flags), forwarded verbatim. Our flags come before it.
  if (command === "exec") {
    const sep = rest.indexOf("--");
    const ourArgs = sep === -1 ? rest : rest.slice(0, sep);
    const childArgv = sep === -1 ? [] : rest.slice(sep + 1);
    const { flags } = parseArgs(ourArgs);
    await exec(childArgv, flags);
    return;
  }

  const { positionals, flags } = parseArgs(rest);

  switch (command) {
    case "install":
      await install(flags);
      break;
    case "uninstall":
      await uninstall(flags);
      break;
    case "capture":
      await capture(flags);
      break;
    case "forward":
      await forward(flags);
      break;
    case "ca":
      ca(flags);
      break;
    case "serve":
      await serve(flags);
      break;
    case "build":
      await build(positionals, flags);
      break;
    default:
      console.error(
        [
          "agent-trace — capture agent trajectories for SFT/RL",
          "",
          "Usage:",
          "  agent-trace install   [--command claude] [--save-dir DIR] [--shell bash|zsh] [--rc PATH] [--print]",
          "  agent-trace uninstall [--command claude] [--shell bash|zsh] [--rc PATH]",
          "  agent-trace exec      [--mitm] [--verbose] [--port N] [--save-dir DIR] [--session-id ID] -- <agent command...>",
          "  agent-trace capture   [--port 8787] [--anthropic-base-url URL] [--openai-base-url URL]",
          "                       [--upstream-base-url URL] [--session-id ID] [--save-dir DIR]",
          "  agent-trace forward   [--port 8788] [--mitm-host h1,h2] [--mitm-all] [--save-dir DIR] [--session-id ID]",
          "  agent-trace ca        [--print] [--ca-dir DIR]",
          "  agent-trace serve     --inference-base-url URL --model NAME [--engine vllm] [--port 8080] [--save-dir DIR]",
          "  agent-trace build     <records-dir> [--builder prefix_merging] [--format rl|sft|trajectory] [--out FILE] [--include-tokens]",
          "",
          "`install` adds a transparent shell wrapper so just running `claude`",
          "interactively auto-captures to a default dir (~/.agent-trace/captures) —",
          "no per-run typing, and your canonical ANTHROPIC_BASE_URL is left untouched.",
          "`exec` is silent by default (never touches the agent's TUI); pass --verbose",
          "or set AGENT_TRACE_VERBOSE=1 to see routing + per-request capture logs.",
          "Under the hood it uses `exec`, which routes only the child's *_BASE_URL",
          "through a capture proxy and forwards to your real backend. `capture` runs",
          "the proxy standalone; `serve` is the self-hosted path that also captures",
          "token ids + logprobs for token-level RL.",
          "",
          "`forward` (and `exec --mitm`) is the TLS-intercepting path for clients that",
          "ignore *_BASE_URL or speak a non-JSON wire (e.g. cursor-agent's protobuf):",
          "the agent uses HTTPS_PROXY + trusts the local CA (`agent-trace ca`), and",
          "recognized turns are decoded to OpenAI-chat and saved like any capture.",
        ].join("\n"),
      );
      process.exit(command ? 1 : 0);
  }
}

void main();
