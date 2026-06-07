/**
 * Auto-capture install: a transparent shell wrapper so that simply running
 * `claude` (or any env-configured runtime) interactively goes through the
 * capture proxy and persists to a default directory — no per-invocation
 * `agent-trace exec` typing, and the user's canonical ANTHROPIC_BASE_URL is left
 * untouched (the wrapper relies on `exec`, which only rewires the child env).
 *
 * The wrapper resolves the *real* binary via `type -P`/`whence -p` (skipping
 * this very function), and degrades to running the real binary directly if
 * `agent-trace` isn't on PATH — so it can never break the wrapped command.
 *
 * All functions here are pure string transforms (no fs), so they're trivially
 * testable; the CLI does the actual file read/write.
 */

export type ShellKind = "bash" | "zsh";

export interface WrapperOptions {
  /** Command to wrap, e.g. "claude". */
  command: string;
  /** Default save directory (shell expression; `$HOME` expands at runtime). */
  saveDir: string;
  /** Target shell (controls how the real binary is resolved). */
  shell: ShellKind;
  /** Name of the agent-trace CLI on PATH (default "agent-trace"). */
  cliName?: string;
}

export interface BlockMarkers {
  begin: string;
  end: string;
}

/** Marker comments delimiting one command's wrapper block (idempotent upserts). */
export function blockMarkers(command: string): BlockMarkers {
  return {
    begin: `# >>> agent-trace auto-capture (${command}) >>>`,
    end: `# <<< agent-trace auto-capture (${command}) <<<`,
  };
}

/** How to resolve the real PATH binary, skipping shell functions/aliases. */
function realBinaryLookup(shell: ShellKind, command: string): string {
  return shell === "zsh" ? `whence -p ${command}` : `type -P ${command}`;
}

/** Render the full, marker-delimited wrapper block for a command. */
export function renderWrapperBlock(opts: WrapperOptions): string {
  const cli = opts.cliName ?? "agent-trace";
  const { begin, end } = blockMarkers(opts.command);
  const lookup = realBinaryLookup(opts.shell, opts.command);
  // Note: $(...) and $HOME are intentionally left unexpanded — they evaluate at
  // runtime so the wrapper follows PATH and per-user $HOME.
  return [
    begin,
    "# Added by `agent-trace install`; remove with `agent-trace uninstall`.",
    `${opts.command}() {`,
    `  local __ccp_bin __ccp_cli`,
    `  __ccp_bin="$(${lookup})"`,
    `  __ccp_cli="$(command -v ${cli})"`,
    `  if [ -z "$__ccp_bin" ]; then`,
    `    echo "agent-trace: real '${opts.command}' not found in PATH" >&2; return 127`,
    `  fi`,
    `  if [ -z "$__ccp_cli" ]; then "$__ccp_bin" "$@"; return; fi`,
    `  "$__ccp_cli" exec --save-dir "${opts.saveDir}" \\`,
    `    --session-id "${opts.command}-$(date +%Y%m%d-%H%M%S)-$$-$RANDOM" -- "$__ccp_bin" "$@"`,
    `}`,
    end,
  ].join("\n");
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Remove an existing block (by markers) from rc content, trimming blank gaps. */
export function removeBlock(rc: string, markers: BlockMarkers): string {
  const pattern = new RegExp(
    `\\n*${escapeRegExp(markers.begin)}[\\s\\S]*?${escapeRegExp(markers.end)}\\n*`,
    "g",
  );
  const out = rc.replace(pattern, "\n");
  return out;
}

/**
 * Insert or replace a command's wrapper block (idempotent). Returns the new rc
 * content and whether this was a fresh add vs. a replacement.
 */
export function upsertBlock(
  rc: string,
  block: string,
  markers: BlockMarkers,
): { content: string; replaced: boolean } {
  const had = rc.includes(markers.begin);
  const base = had ? removeBlock(rc, markers).replace(/\s*$/, "") : rc.replace(/\s*$/, "");
  const prefix = base.length > 0 ? `${base}\n\n` : "";
  return { content: `${prefix}${block}\n`, replaced: had };
}
