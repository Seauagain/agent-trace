/**
 * Auto-capture shell wrapper: the generated block routes through `agent-trace
 * exec` (non-destructive), resolves the real binary (no recursion), degrades
 * gracefully, and upserts idempotently with per-command markers.
 */

import { describe, expect, it } from "vitest";

import {
  blockMarkers,
  removeBlock,
  renderWrapperBlock,
  upsertBlock,
} from "../src/install/shellWrapper.js";

describe("renderWrapperBlock", () => {
  const block = renderWrapperBlock({
    command: "claude",
    saveDir: "$HOME/.agent-trace/captures",
    shell: "bash",
  });

  it("wraps the command via `agent-trace exec`, not by overwriting base URLs", () => {
    expect(block).toContain("claude() {");
    expect(block).toContain('"$__ccp_cli" exec --save-dir "$HOME/.agent-trace/captures"');
    expect(block).toContain('-- "$__ccp_bin" "$@"');
    // It must NOT export/clobber the canonical base URL anywhere.
    expect(block).not.toContain("ANTHROPIC_BASE_URL=");
    expect(block).not.toContain("export ANTHROPIC_BASE_URL");
  });

  it("resolves the real binary with `type -P` (bash) to avoid recursing the function", () => {
    expect(block).toContain('__ccp_bin="$(type -P claude)"');
  });

  it("degrades to the real binary when agent-trace is absent", () => {
    expect(block).toContain('if [ -z "$__ccp_cli" ]; then "$__ccp_bin" "$@"; return; fi');
  });

  it("uses `whence -p` for zsh", () => {
    const z = renderWrapperBlock({ command: "claude", saveDir: "~/c", shell: "zsh" });
    expect(z).toContain('__ccp_bin="$(whence -p claude)"');
  });
});

describe("upsertBlock / removeBlock", () => {
  const markers = blockMarkers("claude");
  const block = renderWrapperBlock({
    command: "claude",
    saveDir: "$HOME/.agent-trace/captures",
    shell: "bash",
  });

  it("adds the block to existing rc content (fresh)", () => {
    const rc = "export PATH=$PATH:/usr/local/bin\n";
    const { content, replaced } = upsertBlock(rc, block, markers);
    expect(replaced).toBe(false);
    expect(content).toContain("export PATH=");
    expect(content).toContain(markers.begin);
    expect(content).toContain(markers.end);
  });

  it("is idempotent: re-installing replaces the block, not duplicates it", () => {
    const rc = "alias x=y\n";
    const once = upsertBlock(rc, block, markers).content;
    const twice = upsertBlock(once, block, markers);
    expect(twice.replaced).toBe(true);
    const beginCount = twice.content.split(markers.begin).length - 1;
    expect(beginCount).toBe(1);
  });

  it("removeBlock strips the block and preserves surrounding content", () => {
    const rc = `line-before\n\n${block}\n\nline-after\n`;
    const out = removeBlock(rc, markers);
    expect(out).not.toContain(markers.begin);
    expect(out).not.toContain("claude() {");
    expect(out).toContain("line-before");
    expect(out).toContain("line-after");
  });

  it("keeps multiple commands' blocks independent", () => {
    const codexBlock = renderWrapperBlock({ command: "codex", saveDir: "~/c", shell: "bash" });
    let rc = upsertBlock("", block, markers).content;
    rc = upsertBlock(rc, codexBlock, blockMarkers("codex")).content;
    // Removing claude leaves codex intact.
    const afterRemove = removeBlock(rc, markers);
    expect(afterRemove).not.toContain("claude() {");
    expect(afterRemove).toContain("codex() {");
  });
});
