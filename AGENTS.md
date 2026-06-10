# AGENTS.md

Guidance for AI agents working on `agent-trace` — a runtime-agnostic capture
proxy that records LLM traces and serializes them into SFT/RL-trainable records.

## Commands

- Build: `npm run build` (tsup → `dist/`)
- Typecheck: `npm run typecheck` (`tsc --noEmit`)
- Test: `npm test` (vitest, **not** jest) · watch: `npm run test:watch`
- Run CLI from source: `npm run dev -- <args>` (`tsx src/cli.ts`)
- Always run `npm test` and `npm run typecheck` before declaring a change done.

## Project conventions

- **Pure ESM** (`"type": "module"`). No `require`/CJS. Use `import`/`export`,
  and keep `.js` extensions in relative import specifiers where present.
- **Strict TypeScript**: `verbatimModuleSyntax` is on — use `import type` for
  type-only imports. `noUncheckedIndexedAccess` is on — guard array/object index
  access. Target Node 20+ (relies on global `fetch`).
- Validation uses **zod**; prefer schema-driven parsing over hand-rolled checks.

## Architecture (where changes belong)

- `src/capture/` — the proxy: HTTP server, passthrough/MITM forwarding, and
  on-wire protocol decoders (`forward/decoders/`, incl. Cursor Connect-RPC).
- `src/trajectory/` — stitches captured completions into trajectories
  (builders: `per_request`, `prefix_merging`) + registry/models.
- `src/serialize/` — turns trajectories into RL / SFT JSONL samples.
- `src/install/` — shell-wrapper install logic. `src/cli.ts` — CLI entry.
- `tests/` — vitest regression tests, one per decoder/proxy/builder. Add tests
  here when touching capture or serialization logic.

## Invariants — do NOT break these

- **Forward requests verbatim.** The proxy must relay the agent's request
  unchanged and pass the real API key through to the upstream untouched.
- **Never modify the user's canonical `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`.**
  Routing is applied only to the child process env (via `exec`), or via the
  opt-in `*_BASE_URL_TRACE_PROXY` vars.
- **Capture is lossless** — read from the wire (tee the stream); never
  reconstruct traces from lossy local logs.
- Preserve the three modes' semantics: `capture` (passthrough, message-level),
  `forward` (TLS MITM, SFT-grade), `serve` (inference, token-level RL).

## Docs

- Keep `README.md` in sync when changing CLI flags, modes, or output formats.
