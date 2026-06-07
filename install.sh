#!/usr/bin/env bash
#
# agent-trace one-line installer.
#
#   curl -fsSL https://raw.githubusercontent.com/Seauagain/agent-trace/main/install.sh | bash
#
# Options (pass after `bash -s --`):
#   --command <name>   wrap this command (repeatable; default: claude)
#   --all              wrap the common set (claude + codex)
#   --save-dir <dir>   default capture dir (default: ~/.agent-trace/captures)
#   --shell bash|zsh   target shell rc (default: from $SHELL)
#   --ref <git-ref>    branch/tag to install (default: main)
#
# Env overrides: AGENT_TRACE_REPO, AGENT_TRACE_HOME
set -euo pipefail

REPO_URL="${AGENT_TRACE_REPO:-https://github.com/Seauagain/agent-trace.git}"
HOME_DIR="${AGENT_TRACE_HOME:-$HOME/.agent-trace}"
APP_DIR="$HOME_DIR/app"
REF="main"
ALL=0
COMMANDS=()
PASS=()   # passthrough flags for `agent-trace install`

log() { printf '\033[1;36m[agent-trace]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[agent-trace]\033[0m %s\n' "$*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --command)    COMMANDS+=("$2"); shift 2;;
    --command=*)  COMMANDS+=("${1#*=}"); shift;;
    --all)        ALL=1; shift;;
    --ref)        REF="$2"; shift 2;;
    --ref=*)      REF="${1#*=}"; shift;;
    --save-dir)   PASS+=(--save-dir "$2"); shift 2;;
    --save-dir=*) PASS+=(--save-dir "${1#*=}"); shift;;
    --shell)      PASS+=(--shell "$2"); shift 2;;
    --shell=*)    PASS+=(--shell "${1#*=}"); shift;;
    -h|--help)    sed -n '2,16p' "$0" 2>/dev/null || true; exit 0;;
    *) err "unknown argument: $1"; exit 2;;
  esac
done

[ "$ALL" = "1" ] && COMMANDS+=(claude codex)
[ "${#COMMANDS[@]}" -eq 0 ] && COMMANDS=(claude)

for bin in git node npm; do
  command -v "$bin" >/dev/null 2>&1 || { err "missing '$bin' — please install it first."; exit 1; }
done

mkdir -p "$HOME_DIR"
if [ -d "$APP_DIR/.git" ]; then
  log "updating source ($REF) in $APP_DIR"
  git -C "$APP_DIR" remote set-url origin "$REPO_URL"
  git -C "$APP_DIR" fetch -q --depth 1 origin "$REF"
  git -C "$APP_DIR" reset -q --hard FETCH_HEAD
else
  log "cloning $REPO_URL ($REF) -> $APP_DIR"
  rm -rf "$APP_DIR"
  git clone -q --depth 1 --branch "$REF" "$REPO_URL" "$APP_DIR"
fi

log "installing dependencies + building"
( cd "$APP_DIR" && npm ci --silent && npm run build --silent )

# Link the CLI onto a PATH dir (prefer one already on PATH).
choose_bindir() {
  local d
  for d in "$HOME/.local/bin" "$HOME/bin"; do
    case ":$PATH:" in *":$d:"*) printf '%s' "$d"; return;; esac
  done
  printf '%s' "$HOME/.local/bin"
}
BIN_DIR="$(choose_bindir)"
mkdir -p "$BIN_DIR"
chmod +x "$APP_DIR/dist/cli.js"
ln -sf "$APP_DIR/dist/cli.js" "$BIN_DIR/agent-trace"
log "linked CLI: $BIN_DIR/agent-trace -> $APP_DIR/dist/cli.js"

case ":$PATH:" in
  *":$BIN_DIR:"*) :;;
  *) err "note: $BIN_DIR is not on PATH — add this to your shell rc:"
     err "      export PATH=\"$BIN_DIR:\$PATH\"";;
esac

for cmd in "${COMMANDS[@]}"; do
  log "wiring auto-capture wrapper for: $cmd"
  "$BIN_DIR/agent-trace" install --command "$cmd" ${PASS[@]+"${PASS[@]}"}
done

log "done — open a new shell (or 'source' your rc). Then just run: ${COMMANDS[*]}"
