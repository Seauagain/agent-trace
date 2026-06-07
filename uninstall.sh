#!/usr/bin/env bash
#
# agent-trace uninstaller — removes shell wrappers, the CLI symlink, and
# (optionally) the app + captured data.
#
#   curl -fsSL https://raw.githubusercontent.com/Seauagain/agent-trace/main/uninstall.sh | bash
#
# Options:
#   --command <name>   remove this command's wrapper (repeatable; default: claude)
#   --all              remove the common set (claude + codex)
#   --shell bash|zsh   target shell rc (default: from $SHELL)
#   --purge            also delete ~/.agent-trace (app + captured trajectories)
set -euo pipefail

HOME_DIR="${AGENT_TRACE_HOME:-$HOME/.agent-trace}"
ALL=0
PURGE=0
COMMANDS=()
PASS=()

log() { printf '\033[1;36m[agent-trace]\033[0m %s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --command)   COMMANDS+=("$2"); shift 2;;
    --command=*) COMMANDS+=("${1#*=}"); shift;;
    --all)       ALL=1; shift;;
    --shell)     PASS+=(--shell "$2"); shift 2;;
    --shell=*)   PASS+=(--shell "${1#*=}"); shift;;
    --purge)     PURGE=1; shift;;
    -h|--help)   sed -n '2,14p' "$0" 2>/dev/null || true; exit 0;;
    *) echo "agent-trace uninstall: unknown argument: $1" >&2; exit 2;;
  esac
done

[ "$ALL" = "1" ] && COMMANDS+=(claude codex)
[ "${#COMMANDS[@]}" -eq 0 ] && COMMANDS=(claude)

CLI="$(command -v agent-trace || true)"
if [ -n "$CLI" ]; then
  for cmd in "${COMMANDS[@]}"; do
    log "removing wrapper for: $cmd"
    "$CLI" uninstall --command "$cmd" ${PASS[@]+"${PASS[@]}"} || true
  done
fi

for d in "$HOME/.local/bin" "$HOME/bin"; do
  if [ -L "$d/agent-trace" ]; then rm -f "$d/agent-trace"; log "removed CLI symlink: $d/agent-trace"; fi
done

if [ "$PURGE" = "1" ]; then
  rm -rf "$HOME_DIR"
  log "purged $HOME_DIR (app + captured trajectories)"
else
  log "kept $HOME_DIR (app + captures). Use --purge to delete it."
fi

log "done — open a new shell (or 'source' your rc)."
