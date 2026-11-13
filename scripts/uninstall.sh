#!/usr/bin/env bash
#
# Remove gustav from this machine.
#
# It takes away what the installer put there — the virtualenv, the `ai` symlink, the
# systemd user unit — and leaves everything *you* accumulated: the database, the
# adapters it trained, the tools you approved, the backups. Those are not the
# installer's to delete, and a machine that deletes a year of learning because you
# asked it to uninstall a program has made a decision that was not offered.
#
# --purge deletes them too, after saying exactly what and asking.
#
# Like the installer, it writes down everything it does — what it removed, how big it
# was, what it decided to leave — in a log whose path it prints before it starts. An
# uninstall is the worst possible time to be unsure what just happened.
set -euo pipefail

# fd 3 and 4 are the screen; once the log is open, ordinary output belongs to the file.
exec 3>&1 4>&2

# See install.sh: the one-liner pipes this script into bash, so the question has to go
# to the terminal rather than to stdin.
TTY=""
if [ -t 0 ]; then
  TTY="/dev/stdin"
elif { : < /dev/tty; } 2>/dev/null; then
  # The device node exists on machines with no controlling terminal too, so the
  # only honest test is to open it.
  TTY="/dev/tty"
fi
asking() { [ -n "$TTY" ]; }
ask() {  # ask <variable> <prompt>
  local __reply=""
  asking || return 1
  if [ "$TTY" = /dev/tty ]; then
    printf '%s' "$2" > /dev/tty
    IFS= read -r __reply < /dev/tty || true
  else
    printf '%s' "$2" >&3
    IFS= read -r __reply || true
  fi
  log "asked: $2[$__reply]"
  printf -v "$1" '%s' "$__reply"
}

# See install.sh: bumped by hand, so a paste says which copy of this ran.
VERSION="2026-11-13.4"

ARGV=("$@")
DIR="${GUSTAV_DIR:-$HOME/gustav}"
BIN_DIR="${GUSTAV_BIN:-$HOME/.local/bin}"
PURGE=false
YES=false
DRY_RUN=false

LOG=""
LOG_READY=false
GRACEFUL=false
STEP=0
STEP_TOTAL=4

now()  { date '+%Y-%m-%d %H:%M:%S'; }
log()  { if [ "$LOG_READY" = true ]; then printf '%s  %s\n' "$(now)" "$*"; fi; }
say()  { log "$*"; printf '%s\n' "$*" >&3; }
step() {
  STEP=$((STEP + 1))
  log "--- step $STEP/$STEP_TOTAL: $* ------------------------------------"
  printf '\n\033[1m[%d/%d] %s\033[0m\n' "$STEP" "$STEP_TOTAL" "$*" >&3
}
warn() { log "warning: $*"; printf '\033[33m%s\033[0m\n' "$*" >&4; }
die()  { GRACEFUL=true; log "error: $*"; printf '\033[31merror:\033[0m %s\n' "$*" >&4; exit 1; }
fail() { log "error: $*"; printf '\033[31merror:\033[0m %s\n' "$*" >&4; exit 1; }

run() {
  log "\$ $*"
  if [ "$DRY_RUN" = true ]; then
    printf '  would run: %s\n' "$*" >&3
    return 0
  fi
  "$@" < /dev/null
}

size_of() { du -sh "$1" 2>/dev/null | cut -f1 || printf '?'; }

on_exit() {
  local rc=$?
  if [ "$LOG_READY" != true ]; then exit "$rc"; fi
  if [ "$rc" -eq 0 ]; then
    log "finished (exit 0)"
    printf '\nWhat it did, in full: %s\n' "$LOG" >&3
    exit 0
  fi
  log "stopped (exit $rc)"
  printf '\n\033[31mThat did not finish.\033[0m\n' >&4
  if [ "$GRACEFUL" != true ]; then
    printf 'The last of the log (%s):\n\n' "$LOG" >&4
    tail -n 25 "$LOG" | sed 's/^/  /' >&4
    printf '\n' >&4
  else
    printf 'The whole log is in %s\n' "$LOG" >&4
  fi
  printf 'Whatever it had not removed yet is still there.\n' >&4
  exit "$rc"
}

usage() {
  cat <<EOF
Usage: uninstall.sh [options]

  --dir <path>   where it was installed (default: $DIR)
  --purge        also delete the database, adapters, tools and backups
  --yes          do not ask
  --log <path>   where to write the log (default: $logdefault)
  --version      print the version of this script and stop
  --dry-run      print what would happen and change nothing
  -h, --help     this

Environment: GUSTAV_DIR, GUSTAV_BIN, GUSTAV_LOG
EOF
}

LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/gustav"
logdefault="$LOG_DIR/uninstall-<date>.log"
LOG="${GUSTAV_LOG:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)     DIR="${2:-}"; shift 2 ;;
    --purge)   PURGE=true; shift ;;
    --yes|-y)  YES=true; shift ;;
    --log)     LOG="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --version) printf 'uninstall.sh %s\n' "$VERSION"; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *)         die "unknown option: $1 (try --help)" ;;
  esac
done

# --- the log -----------------------------------------------------------------
if [ -z "$LOG" ]; then
  if mkdir -p "$LOG_DIR" 2>/dev/null; then
    LOG="$LOG_DIR/uninstall-$(date +%Y%m%d-%H%M%S).log"
  else
    LOG="${TMPDIR:-/tmp}/gustav-uninstall-$(date +%Y%m%d-%H%M%S).log"
  fi
else
  mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
fi
: > "$LOG" || die "cannot write the log at $LOG (pass --log somewhere writable)"
exec >>"$LOG" 2>&1
LOG_READY=true
trap on_exit EXIT

log "uninstall.sh $VERSION"
log "  argv:    ${ARGV[*]:-}"
log "  user:    $(id -un) (uid $(id -u))"
log "  machine: $(uname -n) — $(uname -srm)"
log "  dir:     $DIR"
log "  bin:     $BIN_DIR"
log "  purge:   $PURGE   dry-run: $DRY_RUN   terminal: ${TTY:-none}"

printf 'gustav uninstaller %s — writing all of this to %s\n' "$VERSION" "$LOG" >&3

if [ -d "$LOG_DIR" ]; then
  ls -1t "$LOG_DIR"/uninstall-*.log 2>/dev/null | tail -n +11 | while IFS= read -r old; do
    rm -f "$old" && log "pruned old log $old"
  done || true
fi

[ -d "$DIR" ] || die "$DIR is not there. Pass --dir if you installed somewhere else."

# --- what is about to happen -------------------------------------------------
step "What this will take away"
say "  - the virtualenv ($DIR/.venv)"
say "  - the 'ai' command ($BIN_DIR/ai), if it points into $DIR"
say "  - the systemd user service, if it is installed"
if [ "$PURGE" = true ]; then
  for part in var backups adapters tools models; do
    if [ -e "$DIR/$part" ]; then say "  - $DIR/$part  ($(size_of "$DIR/$part"))"; fi
  done
  warn "  --purge: the database, what it learned and your approved tools go too."
else
  say ""
  say "Keeping (delete them yourself, or pass --purge):"
  for part in var backups adapters tools models config.yaml; do
    if [ -e "$DIR/$part" ]; then say "  $DIR/$part  ($(size_of "$DIR/$part"))"; fi
  done
fi

if [ "$YES" = false ] && [ "$DRY_RUN" = false ]; then
  if asking; then
    ask answer $'\nGo ahead? [y/N]: '
    case "$answer" in [yY]*) ;; *) say "nothing was changed"; exit 0 ;; esac
  else
    die "pass --yes (nothing is attached to confirm with)"
  fi
fi

# --- the service -------------------------------------------------------------
step "The systemd user service"
if command -v systemctl >/dev/null 2>&1; then
  if systemctl --user list-unit-files gustav.service >/dev/null 2>&1; then
    run systemctl --user disable --now gustav.service || \
      warn "  systemctl would not stop it; carrying on"
    say "  stopped and disabled gustav.service"
  else
    say "  no gustav.service was registered"
  fi
else
  say "  no systemctl here; nothing to stop"
fi
unit="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/gustav.service"
if [ -f "$unit" ]; then
  run rm -f "$unit"
  say "  removed $unit"
else
  say "  no unit file at $unit"
fi

# --- the command -------------------------------------------------------------
step "The 'ai' command"
if [ -L "$BIN_DIR/ai" ]; then
  # Only if it points into the install being removed. With two checkouts on one machine
  # — a server and a working copy, say — uninstalling one must not take away the
  # other's command.
  target="$(readlink -f "$BIN_DIR/ai" 2>/dev/null || true)"
  log "$BIN_DIR/ai -> $target"
  case "$target" in
    "$(readlink -f "$DIR" 2>/dev/null)"/*)
      run rm -f "$BIN_DIR/ai"
      say "  removed $BIN_DIR/ai" ;;
    *) warn "$BIN_DIR/ai points at $target, which is not in $DIR; leaving it alone" ;;
  esac
elif [ -e "$BIN_DIR/ai" ]; then
  warn "$BIN_DIR/ai is not a symlink; leaving it alone"
else
  say "  nothing at $BIN_DIR/ai"
fi

# --- the files ---------------------------------------------------------------
also=""
if [ "$PURGE" = true ]; then also=", and what --purge takes with it"; fi
step "The virtualenv$also"
if [ -d "$DIR/.venv" ]; then
  say "  removing $DIR/.venv ($(size_of "$DIR/.venv"))"
  run rm -rf "$DIR/.venv"
else
  say "  no virtualenv at $DIR/.venv"
fi

if [ "$PURGE" = true ]; then
  for part in var backups adapters tools models; do
    if [ -e "$DIR/$part" ]; then
      say "  removing $DIR/$part ($(size_of "$DIR/$part"))"
      run rm -rf "${DIR:?}/$part"
    fi
  done
fi

printf '\n\033[1mDone.\033[0m\n' >&3
log "=== done ==="
if [ "$PURGE" = false ]; then
  say "What it learned is still in $DIR. Reinstalling there picks up where it left off."
fi
