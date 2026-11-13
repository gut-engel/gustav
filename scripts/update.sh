#!/usr/bin/env bash
#
# Update a gustav install that is already here.
#
#   curl -fsSL https://raw.githubusercontent.com/gut-engel/gustav/main/scripts/update.sh | bash
#
# The installer does this too — running it again updates in place — but it is a big
# script that re-checks a machine it has already checked. This one does the thing you
# actually want a week later: move the checkout to what has been published, reinstall
# the package, make sure it imports, and restart the service if it was running.
#
# It is deliberately thin. Everything it decides is decided by `ai update`, which lives
# in the package where it can be tested; this script exists because `ai update` is
# itself part of what needs updating, and because a one-liner is the one thing you can
# paste into a machine you have not logged into for a month. The install is editable,
# so the moment the checkout moves, the `ai` in the virtualenv *is* the new code.
#
# Nothing here touches the database, the adapters, the weights or your config.
set -euo pipefail

VERSION="2026-11-13.5"

exec 3>&1 4>&2

TTY=""
if [ -t 0 ]; then
  TTY="/dev/stdin"
elif { : < /dev/tty; } 2>/dev/null; then
  TTY="/dev/tty"
fi
asking() { [ -n "$TTY" ]; }

ARGV=("$@")
DIR="${GUSTAV_DIR:-$HOME/gustav}"
REF="${GUSTAV_REF:-}"
RESET=false
CHECK=false
RESTART=true
DRY_RUN=false
QUIET=false

LOG=""
LOG_READY=false
GRACEFUL=false

now()  { date '+%Y-%m-%d %H:%M:%S'; }
log()  { if [ "$LOG_READY" = true ]; then printf '%s  %s\n' "$(now)" "$*"; fi; }
say()  { log "$*"; if [ "$QUIET" != true ]; then printf '%s\n' "$*" >&3; fi; }
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

on_exit() {
  local rc=$?
  if [ "$LOG_READY" != true ]; then exit "$rc"; fi
  if [ "$rc" -eq 0 ]; then
    log "finished (exit 0)"
    if [ "$QUIET" != true ]; then printf '\nWhat it did, in full: %s\n' "$LOG" >&3; fi
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
  printf 'The install was left as it is. Nothing of yours was touched.\n' >&4
  exit "$rc"
}

usage() {
  cat <<EOF
Usage: update.sh [options]

  --dir <path>    the install to update (default: $DIR)
  --ref <git-ref> branch or tag to follow (default: the branch it is on)
  --check         say what is new and change nothing
  --reset         local changes go to a git stash and the checkout matches the ref
  --no-restart    leave the service alone
  --log <path>    where to write the log
  --dry-run       print what would happen and change nothing
  --quiet         only errors
  --version       print the version of this script and stop
  -h, --help      this

Environment: GUSTAV_DIR, GUSTAV_REF, GUSTAV_LOG
EOF
}

LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/gustav"
LOG="${GUSTAV_LOG:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)        DIR="${2:-}"; shift 2 ;;
    --ref)        REF="${2:-}"; shift 2 ;;
    --check)      CHECK=true; shift ;;
    --reset)      RESET=true; shift ;;
    --no-restart) RESTART=false; shift ;;
    --log)        LOG="${2:-}"; shift 2 ;;
    --dry-run)    DRY_RUN=true; shift ;;
    --quiet)      QUIET=true; shift ;;
    --version)    printf 'update.sh %s\n' "$VERSION"; exit 0 ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown option: $1 (try --help)" ;;
  esac
done

if [ -z "$LOG" ]; then
  if mkdir -p "$LOG_DIR" 2>/dev/null; then
    LOG="$LOG_DIR/update-$(date +%Y%m%d-%H%M%S).log"
  else
    LOG="${TMPDIR:-/tmp}/gustav-update-$(date +%Y%m%d-%H%M%S).log"
  fi
else
  mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
fi
: > "$LOG" || die "cannot write the log at $LOG (pass --log somewhere writable)"
exec >>"$LOG" 2>&1
LOG_READY=true
trap on_exit EXIT

log "update.sh $VERSION"
log "  argv:    ${ARGV[*]:-}"
log "  user:    $(id -un) (uid $(id -u))"
log "  machine: $(uname -n) — $(uname -srm)"
log "  dir:     $DIR"
log "  ref:     ${REF:-<the branch it is on>}   reset: $RESET   check: $CHECK"

if [ "$QUIET" != true ]; then
  printf 'gustav updater %s — writing all of this to %s\n' "$VERSION" "$LOG" >&3
fi

if [ -d "$LOG_DIR" ]; then
  ls -1t "$LOG_DIR"/update-*.log 2>/dev/null | tail -n +11 | while IFS= read -r old; do
    rm -f "$old" && log "pruned old log $old"
  done || true
fi

# --- is there an install here at all? ----------------------------------------
[ -d "$DIR" ] || die "$DIR is not there. Pass --dir if you installed somewhere else."
[ -d "$DIR/.git" ] || die "$DIR is not a git checkout, so there is nothing to pull.
       Reinstall with scripts/install.sh, which can create one."
AI="$DIR/.venv/bin/ai"
[ -x "$AI" ] || die "there is no virtualenv at $DIR/.venv. Run scripts/install.sh first."

command -v git >/dev/null 2>&1 || die "git is needed to update the checkout"

# --- move the source ---------------------------------------------------------
# Only far enough to have the current 'ai update' in hand: the install is editable, so
# the command below *is* the version that has just been fetched. Everything after this
# — what to reinstall, what to put back, whether to restart — is its decision, made in
# code that has tests, rather than a second set of rules living in a shell script.
if [ -z "$REF" ]; then
  REF="$(git -C "$DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'main')"
  [ "$REF" = "HEAD" ] && die "$DIR is not on a branch (detached HEAD). Pass --ref."
fi
say "Fetching $REF"
run git -C "$DIR" fetch --quiet origin "$REF" || fail "could not fetch $REF from origin"

if [ "$CHECK" = false ] && [ "$DRY_RUN" = false ]; then
  if [ "$RESET" = true ]; then
    log "resetting the checkout to origin/$REF"
    git -C "$DIR" stash push --include-untracked --quiet \
      --message "update.sh $(now)" || true
    git -C "$DIR" checkout --quiet "$REF" || fail "could not switch $DIR to $REF"
    git -C "$DIR" reset --hard --quiet "origin/$REF" || fail "could not reset $DIR"
  elif ! git -C "$DIR" merge --ff-only --quiet "origin/$REF" 2>/dev/null; then
    # Not fatal here: 'ai update' says exactly what is in the way, and in the words
    # the rest of the project uses. This only means it will be the old copy saying it.
    log "could not fast-forward; leaving that to 'ai update' to explain"
  fi
fi

# --- and let the package do the rest -----------------------------------------
# The same log file, not a second one: it appends, and the transcript reads as one
# story — what the shell did, then what the package did.
forward=(--log "$LOG")
[ -n "$REF" ]          && forward+=(--ref "$REF")
[ "$CHECK" = true ]    && forward+=(--check)
[ "$RESET" = true ]    && forward+=(--reset)
[ "$RESTART" = false ] && forward+=(--no-restart)
[ "$DRY_RUN" = true ]  && forward+=(--dry-run)

if ! "$AI" update --help >/dev/null 2>&1; then
  die "the 'ai' in $DIR is too old to know 'ai update'.
       Run the installer once more and it will be there afterwards:
         curl -fsSL https://raw.githubusercontent.com/gut-engel/gustav/main/scripts/install.sh | bash"
fi

say ""
log "\$ $AI update ${forward[*]:-}"
if [ "$QUIET" = true ]; then
  "$AI" update "${forward[@]}" < /dev/null || fail "'ai update' did not finish"
else
  "$AI" update "${forward[@]}" < /dev/null 2>&1 | tee /dev/fd/3 || fail "'ai update' did not finish"
fi
