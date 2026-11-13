#!/usr/bin/env bash
#
# Install gustav on this machine.
#
#   server:  everything, including the CUDA wheels, plus a systemd *user* service
#   client:  just the `ai` command, pointed at a server you already run
#
# It installs into your home directory and runs as you. It never asks for sudo, never
# writes outside the directories it tells you about, and refuses rather than guessing
# when something is in the way. Run it again any time to update.
#
# Everything it does goes into a log: every command, every line those commands printed,
# how long each step took. The screen gets the short version — numbered steps, and a
# line that keeps moving while pip spends ten minutes on four gigabytes of CUDA. When
# something breaks, the last of the log is printed right there, and the whole of it is
# in a file whose path is on the screen before anything has happened yet.
#
# The repository is public, so this needs no credential. If you have made your own fork
# private, set GUSTAV_TOKEN to a token with read access to it, or pass --ssh.
set -euo pipefail

# fd 3 and 4 are the screen, for good. Once the log is open, plain stdout and stderr
# belong to the log file — so anything a command prints is recorded without being asked
# — and everything meant for the person watching is written to 3 (or 4 for trouble).
exec 3>&1 4>&2

# Asking a question through /dev/tty rather than stdin. The one-liner pipes this script
# into bash, so stdin is the download, not the person at the keyboard — reading from it
# would consume the rest of the script. There is still a terminal; it is just over here.
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

ARGV=("$@")
REPO_SLUG="${GUSTAV_REPO:-gut-engel/gustav}"
REF="${GUSTAV_REF:-main}"
DIR="${GUSTAV_DIR:-$HOME/gustav}"
MODE=""            # server | client
SOURCE=""          # a local path, instead of cloning
URL="${GUSTAV_URL:-}"
TOKEN="${GUSTAV_TOKEN:-}"
USE_SSH=false
WITH_SERVICE=""    # yes | no, asked if neither flag is given
RESET=""           # yes | no, asked if the checkout cannot be fast-forwarded
DRY_RUN=false
QUIET=false
PYTHON="${GUSTAV_PYTHON:-}"
BIN_DIR="${GUSTAV_BIN:-$HOME/.local/bin}"

LOG=""
LOG_READY=false
GRACEFUL=false     # true once something has already been explained properly
STEP=0
STEP_TOTAL=6

now()  { date '+%Y-%m-%d %H:%M:%S'; }
# log() writes to plain stdout, which is the log file from the moment it is open and
# nowhere at all before that.
log()  { if [ "$LOG_READY" = true ]; then printf '%s  %s\n' "$(now)" "$*"; fi; }
say()  { log "$*"; if [ "$QUIET" != true ]; then printf '%s\n' "$*" >&3; fi; }
step() {
  STEP=$((STEP + 1))
  log "--- step $STEP/$STEP_TOTAL: $* ------------------------------------"
  if [ "$QUIET" != true ]; then
    printf '\n\033[1m[%d/%d] %s\033[0m\n' "$STEP" "$STEP_TOTAL" "$*" >&3
  fi
}
warn() { log "warning: $*"; printf '\033[33m%s\033[0m\n' "$*" >&4; }
# die() is for something the person did and can fix from the message alone: the log
# tail would only bury it. fail() is for a command that broke, where the last thing
# it printed is the whole point.
die()  { GRACEFUL=true; log "error: $*"; printf '\033[31merror:\033[0m %s\n' "$*" >&4; exit 1; }
fail() { log "error: $*"; printf '\033[31merror:\033[0m %s\n' "$*" >&4; exit 1; }

# Any credential is masked everywhere it could be written down — the screen, the log,
# and the "would run" of a dry run all end up pasted into issues and scrollback.
mask()    { printf '%s' "${*//x-access-token:*@github.com/x-access-token:***@github.com}"; }
log_cmd() { log "\$ $(mask "$@")"; }

fmt_secs() { printf '%d:%02d' $(($1 / 60)) $(($1 % 60)); }

# Commands run with their stdin closed. In the one-liner, this script's stdin is the
# script itself; a child that reads from it would swallow the rest of the install.
run() {
  log_cmd "$@"
  if [ "$DRY_RUN" = true ]; then
    printf '  would run: %s\n' "$(mask "$@")" >&3
    return 0
  fi
  "$@" < /dev/null
}

# Like run, but the command's own output is worth watching (`ai init` says what it kept).
runv() {
  log_cmd "$@"
  if [ "$DRY_RUN" = true ]; then
    printf '  would run: %s\n' "$(mask "$@")" >&3
    return 0
  fi
  if [ "$QUIET" = true ]; then
    "$@" < /dev/null
    return
  fi
  "$@" < /dev/null 2>&1 | tee /dev/fd/3
}

# The last thing worth repeating out of what the running command said. pip -v writes
# down everything it does, which is what the log is for and not what a one-line status
# is for: this picks the last line that means something to a person, and falls back to
# the last line of any kind.
last_log_line() {
  local text picked
  text="$(tail -c 32768 "$LOG" 2>/dev/null | tr '\r' '\n')"
  picked="$(printf '%s\n' "$text" | grep -E \
    '^[[:space:]]*(Collecting|Downloading|Using cached|Obtaining|Building|Preparing|Processing|Installing|Successfully|Receiving|Resolving|Compressing|Counting|Cloning|error|ERROR|WARNING)' \
    | tail -n 1 || true)"
  if [ -z "$picked" ]; then
    picked="$(printf '%s\n' "$text" | grep -v '^[[:space:]]*$' | tail -n 1 || true)"
  fi
  printf '%s' "${picked#"${picked%%[![:space:]]*}"}"
}

progress() {  # progress <seconds so far>
  if [ "$QUIET" = true ]; then return 0; fi
  local detail
  detail="$(last_log_line)"
  # Our own lines carry a timestamp; the command's do not.
  detail="${detail#[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]  }"
  if [ -t 3 ]; then
    printf '\r  \033[2m%-7s %-62.62s\033[0m' "[$(fmt_secs "$1")]" "$detail" >&3
  else
    printf '  [%s] %.62s\n' "$(fmt_secs "$1")" "$detail" >&3
  fi
}
progress_clear() {
  if [ "$QUIET" != true ] && [ -t 3 ]; then printf '\r%*s\r' 74 '' >&3; fi
}

# A command that takes minutes: it runs in the background with its output going to the
# log, while the screen gets the clock and whatever it last printed. A long silence is
# the one thing an installer must never look like.
run_long() {  # run_long <what> <command...>
  local what="$1"; shift
  log_cmd "$@"
  if [ "$DRY_RUN" = true ]; then
    printf '  would run: %s\n' "$(mask "$@")" >&3
    return 0
  fi
  local start=$SECONDS rc=0 pid tick=0 every=5
  [ -t 3 ] || every=30    # not a terminal: a line every half minute, not a flood
  "$@" < /dev/null &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    tick=$((tick + 1))
    if [ $((tick % every)) -eq 0 ]; then progress $((SECONDS - start)); fi
  done
  wait "$pid" || rc=$?
  progress_clear
  if [ "$rc" -eq 0 ]; then
    say "  $what: done in $(fmt_secs $((SECONDS - start)))"
  else
    log "$what failed after $(fmt_secs $((SECONDS - start))) (exit $rc)"
  fi
  return "$rc"
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
  printf 'Nothing was taken away. Fix what it is complaining about and run it again.\n' >&4
  exit "$rc"
}

usage() {
  cat <<EOF
Usage: install.sh [--server | --client] [options]

  --server            the daemon and everything it needs (needs an NVIDIA GPU)
  --client            only the 'ai' command, to talk to a server you already run
  --url <url>         the server the client should use, e.g. https://box:8713
  --dir <path>        where to install (default: $DIR)
  --ref <git-ref>     branch or tag to install (default: $REF)
  --source <path>     install from a local checkout instead of cloning
  --ssh               clone over SSH (for a fork you have made private)
  --service / --no-service
                      install (or skip) the systemd user service, server only
  --reset             if the checkout has local changes: stash them and match the ref
  --keep-local        the opposite: install what is in the checkout, changes and all
  --python <path>     interpreter to build the virtualenv with
  --log <path>        where to write the log (default: $logdefault)
  --dry-run           print what would happen and change nothing
  --quiet             only errors (the log is written in full either way)
  -h, --help          this

Environment: GUSTAV_TOKEN, GUSTAV_REPO, GUSTAV_REF, GUSTAV_DIR, GUSTAV_URL,
             GUSTAV_PYTHON, GUSTAV_BIN, GUSTAV_LOG
EOF
}

LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/gustav"
logdefault="$LOG_DIR/install-<date>.log"
LOG="${GUSTAV_LOG:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --server)     MODE=server; shift ;;
    --client)     MODE=client; shift ;;
    --url)        URL="${2:-}"; shift 2 ;;
    --dir)        DIR="${2:-}"; shift 2 ;;
    --ref)        REF="${2:-}"; shift 2 ;;
    --source)     SOURCE="${2:-}"; shift 2 ;;
    --ssh)        USE_SSH=true; shift ;;
    --service)    WITH_SERVICE=yes; shift ;;
    --no-service) WITH_SERVICE=no; shift ;;
    --reset)      RESET=yes; shift ;;
    --keep-local) RESET=no; shift ;;
    --python)     PYTHON="${2:-}"; shift 2 ;;
    --log)        LOG="${2:-}"; shift 2 ;;
    --dry-run)    DRY_RUN=true; shift ;;
    --quiet)      QUIET=true; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown option: $1 (try --help)" ;;
  esac
done

# --- the log -----------------------------------------------------------------
# Opened before anything is checked, so that even "python is too old" is written down
# somewhere with the machine it was said about.
if [ -z "$LOG" ]; then
  if mkdir -p "$LOG_DIR" 2>/dev/null; then
    LOG="$LOG_DIR/install-$(date +%Y%m%d-%H%M%S).log"
  else
    LOG="${TMPDIR:-/tmp}/gustav-install-$(date +%Y%m%d-%H%M%S).log"
  fi
else
  mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
fi
: > "$LOG" || die "cannot write the log at $LOG (pass --log somewhere writable)"
exec >>"$LOG" 2>&1
LOG_READY=true
trap on_exit EXIT

log "install.sh"
log "  argv:    $(mask "${ARGV[@]:-}")"
log "  user:    $(id -un) (uid $(id -u))"
log "  machine: $(uname -n) — $(uname -srm)"
if [ -r /etc/os-release ]; then
  log "  os:      $( . /etc/os-release 2>/dev/null; printf '%s' "${PRETTY_NAME:-unknown}" )"
fi
log "  bash:    $BASH_VERSION"
log "  dir:     $DIR"
log "  bin:     $BIN_DIR"
log "  ref:     $REF   repo: $REPO_SLUG   source: ${SOURCE:-<clone>}"
log "  dry-run: $DRY_RUN   quiet: $QUIET   terminal: ${TTY:-none}"

if [ "$QUIET" != true ]; then
  printf 'Writing a log of all of this to %s\n' "$LOG" >&3
fi

# Keep the last ten. A log nobody can find and nobody deletes is just a slow leak.
if [ -d "$LOG_DIR" ]; then
  ls -1t "$LOG_DIR"/install-*.log 2>/dev/null | tail -n +11 | while IFS= read -r old; do
    rm -f "$old" && log "pruned old log $old"
  done || true
fi

if [ "$(id -u)" -eq 0 ]; then
  warn "running as root. gustav is meant to run as you: it needs your GPU, your files and your home directory. Installing as root will work and is not what you want."
fi

# --- what are we installing? -------------------------------------------------
if [ -z "$MODE" ]; then
  if asking; then
    say "What is this machine?"
    say "  1) a server — runs the model, sleeps, learns  (needs an NVIDIA GPU)"
    say "  2) a client — just the 'ai' command, talking to a server you already run"
    ask choice 'Choose [1/2]: '
    case "$choice" in
      1) MODE=server ;;
      2) MODE=client ;;
      *) die "expected 1 or 2" ;;
    esac
  else
    die "say --server or --client (nothing is attached to read an answer from)"
  fi
fi
log "mode: $MODE"

# --- prerequisites -----------------------------------------------------------
step "Checking what is here"

find_python() {
  for candidate in "$PYTHON" python3.14 python3.13 python3.12 python3; do
    [ -n "$candidate" ] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
      printf '%s' "$(command -v "$candidate")"
      return 0
    fi
  done
  return 1
}

PYTHON="$(find_python)" || die "python 3.12 or newer is needed; none found on PATH"
say "  python     $PYTHON ($("$PYTHON" -c 'import platform; print(platform.python_version())'))"

if [ -z "$SOURCE" ]; then
  command -v git >/dev/null 2>&1 || die "git is needed to fetch the source (or pass --source)"
  say "  git        $(git --version | awk '{print $3}')"
fi

if [ "$MODE" = server ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    say "  gpu        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
    log "nvidia-smi says:"
    nvidia-smi || log "  (nvidia-smi would not run a second time)"
  else
    warn "  gpu        no nvidia-smi found. The server needs an NVIDIA GPU; install will"
    warn "             continue, but 'ai plan-hardware' will tell you it cannot run here."
  fi
  free_kb=$(df -Pk "$(dirname "$DIR")" | awk 'NR==2 {print $4}')
  if [ "${free_kb:-0}" -lt 31457280 ]; then
    warn "  disk       $(( free_kb / 1048576 )) GiB free where this is going. The wheels are"
    warn "             about 4 GiB and the smallest useful model about 1 GiB more."
  else
    say "  disk       $(( free_kb / 1048576 )) GiB free"
  fi
fi

# --- the source --------------------------------------------------------------
step "Putting the source in $DIR"

if [ -n "$SOURCE" ]; then
  SOURCE="$(cd "$SOURCE" && pwd)"
  [ -f "$SOURCE/pyproject.toml" ] || die "$SOURCE does not look like a gustav checkout"
  if [ "$SOURCE" != "$DIR" ]; then
    run mkdir -p "$DIR"
    # Only the source. A checkout that has been *used* also holds a virtualenv, model
    # weights, adapters, backups and a database — gigabytes of it, and all of it either
    # rebuilt here or someone else's. Copying var/ would hand this machine another
    # machine's conversations.
    if [ "$DRY_RUN" = true ]; then
      say "  would copy the source from $SOURCE (without .venv, var, models, backups)"
    else
      log "copying $SOURCE -> $DIR"
      tar -C "$SOURCE" -cf - \
        --exclude=./.venv --exclude=./.git --exclude=./var --exclude=./backups \
        --exclude=./models --exclude=./adapters --exclude=./tools \
        --exclude=./.pytest_cache --exclude='*.egg-info' --exclude='__pycache__' \
        . | tar -C "$DIR" -xvf -
    fi
  fi
  say "  from $SOURCE"
elif [ -d "$DIR/.git" ]; then
  say "  updating the checkout that is already there"
  was="$(git -C "$DIR" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
  log "HEAD before: $(git -C "$DIR" log -1 --oneline 2>/dev/null || printf 'unknown')"
  run git -C "$DIR" fetch --quiet origin "$REF" || fail "could not fetch $REF from origin"

  # What is in the way, if anything. This is the part that used to shrug and carry on:
  # a checkout with an edit in it would be installed as it stood, and the first anyone
  # heard of it was the daemon dying on a SyntaxError in a file they had never opened.
  changed="$(git -C "$DIR" status --porcelain --untracked-files=no 2>/dev/null || true)"
  if [ -n "$changed" ]; then log "local changes:"; printf '%s\n' "$changed"; fi
  ff_ok=true
  if git -C "$DIR" rev-parse --verify --quiet "origin/$REF^{commit}" >/dev/null; then
    git -C "$DIR" merge-base --is-ancestor HEAD "origin/$REF" || ff_ok=false
  fi
  ahead="$(git -C "$DIR" rev-list --count "origin/$REF..HEAD" 2>/dev/null || printf 0)"
  log "fast-forward possible: $ff_ok   commits ahead of origin/$REF: $ahead"

  updated=false
  if [ "$ff_ok" = true ]; then
    if [ "$DRY_RUN" = true ]; then
      say "  would fast-forward to origin/$REF"
      updated=true
    elif { log_cmd git -C "$DIR" checkout "$REF"; git -C "$DIR" checkout --quiet "$REF"; } \
      && { log_cmd git -C "$DIR" merge --ff-only "origin/$REF"; git -C "$DIR" merge --ff-only --quiet "origin/$REF"; }; then
      updated=true
      say "  now at $(git -C "$DIR" log -1 --format='%h %s')"
      if [ -n "$changed" ]; then
        warn "  you have local changes here; they were left alone:"
        printf '%s\n' "$changed" | sed 's/^/      /' >&4
      fi
    fi
  fi

  if [ "$updated" = false ]; then
    warn "  $DIR cannot be brought up to $REF as it stands."
    if [ -n "$changed" ]; then
      warn "  changed files in the way:"
      printf '%s\n' "$changed" | sed 's/^/      /' >&4
    fi
    if [ "$ahead" != 0 ]; then
      warn "  and it is $ahead commit(s) that origin/$REF does not have."
    fi
    warn "  Installing on top of this installs those files, not $REF."
    choice="$RESET"
    if [ -z "$choice" ] && asking; then
      ask answer "  Put them aside (git stash) and match origin/$REF? [Y/n]: "
      case "$answer" in [nN]*) choice=no ;; *) choice=yes ;; esac
    fi
    case "$choice" in
      yes)
        run git -C "$DIR" stash push --include-untracked --quiet --message "install.sh $(now)" || true
        run git -C "$DIR" checkout --quiet "$REF" || fail "could not switch $DIR to $REF"
        run git -C "$DIR" reset --hard --quiet "origin/$REF" || fail "could not reset $DIR to origin/$REF"
        say "  reset to origin/$REF"
        say "  what was there: 'git -C $DIR stash list', and the old commit was $was"
        ;;
      no)
        warn "  leaving the checkout as it is. If the daemon then dies on an import, this is why."
        ;;
      *)
        die "$DIR has changes of its own and there is nothing attached to ask with.
       --reset       put them in a git stash and match origin/$REF
       --keep-local  install them as they are, whatever is in them"
        ;;
    esac
  fi
else
  [ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ] && \
    die "$DIR exists and is not a git checkout. Move it, or pass --dir."
  if [ "$USE_SSH" = true ]; then
    remote="git@github.com:${REPO_SLUG}.git"
    say "  cloning over SSH"
  elif [ -n "$TOKEN" ]; then
    remote="https://x-access-token:${TOKEN}@github.com/${REPO_SLUG}.git"
    say "  cloning with the token from GUSTAV_TOKEN"
  else
    remote="https://github.com/${REPO_SLUG}.git"
    say "  cloning $REPO_SLUG"
  fi
  run_long "the clone" git clone --progress --branch "$REF" --depth 1 "$remote" "$DIR" \
    || fail "could not clone $REPO_SLUG at $REF"
  # The token must not survive in .git/config, where it would end up in backups.
  if [ -n "$TOKEN" ] && [ "$DRY_RUN" = false ]; then
    git -C "$DIR" remote set-url origin "https://github.com/${REPO_SLUG}.git"
  fi
fi

# --- the virtualenv ----------------------------------------------------------
step "Building the virtualenv"
VENV="$DIR/.venv"
if [ -x "$VENV/bin/python" ]; then
  say "  reusing the one in $VENV"
else
  run "$PYTHON" -m venv "$VENV" || fail "could not create a virtualenv in $VENV"
  say "  new virtualenv in $VENV"
fi

# This is the step that takes ten minutes and the one that goes wrong on a machine
# nobody can look at, so everything that decides its outcome is written down first:
# which python, which pip, which index, what proxy, how much room is left.
if [ "$DRY_RUN" = false ]; then
  log "the interpreter and the packaging tools it will use:"
  "$VENV/bin/python" -VV || true
  "$VENV/bin/python" -m pip --version || true
  log "the pip settings in this environment:"
  # Credentials live in index URLs. They are masked here the same way they are
  # everywhere else this script writes something down.
  env | grep -E '^(PIP_|PYTHON|VIRTUAL_ENV|CUDA|LD_LIBRARY_PATH|(HTTP|HTTPS|NO)_PROXY|http_proxy|https_proxy)' \
    | sed -E 's#://[^/@ ]*:[^/@ ]*@#://***@#g' | sort || log "  (none set)"
  "$VENV/bin/python" -m pip config list || true
  log "room where the wheels are going:"
  df -Ph "$DIR" || true
fi

run_long "pip itself" "$VENV/bin/python" -m pip install --upgrade --disable-pip-version-check \
  --progress-bar off pip || fail "could not update pip inside the virtualenv"

# -v so that the log holds the whole of it: every candidate considered, every wheel
# downloaded, and — the one that matters — the entire output of a build that failed.
# The screen is not any noisier for it; it shows the lines a person would have read.
if [ "$MODE" = server ]; then
  say "  installing with the ML extras — this is the slow part, about 4 GiB of wheels"
  run_long "the wheels" "$VENV/bin/python" -m pip install --disable-pip-version-check \
    --progress-bar off -v -e "$DIR[ml]" \
    || fail "pip could not install the server packages (the whole attempt is in $LOG)"
else
  say "  installing the client only: no torch, no CUDA, a few megabytes"
  run_long "the packages" "$VENV/bin/python" -m pip install --disable-pip-version-check \
    --progress-bar off -v -e "$DIR" \
    || fail "pip could not install the client packages (the whole attempt is in $LOG)"
fi

# And what it ended up with, by name and version. "It worked on mine" is answerable
# from this and nothing else.
if [ "$DRY_RUN" = false ]; then
  log "everything in the virtualenv now:"
  "$VENV/bin/python" -m pip list --format=freeze || true
  log "pip's own opinion of those versions:"
  "$VENV/bin/python" -m pip check || log "  (pip check is unhappy; see just above)"
  installed="$("$VENV/bin/python" -m pip list --format=freeze 2>/dev/null | wc -l)"
  say "  $installed packages, $(du -sh "$VENV" 2>/dev/null | cut -f1) in $VENV"
fi

# --- is what we installed actually sound? ------------------------------------
step "Checking what was installed"
if [ "$DRY_RUN" = true ]; then
  say "  would byte-compile $DIR/src and run 'ai --help'"
else
  # A checkout with one broken file in it installs perfectly happily and then falls over
  # at the first import — a daemon that dies on startup with a SyntaxError in a file
  # nobody meant to edit. Compiling every file here turns that into a failure now, next
  # to the thing that caused it, instead of an hour later on a different machine.
  sources=("$DIR/src")
  [ -f "$DIR/daemon.py" ] && sources+=("$DIR/daemon.py")
  if "$VENV/bin/python" -m compileall -q "${sources[@]}"; then
    say "  every source file compiles"
  else
    fail "some of the source in $DIR does not compile — see the lines above.
       If you have edited files there, that is where to look; 'cd $DIR && git status'
       says which ones. Re-running with --reset puts them aside and installs $REF."
  fi
  if [ "$MODE" = server ]; then
    # The whole point of the server install. If this says available=False, the daemon
    # will plan for a GPU it cannot reach, and that is much easier to understand here
    # than three screens into a hardware plan.
    log "what torch can see:"
    if torch_says="$("$VENV/bin/python" -c 'import torch; print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")')"; then
      say "  $torch_says"
      case "$torch_says" in
        *available=False*) warn "  torch cannot reach the GPU. The driver, or a torch built without CUDA." ;;
      esac
    else
      warn "  torch did not import — the log says why. The daemon will not run without it."
    fi
  fi
  if "$VENV/bin/ai" --help >/dev/null 2>&1; then
    say "  'ai' runs"
  else
    "$VENV/bin/ai" --help >/dev/null || true   # again, for the log
    fail "the 'ai' command does not run — see the lines above"
  fi
fi

# --- the ai command ----------------------------------------------------------
step "Putting 'ai' on your PATH"
run mkdir -p "$BIN_DIR"
if [ "$DRY_RUN" = false ]; then
  if [ -e "$BIN_DIR/ai" ] && [ ! -L "$BIN_DIR/ai" ]; then
    warn "  $BIN_DIR/ai exists and is not a symlink; leaving it alone"
  else
    ln -sfn "$VENV/bin/ai" "$BIN_DIR/ai"
    say "  $BIN_DIR/ai -> $VENV/bin/ai"
  fi
fi
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "  $BIN_DIR is not on your PATH. Add this to your shell profile:"
     # Name the directory it actually used, not the default: --dir and GUSTAV_BIN
     # move it, and advice that points somewhere else is worse than none.
     warn "      export PATH=\"${BIN_DIR/#$HOME/\$HOME}:\$PATH\"" ;;
esac

# --- make it a project -------------------------------------------------------
if [ "$MODE" = server ]; then
  step "Setting up the project"
  runv "$VENV/bin/ai" init "$DIR" || fail "'ai init' did not finish"

  if [ -z "$WITH_SERVICE" ]; then
    if asking; then
      ask answer 'Run it as a systemd user service, so it is always on? [Y/n]: '
      case "$answer" in [nN]*) WITH_SERVICE=no ;; *) WITH_SERVICE=yes ;; esac
    else
      WITH_SERVICE=no
    fi
  fi
  log "service: $WITH_SERVICE"
  if [ "$WITH_SERVICE" = yes ]; then
    runv "$VENV/bin/ai" service install --directory "$DIR" --python "$VENV/bin/python" \
      || fail "could not write the systemd user unit"
  fi
else
  step "Pointing the client at your server"
  if [ -z "$URL" ] && asking; then
    ask URL 'Server URL (e.g. https://box.example:8713), blank to do it later: '
  fi
  if [ -z "$URL" ]; then
    say "  no server named yet: 'ai login <url>' when you have one"
  else
    say "  ai login $URL"
    log_cmd "$VENV/bin/ai" login "$URL"
    # login asks for a password, so it keeps the terminal and prints for itself.
    if [ "$DRY_RUN" = false ]; then
      "$VENV/bin/ai" login "$URL" >&3 2>&4 || warn "  login did not finish; 'ai login $URL' when you are ready"
    fi
  fi
fi

# --- what now ----------------------------------------------------------------
if [ "$QUIET" != true ]; then printf '\n\033[1mDone\033[0m\n' >&3; fi
log "=== done ==="
if [ "$MODE" = server ]; then
  say "  cd $DIR"
  say "  ai plan-hardware -v        # what it would run here, and why"
  say "  ai model pull <model>      # the weights the plan chose (uses the network)"
  if [ "$WITH_SERVICE" = yes ]; then
    say "  systemctl --user daemon-reload"
    say "  systemctl --user enable --now gustav"
    say "  journalctl --user -u gustav -f"
  else
    say "  ai start                   # run it in this terminal"
  fi
  say ""
  say "  The API and the web interface are off until you turn them on. To reach this"
  say "  machine from elsewhere, set api.enabled (and web.enabled) in $DIR/config.yaml,"
  say "  then make an account:  ai user add <name> --owner"
else
  say "  ai                         # chat"
  say "  ai login <url>             # if you skipped it, or to point somewhere else"
  say "  ai whoami                  # which server and account you are using"
fi
