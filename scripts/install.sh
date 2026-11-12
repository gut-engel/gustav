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
# The repository is private, so fetching it needs a credential. Set GUSTAV_TOKEN to a
# GitHub token with read access to this one repository, or have SSH access configured
# and pass --ssh. See the README for how to mint a token that can do nothing else.
set -euo pipefail

REPO_SLUG="${GUSTAV_REPO:-gut-engel/gustav}"
REF="${GUSTAV_REF:-main}"
DIR="${GUSTAV_DIR:-$HOME/gustav}"
MODE=""            # server | client
SOURCE=""          # a local path, instead of cloning
URL="${GUSTAV_URL:-}"
TOKEN="${GUSTAV_TOKEN:-}"
USE_SSH=false
WITH_SERVICE=""    # yes | no, asked if neither flag is given
DRY_RUN=false
QUIET=false
PYTHON="${GUSTAV_PYTHON:-}"
BIN_DIR="${GUSTAV_BIN:-$HOME/.local/bin}"

say()  { [ "$QUIET" = true ] || printf '%s\n' "$*"; }
step() { [ "$QUIET" = true ] || printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY_RUN" = true ]; then printf '  would run: %s\n' "$*"; else "$@"; fi; }

usage() {
  cat <<EOF
Usage: install.sh [--server | --client] [options]

  --server            the daemon and everything it needs (needs an NVIDIA GPU)
  --client            only the 'ai' command, to talk to a server you already run
  --url <url>         the server the client should use, e.g. https://box:8713
  --dir <path>        where to install (default: $DIR)
  --ref <git-ref>     branch or tag to install (default: $REF)
  --source <path>     install from a local checkout instead of cloning
  --ssh               clone over SSH instead of using GUSTAV_TOKEN
  --service / --no-service
                      install (or skip) the systemd user service, server only
  --python <path>     interpreter to build the virtualenv with
  --dry-run           print what would happen and change nothing
  --quiet             only errors
  -h, --help          this

Environment: GUSTAV_TOKEN, GUSTAV_REPO, GUSTAV_REF, GUSTAV_DIR, GUSTAV_URL, GUSTAV_PYTHON
EOF
}

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
    --python)     PYTHON="${2:-}"; shift 2 ;;
    --dry-run)    DRY_RUN=true; shift ;;
    --quiet)      QUIET=true; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown option: $1 (try --help)" ;;
  esac
done

[ "$(id -u)" -eq 0 ] && warn "running as root. gustav is meant to run as you: it needs your GPU, your files and your home directory. Installing as root will work and is not what you want."

# --- what are we installing? -------------------------------------------------
if [ -z "$MODE" ]; then
  if [ -t 0 ]; then
    say "What is this machine?"
    say "  1) a server — runs the model, sleeps, learns  (needs an NVIDIA GPU)"
    say "  2) a client — just the 'ai' command, talking to a server you already run"
    printf 'Choose [1/2]: '
    read -r choice
    case "$choice" in
      1) MODE=server ;;
      2) MODE=client ;;
      *) die "expected 1 or 2" ;;
    esac
  else
    die "say --server or --client (nothing is attached to read an answer from)"
  fi
fi

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
      tar -C "$SOURCE" -cf - \
        --exclude=./.venv --exclude=./.git --exclude=./var --exclude=./backups \
        --exclude=./models --exclude=./adapters --exclude=./tools \
        --exclude=./.pytest_cache --exclude='*.egg-info' --exclude='__pycache__' \
        . | tar -C "$DIR" -xf -
    fi
  fi
  say "  from $SOURCE"
elif [ -d "$DIR/.git" ]; then
  say "  updating the checkout that is already there"
  run git -C "$DIR" fetch --quiet origin "$REF"
  run git -C "$DIR" checkout --quiet "$REF"
  run git -C "$DIR" merge --ff-only --quiet "origin/$REF" 2>/dev/null || \
    warn "  could not fast-forward; your local changes were left alone"
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
    die "this repository is private: set GUSTAV_TOKEN, or pass --ssh, or use --source"
  fi
  run git clone --quiet --branch "$REF" --depth 1 "$remote" "$DIR"
  # The token must not survive in .git/config, where it would end up in backups.
  [ -n "$TOKEN" ] && [ "$DRY_RUN" = false ] && \
    git -C "$DIR" remote set-url origin "https://github.com/${REPO_SLUG}.git"
fi

# --- the virtualenv ----------------------------------------------------------
step "Building the virtualenv"
VENV="$DIR/.venv"
[ -x "$VENV/bin/python" ] || run "$PYTHON" -m venv "$VENV"
run "$VENV/bin/python" -m pip install --quiet --upgrade pip

if [ "$MODE" = server ]; then
  say "  installing with the ML extras — this is the slow part, about 4 GiB of wheels"
  run "$VENV/bin/python" -m pip install --quiet -e "$DIR[ml]"
else
  say "  installing the client only: no torch, no CUDA, a few megabytes"
  run "$VENV/bin/python" -m pip install --quiet -e "$DIR"
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
     warn "      export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac

# --- make it a project -------------------------------------------------------
if [ "$MODE" = server ]; then
  step "Setting up the project"
  run "$VENV/bin/ai" init "$DIR"

  if [ -z "$WITH_SERVICE" ]; then
    if [ -t 0 ]; then
      printf 'Run it as a systemd user service, so it is always on? [Y/n]: '
      read -r answer
      case "$answer" in [nN]*) WITH_SERVICE=no ;; *) WITH_SERVICE=yes ;; esac
    else
      WITH_SERVICE=no
    fi
  fi
  if [ "$WITH_SERVICE" = yes ]; then
    run "$VENV/bin/ai" service install --directory "$DIR" --python "$VENV/bin/python"
  fi
else
  step "Pointing the client at your server"
  if [ -z "$URL" ] && [ -t 0 ]; then
    printf 'Server URL (e.g. https://box.example:8713), blank to do it later: '
    read -r URL
  fi
  if [ -n "$URL" ]; then
    say "  ai login $URL"
    [ "$DRY_RUN" = false ] && "$VENV/bin/ai" login "$URL" || true
  fi
fi

# --- what now ----------------------------------------------------------------
step "Done"
if [ "$MODE" = server ]; then
  cat <<EOF
  cd $DIR
  ai plan-hardware -v        # what it would run here, and why
  ai model pull qwen3-4b     # the weights (uses the network)
EOF
  if [ "$WITH_SERVICE" = yes ]; then
    cat <<EOF
  systemctl --user daemon-reload
  systemctl --user enable --now gustav
  journalctl --user -u gustav -f
EOF
  else
    say "  ai start                   # run it in this terminal"
  fi
  cat <<EOF

  The API and the web interface are off until you turn them on. To reach this
  machine from elsewhere, set api.enabled (and web.enabled) in $DIR/config.yaml,
  then make an account:  ai user add <name> --owner
EOF
else
  say "  ai                         # chat"
  say "  ai login <url>             # if you skipped it, or to point somewhere else"
  say "  ai whoami                  # which server and account you are using"
fi
