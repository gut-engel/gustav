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
set -euo pipefail

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
    printf '%s' "$2"
    IFS= read -r __reply || true
  fi
  printf -v "$1" '%s' "$__reply"
}

DIR="${GUSTAV_DIR:-$HOME/gustav}"
BIN_DIR="${GUSTAV_BIN:-$HOME/.local/bin}"
PURGE=false
YES=false
DRY_RUN=false

say()  { printf '%s\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY_RUN" = true ]; then printf '  would run: %s\n' "$*"; else "$@"; fi; }

usage() {
  cat <<EOF
Usage: uninstall.sh [options]

  --dir <path>   where it was installed (default: $DIR)
  --purge        also delete the database, adapters, tools and backups
  --yes          do not ask
  --dry-run      print what would happen and change nothing
  -h, --help     this
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)     DIR="${2:-}"; shift 2 ;;
    --purge)   PURGE=true; shift ;;
    --yes|-y)  YES=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         die "unknown option: $1 (try --help)" ;;
  esac
done

[ -d "$DIR" ] || die "$DIR is not there. Pass --dir if you installed somewhere else."

say "About to remove gustav from $DIR:"
say "  - the virtualenv ($DIR/.venv)"
say "  - the 'ai' command ($BIN_DIR/ai), if it points into $DIR"
say "  - the systemd user service, if it is installed"
if [ "$PURGE" = true ]; then
  for part in var backups adapters tools models; do
    [ -e "$DIR/$part" ] && say "  - $DIR/$part  ($(du -sh "$DIR/$part" 2>/dev/null | cut -f1))"
  done
  warn "  --purge: the database, what it learned and your approved tools go too."
else
  say
  say "Keeping (delete them yourself, or pass --purge):"
  for part in var backups adapters tools models config.yaml; do
    [ -e "$DIR/$part" ] && say "  $DIR/$part"
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

if command -v systemctl >/dev/null 2>&1; then
  if systemctl --user list-unit-files gustav.service >/dev/null 2>&1; then
    run systemctl --user disable --now gustav.service || true
  fi
fi
unit="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/gustav.service"
[ -f "$unit" ] && run rm -f "$unit"

if [ -L "$BIN_DIR/ai" ]; then
  # Only if it points into the install being removed. With two checkouts on one machine
  # — a server and a working copy, say — uninstalling one must not take away the
  # other's command.
  target="$(readlink -f "$BIN_DIR/ai" 2>/dev/null || true)"
  case "$target" in
    "$(readlink -f "$DIR" 2>/dev/null)"/*) run rm -f "$BIN_DIR/ai" ;;
    *) warn "$BIN_DIR/ai points at $target, which is not in $DIR; leaving it alone" ;;
  esac
elif [ -e "$BIN_DIR/ai" ]; then
  warn "$BIN_DIR/ai is not a symlink; leaving it alone"
fi

[ -d "$DIR/.venv" ] && run rm -rf "$DIR/.venv"

if [ "$PURGE" = true ]; then
  for part in var backups adapters tools models; do
    [ -e "$DIR/$part" ] && run rm -rf "${DIR:?}/$part"
  done
fi

say
say "Done."
if [ "$PURGE" = false ]; then
  say "What it learned is still in $DIR. Reinstalling there picks up where it left off."
fi
