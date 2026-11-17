#!/usr/bin/env bash
#
# Put a gustav satellite on this machine: a microphone and a speaker in another room.
#
#   curl -fsSL https://raw.githubusercontent.com/gut-engel/gustav/main/scripts/satellite.sh \
#     | bash -s -- --url http://box.local:8713 --token <token>
#
# It is one Python file with **no dependencies** — not numpy, not sounddevice, not
# websockets — because every wheel that has to be compiled on an ARM board is a minute
# of someone's evening and a new way for an install to fail. Recording and playback are
# arecord and aplay, which are already there; the WebSocket client is two hundred lines
# inside the file.
#
# So this script does four small things: check the sound card, fetch one file, write a
# config, and install a systemd service. On a Pi Zero it takes about fifteen seconds,
# and almost all of that is apt installing alsa-utils if it is missing.
#
# Get the token on the server with:
#     ai user add kitchen --owner        # or a plain user
#     ai user token kitchen
set -euo pipefail

exec 3>&1 4>&2

VERSION="2026-11-17.1"

ARGV=("$@")
URL="${GUSTAV_URL:-}"
TOKEN="${GUSTAV_TOKEN:-}"
NAME="$(hostname -s 2>/dev/null || echo satellite)"
DIR="${GUSTAV_SATELLITE_DIR:-$HOME/.local/share/gustav-satellite}"
BIN_DIR="${GUSTAV_BIN:-$HOME/.local/bin}"
CONFIG="${GUSTAV_SATELLITE_CONFIG:-$HOME/.config/gustav-satellite.conf}"
REPO_SLUG="${GUSTAV_REPO:-gut-engel/gustav}"
REF="${GUSTAV_REF:-main}"
SOURCE=""
INPUT=""
OUTPUT=""
THRESHOLD=""
WITH_SERVICE=""
DRY_RUN=false
QUIET=false
SKIP_SOUND_CHECK=false

now()  { date '+%H:%M:%S'; }
say()  { if [ "$QUIET" != true ]; then printf '%s  %s\n' "$(now)" "$*" >&3; fi; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&4; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&4; exit 1; }

TTY=""
if [ -t 0 ]; then TTY=/dev/stdin
elif { : < /dev/tty; } 2>/dev/null; then TTY=/dev/tty
fi
ask() {  # ask <variable> <prompt>
  local __reply=""
  [ -n "$TTY" ] || return 1
  if [ "$TTY" = /dev/tty ]; then
    printf '%s' "$2" > /dev/tty; IFS= read -r __reply < /dev/tty || true
  else
    printf '%s' "$2" >&3; IFS= read -r __reply || true
  fi
  printf -v "$1" '%s' "$__reply"
}

usage() {
  cat <<EOF
Usage: satellite.sh --url <http://server:8713> [--token <token>] [options]

  --url <url>         the gustav daemon this satellite talks to
  --token <token>     from 'ai user token <name>' on the server
  --name <name>       what to call this satellite (default: $NAME)
  --input <device>    ALSA capture device, e.g. plughw:1,0 (default: the default)
  --output <device>   ALSA playback device
  --threshold <0..1>  loudness that counts as speech; --calibrate measures it
  --service / --no-service
                      install (or skip) the systemd user service
  --skip-sound-check  install anyway on a machine with no sound card yet
  --source <path>     install from a checkout instead of downloading
  --dir <path>        where the program goes (default: $DIR)
  --dry-run           say what it would do and change nothing
  --quiet             only errors
  --version           print the version and stop
  -h, --help          this

Afterwards:
  gustav-satellite --calibrate     measure the room, put the number in the config
  gustav-satellite --say "hello"   make the daemon talk through this speaker
  gustav-satellite -v              run it in the foreground and watch
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --url)        URL="${2:-}"; shift 2 ;;
    --token)      TOKEN="${2:-}"; shift 2 ;;
    --name)       NAME="${2:-}"; shift 2 ;;
    --input)      INPUT="${2:-}"; shift 2 ;;
    --output)     OUTPUT="${2:-}"; shift 2 ;;
    --threshold)  THRESHOLD="${2:-}"; shift 2 ;;
    --service)    WITH_SERVICE=yes; shift ;;
    --no-service) WITH_SERVICE=no; shift ;;
    --skip-sound-check) SKIP_SOUND_CHECK=true; shift ;;
    --source)     SOURCE="${2:-}"; shift 2 ;;
    --dir)        DIR="${2:-}"; shift 2 ;;
    --config)     CONFIG="${2:-}"; shift 2 ;;
    --dry-run)    DRY_RUN=true; shift ;;
    --quiet)      QUIET=true; shift ;;
    --version)    printf 'satellite.sh %s\n' "$VERSION"; exit 0 ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown option: $1 (try --help)" ;;
  esac
done

say "gustav satellite installer $VERSION"

# --- 1. can this machine hear and speak? ------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
      PYTHON="$(command -v "$candidate")"
      break
    fi
  fi
done
[ -n "$PYTHON" ] || die "python 3.9 or newer is needed; none found on PATH"
say "  python     $PYTHON"

if ! command -v arecord >/dev/null 2>&1 || ! command -v aplay >/dev/null 2>&1; then
  warn "  alsa-utils is not installed, which is how this records and plays."
  if command -v apt-get >/dev/null 2>&1 && [ "$DRY_RUN" != true ] \
     && [ "$SKIP_SOUND_CHECK" != true ]; then
    say "  installing alsa-utils (this is the only thing that needs sudo)"
    sudo apt-get update -qq && sudo apt-get install -y -qq alsa-utils \
      || die "could not install alsa-utils; install it and run this again"
  elif [ "$DRY_RUN" = true ] || [ "$SKIP_SOUND_CHECK" = true ]; then
    # A dry run says what it would do on the machine it is run on, and one of the
    # things it would do here is install alsa-utils. Saying so beats stopping. The
    # same goes for a box whose USB microphone has not arrived yet.
    warn "  install it before this can hear anything: apt install alsa-utils"
  else
    die "install alsa-utils (apt install alsa-utils) and run this again,
       or pass --skip-sound-check if the microphone is arriving later"
  fi
else
  say "  sound      arecord and aplay are here"
fi

if [ "$DRY_RUN" != true ]; then
  if ! arecord -l 2>/dev/null | grep -q card; then
    warn "  arecord -l lists no capture device. A USB microphone is the usual fix;"
    warn "  the Raspberry Pi has no microphone input of its own."
  else
    say "  capture    $(arecord -l 2>/dev/null | grep -m1 card | cut -c1-60)"
  fi
fi

# --- 2. where does it point? -------------------------------------------------
if [ -z "$URL" ]; then
  ask URL '  The gustav daemon (e.g. http://box.local:8713): ' \
    || die "say --url http://your-server:8713"
fi
[ -n "$URL" ] || die "no server address"
case "$URL" in http://*|https://*) ;; *) URL="http://$URL" ;; esac

if [ -z "$TOKEN" ]; then
  warn "  No token. The daemon will refuse unless it has accounts switched off."
  warn "  On the server:  ai user add $NAME && ai user token $NAME"
  ask TOKEN '  Token (blank to carry on without one): ' || true
fi

# --- 3. the program ----------------------------------------------------------
say "  installing into $DIR"
if [ "$DRY_RUN" = true ]; then
  say "  would write $DIR/gustav_satellite.py, $CONFIG and the service"
else
  mkdir -p "$DIR" "$BIN_DIR" "$(dirname "$CONFIG")"
  if [ -n "$SOURCE" ]; then
    cp "$SOURCE/satellite/gustav_satellite.py" "$DIR/gustav_satellite.py"
  else
    URL_RAW="https://raw.githubusercontent.com/$REPO_SLUG/$REF/satellite/gustav_satellite.py"
    curl -fsSL "$URL_RAW" -o "$DIR/gustav_satellite.py" \
      || die "could not download $URL_RAW"
  fi
  chmod +x "$DIR/gustav_satellite.py"

  cat > "$BIN_DIR/gustav-satellite" <<EOF
#!/bin/sh
exec "$PYTHON" "$DIR/gustav_satellite.py" "\$@"
EOF
  chmod +x "$BIN_DIR/gustav-satellite"
  say "  $BIN_DIR/gustav-satellite"
fi

# --- 4. the config -----------------------------------------------------------
if [ "$DRY_RUN" != true ]; then
  if [ -f "$CONFIG" ]; then
    say "  keeping your $CONFIG"
  else
    {
      echo "# gustav satellite — $NAME"
      echo "# Everything else (language, wake words, how long a question may be) comes"
      echo "# from the daemon, so there is nothing here to get out of step."
      echo "url = $URL"
      echo "token = $TOKEN"
      [ -n "$INPUT" ]  && echo "input_device = $INPUT"
      [ -n "$OUTPUT" ] && echo "output_device = $OUTPUT"
      echo "# Loudness that counts as speech, 0..1. Measure it where this will live:"
      echo "#   gustav-satellite --calibrate"
      echo "threshold = ${THRESHOLD:-0.02}"
      echo "# Quiet this long ends a question. Lower feels snappier and cuts people off."
      echo "silence_ms = 700"
      echo "# Audio kept from before speech was detected, so the first word survives."
      echo "preroll_ms = 240"
    } > "$CONFIG"
    chmod 600 "$CONFIG"          # it has a token in it
    say "  wrote $CONFIG"
  fi
fi

# --- 5. does it work? --------------------------------------------------------
if [ "$DRY_RUN" != true ]; then
  say "  asking $URL what it sounds like"
  if "$PYTHON" - "$URL" "$TOKEN" <<'CHECK'
import json, sys, urllib.request
url, token = sys.argv[1].rstrip("/"), sys.argv[2]
request = urllib.request.Request(url + "/voice/config")
if token:
    request.add_header("Authorization", "Bearer " + token)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        found = json.loads(response.read().decode())
except Exception as exc:
    print(f"    could not reach it: {exc}")
    raise SystemExit(1)
print(f"    {found.get('instance', 'gustav')}: language {found.get('language')}, "
      f"{found['stt']['sample_rate']} Hz")
for part in ("stt", "tts", "socket"):
    state = found.get(part) or {}
    if state and not state.get("ok", True):
        print(f"    {part}: {state.get('detail')}")
raise SystemExit(0)
CHECK
  then
    say "  the daemon is there"
  else
    warn "  The daemon did not answer. The satellite will keep trying once it runs;"
    warn "  check the url and the token in $CONFIG."
    warn "  (voice.enabled must be true in the daemon's config.yaml.)"
  fi
fi

# --- 6. always on ------------------------------------------------------------
if [ -z "$WITH_SERVICE" ]; then
  if [ -n "$TTY" ]; then
    ask answer '  Start it on boot (systemd user service)? [Y/n]: '
    case "$answer" in [nN]*) WITH_SERVICE=no ;; *) WITH_SERVICE=yes ;; esac
  else
    WITH_SERVICE=yes
  fi
fi

if [ "$WITH_SERVICE" = yes ] && [ "$DRY_RUN" != true ]; then
  if ! command -v systemctl >/dev/null 2>&1; then
    warn "  no systemd here; start it yourself with: gustav-satellite"
  else
    UNIT_DIR="$HOME/.config/systemd/user"
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/gustav-satellite.service" <<EOF
[Unit]
Description=gustav satellite ($NAME)
After=network-online.target sound.target

[Service]
Type=simple
ExecStart=$BIN_DIR/gustav-satellite
Restart=always
RestartSec=3
# It is a microphone in a room: nothing it does needs anything of yours.
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now gustav-satellite >/dev/null 2>&1 || true
    say "  gustav-satellite.service is running"
    # Without this it stops when you log out, which on a headless Pi is immediately.
    if command -v loginctl >/dev/null 2>&1; then
      loginctl enable-linger "$(id -un)" >/dev/null 2>&1 \
        || warn "  could not enable lingering: it will stop when you log out"
    fi
  fi
fi

say ""
say "Done."
say "  gustav-satellite --calibrate      measure this room, then put the number in"
say "                                    $CONFIG"
say "  gustav-satellite --say \"hello\"     make the daemon talk through this speaker"
say "  journalctl --user -u gustav-satellite -f"
