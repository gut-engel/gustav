#!/usr/bin/env bash
#
# Install the current NVIDIA driver (and, if you ask, the CUDA toolkit) on this machine.
#
#   curl -fsSL https://raw.githubusercontent.com/gut-engel/gustav/main/scripts/driver.sh | sudo bash
#
# This is the one script in gustav that asks for root, and it is separate from the
# installer for that reason alone: the installer puts a daemon in your home directory
# and never needs more than you, while a kernel module needs the kernel. Nothing here
# touches gustav — no checkout, no virtualenv, no weights, no config. It installs a
# driver and gets out of the way.
#
# It is the other half of the error that sends people here:
#
#   RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
#
# which means torch was built against a newer CUDA than this driver can run. The cheap
# cure is a torch that matches the driver — `ai doctor --fix-torch`, no root, no reboot.
# This is the other cure, the one that leaves the machine current instead of pinned.
#
# What it does per distribution is what that distribution's own maintainers recommend,
# not a single trick repeated five times:
#
#   Debian/Ubuntu  NVIDIA's CUDA repository + cuda-drivers, falling back to the
#                  distribution's own driver when the repository has nothing for this
#                  release
#   Fedora         RPM Fusion's akmod-nvidia, which rebuilds on every kernel
#   RHEL and kin   NVIDIA's CUDA repository + cuda-drivers, module stream disabled
#   Arch and kin   nvidia-open-dkms (or nvidia-dkms on pre-Turing cards) + nvidia-utils
#   openSUSE       NVIDIA's openSUSE repository + the G06 driver
#   anything else  NVIDIA's own .run installer, newest version, with --dkms
#
# And what it refuses to do, which matters just as much: there is no NVIDIA driver for
# macOS and has not been since 10.13; a WSL2 Linux uses the Windows host's driver and
# installing one in here breaks the one that works; a container shares the host's
# kernel, so it shares the host's driver. In each case it says so and stops.
#
# Everything goes into a log: every command, every line of output, how long it took.
set -euo pipefail

exec 3>&1 4>&2

VERSION="2026-11-17.1"

ARGV=("$@")
TOOLKIT=false
ASSUME_YES=false
DRY_RUN=false
QUIET=false
FORCE=false
REBOOT_NEEDED=false
FAMILY=""
PRETTY=""
DISTRO=""
DISTRO_LIKE=""
VERSION_ID=""
ARCH="$(uname -m)"
REPO_DISTRO=""
IS_WSL=false
CONTAINER=""
SUDO=""

LOG=""
LOG_READY=false
GRACEFUL=false
STEP=0
STEP_TOTAL=5

# Two hooks, both only for the tests: one machine cannot be five distributions, and the
# per-distribution choices are the whole content of this script, so they have to be
# provokable without owning a server farm.
OS_RELEASE="${GUSTAV_OS_RELEASE:-/etc/os-release}"
FAKE_SYSTEM="${GUSTAV_FAKE_SYSTEM:-}"     # Linux | Darwin | Windows
FAKE_KERNEL="${GUSTAV_FAKE_KERNEL:-}"     # stands in for `uname -r` (WSL detection)

now()  { date '+%Y-%m-%d %H:%M:%S'; }
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
die()  { GRACEFUL=true; log "error: $*"; printf '\033[31merror:\033[0m %s\n' "$*" >&4; exit 1; }
fail() { log "error: $*"; printf '\033[31merror:\033[0m %s\n' "$*" >&4; exit 1; }

fmt_secs() { printf '%d:%02d' $(($1 / 60)) $(($1 % 60)); }

TTY=""
if [ -t 0 ]; then
  TTY="/dev/stdin"
elif { : < /dev/tty; } 2>/dev/null; then
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

run() {
  log "\$ $*"
  if [ "$DRY_RUN" = true ]; then
    printf '  would run: %s\n' "$*" >&3
    return 0
  fi
  "$@" < /dev/null
}

# A package manager downloading a driver is minutes of silence otherwise.
run_long() {  # run_long <what> <command...>
  local what="$1"; shift
  log "\$ $*"
  if [ "$DRY_RUN" = true ]; then
    printf '  would run: %s\n' "$*" >&3
    return 0
  fi
  local start=$SECONDS rc=0 pid tick=0 every=5
  [ -t 3 ] || every=30
  "$@" < /dev/null &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    tick=$((tick + 1))
    if [ $((tick % every)) -eq 0 ] && [ "$QUIET" != true ]; then
      if [ -t 3 ]; then
        printf '\r  \033[2m[%s] %-60.60s\033[0m' "$(fmt_secs $((SECONDS - start)))" "$what" >&3
      fi
    fi
  done
  wait "$pid" || rc=$?
  if [ "$QUIET" != true ] && [ -t 3 ]; then printf '\r%*s\r' 74 '' >&3; fi
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
    printf 'Your driver has not been removed: a failed install leaves what was working.\n' >&4
  else
    printf 'The whole log is in %s\n' "$LOG" >&4
  fi
  exit "$rc"
}

usage() {
  cat <<EOF
Usage: sudo bash driver.sh [options]

Installs the current NVIDIA driver using whatever this distribution's maintainers
recommend, and says what it is doing while it does it.

  --toolkit           also install the full CUDA toolkit (nvcc, headers: ~4 GiB).
                      Off by default — every torch wheel carries its own CUDA
                      runtime, so nothing gustav does needs the compiler.
  --open / --closed   on Arch and kin: the open kernel module (Turing and newer,
                      the default when the card supports it) or the proprietary one
  --yes               do not ask before changing anything
  --force             carry on even where this script would normally refuse
  --log <path>        where to write the log (default: $logdefault)
  --version           print the version of this script and stop
  --dry-run           print every command it would run, change nothing
  --quiet             only errors (the log is written in full either way)
  -h, --help          this

After it finishes, reboot. The running kernel keeps the module it booted with, so
nvidia-smi goes on reporting the old driver until you do.
EOF
}

LOG_DIR="${XDG_STATE_HOME:-${HOME:-/root}/.local/state}/gustav"
logdefault="$LOG_DIR/driver-<date>.log"
LOG="${GUSTAV_LOG:-}"
MODULE_FLAVOUR=""     # open | closed, Arch only; empty means decide from the card

while [ $# -gt 0 ]; do
  case "$1" in
    --toolkit)    TOOLKIT=true; shift ;;
    --open)       MODULE_FLAVOUR=open; shift ;;
    --closed)     MODULE_FLAVOUR=closed; shift ;;
    --yes|-y)     ASSUME_YES=true; shift ;;
    --force)      FORCE=true; shift ;;
    --log)        LOG="${2:-}"; shift 2 ;;
    --dry-run)    DRY_RUN=true; shift ;;
    --quiet)      QUIET=true; shift ;;
    --version)    printf 'driver.sh %s\n' "$VERSION"; exit 0 ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown option: $1 (try --help)" ;;
  esac
done

# --- the log -----------------------------------------------------------------
if [ -z "$LOG" ]; then
  if mkdir -p "$LOG_DIR" 2>/dev/null; then
    LOG="$LOG_DIR/driver-$(date +%Y%m%d-%H%M%S).log"
  else
    LOG="${TMPDIR:-/tmp}/gustav-driver-$(date +%Y%m%d-%H%M%S).log"
  fi
else
  mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
fi
: > "$LOG" || die "cannot write the log at $LOG (pass --log somewhere writable)"
exec >>"$LOG" 2>&1
LOG_READY=true
trap on_exit EXIT

log "driver.sh $VERSION"
log "  argv:    ${ARGV[*]:-}"
log "  user:    $(id -un) (uid $(id -u))"
log "  machine: $(uname -n) — $(uname -srm)"
log "  bash:    $BASH_VERSION"
log "  dry-run: $DRY_RUN   toolkit: $TOOLKIT   terminal: ${TTY:-none}"

if [ "$QUIET" != true ]; then
  printf 'gustav driver installer %s — writing all of this to %s\n' "$VERSION" "$LOG" >&3
fi

if [ -d "$LOG_DIR" ]; then
  ls -1t "$LOG_DIR"/driver-*.log 2>/dev/null | tail -n +11 | while IFS= read -r old; do
    rm -f "$old" && log "pruned old log $old"
  done || true
fi

# --- 1. what is this machine ------------------------------------------------
step "Working out what this machine is"

SYSTEM="${FAKE_SYSTEM:-$(uname -s)}"
KERNEL="${FAKE_KERNEL:-$(uname -r)}"

case "$SYSTEM" in
  Darwin)
    GRACEFUL=true
    say "  macOS"
    die "there is no NVIDIA driver for macOS.

       Apple stopped shipping them after 10.13 and NVIDIA stopped building them; no
       version of CUDA runs on Apple silicon at all. There is nothing this script can
       install here.

       What works: install gustav on this Mac as a client, and point it at the Linux or
       Windows machine that has the GPU in it.

           curl -fsSL .../scripts/install.sh | bash -s -- --client --url https://box:8713"
    ;;
  Linux) ;;
  *)
    GRACEFUL=true
    die "this script is for Linux; '$SYSTEM' is not one.

       On Windows, install the driver from NVIDIA's own installer — 'winget install
       --id Nvidia.GeForceExperience', or the .exe from nvidia.com/Download — and run
       gustav's server side under WSL2, where that Windows driver is the one that counts."
    ;;
esac

if [ -r "$OS_RELEASE" ]; then
  # Read as data, not sourced: an /etc/os-release with a command substitution in it is
  # not something a root script should be evaluating.
  DISTRO="$(sed -n 's/^ID=//p' "$OS_RELEASE" | tr -d '"'"'" | head -1)"
  DISTRO_LIKE="$(sed -n 's/^ID_LIKE=//p' "$OS_RELEASE" | tr -d '"'"'" | head -1)"
  VERSION_ID="$(sed -n 's/^VERSION_ID=//p' "$OS_RELEASE" | tr -d '"'"'" | head -1)"
  PRETTY="$(sed -n 's/^PRETTY_NAME=//p' "$OS_RELEASE" | tr -d '"'"'" | head -1)"
fi
PRETTY="${PRETTY:-Linux $ARCH}"

family_of() {  # family_of <id> <id_like...>
  local candidate
  for candidate in "$@"; do
    case "$candidate" in
      debian|ubuntu|linuxmint|pop|elementary|raspbian|zorin|kali|devuan) echo debian; return ;;
      fedora|nobara)                                                     echo fedora; return ;;
      rhel|centos|rocky|almalinux|ol|oracle|amzn|scientific)             echo rhel;   return ;;
      arch|archarm|manjaro|endeavouros|garuda|artix|cachyos)             echo arch;   return ;;
      suse|opensuse|opensuse-leap|opensuse-tumbleweed|sles|sled)         echo suse;   return ;;
    esac
  done
  echo linux
}
# shellcheck disable=SC2086 - ID_LIKE is a space-separated list and is meant to split
FAMILY="$(family_of "$DISTRO" $DISTRO_LIKE)"

case "$KERNEL" in
  *microsoft*|*WSL*|*wsl*) IS_WSL=true ;;
esac
if [ -f /.dockerenv ]; then CONTAINER=docker
elif [ -f /run/.containerenv ]; then CONTAINER=podman
elif grep -qE '(docker|lxc|kubepods)' /proc/1/cgroup 2>/dev/null; then CONTAINER=container
fi

say "  system     $PRETTY ($ARCH)"
say "  family     $FAMILY${DISTRO:+  (ID=$DISTRO${DISTRO_LIKE:+, ID_LIKE=$DISTRO_LIKE})}"
say "  kernel     $KERNEL"

if [ "$IS_WSL" = true ] && [ "$FORCE" != true ]; then
  GRACEFUL=true
  die "this is WSL, where the driver is not yours to install.

       WSL2 reaches the GPU through /dev/dxg using the driver installed in Windows.
       Putting a Linux driver in here replaces libcuda.so with one that has no GPU
       behind it, which breaks the thing that currently works.

       Update it on the Windows side instead:

           winget install --id Nvidia.GeForceExperience      # in PowerShell
           wsl --shutdown                                    # then start this again

       Only the CUDA *toolkit* is ever installed inside WSL, and torch does not need it.
       (--force overrides this, and you will have to put it back.)"
fi

if [ -n "$CONTAINER" ] && [ "$FORCE" != true ]; then
  if [ "$DRY_RUN" = true ]; then
    warn "  this is a $CONTAINER container: a real run would refuse (the driver belongs"
    warn "  to the host, and a container shares the host's kernel)"
  else
    GRACEFUL=true
    die "this is a $CONTAINER container, which shares the host's kernel and therefore the
       host's driver. Install the driver on the host, then hand the GPU in with the
       NVIDIA container toolkit (docker run --gpus all). (--force overrides this.)"
  fi
fi

if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
    say "  root       not yet: every command that changes something goes through sudo"
  else
    die "this needs root and there is no sudo here. Run it as root:  su -c 'bash $0 ${ARGV[*]:-}'"
  fi
fi

# A machine with no NVIDIA card in it has nothing to gain and a display driver to lose.
if command -v lspci >/dev/null 2>&1; then
  if lspci -nn 2>/dev/null | grep -qiE 'vga|3d|display' ; then
    if lspci -nn 2>/dev/null | grep -iE 'vga|3d|display' | grep -qi nvidia; then
      say "  card       $(lspci 2>/dev/null | grep -iE 'vga|3d|display' | grep -i nvidia \
                           | head -1 | sed 's/^[^ ]* //')"
    elif [ "$FORCE" != true ] && [ "$DRY_RUN" != true ]; then
      GRACEFUL=true
      die "lspci finds no NVIDIA display controller on this machine.

       Installing an NVIDIA driver on a machine without an NVIDIA card can replace the
       driver that is currently painting your screen. If you know better than lspci —
       a card that is passed through, or one that is powered down — pass --force."
    else
      # A dry run says what it would do on the machine it is run on, and one of the
      # things it would do here is refuse. Saying so is more use than stopping.
      warn "  lspci finds no NVIDIA display controller: a real run would refuse here"
    fi
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  say "  driver     $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
                      | head -1 || echo 'installed but not answering')"
  log "nvidia-smi says:"
  nvidia-smi || log "  (nvidia-smi would not run)"
else
  say "  driver     none installed (no nvidia-smi)"
fi

# --- 2. what Secure Boot will make of it ------------------------------------
step "Checking what will stop the module from loading"

SECURE_BOOT=unknown
if command -v mokutil >/dev/null 2>&1; then
  case "$(mokutil --sb-state 2>/dev/null)" in
    *enabled*)  SECURE_BOOT=on ;;
    *disabled*) SECURE_BOOT=off ;;
  esac
elif [ -d /sys/firmware/efi ]; then
  for var in /sys/firmware/efi/efivars/SecureBoot-*; do
    [ -r "$var" ] || continue
    if [ "$(od -An -t u1 "$var" 2>/dev/null | tr -s ' ' | awk '{print $NF}')" = "1" ]; then
      SECURE_BOOT=on
    else
      SECURE_BOOT=off
    fi
    break
  done
fi
say "  secure boot  $SECURE_BOOT"

if [ "$SECURE_BOOT" = on ]; then
  warn "  Secure Boot is on. A module built on this machine — DKMS, akmods, the .run"
  warn "  installer — is unsigned, and the firmware will refuse to load it: the install"
  warn "  looks fine, and after the reboot nvidia-smi says 'no devices were found'."
  warn "  Either enrol a key (mokutil --import), or turn Secure Boot off in the"
  warn "  firmware, or stay on a distribution package that ships a signed module."
  if [ "$ASSUME_YES" != true ] && asking; then
    ask answer "  Carry on anyway? [y/N]: "
    case "$answer" in [yY]*) ;; *) die "stopped before building a module the firmware would refuse" ;; esac
  fi
fi

# --- 3. confirm -------------------------------------------------------------
step "What it is about to do"

say "  install the current NVIDIA driver the $FAMILY way"
if [ "$TOOLKIT" = true ]; then
  say "  and the full CUDA toolkit (nvcc and headers, about 4 GiB)"
else
  say "  not the CUDA toolkit: torch carries its own runtime (--toolkit if you want nvcc)"
fi
say "  nothing in gustav is touched: no checkout, no virtualenv, no weights, no config"

if [ "$DRY_RUN" != true ] && [ "$ASSUME_YES" != true ]; then
  if asking; then
    ask answer "  Go ahead? [y/N]: "
    case "$answer" in
      [yY]*) ;;
      *) die "nothing was changed" ;;
    esac
  else
    die "nothing is attached to confirm with. Pass --yes if you meant it, or --dry-run to look first."
  fi
fi

# --- 4. install -------------------------------------------------------------
step "Installing"

repo_distro() {
  # NVIDIA's name for a distribution in its CUDA repository paths: the ID and the
  # version with the dots taken out — ubuntu2404, debian12, rhel9.
  local base="$DISTRO" version="${VERSION_ID//./}"
  case "$FAMILY" in
    debian)
      case "$DISTRO $DISTRO_LIKE" in
        *ubuntu*) base=ubuntu ;;
        *)        base=debian; version="${version:0:2}" ;;
      esac
      ;;
    rhel)
      # Amazon Linux names itself by year (amzn2023); the rest of the family is
      # major-version only (rhel9), whatever the minor says.
      case "$DISTRO" in
        amzn*) base=amzn ;;
        *)     base=rhel; version="${version%%-*}"; version="${version:0:1}" ;;
      esac
      ;;
    fedora) base=fedora ;;
  esac
  printf '%s%s' "$base" "$version"
}
REPO_DISTRO="$(repo_distro)"
REPO_ARCH="$ARCH"
case "$ARCH" in aarch64|arm64) REPO_ARCH="sbsa" ;; esac
log "nvidia repository path would be $REPO_DISTRO/$REPO_ARCH"

install_debian() {
  local keyring="cuda-keyring_1.1-1_all.deb"
  local base="https://developer.download.nvidia.com/compute/cuda/repos/$REPO_DISTRO/$REPO_ARCH"
  export DEBIAN_FRONTEND=noninteractive
  run_long "apt-get update" $SUDO apt-get update || warn "  apt-get update was unhappy; carrying on"
  run_long "build tools and headers" $SUDO apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg "linux-headers-$(uname -r)" build-essential dkms \
    || warn "  could not install the headers for $(uname -r); DKMS may have nothing to build against"

  local tmp; tmp="$(mktemp -d)"
  say "  fetching NVIDIA's repository keyring for $REPO_DISTRO"
  if [ "$DRY_RUN" = true ]; then
    # The real run only learns here whether NVIDIA publishes a repository for this
    # release, so a dry run has to name both routes rather than pick one.
    say "  would run: curl -fsSL $base/$keyring -o $tmp/$keyring"
    say "  would run: sudo dpkg -i $keyring"
    say "  would run: sudo apt-get update"
    say "  would run: sudo apt-get install -y cuda-drivers$([ "$TOOLKIT" = true ] && printf ' cuda-toolkit')"
    say "  and if NVIDIA publishes nothing for $REPO_DISTRO, the distribution's own instead:"
    say "  would run: sudo ubuntu-drivers install   # or apt-get install nvidia-driver-cuda"
    rm -rf "$tmp" 2>/dev/null || true
    REBOOT_NEEDED=true
    return 0
  elif curl -fsSL "$base/$keyring" -o "$tmp/$keyring"; then
    run $SUDO dpkg -i "$tmp/$keyring" || fail "the keyring package would not install"
    run_long "apt-get update (with NVIDIA's repository)" $SUDO apt-get update \
      || fail "apt-get could not read NVIDIA's repository"
    local packages=(cuda-drivers)
    [ "$TOOLKIT" = true ] && packages+=(cuda-toolkit)
    if run_long "the driver" $SUDO apt-get install -y "${packages[@]}"; then
      REBOOT_NEEDED=true
      rm -rf "$tmp"
      return 0
    fi
    warn "  NVIDIA's repository has no driver for this release; using the distribution's"
  else
    warn "  NVIDIA publishes no CUDA repository for '$REPO_DISTRO'; using the distribution's"
  fi
  rm -rf "$tmp" 2>/dev/null || true

  # The fallback. Ubuntu knows which driver suits the card it can see; Debian has one
  # package that pulls the current one from its own archive.
  if command -v ubuntu-drivers >/dev/null 2>&1; then
    run_long "ubuntu-drivers install" $SUDO ubuntu-drivers install \
      || fail "ubuntu-drivers could not install a driver"
  else
    run_long "nvidia-driver from the distribution" $SUDO apt-get install -y \
      nvidia-driver-cuda nvidia-kernel-dkms \
      || fail "apt-get could not install a driver from this distribution either"
  fi
  REBOOT_NEEDED=true
}

install_fedora() {
  local release
  release="$(rpm -E %fedora 2>/dev/null || echo "${VERSION_ID}")"
  run_long "RPM Fusion" $SUDO dnf install -y \
    "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$release.noarch.rpm" \
    "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$release.noarch.rpm" \
    || warn "  RPM Fusion is already there, or would not install; carrying on"
  local packages=(akmod-nvidia xorg-x11-drv-nvidia-cuda kernel-devel)
  [ "$TOOLKIT" = true ] && packages+=(cuda-toolkit)
  run_long "the driver" $SUDO dnf install -y "${packages[@]}" \
    || fail "dnf could not install akmod-nvidia"
  # akmods builds the module here and now. Without this the first boot after the
  # install comes up with no module and an unexplained 'no devices were found'.
  run_long "building the module for this kernel" $SUDO akmods --force \
    || warn "  akmods did not finish; it will try again on the next boot"
  REBOOT_NEEDED=true
}

install_rhel() {
  local base="https://developer.download.nvidia.com/compute/cuda/repos/$REPO_DISTRO/$REPO_ARCH"
  run_long "build tools and headers" $SUDO dnf install -y dnf-plugins-core \
    "kernel-devel-$(uname -r)" "kernel-headers-$(uname -r)" \
    || warn "  could not install the headers for $(uname -r)"
  # dnf5 renamed the subcommand; dnf4 is what RHEL 9 and its rebuilds still ship. Try
  # the old spelling first and the new one after, because the old one is what most of
  # these machines have and a wrong guess here costs a whole repository.
  command -v dnf >/dev/null 2>&1 || [ "$DRY_RUN" = true ] \
    || fail "this looks like a RHEL-family machine but there is no dnf on it"
  run $SUDO dnf config-manager --add-repo "$base/cuda-$REPO_DISTRO.repo" \
    || run $SUDO dnf config-manager addrepo --from-repofile="$base/cuda-$REPO_DISTRO.repo" \
    || fail "could not add NVIDIA's repository for $REPO_DISTRO"
  # Without this the distribution's own module stream wins and nothing changes.
  run $SUDO dnf module disable -y nvidia-driver || true
  local packages=(cuda-drivers)
  [ "$TOOLKIT" = true ] && packages+=(cuda-toolkit)
  run_long "the driver" $SUDO dnf install -y "${packages[@]}" \
    || fail "dnf could not install cuda-drivers"
  REBOOT_NEEDED=true
}

install_arch() {
  # nvidia-open is NVIDIA's open kernel module, supported on Turing (GTX 16xx, RTX 20xx)
  # and newer. On anything older it builds and then finds no card it may drive, so the
  # card decides which package this is — and when the card cannot be read, the
  # proprietary module is the one that works everywhere.
  local flavour="$MODULE_FLAVOUR" capability=""
  if [ -z "$flavour" ]; then
    capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                   | head -1 || true)"
    log "compute capability from nvidia-smi: ${capability:-unknown}"
    case "${capability%%.*}" in
      ''|*[!0-9]*) flavour=closed ;;
      *) if [ "${capability%%.*}" -ge 8 ] 2>/dev/null; then flavour=open
         elif [ "${capability%%.*}" -eq 7 ] && [ "${capability##*.}" -ge 5 ] 2>/dev/null; then flavour=open
         else flavour=closed; fi ;;
    esac
  fi
  say "  kernel module: $flavour ${capability:+(compute capability $capability)}"

  local headers=(linux-headers)
  # DKMS builds against every kernel whose headers are here, and a machine with the LTS
  # kernel installed and no LTS headers boots into no module at all.
  for kernel in linux-lts linux-zen linux-hardened linux-rt; do
    if pacman -Qq "$kernel" >/dev/null 2>&1; then headers+=("$kernel-headers"); fi
  done
  run_long "kernel headers" $SUDO pacman -Sy --needed --noconfirm "${headers[@]}" \
    || fail "pacman could not install the kernel headers"

  local packages=(nvidia-utils)
  if [ "$flavour" = open ]; then packages+=(nvidia-open-dkms); else packages+=(nvidia-dkms); fi
  [ "$TOOLKIT" = true ] && packages+=(cuda)
  run_long "the driver" $SUDO pacman -S --needed --noconfirm "${packages[@]}" \
    || fail "pacman could not install the driver"
  REBOOT_NEEDED=true
}

install_suse() {
  local repo="https://download.nvidia.com/opensuse/${VERSION_ID:-tumbleweed}"
  run_long "kernel development files" $SUDO zypper install -y kernel-devel kernel-default-devel \
    || warn "  could not install the kernel development files"
  run $SUDO zypper addrepo --refresh "$repo" NVIDIA || true
  run $SUDO zypper --gpg-auto-import-keys refresh || warn "  could not refresh NVIDIA's repository"
  local packages=(nvidia-video-G06 nvidia-compute-utils-G06)
  [ "$TOOLKIT" = true ] && packages+=(cuda)
  run_long "the driver" $SUDO zypper install -y --auto-agree-with-licenses "${packages[@]}" \
    || fail "zypper could not install the G06 driver"
  REBOOT_NEEDED=true
}

install_runfile() {
  # No package manager we know carries it. NVIDIA publishes the newest version as one
  # line of text, so nothing here has a version hardcoded in it.
  local latest version url file
  say "  no packaged driver for '$DISTRO'; using NVIDIA's own installer"
  if [ "$DRY_RUN" = true ]; then
    say "  would read https://download.nvidia.com/XFree86/Linux-x86_64/latest.txt"
    say "  would run: sh NVIDIA-Linux-x86_64-<newest>.run --dkms --silent"
    return 0
  fi
  latest="$(curl -fsSL https://download.nvidia.com/XFree86/Linux-x86_64/latest.txt)" \
    || fail "could not reach download.nvidia.com to ask which version is current"
  version="$(printf '%s' "$latest" | awk '{print $1}')"
  [ -n "$version" ] || fail "download.nvidia.com answered something unexpected: $latest"
  file="NVIDIA-Linux-x86_64-$version.run"
  url="https://download.nvidia.com/XFree86/Linux-x86_64/$version/$file"
  say "  newest driver is $version"

  local tmp; tmp="$(mktemp -d)"
  run_long "downloading $file" curl -fsSL "$url" -o "$tmp/$file" \
    || fail "could not download $url"
  if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    warn "  a graphical session is running on this card. The .run installer cannot"
    warn "  replace a driver that is in use: switch to a text console (Ctrl+Alt+F3) or"
    warn "  'systemctl isolate multi-user.target' first, then run this again."
    die "stopped rather than failing halfway through replacing the running driver"
  fi
  run_long "the driver" $SUDO sh "$tmp/$file" --dkms --silent \
    || fail "NVIDIA's installer did not finish (its own log is /var/log/nvidia-installer.log)"
  rm -rf "$tmp"
  REBOOT_NEEDED=true
}

case "$FAMILY" in
  debian) install_debian ;;
  fedora) install_fedora ;;
  rhel)   install_rhel ;;
  arch)   install_arch ;;
  suse)   install_suse ;;
  *)      install_runfile ;;
esac

# --- 5. what it ended up with -----------------------------------------------
step "What is installed now"

if [ "$DRY_RUN" = true ]; then
  say "  nothing was changed, so there is nothing to check"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi after the install:"
    nvidia-smi || log "  (nvidia-smi will not run until the new module is loaded)"
    installed="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    if [ -n "$installed" ]; then
      say "  nvidia-smi reports driver $installed"
      say "  (that is the module the running kernel has; the new one arrives on reboot)"
    fi
  fi
  # What the kernel will actually load next time, which is the number that matters.
  if command -v modinfo >/dev/null 2>&1; then
    staged="$(modinfo -F version nvidia 2>/dev/null | head -1 || true)"
    [ -n "$staged" ] && say "  the module on disk is $staged — this is what boots next"
  fi
fi

say ""
if [ "$REBOOT_NEEDED" = true ] && [ "$DRY_RUN" != true ]; then
  say "Done. Reboot now: the running kernel keeps the module it booted with, so nothing"
  say "changes until you do."
  say ""
  say "After the reboot:"
  say "    nvidia-smi                 # the new driver, and the CUDA it runs"
  say "    ai doctor                  # whether gustav's torch matches it"
  say "    ai status                  # and whether the model loads now"
else
  say "Done."
fi
