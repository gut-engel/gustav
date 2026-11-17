"""The NVIDIA driver: what it is, what it is allowed to run, and how to replace it.

Everything in this file exists because of one sentence that arrives in the chat box
instead of an answer:

    RuntimeError: The NVIDIA driver on your system is too old (found version 12040)

That is torch saying it was built against a CUDA newer than the driver on this machine,
and there is nothing a chat box can do about it. It has exactly two cures, and this
module knows both:

* **Install a newer driver.** The real fix, root, one reboot. Different on every
  distribution, which is why :func:`driver_recipe` exists.
* **Install a torch the driver can already run.** No root, no reboot, four gigabytes of
  wheels. Which torch that is comes from :func:`pick_torch_channel`.

It is deliberately stdlib-only and imports nothing from the rest of gustav: the
installer runs ``python3 src/gustav/hw/driver.py`` on a machine where *nothing* has been
installed yet, and the daemon imports it on the path where torch has just failed. Both
have to work before anything is working.

Nothing here touches the machine. Reading ``nvidia-smi`` and ``/etc/os-release`` is as
far as it goes; the commands it chooses are run by ``scripts/driver.sh``, which is the
only thing in this project that asks for sudo. Every function that has an opinion takes
its facts as arguments, so the whole of it can be tested against a machine nobody owns.
"""

from __future__ import annotations

import argparse
import json
import os
import platform as _platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# what a driver has to be for a given CUDA
# ---------------------------------------------------------------------------

#: The oldest driver each CUDA major runs on, from NVIDIA's own compatibility table.
#: Only ever used to *say* something ("your driver is 550; CUDA 13 wants 580 or newer").
#: Nothing branches on these numbers, so a new CUDA major does not make this file wrong
#: — it only makes one sentence vaguer until the row is added.
CUDA_DRIVER_FLOOR: dict[int, dict[str, str]] = {
    11: {"Linux": "450.80.02", "Windows": "452.39"},
    12: {"Linux": "525.60.13", "Windows": "527.41"},
    13: {"Linux": "580.65.06", "Windows": "580.88"},
}

#: Where the index of CUDA-specific torch wheels lives. PyPI carries exactly one build
#: of torch — the newest — so a machine with an older driver has to be pointed here.
TORCH_INDEX = "https://download.pytorch.org/whl"

#: The newest driver NVIDIA has published for Linux, as plain text, so the fallback
#: ``.run`` path does not need a version hardcoded anywhere.
LATEST_DRIVER_TXT = "https://download.nvidia.com/XFree86/Linux-x86_64/latest.txt"


def decode_api_version(raw: int) -> tuple[int, int]:
    """Turn CUDA's packed driver version into ``(major, minor)``.

    ``12040`` is how the CUDA runtime writes 12.4: ``major * 1000 + minor * 10``. It is
    the number in torch's complaint, and the only place anyone ever meets it.

    >>> decode_api_version(12040)
    (12, 4)
    >>> decode_api_version(13000)
    (13, 0)
    """
    return raw // 1000, (raw % 1000) // 10


def parse_version(text: str) -> tuple[int, ...] | None:
    """``"12.4"`` -> ``(12, 4)``; anything unparseable -> ``None``."""
    found = re.findall(r"\d+", text or "")
    return tuple(int(n) for n in found) if found else None


def format_cuda(version: tuple[int, ...] | None) -> str:
    return ".".join(str(n) for n in version) if version else "unknown"


# ---------------------------------------------------------------------------
# reading the error
# ---------------------------------------------------------------------------

#: Every CUDA failure that is really a driver or a build problem, in the order a match
#: should be tried. The kinds are what :func:`diagnose` branches on.
_PATTERNS: tuple[tuple[str, str], ...] = (
    ("driver_too_old", r"driver on your system is too old \(found version (\d+)\)"),
    ("driver_too_old", r"CUDA driver version is insufficient for the CUDA runtime version"),
    ("not_compiled", r"Torch not compiled with CUDA enabled"),
    ("no_device", r"no CUDA-capable device is detected"),
    ("no_device", r"CUDA unknown error"),
    ("no_driver", r"cannot find (?:the )?(?:nvidia )?driver|libcuda\.so(?:\.1)? "
                  r"(?:is missing|cannot open|not found)"),
    ("wrong_arch", r"no kernel image is available for execution on the device"),
    ("no_torch", r"No module named ['\"]torch['\"]"),
    ("out_of_memory", r"CUDA out of memory"),
)


@dataclass(frozen=True)
class Problem:
    """What the error text actually said, once."""

    kind: str
    detail: str
    #: The CUDA the *driver* supports, when the error was kind enough to say.
    driver_cuda: tuple[int, int] | None = None
    raw_api: int | None = None

    @property
    def is_driver_mismatch(self) -> bool:
        return self.kind in ("driver_too_old", "no_driver")

    def headline(self) -> str:
        return {
            "driver_too_old": (
                "the NVIDIA driver on this machine is older than the CUDA the installed "
                "torch was built against"
            ),
            "not_compiled": "the installed torch has no CUDA support compiled into it",
            "no_device": "the driver is there but no GPU could be opened",
            "no_driver": "there is no NVIDIA driver loaded on this machine",
            "wrong_arch": (
                "this torch has no compiled kernels for this card's architecture"
            ),
            "no_torch": "torch is not installed in this environment",
            "out_of_memory": "the card ran out of memory, which is not a driver problem",
        }.get(self.kind, self.detail or "the model would not load")


def parse_cuda_error(text: str | None) -> Problem | None:
    """Recognise a driver or build failure in whatever torch threw.

    Returns ``None`` for anything this module has no advice about — a missing
    checkpoint, a bad config — because a wrong diagnosis is worse than none.
    """
    if not text:
        return None
    for kind, pattern in _PATTERNS:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        raw = int(match.group(1)) if match.groups() and match.group(1) else None
        return Problem(
            kind=kind,
            detail=text.strip().splitlines()[0],
            driver_cuda=decode_api_version(raw) if raw else None,
            raw_api=raw,
        )
    return None


# ---------------------------------------------------------------------------
# which machine is this
# ---------------------------------------------------------------------------

#: os-release ``ID``/``ID_LIKE`` to the family whose package manager we would use.
_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("debian", ("debian", "ubuntu", "linuxmint", "pop", "elementary", "raspbian",
                "zorin", "kali", "devuan")),
    ("fedora", ("fedora", "nobara")),
    ("rhel", ("rhel", "centos", "rocky", "almalinux", "ol", "oracle", "amzn",
              "scientific")),
    ("arch", ("arch", "archarm", "manjaro", "endeavouros", "garuda", "artix", "cachyos")),
    ("suse", ("suse", "opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles", "sled")),
    ("gentoo", ("gentoo",)),
    ("alpine", ("alpine",)),
    ("nixos", ("nixos",)),
)


@dataclass(frozen=True)
class Platform:
    """Enough about the operating system to choose a package manager, or to refuse."""

    system: str = "Linux"            # Linux | Darwin | Windows
    distro: str = ""                 # os-release ID
    like: tuple[str, ...] = ()       # os-release ID_LIKE, split
    version_id: str = ""
    pretty: str = ""
    family: str = "linux"            # debian | fedora | rhel | arch | suse | ... | macos
    arch: str = "x86_64"
    wsl: bool = False
    container: str = ""              # docker | podman | lxc | ""
    secure_boot: bool | None = None

    def describe(self) -> str:
        bits = [self.pretty or f"{self.system} {self.arch}"]
        if self.wsl:
            bits.append("under WSL")
        if self.container:
            bits.append(f"in a {self.container} container")
        if self.secure_boot:
            bits.append("Secure Boot on")
        return ", ".join(bits)


def _split_like(value: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"[\s,]+", value.strip()) if part)


def family_of(distro: str, like: tuple[str, ...] = ()) -> str:
    """Which package-manager family a distribution belongs to.

    ``ID`` first, then ``ID_LIKE``: a derivative nobody has heard of still says what it
    is derived from, and that is the whole point of the field.
    """
    for candidate in (distro.lower(), *(item.lower() for item in like)):
        if not candidate:
            continue
        for family, members in _FAMILIES:
            if candidate in members:
                return family
    return "linux"


def parse_os_release(text: str) -> dict[str, str]:
    """``/etc/os-release`` as a dict, without sourcing it in a shell."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        found[key.strip()] = value.strip().strip('"').strip("'")
    return found


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _secure_boot() -> bool | None:
    """Whether Secure Boot is on, which decides if an unsigned module can load."""
    mokutil = shutil.which("mokutil")
    if mokutil:
        try:
            said = subprocess.run(
                [mokutil, "--sb-state"], capture_output=True, text=True, timeout=5
            ).stdout.lower()
        except (OSError, subprocess.SubprocessError):
            said = ""
        if "enabled" in said:
            return True
        if "disabled" in said:
            return False
    # The EFI variable is the same answer without a helper installed: the last byte of
    # SecureBoot-8be4df61… is 1 when it is on.
    for name in Path("/sys/firmware/efi/efivars").glob("SecureBoot-*"):
        try:
            data = name.read_bytes()
        except OSError:
            continue
        if data:
            return data[-1] == 1
    return None


def _container() -> str:
    if Path("/.dockerenv").exists():
        return "docker"
    if Path("/run/.containerenv").exists():
        return "podman"
    cgroup = _read("/proc/1/cgroup")
    for needle in ("docker", "lxc", "kubepods"):
        if needle in cgroup:
            return "lxc" if needle == "lxc" else "docker"
    if os.environ.get("container"):
        return os.environ["container"]
    return ""


def detect_platform(
    *,
    system: str | None = None,
    os_release: str | None = None,
    kernel: str | None = None,
    arch: str | None = None,
    container: str | None = None,
    secure_boot: bool | None = None,
) -> Platform:
    """What this machine is. Every fact can be supplied instead, which is how it is tested."""
    system = system or _platform.system() or "Linux"
    arch = arch or _platform.machine() or "x86_64"
    if system == "Darwin":
        return Platform(
            system="Darwin", family="macos", arch=arch,
            pretty=f"macOS {_platform.mac_ver()[0] or ''}".strip(),
        )
    if system == "Windows":
        return Platform(
            system="Windows", family="windows", arch=arch,
            pretty=f"Windows {_platform.release()}",
        )

    if os_release is None:
        os_release = _read(os.environ.get("GUSTAV_OS_RELEASE") or "/etc/os-release")
    fields = parse_os_release(os_release)
    distro = fields.get("ID", "")
    like = _split_like(fields.get("ID_LIKE", ""))
    if kernel is None:
        kernel = f"{_platform.release()} {_read('/proc/version')}"
    return Platform(
        system="Linux",
        distro=distro,
        like=like,
        version_id=fields.get("VERSION_ID", ""),
        pretty=fields.get("PRETTY_NAME", "") or f"Linux {arch}",
        family=family_of(distro, like),
        arch=arch,
        # Both WSL kernels say so in their release string; WSL1 has no /dev/nvidia at
        # all and WSL2 gets the driver from Windows. Either way we must not install one.
        wsl=bool(re.search(r"microsoft|wsl", kernel or "", re.I)),
        container=_container() if container is None else container,
        secure_boot=_secure_boot() if secure_boot is None else secure_boot,
    )


# ---------------------------------------------------------------------------
# how this machine installs a driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """The commands that put the current driver on this particular machine.

    ``scripts/driver.sh`` runs the same commands for the same families — it has to, it
    is the thing that works when nothing is installed yet. This copy is what ``ai
    doctor`` prints, so that the manual route is on the screen next to the automatic
    one, and a test holds the two to each other.
    """

    family: str
    summary: str
    steps: tuple[str, ...] = ()
    supported: bool = True
    reboot: bool = True
    notes: tuple[str, ...] = ()

    #: The binary a test can look for in driver.sh to know both know the same family.
    @property
    def tool(self) -> str:
        return {
            "debian": "apt-get", "fedora": "dnf", "rhel": "dnf",
            "arch": "pacman", "suse": "zypper",
        }.get(self.family, "")


def _cuda_repo_distro(platform: Platform) -> str:
    """NVIDIA's name for this distribution in its CUDA repository paths.

    ``ubuntu2404``, ``debian12``, ``rhel9``, ``fedora41`` — version with the dots taken
    out, which is the convention the repository has used since CUDA 10.
    """
    version = (platform.version_id or "").replace(".", "")
    base = {"fedora": "fedora", "rhel": "rhel"}.get(platform.family, platform.distro)
    if platform.family == "debian":
        base = "ubuntu" if "ubuntu" in (platform.distro, *platform.like) else "debian"
        if base == "debian":
            version = version[:2]
    if platform.family == "rhel":
        # Amazon Linux names itself by year (amzn2023); everything else in this family
        # is major-version only (rhel9), whatever the minor says.
        if platform.distro.startswith("amzn"):
            base = "amzn"
        else:
            base, version = "rhel", (version.split("-")[0][:1] or version)
    return f"{base}{version}"


def driver_recipe(platform: Platform, *, toolkit: bool = False) -> Recipe:
    """What to run on this machine to end up on the current driver.

    The driver is what matters: it is the half that cannot be fixed from a virtualenv.
    The CUDA *toolkit* — ``nvcc`` and the headers — is four more gigabytes that torch
    does not need, because every torch wheel carries its own CUDA runtime. So it is off
    unless it was asked for.
    """
    arch = "sbsa" if platform.arch in ("aarch64", "arm64") else platform.arch
    repo = _cuda_repo_distro(platform)

    if platform.system == "Darwin":
        return Recipe(
            "macos", "macOS has no NVIDIA driver to install", supported=False, reboot=False,
            notes=(
                "Apple stopped shipping NVIDIA drivers after macOS 10.13 and NVIDIA "
                "stopped building them; no version of CUDA runs on Apple silicon.",
                "Install gustav here as a client (--client) and point it at a Linux or "
                "Windows box with the GPU in it.",
            ),
        )
    if platform.system == "Windows":
        return Recipe(
            "windows", "install the driver from NVIDIA's installer", reboot=True,
            steps=(
                "winget install --id Nvidia.GeForceExperience --accept-package-agreements",
            ),
            notes=(
                "Or the .exe from https://www.nvidia.com/Download/index.aspx — pick your "
                "card, take the Studio or Game Ready driver, choose a clean install.",
                "gustav's server side is built for Linux; on Windows run it under WSL2, "
                "where the Windows driver is the one that counts.",
            ),
        )
    if platform.wsl:
        return Recipe(
            "wsl", "the driver belongs to Windows, not to this Linux", supported=False,
            reboot=False,
            steps=(
                "# in Windows PowerShell, not here:",
                "winget install --id Nvidia.GeForceExperience",
            ),
            notes=(
                "WSL2 uses the Windows host's driver through /dev/dxg. Installing a "
                "Linux driver in here replaces libcuda.so with one that has no GPU "
                "behind it, and that breaks what does work.",
                "Update the driver in Windows, restart WSL (`wsl --shutdown`), and this "
                "side picks it up. Only the CUDA *toolkit* is ever installed in here.",
            ),
        )
    if platform.container:
        return Recipe(
            platform.family,
            f"this is a {platform.container} container: the driver lives on the host",
            supported=False, reboot=False,
            notes=(
                "A container shares the host's kernel, so it shares the host's driver. "
                "Install the driver on the host and pass the GPU in with the NVIDIA "
                "container toolkit (`--gpus all`).",
            ),
        )

    if platform.family == "debian":
        return Recipe(
            "debian", "NVIDIA's CUDA repository, then the cuda-drivers meta-package",
            steps=(
                "sudo apt-get update",
                "sudo apt-get install -y curl ca-certificates "
                "linux-headers-$(uname -r) build-essential",
                f"curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/"
                f"{repo}/{arch}/cuda-keyring_1.1-1_all.deb",
                "sudo dpkg -i cuda-keyring_1.1-1_all.deb",
                "sudo apt-get update",
                "sudo apt-get install -y cuda-drivers"
                + (" cuda-toolkit" if toolkit else ""),
            ),
            notes=(
                "If the repository has nothing for this release, `sudo ubuntu-drivers "
                "install` (Ubuntu) or `sudo apt-get install -y nvidia-driver-cuda` "
                "(Debian) installs the distribution's own, which is usually a release "
                "or two behind.",
            ),
        )
    if platform.family == "fedora":
        return Recipe(
            "fedora", "RPM Fusion's akmod, which rebuilds itself on every kernel",
            steps=(
                "sudo dnf install -y "
                "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$("
                "rpm -E %fedora).noarch.rpm "
                "https://mirrors.rpmfusion.org/nonfree/fedora/"
                "rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm",
                "sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda"
                + (" cuda-toolkit" if toolkit else ""),
                "sudo akmods --force",
            ),
            notes=(
                "The module is built on this machine, so give it a few minutes after "
                "the install finishes and before the reboot: `modinfo -F version nvidia` "
                "answers once it is ready.",
                "NVIDIA's own CUDA repository is usually a Fedora release behind, which "
                "is why this takes RPM Fusion instead.",
            ),
        )
    if platform.family == "rhel":
        return Recipe(
            "rhel", "NVIDIA's CUDA repository, then the cuda-drivers meta-package",
            steps=(
                "sudo dnf install -y dnf-plugins-core epel-release kernel-devel-$(uname -r) "
                "kernel-headers-$(uname -r)",
                f"sudo dnf config-manager --add-repo https://developer.download.nvidia.com/"
                f"compute/cuda/repos/{repo}/{arch}/cuda-{repo}.repo",
                "sudo dnf module disable -y nvidia-driver || true",
                "sudo dnf install -y cuda-drivers" + (" cuda-toolkit" if toolkit else ""),
            ),
            notes=(
                "`dnf module disable nvidia-driver` is what lets the repository's driver "
                "win over the one in the distribution's module stream; without it the "
                "install succeeds and nothing changes.",
            ),
        )
    if platform.family == "arch":
        return Recipe(
            "arch", "the distribution's own packages, which are always current",
            steps=(
                "sudo pacman -Syu --needed --noconfirm linux-headers",
                "sudo pacman -S --needed --noconfirm nvidia-open-dkms nvidia-utils"
                + (" cuda" if toolkit else ""),
            ),
            notes=(
                "nvidia-open-dkms is the open kernel module, which NVIDIA supports on "
                "Turing (GTX 16xx, RTX 20xx) and newer. On anything older, install "
                "`nvidia-dkms` instead — the same driver, the proprietary module.",
                "Install the headers for every kernel you boot (linux-lts-headers, "
                "linux-zen-headers), or DKMS silently has nothing to build against.",
            ),
        )
    if platform.family == "suse":
        return Recipe(
            "suse", "NVIDIA's openSUSE repository, then the G06 driver",
            steps=(
                "sudo zypper install -y kernel-devel kernel-default-devel",
                "sudo zypper addrepo --refresh "
                "https://download.nvidia.com/opensuse/$(. /etc/os-release; "
                'echo "${VERSION_ID:-tumbleweed}") NVIDIA || true',
                "sudo zypper --gpg-auto-import-keys refresh",
                "sudo zypper install -y --auto-agree-with-licenses nvidia-video-G06 "
                "nvidia-compute-utils-G06" + (" cuda" if toolkit else ""),
            ),
        )
    return Recipe(
        platform.family or "linux",
        "no package manager here that is known to carry the driver: NVIDIA's own installer",
        steps=(
            f"curl -fsSL {LATEST_DRIVER_TXT}",
            "# that prints '<version> Linux-x86_64/<version>/NVIDIA-Linux-x86_64-<version>.run'",
            "curl -fsSLO https://download.nvidia.com/XFree86/Linux-x86_64/<version>/"
            "NVIDIA-Linux-x86_64-<version>.run",
            "sudo sh NVIDIA-Linux-x86_64-<version>.run --dkms",
        ),
        notes=(
            "The .run installer needs the kernel headers for the running kernel, a "
            "compiler, and no X or Wayland session on the card — run it from a text "
            "console (Ctrl+Alt+F3) or after `systemctl isolate multi-user.target`.",
            "It is also the one route that your package manager will not know about "
            "afterwards, so it has to be re-run after every kernel update unless --dkms "
            "takes.",
        ),
    )


def secure_boot_warning(platform: Platform) -> str | None:
    """Secure Boot refuses an unsigned module, and the symptom is a GPU that vanished."""
    if not platform.secure_boot or platform.system != "Linux":
        return None
    return (
        "Secure Boot is on. A kernel module built on this machine (DKMS, akmods, the "
        ".run installer) is unsigned, and the firmware will refuse to load it: the "
        "install will look fine and nvidia-smi will say 'no devices were found' after "
        "the reboot. Either enrol a signing key with mokutil, or turn Secure Boot off "
        "in the firmware, or use a distribution package that ships a signed module."
    )


# ---------------------------------------------------------------------------
# which torch this driver may run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TorchPick:
    channel: str                       # cu124, cu128, …
    version: tuple[int, ...]
    #: True when the channel's CUDA is newer than the driver's. It works by CUDA's
    #: minor-version compatibility, and that is a promise, not a guarantee.
    above_driver: bool = False
    satisfies_floor: bool = True

    @property
    def index_url(self) -> str:
        return f"{TORCH_INDEX}/{self.channel}"

    def describe(self) -> str:
        return f"torch {format_cuda(self.version)} from {self.channel}"


def _fetch(url: str, timeout: float = 15.0) -> str | None:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - a channel that was never built, or no network
        return None


def channel_versions(channel: str, fetch: Callable[[str], str | None] = _fetch) -> list[tuple[int, ...]]:
    """Every torch version published for one CUDA channel, oldest first."""
    page = fetch(f"{TORCH_INDEX}/{channel}/torch/")
    if not page:
        return []
    found = {
        tuple(int(n) for n in version.split("."))
        for version in re.findall(
            r"torch-(\d+\.\d+\.\d+)(?:%2B|\+)" + re.escape(channel) + "-", page
        )
    }
    return sorted(found)


def pick_torch_channel(
    driver_cuda: tuple[int, ...],
    floor: tuple[int, ...] = (),
    fetch: Callable[[str], str | None] = _fetch,
) -> TorchPick | None:
    """The best torch this driver can run, and whether it is new enough for gustav.

    Three passes, in order of how safe they are:

    1. the driver's own CUDA major, at or below its minor — always runs;
    2. the same major above the driver's minor — runs by CUDA's minor-version
       compatibility, which holds for everything torch does but is not a guarantee;
    3. the major below — older, always runs, and the last chance for a machine whose
       driver is a whole generation behind.

    ``None`` means download.pytorch.org has nothing for this driver at all, which is
    different from having something too old: that comes back with ``satisfies_floor``
    false, so the caller can say what *would* work.
    """
    if not driver_cuda:
        return None
    major, minor = driver_cuda[0], (driver_cuda[1] if len(driver_cuda) > 1 else 0)
    best: TorchPick | None = None
    passes: tuple[tuple[int, range, bool], ...] = (
        (major, range(minor, -1, -1), False),
        (major, range(9, minor, -1), True),
        (major - 1, range(9, -1, -1), False),
    )
    for channel_major, minors, above in passes:
        if channel_major < 11:       # nothing older than cu11x was ever on this index
            continue
        for channel_minor in minors:
            channel = f"cu{channel_major}{channel_minor}"
            have = channel_versions(channel, fetch)
            if not have:
                continue
            pick = TorchPick(channel, have[-1], above_driver=above)
            if best is None or have[-1] > best.version:
                best = pick
            if not floor or have[-1] >= tuple(floor):
                return pick
    if best is not None:
        return replace(best, satisfies_floor=False)
    return None


# ---------------------------------------------------------------------------
# what is true on this machine right now
# ---------------------------------------------------------------------------


@dataclass
class Facts:
    """Everything the diagnosis is built from. Supply it yourself and nothing is read."""

    platform: Platform = field(default_factory=Platform)
    #: From nvidia-smi: the driver's own version, and the newest CUDA it can run.
    driver_version: str = ""
    driver_cuda: tuple[int, ...] | None = None
    gpu_names: tuple[str, ...] = ()
    smi_present: bool = False
    #: From torch, if it is installed and would import.
    torch_version: str = ""
    torch_cuda: tuple[int, ...] | None = None
    cuda_available: bool | None = None
    torch_error: str = ""
    #: What the daemon last said when it tried to load the model.
    load_error: str = ""
    torch_floor: tuple[int, ...] = ()
    #: False when nothing about *this* machine was read — explaining an error that
    #: came from somewhere else. Then "no nvidia-smi here" is not a finding.
    inspected: bool = True

    @property
    def problem(self) -> Problem | None:
        return parse_cuda_error(self.load_error) or parse_cuda_error(self.torch_error)


def read_smi(run: Callable[[list[str]], str | None] | None = None) -> dict[str, object]:
    """Ask nvidia-smi for the driver version, its CUDA, and the cards.

    nvidia-smi comes with the driver, so its absence is itself the answer to most of
    what this module is for.
    """
    def _run(argv: list[str]) -> str | None:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    runner = run or _run
    if not shutil.which("nvidia-smi") and run is None:
        return {"smi_present": False}
    query = runner(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    # `--version` is the stable place to ask, and the only one that survived the
    # renaming: the driver that deprecated "CUDA Version" in favour of "CUDA UMD
    # Version" is already out, and the old spelling now carries the word Deprecated
    # where the number used to be.
    versions = runner(["nvidia-smi", "--version"]) or ""
    header = runner(["nvidia-smi"]) or ""
    if query is None and not versions and not header:
        return {"smi_present": False}
    names: list[str] = []
    driver = ""
    for line in (query or "").splitlines():
        if not line.strip():
            continue
        name, _, version = line.partition(",")
        names.append(name.strip())
        driver = driver or version.strip()
    return {
        "smi_present": True,
        "driver_version": driver or _smi_driver_version(versions),
        "driver_cuda": smi_cuda_version(f"{versions}\n{header}"),
        "gpu_names": tuple(names),
    }


def _smi_driver_version(text: str) -> str:
    for pattern in (r"KMD version\s*:\s*([0-9][0-9.]*)", r"NVIDIA-SMI version\s*:\s*([0-9][0-9.]*)"):
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1)
    return ""


def smi_cuda_version(text: str) -> tuple[int, ...] | None:
    """The newest CUDA this driver can run, out of anything nvidia-smi prints.

    Four spellings in the wild, and the number is what matters, not which one said it:

    * ``CUDA UMD version    : 13.4``   — ``nvidia-smi --version``, current drivers
    * ``CUDA version        : 13.4``   — the same, older drivers
    * ``| … CUDA Version: 12.4 |``     — the header of plain ``nvidia-smi``
    * ``| … CUDA UMD Version: 13.4 |`` — that header, current drivers

    A value of ``Deprecated, see …`` is a signpost, not a version, so a pattern that
    matches digits is the whole of the defence.
    """
    for pattern in (
        r"CUDA UMD version\s*:\s*([0-9]+\.[0-9]+)",
        r"CUDA version\s*:\s*([0-9]+\.[0-9]+)",
        r"CUDA(?:\s+UMD)?\s+Version:\s*([0-9]+\.[0-9]+)",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            return parse_version(match.group(1))
    return None


def read_torch() -> dict[str, object]:
    """What torch says about itself, including the exception it raises instead.

    Importing torch is seconds and hundreds of megabytes, which is why this is only
    ever called by ``ai doctor`` — never on a path where speed matters.
    """
    try:
        import torch  # noqa: PLC0415 - the whole point is to find out if this works
    except Exception as exc:  # noqa: BLE001 - a broken install must still be diagnosable
        return {"torch_error": f"{type(exc).__name__}: {exc}"}
    found: dict[str, object] = {
        "torch_version": getattr(torch, "__version__", ""),
        "torch_cuda": parse_version(getattr(torch.version, "cuda", "") or ""),
    }
    try:
        found["cuda_available"] = bool(torch.cuda.is_available())
        if found["cuda_available"]:
            found["gpu_names"] = tuple(
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            )
    except Exception as exc:  # noqa: BLE001 - this is the call that raises the error
        found["cuda_available"] = False
        found["torch_error"] = f"{type(exc).__name__}: {exc}"
    return found


def torch_floor_from(pyproject: Path) -> tuple[int, ...]:
    """gustav's own ``torch>=`` requirement, so advice cannot suggest something too old."""
    match = re.search(r'"torch>=([0-9.]+)"', _read(str(pyproject)))
    return parse_version(match.group(1)) or () if match else ()


def collect(
    *,
    load_error: str = "",
    repo: Path | None = None,
    torch: bool = True,
    smi: bool = True,
) -> Facts:
    """Read this machine. The one function here that touches anything.

    ``smi=False`` is for explaining an error that came from somewhere else: this
    machine's driver has nothing to do with the one that raised it, and mixing the two
    produces advice for a machine that does not exist.
    """
    facts = Facts(platform=detect_platform(), load_error=load_error)
    facts.inspected = smi
    if smi:
        for key, value in read_smi().items():
            setattr(facts, key, value)
    if torch:
        for key, value in read_torch().items():
            if key == "gpu_names" and facts.gpu_names:
                continue
            setattr(facts, key, value)
    root = repo or Path(__file__).resolve().parents[3]
    facts.torch_floor = torch_floor_from(root / "pyproject.toml")
    return facts


# ---------------------------------------------------------------------------
# the diagnosis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """One way out, with the commands that take it."""

    title: str
    why: str
    commands: tuple[str, ...] = ()
    needs_root: bool = False
    reboot: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Diagnosis:
    ok: bool
    headline: str
    facts: Facts
    routes: tuple[Route, ...] = ()
    details: tuple[str, ...] = ()

    @property
    def problem(self) -> Problem | None:
        return self.facts.problem

    def lines(self) -> list[str]:
        """The whole thing, as ``ai doctor`` prints it."""
        out = [self.headline, ""]
        out += [f"  {line}" for line in self.details]
        for index, route in enumerate(self.routes, 1):
            out.append("")
            root = "  (needs sudo)" if route.needs_root else ""
            out.append(f"  {index}. {route.title}{root}")
            out.append(f"     {route.why}")
            for command in route.commands:
                out.append(f"       {command}")
            if route.reboot:
                out.append("       # then reboot: the running kernel keeps the old module")
            for note in route.notes:
                wrapped = _wrap(note, 70)
                out.append(f"     - {wrapped[0]}")
                out += [f"       {piece}" for piece in wrapped[1:]]
        return out

    def chat_reply(self) -> str:
        """The same thing in a sentence, for whoever just asked a question and got this.

        One paragraph: what is wrong, that it is not about what they asked, and the one
        command that looks into it. A chat box is the wrong place for a repository URL.
        """
        first = self.routes[0] if self.routes else None
        headline = self.headline.rstrip(".")
        if first is None:
            return f"I cannot answer right now: {headline}."
        return (
            f"I cannot answer right now: {headline}. The way out is on the server, and "
            f"`ai doctor` prints it — the shortest one here is to {first.title}."
        )


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]


def diagnose(facts: Facts) -> Diagnosis:
    """Turn what is true about this machine into what to do about it.

    The order of the routes is the whole opinion of this function: the cheapest thing
    that would work comes first. A torch that matches the driver is a download; a new
    driver is root and a reboot. When both would work, the download is offered first and
    the driver second, because one of them cannot break the machine someone games on.
    """
    problem = facts.problem
    platform = facts.platform
    recipe = driver_recipe(platform)
    details: list[str] = [
        f"machine     {platform.describe()}",
        "driver      "
        + (
            f"{facts.driver_version or 'unknown version'}, runs CUDA up to "
            f"{format_cuda(facts.driver_cuda)}"
            if facts.smi_present
            else (
                "nvidia-smi is not on this machine, so no driver is installed"
                if facts.inspected
                else "not read: this is about the error, not about this machine"
            )
        ),
        f"gpu         {', '.join(facts.gpu_names) or 'none reported'}",
    ]
    if facts.torch_version or facts.torch_error:
        details.append(
            f"torch       "
            + (
                f"{facts.torch_version}, built for CUDA {format_cuda(facts.torch_cuda)}, "
                f"sees the GPU: {facts.cuda_available}"
                if facts.torch_version
                else facts.torch_error
            )
        )
    if problem:
        details.append(f"the error   {problem.detail}")

    warning = secure_boot_warning(platform)

    def driver_route() -> Route:
        needed = facts.torch_cuda[0] if facts.torch_cuda else None
        floor = CUDA_DRIVER_FLOOR.get(needed or 0, {}).get("Linux", "")
        why = recipe.summary
        if needed and floor:
            why = (
                f"CUDA {needed}.x needs driver {floor} or newer; this is "
                f"{facts.driver_version or 'older'}. {recipe.summary}."
            )
        notes = list(recipe.notes)
        if warning:
            notes.insert(0, warning)
        return Route(
            title="install the current NVIDIA driver",
            why=why,
            commands=("ai doctor --install-driver",) + recipe.steps,
            needs_root=True,
            reboot=recipe.reboot,
            notes=tuple(notes),
        )

    def torch_route() -> Route:
        return Route(
            title="install a torch this driver can already run",
            why=(
                "every torch wheel carries its own CUDA runtime, so a build for the "
                "driver's CUDA needs no root and no reboot — only the download"
            ),
            commands=("ai doctor --fix-torch",),
            notes=(
                "It picks the newest build on download.pytorch.org that this driver "
                "runs, and refuses rather than installing one that would import, "
                "report no GPU, and leave you here again.",
            ),
        )

    # --- nothing to work with -------------------------------------------
    if not recipe.supported:
        return Diagnosis(
            ok=False,
            headline=recipe.summary,
            facts=facts,
            details=tuple(details),
            routes=(
                Route(
                    title=recipe.summary,
                    why="",
                    commands=recipe.steps,
                    notes=recipe.notes,
                ),
            ),
        )

    if problem is None and facts.cuda_available:
        return Diagnosis(
            ok=True,
            headline=(
                f"the GPU is usable: torch {facts.torch_version} on driver "
                f"{facts.driver_version}"
            ),
            facts=facts,
            details=tuple(details),
        )

    if problem is None and facts.cuda_available is None:
        # Nobody has asked the GPU a question: no error to read, and torch was not
        # imported (--no-torch, or it is a client install with no torch to import).
        # Guessing at a cure from here would be inventing a fault.
        return Diagnosis(
            ok=facts.smi_present,
            headline=(
                "nothing here is broken, but nothing has asked the GPU a question "
                "either: no load error from the daemon, and torch was not imported"
                if facts.smi_present
                else "there is no NVIDIA driver on this machine"
            ),
            facts=facts,
            details=tuple(details),
            routes=() if facts.smi_present else (driver_route(),),
        )

    if not facts.smi_present and facts.inspected:
        return Diagnosis(
            ok=False,
            headline="there is no NVIDIA driver on this machine",
            facts=facts,
            details=tuple(details),
            routes=(driver_route(),),
        )

    if problem and problem.kind == "wrong_arch":
        return Diagnosis(
            ok=False,
            headline=(
                "this torch has no kernels for this card: it is either too new for the "
                "card or too old for it"
            ),
            facts=facts,
            details=tuple(details),
            routes=(torch_route(), driver_route()),
        )

    if problem and problem.kind in ("not_compiled", "no_torch"):
        return Diagnosis(
            ok=False,
            headline=problem.headline(),
            facts=facts,
            details=tuple(details),
            routes=(torch_route(),),
        )

    if problem and problem.kind == "out_of_memory":
        return Diagnosis(
            ok=False,
            headline=(
                "the card ran out of memory. That is a size problem, not a driver one: "
                "`ai plan-hardware` sizes the model to this card"
            ),
            facts=facts,
            details=tuple(details),
        )

    if problem and problem.kind == "no_device":
        notes = (warning,) if warning else ()
        return Diagnosis(
            ok=False,
            headline=(
                "the driver is installed but no GPU could be opened — usually a module "
                "that did not load, or a card another container has taken"
            ),
            facts=facts,
            details=tuple(details),
            routes=(
                Route(
                    title="check the module and the device nodes",
                    why="before replacing anything, find out whether the kernel has it",
                    commands=(
                        "nvidia-smi",
                        "lsmod | grep -i nvidia",
                        "sudo dmesg | grep -iE 'nvrm|nvidia' | tail -20",
                        "ls -l /dev/nvidia*",
                    ),
                    notes=notes,
                ),
                driver_route(),
            ),
        )

    # --- the driver is older than the torch it is being asked to run -----
    driver_cuda = facts.driver_cuda or (problem.driver_cuda if problem else None)
    if driver_cuda and facts.torch_cuda:
        headline = (
            f"the driver here runs CUDA up to {format_cuda(driver_cuda)}, and the "
            f"installed torch was built for CUDA {format_cuda(facts.torch_cuda)}"
        )
    elif problem and problem.driver_cuda:
        # The error names the driver's CUDA even when nothing else here does — which is
        # the case when the error was pasted in from another machine.
        headline = (
            f"that driver runs CUDA up to {format_cuda(problem.driver_cuda)}, and the "
            f"torch on it was built for a newer one"
        )
    else:
        headline = problem.headline() if problem else "the model would not load"
    return Diagnosis(
        ok=False,
        headline=headline,
        facts=facts,
        details=tuple(details),
        routes=(torch_route(), driver_route()),
    )


def help_for(load_error: str | None) -> str | None:
    """One sentence for a chat box, or ``None`` if this is not a driver problem.

    Cheap on purpose: it parses the text it was given and reads nothing at all, because
    it runs on the path where someone is waiting for an answer.
    """
    problem = parse_cuda_error(load_error)
    if problem is None:
        return None
    if problem.kind == "out_of_memory":
        return None
    where = ""
    if problem.driver_cuda:
        where = f" This driver runs CUDA up to {format_cuda(problem.driver_cuda)}."
    advice = {
        "no_torch": "Install the server extras: `ai doctor` says how.",
        "not_compiled": "`ai doctor --fix-torch` replaces it with a CUDA build.",
    }.get(
        problem.kind,
        "`ai doctor` on the server says which of the two fixes this machine needs: "
        "a torch built for this driver, or a newer driver.",
    )
    return f"{problem.headline()}.{where} {advice}"


# ---------------------------------------------------------------------------
# the command line, for the installer
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    """``python3 -P driver.py <what>`` — what install.sh calls before anything exists.

    The ``-P`` matters. Running a file as a script puts its own directory first on the
    import path, and this directory holds ``types.py`` — which shadows the stdlib
    module of that name, which ``enum`` imports, which ``re`` imports, and the whole
    thing falls over on the first import with a circular-import error about ``re``.
    ``-P`` leaves the script's directory off the path, and then nothing here is
    shadowing anything.

    Stdout is for the caller to parse, stderr is for the log. Deliberately crude: it is
    read by shell, and shell is bad at everything else.
    """
    parser = argparse.ArgumentParser(prog="driver.py", description=__doc__)
    sub = parser.add_subparsers(dest="what", required=True)

    pick = sub.add_parser("pick-torch", help="which torch this driver can run")
    pick.add_argument("--driver-cuda", required=True, help="e.g. 12.4")
    pick.add_argument("--floor", default="", help="the torch>= gustav needs")

    rec = sub.add_parser("recipe", help="how this machine installs a driver")
    rec.add_argument("--toolkit", action="store_true")
    rec.add_argument("--json", action="store_true")

    sub.add_parser("platform", help="what this machine is, as json")

    exp = sub.add_parser("explain", help="what a CUDA error means")
    exp.add_argument("--error", required=True)

    args = parser.parse_args(argv)

    if args.what == "pick-torch":
        driver_cuda = parse_version(args.driver_cuda) or ()
        floor = parse_version(args.floor) or ()

        def noisy(url: str) -> str | None:
            page = _fetch(url)
            print(f"  {url}: {'nothing' if page is None else 'ok'}", file=sys.stderr)
            return page

        pick_result = pick_torch_channel(driver_cuda, floor, fetch=noisy)
        if pick_result is None:
            return 1
        print(
            f"{pick_result.channel} {format_cuda(pick_result.version)} "
            f"{'above' if pick_result.above_driver else 'within'} "
            f"{'ok' if pick_result.satisfies_floor else 'too-old'}"
        )
        return 0 if pick_result.satisfies_floor else 2

    if args.what == "recipe":
        recipe = driver_recipe(detect_platform(), toolkit=args.toolkit)
        if args.json:
            print(json.dumps(recipe.__dict__, default=list))
            return 0
        print(f"# {recipe.summary}")
        for step in recipe.steps:
            print(step)
        return 0 if recipe.supported else 1

    if args.what == "platform":
        print(json.dumps(detect_platform().__dict__, default=list))
        return 0

    if args.what == "explain":
        found = diagnose(collect(load_error=args.error, torch=False, smi=False))
        print("\n".join(found.lines()))
        return 0 if found.ok else 1

    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess in tests
    sys.exit(_main())
