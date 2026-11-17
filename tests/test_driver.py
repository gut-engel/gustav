"""The NVIDIA driver: reading the error, and knowing what this machine does about it.

The error that started this arrives in a chat box, not a terminal:

    RuntimeError: The NVIDIA driver on your system is too old (found version 12040)

Every test here is about turning that sentence into something a person can act on
without a search engine — and about never giving advice that would make a machine
worse, which is the real risk in a module that tells people to install kernel modules.

Nothing here touches a driver, a package manager or the network: every fact the code
has an opinion about is handed to it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from gustav.hw import driver

REPO = Path(__file__).resolve().parents[1]

REPORTED = (
    "RuntimeError: the model is not available, so I cannot answer: RuntimeError: The "
    "NVIDIA driver on your system is too old (found version 12040). Please update your "
    "GPU driver by downloading and installing a new version from the URL: "
    "http://www.nvidia.com/Download/index.aspx Alternatively, go to: "
    "https://pytorch.org to install a PyTorch version that has been compiled with your "
    "version of the CUDA driver."
)

UBUNTU = 'ID=ubuntu\nID_LIKE=debian\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\n'
DEBIAN = 'ID=debian\nVERSION_ID="12"\nPRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\n'
FEDORA = 'ID=fedora\nVERSION_ID=41\nPRETTY_NAME="Fedora Linux 41"\n'
ROCKY = 'ID="rocky"\nID_LIKE="rhel centos fedora"\nVERSION_ID="9.4"\nPRETTY_NAME="Rocky Linux 9.4"\n'
ARCH = 'ID=arch\nPRETTY_NAME="Arch Linux"\n'
SUSE = 'ID="opensuse-tumbleweed"\nID_LIKE="opensuse suse"\nVERSION_ID="20261115"\n'
VOID = 'ID=void\nPRETTY_NAME="void"\n'


def linux(os_release: str = UBUNTU, **kwargs) -> driver.Platform:
    kwargs.setdefault("kernel", "6.8.0-45-generic")
    kwargs.setdefault("container", "")
    kwargs.setdefault("secure_boot", False)
    return driver.detect_platform(system="Linux", os_release=os_release, **kwargs)


# --- reading the error ----------------------------------------------------
def test_the_packed_driver_version_is_read_as_a_cuda_version():
    assert driver.decode_api_version(12040) == (12, 4)
    assert driver.decode_api_version(11080) == (11, 8)
    assert driver.decode_api_version(13000) == (13, 0)


def test_the_error_from_the_chat_box_is_recognised_for_what_it_is():
    problem = driver.parse_cuda_error(REPORTED)
    assert problem is not None
    assert problem.kind == "driver_too_old"
    assert problem.driver_cuda == (12, 4), "which is what 12040 means"
    assert problem.is_driver_mismatch


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("RuntimeError: CUDA driver version is insufficient for the CUDA runtime version",
         "driver_too_old"),
        ("AssertionError: Torch not compiled with CUDA enabled", "not_compiled"),
        ("RuntimeError: no CUDA-capable device is detected", "no_device"),
        ("RuntimeError: CUDA error: no kernel image is available for execution on the device",
         "wrong_arch"),
        ("ModuleNotFoundError: No module named 'torch'", "no_torch"),
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
         "out_of_memory"),
    ],
)
def test_the_other_ways_cuda_fails_are_told_apart(text, kind):
    problem = driver.parse_cuda_error(text)
    assert problem is not None and problem.kind == kind


def test_an_error_about_something_else_gets_no_diagnosis():
    """A wrong diagnosis is worse than none: everything else must come back empty."""
    assert driver.parse_cuda_error("FileNotFoundError: model.safetensors") is None
    assert driver.parse_cuda_error("") is None
    assert driver.parse_cuda_error(None) is None


def test_the_sentence_for_the_chat_box_says_what_to_run():
    advice = driver.help_for(REPORTED)
    assert advice is not None
    assert "ai doctor" in advice
    assert "CUDA up to 12.4" in advice
    assert "\n" not in advice, "it goes in a chat bubble, not a terminal"


def test_running_out_of_memory_is_not_a_driver_problem_and_says_nothing():
    assert driver.help_for("torch.OutOfMemoryError: CUDA out of memory") is None


# --- reading nvidia-smi ---------------------------------------------------
def test_the_cuda_version_is_found_in_every_spelling_nvidia_smi_uses():
    """The field was renamed; the old name now holds the word Deprecated."""
    current = (
        "NVIDIA-SMI version  : 615.78.08\n"
        "DRIVER version      : Deprecated, see \"KMD version\" instead\n"
        "CUDA version        : Deprecated, see \"CUDA UMD version\" instead\n"
        "KMD version         : 615.78.08\n"
        "CUDA UMD version    : 13.4\n"
    )
    assert driver.smi_cuda_version(current) == (13, 4)
    older = "| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4   |"
    assert driver.smi_cuda_version(older) == (12, 4)
    assert driver.smi_cuda_version("no mention of it at all") is None


def test_the_driver_is_read_without_running_anything():
    calls: list[list[str]] = []

    def fake(argv):
        calls.append(argv)
        if "--query-gpu=name,driver_version" in " ".join(argv):
            return "NVIDIA GeForce GTX 1650, 550.54.14\n"
        if "--version" in argv:
            return "KMD version : 550.54.14\nCUDA UMD version : 12.4\n"
        return ""

    found = driver.read_smi(run=fake)
    assert found["smi_present"] is True
    assert found["driver_version"] == "550.54.14"
    assert found["driver_cuda"] == (12, 4)
    assert found["gpu_names"] == ("NVIDIA GeForce GTX 1650",)
    assert all(argv[0] == "nvidia-smi" for argv in calls)


def test_no_nvidia_smi_means_no_driver():
    assert driver.read_smi(run=lambda argv: None)["smi_present"] is False


# --- which machine is this ------------------------------------------------
@pytest.mark.parametrize(
    ("os_release", "family", "tool"),
    [
        (UBUNTU, "debian", "apt-get"),
        (DEBIAN, "debian", "apt-get"),
        (FEDORA, "fedora", "dnf"),
        (ROCKY, "rhel", "dnf"),
        (ARCH, "arch", "pacman"),
        (SUSE, "suse", "zypper"),
    ],
)
def test_each_family_gets_its_own_package_manager(os_release, family, tool):
    recipe = driver.driver_recipe(linux(os_release))
    assert recipe.family == family
    assert recipe.supported
    assert recipe.steps, "a supported family with no commands helps nobody"
    assert any(tool in step for step in recipe.steps)


def test_a_derivative_nobody_has_heard_of_follows_what_it_says_it_is_like():
    platform = linux('ID=frobnix\nID_LIKE="ubuntu debian"\nVERSION_ID="24.04"\n')
    assert platform.family == "debian"


def test_an_unknown_distribution_falls_back_to_nvidias_own_installer():
    recipe = driver.driver_recipe(linux(VOID))
    assert recipe.supported
    assert any("NVIDIA-Linux-x86_64" in step for step in recipe.steps)
    assert any("--dkms" in step for step in recipe.steps)
    assert any("text console" in note for note in recipe.notes), "it cannot run under X"


def test_the_repository_path_is_nvidias_spelling_of_the_distribution():
    assert any("ubuntu2404" in step for step in driver.driver_recipe(linux(UBUNTU)).steps)
    assert any("debian12" in step for step in driver.driver_recipe(linux(DEBIAN)).steps)
    assert any("rhel9" in step for step in driver.driver_recipe(linux(ROCKY)).steps)


def test_an_arm_machine_is_pointed_at_the_arm_repository():
    recipe = driver.driver_recipe(linux(UBUNTU, arch="aarch64"))
    assert any("sbsa" in step for step in recipe.steps)


def test_the_toolkit_is_only_installed_when_it_is_asked_for():
    """Four gigabytes of nvcc that torch does not need is not a default."""
    without = driver.driver_recipe(linux(UBUNTU))
    with_it = driver.driver_recipe(linux(UBUNTU), toolkit=True)
    assert not any("cuda-toolkit" in step for step in without.steps)
    assert any("cuda-toolkit" in step for step in with_it.steps)


# --- and where it must refuse ---------------------------------------------
def test_macos_is_told_the_truth_and_offered_the_thing_that_works():
    recipe = driver.driver_recipe(driver.detect_platform(system="Darwin"))
    assert not recipe.supported
    assert recipe.steps == ()
    assert any("10.13" in note for note in recipe.notes)
    assert any("--client" in note for note in recipe.notes)


def test_wsl_is_refused_because_the_driver_belongs_to_windows():
    recipe = driver.driver_recipe(
        linux(UBUNTU, kernel="5.15.153.1-microsoft-standard-WSL2")
    )
    assert not recipe.supported
    assert any("/dev/dxg" in note for note in recipe.notes)
    assert any("PowerShell" in step or "winget" in step for step in recipe.steps)


def test_a_container_is_refused_because_it_shares_the_hosts_kernel():
    recipe = driver.driver_recipe(linux(UBUNTU, container="docker"))
    assert not recipe.supported
    assert any("container toolkit" in note for note in recipe.notes)


def test_secure_boot_is_warned_about_before_it_eats_the_install():
    warning = driver.secure_boot_warning(linux(UBUNTU, secure_boot=True))
    assert warning is not None
    assert "no devices were found" in warning
    assert "mokutil" in warning
    assert driver.secure_boot_warning(linux(UBUNTU, secure_boot=False)) is None


# --- which torch this driver may run --------------------------------------
def page(*versions: str, channel: str = "cu124") -> str:
    return " ".join(f"torch-{v}%2B{channel}-cp313-linux_x86_64.whl" for v in versions)


def index(**channels: str):
    def fetch(url: str) -> str | None:
        for channel, body in channels.items():
            if f"/{channel}/torch/" in url:
                return body
        return None

    return fetch


def test_it_prefers_a_build_the_driver_definitely_runs():
    pick = driver.pick_torch_channel(
        (12, 4),
        (2, 10),
        fetch=index(cu124=page("2.14.0", channel="cu124"),
                    cu128=page("2.16.0", channel="cu128")),
    )
    assert pick is not None
    assert pick.channel == "cu124"
    assert pick.above_driver is False
    assert pick.index_url.endswith("/cu124")


def test_it_will_go_above_the_driver_only_when_nothing_below_is_new_enough():
    pick = driver.pick_torch_channel(
        (12, 4),
        (2, 15),
        fetch=index(cu124=page("2.14.0", channel="cu124"),
                    cu128=page("2.16.0", channel="cu128")),
    )
    assert pick.channel == "cu128"
    assert pick.above_driver is True, "and the caller is expected to say so"


def test_a_driver_a_whole_generation_behind_can_still_be_served():
    pick = driver.pick_torch_channel(
        (12, 0), (2, 0), fetch=index(cu118=page("2.7.1", channel="cu118"))
    )
    assert pick.channel == "cu118"


def test_when_nothing_published_is_new_enough_it_says_what_the_best_is():
    pick = driver.pick_torch_channel(
        (12, 4), (2, 15), fetch=index(cu124=page("2.14.0", channel="cu124"))
    )
    assert pick is not None
    assert pick.satisfies_floor is False
    assert pick.version == (2, 14, 0), "so the message can name it"


def test_an_index_that_cannot_be_reached_is_not_a_recommendation():
    assert driver.pick_torch_channel((12, 4), (2, 15), fetch=lambda url: None) is None


def test_a_driver_too_old_for_any_published_wheel_gets_nothing():
    assert driver.pick_torch_channel((10, 2), (2, 0), fetch=index(cu118=page())) is None


# --- the diagnosis --------------------------------------------------------
def facts(**kwargs) -> driver.Facts:
    base = {
        "platform": linux(UBUNTU),
        "driver_version": "550.54.14",
        "driver_cuda": (12, 4),
        "gpu_names": ("NVIDIA GeForce GTX 1650",),
        "smi_present": True,
        "torch_version": "2.15.0+cu130",
        "torch_cuda": (13, 0),
        "cuda_available": False,
        "load_error": REPORTED,
        "torch_floor": (2, 15),
    }
    base.update(kwargs)
    return driver.Facts(**base)


def test_the_reported_machine_is_diagnosed_and_offered_the_cheap_cure_first():
    found = driver.diagnose(facts())
    assert not found.ok
    assert "12.4" in found.headline and "13.0" in found.headline
    assert [route.needs_root for route in found.routes] == [False, True]
    assert found.routes[0].commands == ("ai doctor --fix-torch",)
    assert "ai doctor --install-driver" in found.routes[1].commands


def test_the_driver_route_says_which_driver_would_be_new_enough():
    found = driver.diagnose(facts())
    assert "580.65.06" in found.routes[1].why, "the floor for CUDA 13"
    assert "550.54.14" in found.routes[1].why, "and what is here now"


def test_the_whole_thing_prints_without_losing_a_command():
    text = "\n".join(driver.diagnose(facts()).lines())
    assert "ai doctor --fix-torch" in text
    assert "sudo apt-get install -y cuda-drivers" in text
    assert "then reboot" in text


def test_a_working_machine_is_told_that_it_works():
    found = driver.diagnose(
        facts(load_error="", torch_version="2.15.0+cu124", torch_cuda=(12, 4),
              cuda_available=True)
    )
    assert found.ok
    assert found.routes == ()
    assert "usable" in found.headline


def test_a_machine_with_no_driver_is_not_offered_a_torch():
    found = driver.diagnose(
        facts(smi_present=False, driver_version="", driver_cuda=None, gpu_names=(),
              load_error="RuntimeError: no CUDA-capable device is detected")
    )
    assert "no NVIDIA driver" in found.headline
    assert [route.needs_root for route in found.routes] == [True]


def test_a_card_the_wheel_has_no_kernels_for_gets_both_routes():
    found = driver.diagnose(
        facts(load_error="CUDA error: no kernel image is available for execution on the device")
    )
    assert "no kernels for this card" in found.headline
    assert len(found.routes) == 2


def test_out_of_memory_is_sent_to_the_planner_not_to_a_driver():
    found = driver.diagnose(facts(load_error="torch.OutOfMemoryError: CUDA out of memory"))
    assert "plan-hardware" in found.headline
    assert found.routes == ()


def test_a_cpu_only_torch_is_a_torch_problem_only():
    found = driver.diagnose(
        facts(load_error="AssertionError: Torch not compiled with CUDA enabled",
              torch_version="2.15.0", torch_cuda=None)
    )
    assert len(found.routes) == 1
    assert found.routes[0].needs_root is False


def test_nothing_is_wrong_and_nothing_was_asked_is_its_own_answer():
    """`--no-torch` on a healthy machine must not invent a fault."""
    found = driver.diagnose(
        facts(load_error="", torch_version="", torch_cuda=None, cuda_available=None)
    )
    assert found.ok
    assert found.routes == ()
    assert "nothing has asked the GPU a question" in found.headline


def test_secure_boot_reaches_the_route_that_would_trip_over_it():
    found = driver.diagnose(facts(platform=linux(UBUNTU, secure_boot=True)))
    assert any("Secure Boot is on" in note for note in found.routes[1].notes)


def test_on_a_mac_it_offers_no_commands_at_all():
    found = driver.diagnose(
        facts(platform=driver.detect_platform(system="Darwin"), smi_present=False,
              driver_version="", driver_cuda=None, gpu_names=())
    )
    assert not found.ok
    assert all(not route.needs_root for route in found.routes)
    assert "macOS has no NVIDIA driver" in found.headline


def test_the_chat_reply_is_one_paragraph_that_names_the_next_step():
    reply = driver.diagnose(facts()).chat_reply()
    assert reply.startswith("I cannot answer right now:")
    assert "ai doctor" in reply
    assert "\n" not in reply


# --- the command line the installer uses ----------------------------------
def run_module(*args: str):
    # -P for the same reason install.sh passes it: this directory has a types.py in it.
    return subprocess.run(
        [sys.executable, "-P", str(REPO / "src" / "gustav" / "hw" / "driver.py"), *args],
        capture_output=True, text=True, timeout=60,
    )


def test_the_module_runs_as_a_script_with_nothing_installed():
    """install.sh calls this before the virtualenv exists, so it must be stdlib-only."""
    import ast

    source = (REPO / "src" / "gustav" / "hw" / "driver.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.parse(source).body:        # module level only: the rest is lazy
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "gustav" not in imported, "it runs where gustav is not installed"
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


def test_it_describes_this_machine_as_json():
    done = run_module("platform")
    assert done.returncode == 0
    import json

    payload = json.loads(done.stdout)
    assert {"system", "family", "arch"} <= payload.keys()


def test_it_prints_the_recipe_for_this_machine():
    done = run_module("recipe")
    assert done.returncode in (0, 1)
    assert done.stdout.startswith("#")


def test_explaining_an_error_exits_nonzero_so_a_script_can_tell():
    done = run_module("explain", "--error", REPORTED)
    assert done.returncode == 1
    assert "12.4" in done.stdout
