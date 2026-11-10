from __future__ import annotations

from datetime import time

import pytest

from gustav.config import ConfigError, load_config
from gustav.config.schema import Config, TimeWindow


def test_defaults_are_a_valid_config():
    config = Config()
    assert config.schedule.windows == ["00:00-06:00"]
    assert config.schedule.idle_minutes == 45
    assert config.supervisor.max_attempts == 3


def test_empty_file_loads(tmp_path):
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    config, paths = load_config(tmp_path / "config.yaml")
    assert config.instance.name == "gustav"
    assert paths.data_dir == tmp_path / "var"


def test_unknown_key_is_an_error(tmp_path):
    (tmp_path / "config.yaml").write_text("schedule:\n  idle_minutez: 10\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "config.yaml")
    assert "idle_minutez" in str(exc.value)


def test_data_dir_override(tmp_path):
    (tmp_path / "config.yaml").write_text("instance:\n  data_dir: elsewhere\n", encoding="utf-8")
    _, paths = load_config(tmp_path / "config.yaml")
    assert paths.data_dir == tmp_path / "elsewhere"


def test_profile_is_merged(tmp_path):
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "small-12gb.yaml").write_text(
        "base_model: Qwen3-4B\nlora_rank: 16\n", encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text(
        "hardware:\n  profile: small-12gb\n  lora_rank: 32\n", encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "config.yaml")
    assert config.hardware["base_model"] == "Qwen3-4B"
    # config.yaml is the base, the profile overlays it: the profile wins on conflict.
    assert config.hardware["lora_rank"] == 16


def test_missing_profile_explains_itself(tmp_path):
    (tmp_path / "config.yaml").write_text("hardware:\n  profile: nope\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="nope"):
        load_config(tmp_path / "config.yaml")


@pytest.mark.parametrize(
    "raw,inside,outside",
    [
        ("00:00-06:00", [time(0, 0), time(3, 0), time(5, 59)], [time(6, 0), time(23, 59)]),
        ("22:00-02:00", [time(22, 0), time(23, 59), time(0, 0), time(1, 59)], [time(2, 0), time(12, 0)]),
        ("12:00-12:00", [time(0, 0), time(12, 0)], []),
    ],
)
def test_time_windows(raw, inside, outside):
    window = TimeWindow.parse(raw)
    for t in inside:
        assert window.contains(t), f"{t} should be inside {raw}"
    for t in outside:
        assert not window.contains(t), f"{t} should be outside {raw}"


@pytest.mark.parametrize("raw", ["0600", "25:00-06:00", "00:00/06:00", "00:00-06:99"])
def test_bad_windows_rejected(raw):
    with pytest.raises(ValueError):
        TimeWindow.parse(raw)
