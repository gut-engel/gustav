"""Piper voices: what they are called, where they live, and how to get one.

piper does not come with a voice, the way a text editor does not come with a novel.
It is two files — a model and its JSON — and until one of them is on the disk the
whole speaking half of this is a program that cannot speak. Nothing in the install
fetches one, because which voice you want depends on which language you are going to
talk to it in, and that is a question only the person can answer.

So: a voice has a name like ``de_DE-thorsten-low``, that name *is* its path in
rhasspy's collection, and this module turns one into the other and fetches it.

    de_DE-thorsten-low  ->  de/de_DE/thorsten/low/de_DE-thorsten-low.onnx
    en_GB-alba-medium   ->  en/en_GB/alba/medium/en_GB-alba-medium.onnx

The quality is the size, and on a machine that is also holding a language model the
difference matters: ``low`` is about 20 MB and faster than real time on a Pi,
``medium`` about 60 MB and better, ``high`` about 110 MB and not worth it through a
speaker in a kitchen.
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

#: Where the voices are. A HuggingFace repository, served as plain files.
COLLECTION = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

#: ``<locale>-<speaker>-<quality>``, where the locale is ``xx_YY``.
NAME = re.compile(r"^([a-z]{2,3})_([A-Z]{2})-([a-z0-9_+-]+)-(x_low|low|medium|high)$")

#: One safe suggestion per language, for a panel that has to offer *something*. Not a
#: catalogue: the whole collection is a thousand files and nobody wants to scroll it.
SUGGESTED: dict[str, list[str]] = {
    "en": ["en_GB-alba-medium", "en_US-lessac-medium", "en_US-amy-low"],
    "de": ["de_DE-thorsten-low", "de_DE-thorsten-medium", "de_DE-kerstin-low"],
    "fr": ["fr_FR-siwis-medium", "fr_FR-upmc-medium"],
    "es": ["es_ES-sharvard-medium", "es_MX-ald-medium"],
    "it": ["it_IT-riccardo-x_low"],
    "nl": ["nl_NL-mls-medium", "nl_BE-nathalie-medium"],
    "pt": ["pt_BR-faber-medium"],
    "pl": ["pl_PL-darkman-medium"],
    "ru": ["ru_RU-dmitri-medium"],
    "uk": ["uk_UA-ukrainian_tts-medium"],
    "sv": ["sv_SE-nst-medium"],
    "da": ["da_DK-talesyntese-medium"],
    "no": ["no_NO-talesyntese-medium"],
    "fi": ["fi_FI-harri-medium"],
    "cs": ["cs_CZ-jirka-medium"],
    "tr": ["tr_TR-dfki-medium"],
    "zh": ["zh_CN-huayan-medium"],
}


class VoiceError(RuntimeError):
    """Something about the voice, said in a sentence someone can act on."""


@dataclass(frozen=True)
class Voice:
    name: str
    language: str       # "de"
    locale: str         # "de_DE"
    speaker: str
    quality: str

    @property
    def directory(self) -> str:
        return f"{self.language}/{self.locale}/{self.speaker}/{self.quality}"

    @property
    def model_url(self) -> str:
        return f"{COLLECTION}/{self.directory}/{self.name}.onnx"

    @property
    def config_url(self) -> str:
        return f"{self.model_url}.json"


def parse(name: str) -> Voice:
    """``de_DE-thorsten-low`` into its parts, or a sentence saying why not."""
    match = NAME.match((name or "").strip())
    if match is None:
        raise VoiceError(
            f"{name!r} is not a piper voice name. They look like "
            f"de_DE-thorsten-low or en_GB-alba-medium: a locale, a speaker, and one "
            f"of x_low, low, medium, high."
        )
    language, country, speaker, quality = match.groups()
    return Voice(
        name=name.strip(), language=language, locale=f"{language}_{country}",
        speaker=speaker, quality=quality,
    )


def suggestions(language: str) -> list[str]:
    code = (language or "en").lower().replace("-", "_").split("_")[0]
    return SUGGESTED.get(code, SUGGESTED["en"])


def installed(directory: Path) -> list[dict[str, object]]:
    """The voices on this machine, with the size and whether they are complete.

    A voice is two files. Half a voice is what you get when a download was
    interrupted, and it fails at the moment someone speaks rather than here, so it is
    reported rather than ignored.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = []
    for model in sorted(directory.glob("*.onnx")):
        settings = model.with_suffix(".onnx.json")
        entry: dict[str, object] = {
            "name": model.stem,
            "path": str(model),
            "size_mb": round(model.stat().st_size / (1 << 20), 1),
            "complete": settings.is_file(),
        }
        if settings.is_file():
            try:
                loaded = json.loads(settings.read_text(encoding="utf-8"))
                entry["language"] = (
                    (loaded.get("language") or {}).get("code")
                    or (loaded.get("espeak") or {}).get("voice")
                    or ""
                )
                entry["sample_rate"] = (loaded.get("audio") or {}).get("sample_rate")
            except (OSError, ValueError):
                entry["complete"] = False
        found.append(entry)
    return found


def for_language(directory: Path, language: str) -> str | None:
    """A voice on disk that speaks this language, if there is one."""
    code = (language or "").lower().replace("-", "_").split("_")[0]
    for entry in installed(directory):
        if not entry["complete"]:
            continue
        name = str(entry["name"])
        if name.lower().startswith(f"{code}_"):
            return str(entry["path"])
    return None


def _download(url: str, target: Path, *, open_url=urllib.request.urlopen) -> int:
    """To a temporary name, then renamed: half a file is worse than no file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with open_url(url, timeout=120) as response, part.open("wb") as handle:  # noqa: S310
            shutil.copyfileobj(response, handle)
    except Exception as exc:  # noqa: BLE001
        part.unlink(missing_ok=True)
        raise VoiceError(f"could not fetch {url}: {exc}") from exc
    size = part.stat().st_size
    if size < 1024:
        part.unlink(missing_ok=True)
        raise VoiceError(f"{url} gave back {size} bytes, which is not a voice")
    part.replace(target)
    return size


def install(name: str, directory: Path, *, open_url=urllib.request.urlopen) -> dict[str, object]:
    """Fetch a voice and its settings. Returns what landed where."""
    voice = parse(name)
    directory = Path(directory)
    model = directory / f"{voice.name}.onnx"
    settings = directory / f"{voice.name}.onnx.json"
    if model.is_file() and settings.is_file():
        return {"name": voice.name, "path": str(model), "size_mb":
                round(model.stat().st_size / (1 << 20), 1), "already": True}
    # One of the two is there: a download that was interrupted. Both are fetched
    # again rather than trusting whichever half survived.
    model.unlink(missing_ok=True)
    settings.unlink(missing_ok=True)

    size = _download(voice.model_url, model, open_url=open_url)
    try:
        _download(voice.config_url, settings, open_url=open_url)
    except VoiceError:
        # A model without its JSON cannot be loaded, and leaving it there means the
        # next listing claims a voice that does not work.
        model.unlink(missing_ok=True)
        raise
    return {
        "name": voice.name,
        "path": str(model),
        "size_mb": round(size / (1 << 20), 1),
        "already": False,
    }
