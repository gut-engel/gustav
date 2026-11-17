"""The thing it says while it is still thinking.

Silence from a speaker reads as broken, not busy. Three hundred milliseconds of it is
enough for someone to say "hello?" over the top of the answer. So if the model has not
produced anything speakable by ``voice.filler_after_ms``, a short phrase goes out
instead — and because it is always the same handful of phrases, they are synthesised
once, written to ``var/voice/fillers/`` and read from disk after that.

Pre-rendering is the whole point. Asking the voice to synthesise "one moment" at the
exact moment you have no time to spare would be an odd way to save time.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from gustav.voice.audio import Audio
from gustav.voice.engines import Tts

#: Per language, because a German speaker hearing "one moment" is a bug. Short on
#: purpose: a filler that takes a second to say is not covering a gap, it is the gap.
PHRASES: dict[str, list[str]] = {
    "en": ["One moment.", "Let me think.", "Checking."],
    "de": ["Einen Moment.", "Ich überlege.", "Ich schaue nach."],
    "fr": ["Un instant.", "Je réfléchis."],
    "es": ["Un momento.", "Déjame pensar."],
    "it": ["Un momento.", "Ci penso."],
    "nl": ["Een moment.", "Ik denk even na."],
    "pt": ["Um momento.", "Deixa-me pensar."],
    "pl": ["Chwileczkę.", "Zastanawiam się."],
    "sv": ["Ett ögonblick.", "Låt mig tänka."],
    "da": ["Et øjeblik.", "Lad mig tænke."],
    "nb": ["Et øyeblikk.", "La meg tenke."],
    "fi": ["Hetki.", "Mietin hetken."],
    "cs": ["Momentík.", "Přemýšlím."],
    "tr": ["Bir saniye.", "Düşüneyim."],
    "ru": ["Минуту.", "Сейчас подумаю."],
    "uk": ["Хвилинку.", "Зараз подумаю."],
    "ja": ["少々お待ちください。", "考えています。"],
    "zh": ["请稍等。", "我想一下。"],
}


def phrases_for(language: str, configured: list[str] | None = None) -> list[str]:
    """What to say, in the voice language — configured, or the built-in ones."""
    if configured:
        return list(configured)
    code = (language or "en").lower().replace("_", "-").split("-")[0]
    return PHRASES.get(code, PHRASES["en"])


class Fillers:
    """Pre-rendered phrases, kept on disk between runs.

    The cache key is the phrase, the voice and the language, so changing any of them
    re-renders rather than playing yesterday's voice.
    """

    def __init__(self, tts: Tts, directory: Path, *, language: str = "en",
                 configured: list[str] | None = None, voice: str = ""):
        self.tts = tts
        self.directory = Path(directory)
        self.language = language
        self.configured = configured or []
        self.voice = voice
        self.last_text = ""
        self._cache: dict[str, Audio] = {}
        self._random = random.Random(0xF177E4)

    def _path(self, phrase: str, language: str) -> Path:
        key = hashlib.sha256(
            f"{phrase}|{language}|{self.voice}|{self.tts.name}".encode("utf-8")
        ).hexdigest()[:16]
        return self.directory / f"{key}.wav"

    def prepare(self, language: str = "") -> int:
        """Render everything now, so nothing is rendered when it is needed. Never raises."""
        language = language or self.language
        made = 0
        for phrase in phrases_for(language, self.configured):
            try:
                self._render(phrase, language)
                made += 1
            except Exception:  # noqa: BLE001 - no voice is a quieter assistant, not a dead one
                continue
        return made

    def _render(self, phrase: str, language: str) -> Audio:
        key = f"{phrase}|{language}"
        if key in self._cache:
            return self._cache[key]
        path = self._path(phrase, language)
        if path.is_file():
            audio = Audio.from_wav(path.read_bytes())
        else:
            audio = self.tts.speak(phrase, language=language, voice=self.voice)
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                path.write_bytes(audio.to_wav())
            except OSError:
                pass        # a cache that cannot be written is still a working filler
        self._cache[key] = audio
        return audio

    def pick(self, language: str = "") -> Audio | None:
        """One phrase, at random so it does not sound like a machine on a loop."""
        language = language or self.language
        options = phrases_for(language, self.configured)
        if not options:
            return None
        phrase = self._random.choice(options)
        try:
            audio = self._render(phrase, language)
        except Exception:  # noqa: BLE001 - never let the filler break the answer
            return None
        self.last_text = phrase
        return audio
