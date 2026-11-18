"""Speech in and speech out, behind two very small interfaces.

Everything about the voice path that can be done without a model being installed is
done against these two protocols, which is why the whole pipeline is testable on a
machine with no microphone, no speaker and no gigabyte of weights: ``NullStt`` and
``NullTts`` answer deterministically and instantly.

The real engines are chosen for one reason each:

* **faster-whisper** for listening. It is the Whisper everyone already trusts, compiled
  through CTranslate2, and ``base`` with int8 weights transcribes a two-second question
  in a couple of hundred milliseconds on a laptop CPU — which matters, because the GPU
  on this machine is busy holding the language model.
* **piper** for speaking. A few hundred megabytes, faster than real time on a
  Raspberry Pi 4, and it sounds like a person rather than like 1998.

Both are optional. ``engine: auto`` takes whichever is installed and falls back to the
null one, so a daemon with ``voice.enabled`` and no models says so instead of failing.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from gustav.voice.audio import Audio


@dataclass(frozen=True)
class Heard:
    """What the recogniser made of a piece of audio."""

    text: str
    language: str = ""
    #: How long the recognising took, which is half the latency budget.
    ms: int = 0
    #: 0..1 where the engine offers one. Used only to decide whether to answer at all.
    confidence: float = 1.0


@runtime_checkable
class Stt(Protocol):
    name: str

    def ready(self) -> tuple[bool, str]: ...

    def warm(self) -> None:
        """Load the model now, so the first question does not pay for it."""

    def listen(self, audio: Audio, *, language: str = "") -> Heard: ...


@runtime_checkable
class Tts(Protocol):
    name: str
    sample_rate: int

    def ready(self) -> tuple[bool, str]: ...

    def warm(self) -> None: ...

    def speak(self, text: str, *, language: str = "", voice: str = "") -> Audio: ...


# ---------------------------------------------------------------------------
# the stand-ins
# ---------------------------------------------------------------------------


@dataclass
class NullStt:
    """Hears whatever it was told to hear. The tests' ears."""

    name: str = "null"
    #: Answers in order; the last one repeats once the queue is empty.
    script: list[str] = field(default_factory=lambda: ["hello"])
    language: str = "en"
    heard: list[Audio] = field(default_factory=list)

    def ready(self) -> tuple[bool, str]:
        return True, "the null recogniser, which hears what it is told to hear"

    def warm(self) -> None:
        return None

    def listen(self, audio: Audio, *, language: str = "") -> Heard:
        self.heard.append(audio)
        index = min(len(self.heard) - 1, len(self.script) - 1)
        return Heard(self.script[index] if self.script else "", language or self.language)


@dataclass
class NullTts:
    """Produces silence of a believable length, and remembers what it was asked to say."""

    name: str = "null"
    sample_rate: int = 22050
    said: list[str] = field(default_factory=list)
    #: Roughly the speed of speech, so timing tests mean something.
    seconds_per_character: float = 0.06

    def ready(self) -> tuple[bool, str]:
        return True, "the null voice, which produces silence of the right length"

    def warm(self) -> None:
        return None

    def speak(self, text: str, *, language: str = "", voice: str = "") -> Audio:
        self.said.append(text)
        return Audio.silence(
            max(len(text) * self.seconds_per_character, 0.05), self.sample_rate
        )


# ---------------------------------------------------------------------------
# the real ones
# ---------------------------------------------------------------------------


class WhisperStt:
    """faster-whisper, loaded once and kept.

    The model is loaded on the first use or on ``warm``, never at import: a daemon that
    is not being talked to should not be holding a speech model.
    """

    name = "whisper"

    def __init__(self, *, model: str = "base", device: str = "cpu",
                 compute_type: str = "int8"):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    @staticmethod
    def installed() -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def ready(self) -> tuple[bool, str]:
        if not self.installed():
            return False, (
                "faster-whisper is not installed: pip install faster-whisper, or set "
                "voice.stt.engine to null"
            )
        return True, f"faster-whisper {self.model_name} on {self.device} ({self.compute_type})"

    def warm(self) -> None:
        self._load()

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # noqa: PLC0415 - optional dependency

            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def listen(self, audio: Audio, *, language: str = "") -> Heard:
        import numpy  # noqa: PLC0415 - comes with faster-whisper

        started = time.monotonic()
        model = self._load()
        # Whisper wants float32 at 16 kHz. Handing it the array directly skips a
        # temporary file, which is a surprising amount of the time for a short clip.
        pcm = audio.resampled(16000).pcm
        samples = numpy.frombuffer(pcm, dtype=numpy.int16).astype(numpy.float32) / 32768.0
        segments, info = model.transcribe(
            samples,
            language=language or None,
            # The defaults chase accuracy; a voice assistant is answering one sentence
            # and the clock is running. beam 1 and no condition on previous text is
            # most of the speed and nearly all of the accuracy for this job.
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Heard(
            text=text,
            language=getattr(info, "language", "") or language,
            ms=int((time.monotonic() - started) * 1000),
            confidence=float(getattr(info, "language_probability", 1.0) or 1.0),
        )


class VoiceNotInstalled(RuntimeError):
    """piper is here and has nothing to speak with. A sentence, not a stack trace."""


def _suggest(language: str) -> str:
    from gustav.voice import voices as catalogue

    return catalogue.suggestions(language)[0]


class PiperTts:
    """piper, through its Python package if it is there and its binary if it is not.

    Two paths because the two are installed by completely different people: the Python
    package on a server that already has a virtualenv, the binary on a Raspberry Pi
    from a release tarball. The voice file is the same either way.
    """

    name = "piper"

    def __init__(self, *, voice: str = "", speed: float = 1.0, sample_rate: int = 22050,
                 voices_dir: Path | None = None):
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.voices_dir = voices_dir
        self._voices: dict[str, Any] = {}

    @staticmethod
    def installed() -> bool:
        return (
            importlib.util.find_spec("piper") is not None
            or shutil.which("piper") is not None
        )

    def ready(self) -> tuple[bool, str]:
        if not self.installed():
            return False, (
                "piper is not installed: pip install piper-tts, or set voice.tts.engine "
                "to null"
            )
        # piper does not come with a voice, the way a text editor does not come with a
        # novel. Without one it is a program that cannot speak, and finding that out
        # at the moment someone presses Say it is too late.
        from gustav.voice import voices as catalogue

        if self.voice and Path(self.voice).exists():
            return True, f"piper, voice {Path(self.voice).stem}"
        on_disk = [
            entry for entry in catalogue.installed(self.voices_dir or Path("."))
            if entry["complete"]
        ]
        if not on_disk:
            return False, (
                "piper has no voice installed, so it cannot say anything. Install one "
                "— `ai voice install de_DE-thorsten-low`, or the Voice panel does it "
                "for you."
            )
        if self.voice and not any(entry["name"] == self.voice for entry in on_disk):
            return False, (
                f"voice.tts.voice is {self.voice!r}, which is not installed. On disk: "
                + ", ".join(str(entry["name"]) for entry in on_disk)
            )
        return True, (
            f"piper, voice {self.voice or 'whichever matches the language'} "
            f"({len(on_disk)} installed)"
        )

    def _voice_path(self, language: str) -> str:
        """The voice to speak with: what was configured, or one matching the language.

        Configured names are looked for on disk too, so voice.tts.voice can be
        ``de_DE-thorsten-low`` rather than an absolute path to a file nobody wants to
        type out.
        """
        from gustav.voice import voices as catalogue

        if self.voice:
            if Path(self.voice).exists():
                return self.voice
            if self.voices_dir:
                guess = self.voices_dir / f"{self.voice}.onnx"
                if guess.exists():
                    return str(guess)
            # Named but missing: say which, rather than falling back to another
            # voice and leaving someone wondering who is speaking.
            raise VoiceNotInstalled(
                f"voice.tts.voice is {self.voice!r} and that voice is not installed. "
                f"`ai voice install {self.voice}` fetches it."
            )
        if self.voices_dir:
            found = catalogue.for_language(self.voices_dir, language)
            if found:
                return found
            any_voice = [
                entry for entry in catalogue.installed(self.voices_dir)
                if entry["complete"]
            ]
            if any_voice:
                # Something rather than nothing: a Dutch voice reading German is
                # strange, silence is useless, and the panel says what is installed.
                return str(any_voice[0]["path"])
        raise VoiceNotInstalled(
            "piper has no voice installed, so there is nothing to speak with. "
            f"`ai voice install {_suggest(language)}` fetches one, or use the Voice panel."
        )

    def warm(self) -> None:
        with_voice = self._voice_path("")
        if with_voice:
            self._load(with_voice)

    def _load(self, name: str):
        if name in self._voices:
            return self._voices[name]
        from piper import PiperVoice  # noqa: PLC0415 - optional dependency

        path = name
        if self.voices_dir and not Path(name).exists():
            guess = self.voices_dir / f"{name}.onnx"
            if guess.exists():
                path = str(guess)
        self._voices[name] = PiperVoice.load(path)
        return self._voices[name]

    def speak(self, text: str, *, language: str = "", voice: str = "") -> Audio:
        name = voice or self._voice_path(language)
        if importlib.util.find_spec("piper") is not None and name:
            return self._speak_in_process(text, name)
        return self._speak_by_command(text, name)

    def _speak_in_process(self, text: str, name: str) -> Audio:
        import io
        import wave as wave_module
        from piper import SynthesisConfig

        loaded = self._load(name)
        buffer = io.BytesIO()

        syn_config = SynthesisConfig(
            length_scale=1.0 / max(self.speed, 0.1)
        )

        with wave_module.open(buffer, "wb") as handle:
            loaded.synthesize_wav(
                text,
                handle,
                syn_config=syn_config,
            )

        return Audio.from_wav(buffer.getvalue())

    def _speak_by_command(self, text: str, name: str) -> Audio:
        binary = shutil.which("piper")
        if binary is None:
            raise RuntimeError("piper is not installed")
        with tempfile.TemporaryDirectory() as room:
            out = Path(room) / "say.wav"
            argv = [binary, "--output_file", str(out)]
            if name:
                argv += ["--model", name]
            if self.speed != 1.0:
                argv += ["--length_scale", f"{1.0 / max(self.speed, 0.1):.3f}"]
            subprocess.run(
                argv, input=text.encode("utf-8"), capture_output=True, timeout=60, check=True
            )
            return Audio.from_wav(out.read_bytes())


# ---------------------------------------------------------------------------
# choosing
# ---------------------------------------------------------------------------


def build_stt(config, *, voices_dir: Path | None = None) -> Stt:
    """The recogniser this configuration asks for, or the honest fallback."""
    wanted = (config.engine or "auto").lower()
    if wanted == "null":
        return NullStt()
    if wanted in ("auto", "whisper"):
        engine = WhisperStt(
            model=config.model, device=config.device, compute_type=config.compute_type
        )
        if engine.installed() or wanted == "whisper":
            return engine
        return NullStt(script=[""])
    raise ValueError(f"no speech recogniser called {config.engine!r}")


def build_tts(config, *, voices_dir: Path | None = None) -> Tts:
    wanted = (config.engine or "auto").lower()
    if wanted == "null":
        return NullTts(sample_rate=config.sample_rate)
    if wanted in ("auto", "piper"):
        engine = PiperTts(
            voice=config.voice, speed=config.speed, sample_rate=config.sample_rate,
            voices_dir=voices_dir,
        )
        if engine.installed() or wanted == "piper":
            return engine
        return NullTts(sample_rate=config.sample_rate)
    raise ValueError(f"no voice called {config.engine!r}")
