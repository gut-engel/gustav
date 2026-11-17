"""The voice side of the daemon: what is loaded, what is ready, and what it costs.

Held by the daemon, built the first time anybody speaks, and honest when it cannot do
anything — a machine with ``voice.enabled`` and no models installed says exactly that,
through ``status()``, instead of failing at the first question.

Nothing here is loaded at import or at startup unless ``warm()`` is called: a speech
model and a voice are a few hundred megabytes of RAM that a daemon nobody is talking
to has no business holding.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from gustav.voice.audio import Audio
from gustav.voice.engines import Stt, Tts, build_stt, build_tts
from gustav.voice.fillers import Fillers, phrases_for
from gustav.voice.pipeline import VoicePipeline


#: The settings the interface may change, and nothing else. A browser may pick the
#: language and the voice; it may not point the daemon at a different model directory
#: or turn the whole section off from a stray click.
SETTABLE = {
    "language": str,
    "reply_language": str,
    "filler_after_ms": int,
    "stream_sentences": bool,
    "barge_in": bool,
    "max_reply_tokens": int,
    "wake_words": list,
    "fillers": list,
    "stt.engine": str,
    "stt.model": str,
    "stt.device": str,
    "tts.engine": str,
    "tts.voice": str,
    "tts.speed": float,
}

#: Where the overrides live. Not in config.yaml: that file is hand-written and full of
#: comments explaining itself, and a program that rewrites it loses all of them. The
#: file stays the default; what the interface sets is an overlay on top of it, and the
#: panel says which is which.
META_KEY = "voice.settings"


def apply_overrides(config, stored: dict) -> list[str]:
    """Put stored settings over the configured ones. Returns what it changed."""
    changed = []
    for key, kind in SETTABLE.items():
        if key not in stored:
            continue
        value = stored[key]
        section = config.voice
        name = key
        if "." in key:
            head, name = key.split(".", 1)
            section = getattr(config.voice, head)
        try:
            if kind is bool:
                value = bool(value)
            elif kind is int:
                value = int(value)
            elif kind is float:
                value = float(value)
            elif kind is list:
                value = [str(item) for item in (value or [])]
            else:
                value = str(value)
            setattr(section, name, value)
            changed.append(key)
        except (TypeError, ValueError):
            continue
    return changed


class VoiceService:
    def __init__(self, config, paths, *, ask, stt: Stt | None = None, tts: Tts | None = None):
        self.config = config
        self.paths = paths
        self.ask = ask
        self._stt = stt
        self._tts = tts
        self._fillers: Fillers | None = None
        self._warmed = False
        self.warm_ms = 0

    # --- the pieces, built on demand ------------------------------------
    @property
    def voices_dir(self) -> Path:
        return Path(self.paths.root) / "voices"

    @property
    def stt(self) -> Stt:
        if self._stt is None:
            self._stt = build_stt(self.config.voice.stt, voices_dir=self.voices_dir)
        return self._stt

    @property
    def tts(self) -> Tts:
        if self._tts is None:
            self._tts = build_tts(self.config.voice.tts, voices_dir=self.voices_dir)
        return self._tts

    @property
    def fillers(self) -> Fillers:
        if self._fillers is None:
            self._fillers = Fillers(
                self.tts,
                Path(self.paths.data_dir) / "voice" / "fillers",
                language=self.config.voice.language,
                configured=list(self.config.voice.fillers),
                voice=self.config.voice.tts.voice,
            )
        return self._fillers

    def pipeline(self) -> VoicePipeline:
        return VoicePipeline(
            stt=self.stt, tts=self.tts, config=self.config.voice,
            ask=self.ask, fillers=self.fillers,
        )

    # --- lifecycle -------------------------------------------------------
    def warm(self) -> dict[str, Any]:
        """Load both models and render the fillers, so the first question pays nothing.

        Called in a thread by the daemon when voice is switched on, and by the first
        request if that has not happened. Never raises: a voice that will not load is
        reported, not thrown.
        """
        started = time.monotonic()
        report: dict[str, Any] = {"stt": "", "tts": "", "fillers": 0}
        try:
            self.stt.warm()
            report["stt"] = self.stt.ready()[1]
        except Exception as exc:  # noqa: BLE001
            report["stt"] = f"would not load: {exc}"
        try:
            self.tts.warm()
            report["tts"] = self.tts.ready()[1]
        except Exception as exc:  # noqa: BLE001
            report["tts"] = f"would not load: {exc}"
        report["fillers"] = self.fillers.prepare()
        self._warmed = True
        self.warm_ms = int((time.monotonic() - started) * 1000)
        report["ms"] = self.warm_ms
        return report

    @staticmethod
    def socket_ready() -> tuple[bool, str]:
        """Whether the fast path can work at all.

        uvicorn speaks WebSocket only when websockets or wsproto is installed. Without
        one, /voice/stream refuses the upgrade and a satellite sees "unsupported
        upgrade request", which is a sentence nobody has ever debugged on the first
        try. Said here instead.
        """
        import importlib.util

        for name in ("websockets", "wsproto"):
            if importlib.util.find_spec(name) is not None:
                return True, f"the socket path is available ({name})"
        return False, (
            "uvicorn cannot speak WebSocket: pip install 'gustav[voice]' (or websockets). "
            "POST /voice/ask still works, a turn at a time."
        )

    def status(self) -> dict[str, Any]:
        """What a browser, a satellite or `ai status` needs to know about the voice."""
        voice = self.config.voice
        stt_ok, stt_why = self.stt.ready()
        tts_ok, tts_why = self.tts.ready()
        socket_ok, socket_why = self.socket_ready()

        # "auto" that fell back to the null engine is not ready, whatever the null
        # engine says about itself. It reports ready because it is — at hearing what
        # it is told to hear, which is a thing for tests and not for a room with a
        # microphone in it. Saying "ready: yes" over a daemon that cannot hear is
        # worse than saying nothing.
        def asked_for(configured: str, engine_name: str) -> bool:
            return engine_name != "null" or (configured or "auto").lower() == "null"

        stt_real = asked_for(voice.stt.engine, self.stt.name)
        tts_real = asked_for(voice.tts.engine, self.tts.name)
        if not stt_real:
            stt_ok, stt_why = False, (
                "no recogniser is installed, so it would hear nothing: "
                "pip install -e '.[voice]' on the server, then restart it"
            )
        if not tts_real:
            tts_ok, tts_why = False, (
                "no voice is installed, so it would say nothing out loud: "
                "pip install -e '.[voice]' on the server, then restart it"
            )
        return {
            "enabled": voice.enabled,
            # Ready means a person could talk to it: hear, speak, and the socket the
            # satellite uses. /voice/ask works without the last one, which is why it
            # is reported separately rather than folded in here.
            "ready": bool(stt_ok and tts_ok),
            "ready_for_a_satellite": bool(stt_ok and tts_ok and socket_ok),
            "warmed": self._warmed,
            "warm_ms": self.warm_ms,
            "language": voice.language,
            "reply_language": voice.reply_language or voice.language,
            "socket": {"ok": socket_ok, "detail": socket_why},
            "stt": {"engine": self.stt.name, "ok": stt_ok, "detail": stt_why,
                    "sample_rate": voice.stt.sample_rate},
            "tts": {"engine": self.tts.name, "ok": tts_ok, "detail": tts_why,
                    "sample_rate": getattr(self.tts, "sample_rate", voice.tts.sample_rate)},
            "wake_words": list(voice.wake_words),
            "filler_after_ms": voice.filler_after_ms,
            "fillers": phrases_for(voice.language, list(voice.fillers)),
            "stream_sentences": voice.stream_sentences,
            "barge_in": voice.barge_in,
            "max_utterance_seconds": voice.stt.max_utterance_seconds,
        }

    # --- the two one-shot things ----------------------------------------
    def transcribe(self, audio: Audio, *, language: str = "") -> dict[str, Any]:
        heard = self.stt.listen(audio, language=language or self.config.voice.language)
        return {"text": heard.text, "language": heard.language, "ms": heard.ms,
                "confidence": heard.confidence}

    def speak(self, text: str, *, language: str = "", voice: str = "") -> Audio:
        return self.tts.speak(
            text,
            language=language or self.config.voice.language,
            voice=voice or self.config.voice.tts.voice,
        )
