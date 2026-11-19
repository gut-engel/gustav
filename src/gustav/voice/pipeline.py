"""One spoken turn, and the four tricks that make it feel instant.

The budget is the gap between someone finishing a sentence and hearing the first
syllable back. Everything here exists to shrink it:

1. **Speak the first sentence before the rest exists.** The model streams tokens; as
   soon as a sentence is complete it is synthesised and sent. The answer's second
   half is generated while its first half is being read aloud, so the length of the
   answer stops mattering.
2. **Say something human while it thinks.** If nothing is ready by
   ``voice.filler_after_ms``, a short pre-rendered phrase goes out — *"one moment"*.
   A person who is thinking says "hmm" too, and silence on a speaker reads as broken
   rather than busy. It is pre-rendered, so it costs nothing at the moment it is
   needed.
3. **Keep the answer short.** ``voice.max_reply_tokens`` is not a quality setting: it
   is an acknowledgement that speech is slow and nobody wants eight paragraphs read
   to them.
4. **Stop the moment the person speaks.** Barge-in is handled by the caller — this
   generator simply stops being iterated — but it only works because the audio leaves
   here a sentence at a time rather than in one block at the end.

Everything is measured. ``Timings`` is the point of the exercise and is returned with
every turn, because a latency claim that nobody measures stops being true within a
month.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from gustav.voice.audio import Audio, sentences
from gustav.voice.engines import Heard, Stt, Tts


@dataclass
class Timings:
    """Milliseconds, from the moment the audio was complete."""

    transcribe_ms: int = 0
    #: The model's first token. Includes loading it, if this was the first question.
    first_token_ms: int = 0
    #: The thing the person actually experiences: when sound started coming back.
    first_audio_ms: int = 0
    #: Whether that first sound was a filler rather than the answer.
    first_audio_was_filler: bool = False
    total_ms: int = 0
    spoken_sentences: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    """What was said, what was answered, and how long each part took."""

    transcript: str = ""
    language: str = ""
    reply: str = ""
    audio: list[Audio] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    ignored: str = ""          # set when the wake word was not heard


def wanted(text: str, wake_words: list[str]) -> bool:
    """Whether a transcript is addressed to us.

    Only used when ``voice.wake_words`` is set. The comparison is deliberately loose —
    speech recognition will give you "hey gustav," with a comma, and "hey, gus tav" on
    a bad day — so it asks whether the opening of the sentence contains the word, not
    whether it equals it.
    """
    if not wake_words:
        return True
    opening = "".join(ch for ch in text.lower()[:40] if ch.isalnum() or ch.isspace())
    return any(
        "".join(ch for ch in word.lower() if ch.isalnum() or ch.isspace()) in opening
        for word in wake_words
    )


def strip_wake_word(text: str, wake_words: list[str]) -> str:
    """"Hey gustav, what time is it" -> "what time is it"."""
    lowered = text.lower()
    for word in sorted(wake_words, key=len, reverse=True):
        at = lowered.find(word.lower())
        if at != -1 and at < 20:
            rest = text[at + len(word) :]
            return rest.lstrip(" ,.:;!?-–—").strip()
    return text.strip()


class VoicePipeline:
    """Audio in, audio out, over whatever answers questions.

    ``ask`` is the only thing it needs from the daemon: an async iterator of dicts
    with ``{"type": "chunk", "text": ...}`` in them — which is exactly what
    ``Daemon.chat_stream`` yields, and exactly what a test can fake.
    """

    def __init__(
        self,
        *,
        stt: Stt,
        tts: Tts,
        config,
        ask: Callable[..., AsyncIterator[dict[str, Any]]],
        fillers=None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.stt = stt
        self.tts = tts
        self.config = config
        self.ask = ask
        self.fillers = fillers
        self.clock = clock

    # --- the parts ------------------------------------------------------
    def transcribe(self, audio: Audio) -> Heard:
        return self.stt.listen(audio, language=self.config.language)

    def say(self, text: str, *, language: str = "") -> Audio:
        return self.tts.speak(
            text, language=language or self.config.language, voice=self.config.tts.voice
        )

    # --- the whole turn -------------------------------------------------
    async def converse(
        self, audio: Audio, *, session: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield events for one spoken turn, in the order a speaker should play them.

        ``transcript`` → (``filler``) → ``audio`` per sentence, with ``reply`` deltas
        in between for anything that is also showing the words. Always ends with
        ``done`` and the timings, even when nothing was said.
        """
        started = self.clock()
        timings = Timings()

        heard = await asyncio.to_thread(self.transcribe, audio)
        timings.transcribe_ms = int((self.clock() - started) * 1000)
        language = heard.language or self.config.language
        yield {"type": "transcript", "text": heard.text, "language": language,
               "ms": timings.transcribe_ms}

        if not heard.text.strip():
            timings.total_ms = int((self.clock() - started) * 1000)
            yield {"type": "done", "ignored": "nothing was said", "timings": timings.to_dict()}
            return

        if not wanted(heard.text, list(self.config.wake_words)):
            timings.total_ms = int((self.clock() - started) * 1000)
            yield {"type": "done", "ignored": "not addressed to me",
                   "timings": timings.to_dict()}
            return

        question = strip_wake_word(heard.text, list(self.config.wake_words))
        if self.config.wake_words and not question:
            timings.total_ms = int((self.clock() - started) * 1000)
            yield {"type": "done", "ignored": "wake word only",
                   "timings": timings.to_dict()}
            return
        if question.lower().strip(" .,!?") in {
            "music mode", "put yourself in music mode", "put yourself into music mode",
            "switch to music mode", "switch into music mode", "go into music mode",
            "start music mode", "enable music mode",
        }:
            timings.total_ms = int((self.clock() - started) * 1000)
            yield {"type": "mode", "mode": "music"}
            yield {"type": "done", "transcript": heard.text,
                   "reply": "Connecting your phone. I will listen again when it disconnects.",
                   "timings": timings.to_dict()}
            return
        if self.config.reply_language:
            question = (
                f"{question}\n\n(Answer in {self.config.reply_language}. "
                f"This answer is going to be read aloud, so keep it to a few sentences.)"
            )

        spoken_any = False
        pending = ""
        reply = ""
        first_token_at = 0.0

        async for event in self.ask(question, session=session):
            if event.get("type") == "error":
                yield {"type": "error", "error": event.get("error", "")}
                break
            text = event.get("text") or (
                event.get("reply", "") if event.get("type") == "command" else ""
            )
            if not text:
                continue
            if not first_token_at:
                first_token_at = self.clock()
                timings.first_token_ms = int((first_token_at - started) * 1000)
            reply += text
            pending += text
            yield {"type": "reply", "delta": text}

            # Nothing to say yet and the silence is getting rude: fill it.
            if (not spoken_any and self.fillers is not None
                    and (self.clock() - started) * 1000 >= self.config.filler_after_ms):
                filler = self.fillers.pick(language)
                if filler is not None:
                    spoken_any = True
                    timings.first_audio_ms = int((self.clock() - started) * 1000)
                    timings.first_audio_was_filler = True
                    yield {"type": "filler", "audio": filler, "text": self.fillers.last_text}

            if not self.config.stream_sentences:
                continue
            finished, pending = sentences(pending)
            for piece in finished:
                clip = await asyncio.to_thread(self.say, piece, language=language)
                if not spoken_any:
                    timings.first_audio_ms = int((self.clock() - started) * 1000)
                    spoken_any = True
                timings.spoken_sentences += 1
                yield {"type": "audio", "audio": clip, "text": piece}

        tail = pending.strip() if self.config.stream_sentences else reply.strip()
        if tail:
            clip = await asyncio.to_thread(self.say, tail, language=language)
            if not spoken_any:
                timings.first_audio_ms = int((self.clock() - started) * 1000)
                spoken_any = True
            timings.spoken_sentences += 1
            yield {"type": "audio", "audio": clip, "text": tail}

        timings.total_ms = int((self.clock() - started) * 1000)
        yield {"type": "done", "transcript": heard.text, "reply": reply,
               "timings": timings.to_dict()}

    async def turn(self, audio: Audio, *, session: str | None = None) -> Turn:
        """The same thing collected into one object, for callers that cannot stream."""
        result = Turn()
        async for event in self.converse(audio, session=session):
            kind = event.get("type")
            if kind == "transcript":
                result.transcript = event["text"]
                result.language = event["language"]
            elif kind in ("audio", "filler"):
                result.audio.append(event["audio"])
            elif kind == "done":
                result.reply = event.get("reply", "")
                result.ignored = event.get("ignored", "")
                result.timings = Timings(**event["timings"])
        return result
