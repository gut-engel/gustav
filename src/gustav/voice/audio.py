"""PCM in, PCM out, and nothing clever in between.

Audio here is always **16-bit signed little-endian mono**, because that is what every
microphone, every speech recogniser and every speaker agrees on. Sample rates differ —
a satellite records at 16 kHz, piper speaks at 22.05 kHz — so there is one resampler,
and it is linear interpolation, which is right at the level of a voice assistant and
wrong for music nobody is playing through this.

``audioop`` would have done most of this, and did until Python 3.13 removed it. So it
is here, in about eighty lines, with no dependency on anything.
"""

from __future__ import annotations

import array
import io
import math
import wave
from dataclasses import dataclass

#: One sample is two bytes. Said once so the arithmetic below reads.
WIDTH = 2


@dataclass(frozen=True)
class Audio:
    """A piece of sound, and the only shape of it this project passes around."""

    pcm: bytes
    sample_rate: int

    @property
    def samples(self) -> int:
        return len(self.pcm) // WIDTH

    @property
    def seconds(self) -> float:
        return self.samples / self.sample_rate if self.sample_rate else 0.0

    def resampled(self, rate: int) -> "Audio":
        return self if rate == self.sample_rate else Audio(
            resample(self.pcm, self.sample_rate, rate), rate
        )

    def to_wav(self) -> bytes:
        return to_wav(self.pcm, self.sample_rate)

    @classmethod
    def from_wav(cls, data: bytes) -> "Audio":
        pcm, rate = from_wav(data)
        return cls(pcm, rate)

    @classmethod
    def silence(cls, seconds: float, rate: int = 16000) -> "Audio":
        return cls(b"\x00\x00" * int(seconds * rate), rate)


def to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """A RIFF header around raw samples. Browsers and `aplay` both want one."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def from_wav(data: bytes) -> tuple[bytes, int]:
    """Samples and rate out of a WAV, downmixed to mono if it arrives in stereo."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != WIDTH:
        raise ValueError(f"only 16-bit audio is handled here, not {width * 8}-bit")
    if channels == 1:
        return frames, rate
    return _downmix(frames, channels), rate


def _downmix(pcm: bytes, channels: int) -> bytes:
    samples = array.array("h")
    samples.frombytes(pcm)
    mono = array.array("h", [0]) * (len(samples) // channels)
    for index in range(len(mono)):
        total = 0
        for channel in range(channels):
            total += samples[index * channels + channel]
        mono[index] = int(total / channels)
    return mono.tobytes()


def resample(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linear interpolation. Honest about what it is: good enough for speech."""
    if source_rate == target_rate or not pcm:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm)
    count = len(samples)
    wanted = max(int(count * target_rate / source_rate), 1)
    out = array.array("h", [0]) * wanted
    step = (count - 1) / (wanted - 1) if wanted > 1 else 0.0
    for index in range(wanted):
        position = index * step
        left = int(position)
        right = min(left + 1, count - 1)
        fraction = position - left
        out[index] = int(samples[left] * (1 - fraction) + samples[right] * fraction)
    return out.tobytes()


def rms(pcm: bytes) -> float:
    """Loudness, 0..1. What a satellite's silence detection is built on."""
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % WIDTH])
    if not samples:
        return 0.0
    total = sum(sample * sample for sample in samples)
    return math.sqrt(total / len(samples)) / 32768.0


def join(pieces: list[Audio]) -> Audio:
    """One stream out of several, resampled to the first one's rate."""
    if not pieces:
        return Audio(b"", 16000)
    rate = pieces[0].sample_rate
    return Audio(b"".join(piece.resampled(rate).pcm for piece in pieces), rate)


#: Where a sentence ends, for the benefit of speaking an answer before it is finished.
#: Deliberately not a parser: it looks for a full stop followed by a space, and it does
#: not care about "e.g." — a wrongly split sentence costs a breath, not an error.
_ENDS = ".!?。！？\n"


def sentences(text: str, *, minimum: int = 12) -> tuple[list[str], str]:
    """Split off every finished sentence, and return the unfinished remainder.

    ``minimum`` stops "Yes." from being spoken on its own while the rest of the answer
    is still arriving — below that it waits for more, because the gap between two
    pieces of synthesised speech is audible and a one-word first piece makes the whole
    answer sound stitched together.
    """
    out: list[str] = []
    start = 0
    for index, character in enumerate(text):
        if character not in _ENDS:
            continue
        # A decimal point, a version number, an ellipsis: not the end of anything.
        after = text[index + 1 : index + 2]
        if character == "." and after and not after.isspace():
            continue
        piece = text[start : index + 1].strip()
        if len(piece) >= minimum:
            out.append(piece)
            start = index + 1
    return out, text[start:]
