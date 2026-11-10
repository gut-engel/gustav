"""Stop sequences, done properly.

A stop string almost never lines up with token boundaries: a model emitting
``"</tool"`` + ``">"`` has produced ``"</tool>"`` even though neither token contains it.
So matching happens on the decoded text, and any partial match at the tail is held back
rather than streamed — otherwise the client sees the first half of a stop sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StopResult:
    #: Text safe to emit now.
    emit: str
    #: True once a stop sequence completed.
    stopped: bool = False
    #: Which sequence matched.
    matched: str | None = None


@dataclass
class StopSequenceMatcher:
    """Feed it decoded text; it tells you what is safe to stream."""

    stop: tuple[str, ...] = ()
    #: Everything emitted so far, for matching across chunk boundaries.
    text: str = ""
    _held: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.stop = tuple(s for s in self.stop if s)

    def feed(self, piece: str) -> StopResult:
        if not piece:
            return StopResult(emit="")
        if not self.stop:
            self.text += piece
            return StopResult(emit=piece)

        buffer = self._held + piece
        # Earliest match wins; on a tie the longer sequence does. Iterating the stop
        # list in order would otherwise let a later-occurring sequence pre-empt one the
        # model actually produced first.
        hit: tuple[int, int, str] | None = None
        for sequence in self.stop:
            index = buffer.find(sequence)
            if index < 0:
                continue
            candidate = (index, -len(sequence), sequence)
            if hit is None or candidate < hit:
                hit = candidate
        if hit is not None:
            index, _, sequence = hit
            emit = buffer[:index]
            self.text += emit
            self._held = ""
            return StopResult(emit=emit, stopped=True, matched=sequence)

        # Hold back anything that could still become a stop sequence.
        hold = self._longest_partial_suffix(buffer)
        emit = buffer[: len(buffer) - hold] if hold else buffer
        self._held = buffer[len(buffer) - hold :] if hold else ""
        self.text += emit
        return StopResult(emit=emit)

    def flush(self) -> str:
        """Generation ended without a stop match: release what was held back."""
        held, self._held = self._held, ""
        self.text += held
        return held

    def _longest_partial_suffix(self, buffer: str) -> int:
        longest = 0
        for sequence in self.stop:
            # A suffix of the buffer that is a proper prefix of this stop sequence.
            for length in range(min(len(sequence) - 1, len(buffer)), 0, -1):
                if buffer.endswith(sequence[:length]):
                    longest = max(longest, length)
                    break
        return longest
