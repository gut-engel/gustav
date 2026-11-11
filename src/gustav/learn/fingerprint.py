"""Content fingerprints.

Every learned thing is keyed by one, so the same fact extracted on three nights is
stored once and a study example rediscovered after a re-fetch does not inflate the
dataset. Normalisation is deliberately aggressive — case, whitespace and trailing
punctuation are not meaningful differences between two statements of the same thing.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _WS.sub(" ", text).strip().lower()
    return text.rstrip(" .;,!?")


def fingerprint(*parts: str | None) -> str:
    """A stable hex digest over the normalised parts."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(normalise(part or "").encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def bucket(fp: str) -> float:
    """A deterministic 0..1 position derived from a fingerprint.

    Used to assign a train/held-out split: the same example always lands on the same
    side of the line, whichever night rediscovers it. A random split would leak
    training data into the gate as soon as a duplicate appeared.
    """
    return int(fp[:8], 16) / 0xFFFFFFFF
