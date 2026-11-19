#!/usr/bin/env python3
"""A gustav satellite: a microphone and a speaker in another room.

It records, decides when you have stopped talking, sends the audio to the daemon, and
plays the answer as it arrives — sentence by sentence, so the speaker starts before
the sentence after it has been generated.

**It has no dependencies.** Not numpy, not sounddevice, not websockets: a Raspberry Pi
Zero should be able to run this ten seconds after the installer finishes, and every
wheel that has to be compiled on an ARM board is a minute of someone's evening and a
new way for an install to fail. Recording and playback are `arecord` and `aplay`, which
are on every Linux with a sound card, and the WebSocket client is the two hundred lines
at the bottom of this file.

What makes it feel fast, in the order it matters:

* **Pre-roll.** The two hundred milliseconds *before* speech was detected are sent too.
  Without it every utterance loses its first consonant and "play" becomes "lay".
* **Streaming up.** Frames go out while you are still talking, so when you stop, the
  server already has all but the last breath of it.
* **Streaming down.** The answer is played as each sentence arrives.
* **No feedback loop.** The microphone stays open for ALSA, but frames captured while
  Gustav is speaking are discarded before the next turn, so his answer cannot become
  the next question.

Run `gustav-satellite configure` to set the server, token and sound devices in a short
wizard. `--calibrate` measures the room's speech threshold when you want to tune it.
"""

from __future__ import annotations

import argparse
import array
import base64
import json
import math
import os
import queue
import re
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

VERSION = "2026-11-18.1"

#: 20 ms of 16 kHz 16-bit mono. The unit everything here counts in.
FRAME_MS = 20
DEFAULT_RATE = 16000


def frame_bytes(rate: int) -> int:
    return int(rate * FRAME_MS / 1000) * 2


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """What this box needs to know. Everything else comes from the server."""

    url: str = "http://127.0.0.1:8713"
    token: str = ""
    #: ALSA device names, e.g. "plughw:1,0". Empty means the default.
    input_device: str = ""
    output_device: str = ""
    #: Loudness that counts as speech, 0..1. `--calibrate` measures it for your room.
    threshold: float = 0.02
    #: How much quiet ends an utterance. Shorter feels snappier and cuts people off.
    silence_ms: int = 700
    #: The audio kept from before speech started, so the first word survives.
    preroll_ms: int = 240
    #: Speech this short is a cough or a door.
    min_speech_ms: int = 250
    #: Stop recording after this, whatever happens.
    max_speech_ms: int = 20000
    rate: int = DEFAULT_RATE
    #: Say something back when the connection is lost, rather than going silent.
    announce_problems: bool = True
    verbose: bool = False

    @classmethod
    def load(cls, path: Path) -> "Settings":
        found = cls()
        if not path.is_file():
            return found
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if not hasattr(found, key):
                continue
            current = getattr(found, key)
            if isinstance(current, bool):
                setattr(found, key, value.lower() in ("1", "true", "yes", "on"))
            elif isinstance(current, int):
                setattr(found, key, int(float(value)))
            elif isinstance(current, float):
                setattr(found, key, float(value))
            else:
                setattr(found, key, value)
        return found


def say(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    print(f"{time.strftime('%H:%M:%S')}  {message}", file=stream, flush=True)


def why(exc: Exception) -> str:
    """What the daemon said, not just the status line it said it with.

    An HTTPError carries the response body, and for this daemon that body is a
    sentence naming the thing to do about it — "piper has no voice installed: `ai
    voice install …`". Printing "503 Service Unavailable" and throwing that away
    leaves someone with a number and no next step.
    """
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        try:
            body = exc.read().decode("utf-8", "replace")
            try:
                detail = (json.loads(body) or {}).get("detail", "") or body
            except ValueError:
                detail = body
        except Exception:  # noqa: BLE001
            detail = ""
        detail = " ".join(detail.split())[:400]
        return f"{exc.code} {exc.reason}" + (f" — {detail}" if detail else "")
    return str(exc)


# ---------------------------------------------------------------------------
# sound
# ---------------------------------------------------------------------------


def rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0


class Microphone:
    """`arecord` in a thread, 20 ms at a time, into a queue.

    A thread because the main loop has to be able to listen while the speaker is
    talking — that is what barge-in is — and because a blocking read on a pipe is the
    simplest correct way to get audio out of ALSA.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        #: What ALSA said, if arecord would not start at all.
        self.failed = ""
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        argv = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1",
                "-r", str(self.settings.rate)]
        if self.settings.input_device:
            argv += ["-D", self.settings.input_device]
        self._process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # arecord fails in a few milliseconds when the device is wrong, and the
        # complaint goes to a pipe nobody was reading. Catching it here turns "no
        # audio arrived" into the line ALSA actually printed.
        time.sleep(0.15)
        if self._process.poll() is not None:
            self.failed = (self._process.stderr.read() or b"").decode(
                "utf-8", "replace").strip()
            return
        self._thread = threading.Thread(target=self._pump, name="microphone", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        size = frame_bytes(self.settings.rate)
        stream = self._process.stdout
        while not self._stop.is_set():
            chunk = stream.read(size)
            if not chunk:
                break
            try:
                self.frames.put_nowait(chunk)
            except queue.Full:
                # Dropping the oldest frame is better than growing without bound: this
                # happens when the Pi is briefly busy, and old audio is worth nothing.
                try:
                    self.frames.get_nowait()
                    self.frames.put_nowait(chunk)
                except queue.Empty:
                    pass

    def read(self, timeout: float = 1.0) -> bytes | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()


class Speaker:
    """`aplay`, opened for a turn and fed as the audio arrives."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._process: subprocess.Popen | None = None
        self.rate = 0

    def play(self, pcm: bytes, rate: int) -> None:
        if self._process is None or self.rate != rate:
            self.stop()
            argv = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(rate)]
            if self.settings.output_device:
                argv += ["-D", self.settings.output_device]
            self._process = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL)
            self.rate = rate
        try:
            self._process.stdin.write(pcm)
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._process = None

    @property
    def talking(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def finish(self) -> None:
        """Close the pipe and let what is buffered finish playing."""
        if self._process is None:
            return
        try:
            self._process.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def stop(self) -> None:
        """Stop now, mid-word, when playback has to be interrupted."""
        if self._process is None:
            return
        self._process.kill()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        self._process = None


class BluetoothMusic:
    """Let a phone play through the satellite until it disconnects."""

    @staticmethod
    def source() -> str:
        """Return the Bluetooth audio source PulseAudio exposes, if one is connected."""
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"], capture_output=True,
                text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) > 1 and fields[1].startswith(("bluez_input", "bluez_source")):
                return fields[1]
        return ""

    @classmethod
    def wait_for_phone(cls) -> None:
        """Open pairing, route a connected phone to the normal speaker, then return."""
        if shutil.which("bluetoothctl") is None or shutil.which("pactl") is None:
            say("music mode needs bluetoothctl and pactl; returning to Gustav", error=True)
            return
        try:
            subprocess.run(["bluetoothctl", "power", "on"], check=True,
                           capture_output=True, timeout=10)
            subprocess.run(["bluetoothctl", "agent", "NoInputNoOutput"], check=True,
                           capture_output=True, timeout=10)
            subprocess.run(["bluetoothctl", "default-agent"], check=True,
                           capture_output=True, timeout=10)
            subprocess.run(["bluetoothctl", "pairable", "on"], check=True,
                           capture_output=True, timeout=10)
            subprocess.run(["bluetoothctl", "discoverable", "on"], check=True,
                           capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            say(f"Bluetooth could not be opened: {exc}; returning to Gustav", error=True)
            return
        say("music mode: pair your phone with this satellite")
        module = ""
        connected = False
        try:
            while True:
                source = cls.source()
                if source:
                    connected = True
                    if not module:
                        try:
                            loaded = subprocess.run(
                                ["pactl", "load-module", "module-loopback",
                                 f"source={source}", "latency_msec=80"],
                                capture_output=True, text=True, timeout=10, check=True,
                            )
                            module = loaded.stdout.strip()
                            say("phone connected; playing through the satellite speaker")
                        except (OSError, subprocess.SubprocessError) as exc:
                            say(f"the phone connected but could not be routed: {exc}",
                                error=True)
                            break
                elif connected:
                    say("phone disconnected; listening to Gustav again")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            say("leaving music mode")
        finally:
            if module:
                subprocess.run(["pactl", "unload-module", module],
                               capture_output=True, check=False)
            subprocess.run(["bluetoothctl", "discoverable", "off"],
                           capture_output=True, check=False)
            subprocess.run(["bluetoothctl", "pairable", "off"],
                           capture_output=True, check=False)

# ---------------------------------------------------------------------------
# a WebSocket client, because one file with no dependencies beats one with four
# ---------------------------------------------------------------------------

OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x1, 0x2, 0x8, 0x9, 0xA


class WebSocketError(RuntimeError):
    pass


class WebSocket:
    """RFC 6455, client side, the parts a satellite uses.

    Text and binary frames, ping and pong, close, and masking — which the standard
    requires of clients and every server enforces. No extensions, no fragmentation on
    send, and received fragments are reassembled.
    """

    def __init__(self, url: str, *, headers: dict[str, str] | None = None,
                 timeout: float = 30.0):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._buffer = b""

    def connect(self) -> None:
        parsed = urlparse(self.url)
        secure = parsed.scheme in ("wss", "https")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if secure else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        raw = socket.create_connection((host, port), timeout=self.timeout)
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if secure:
            context = ssl.create_default_context()
            raw = context.wrap_socket(raw, server_hostname=host)
        self._socket = raw

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        lines += [f"{name}: {value}" for name, value in self.headers.items()]
        raw.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))

        response = b""
        while b"\r\n\r\n" not in response:
            piece = raw.recv(4096)
            if not piece:
                raise WebSocketError("the server closed the connection during the handshake")
            response += piece
        head, _, rest = response.partition(b"\r\n\r\n")
        status = head.split(b"\r\n")[0].decode("latin-1")
        if "101" not in status:
            raise WebSocketError(f"the server would not upgrade: {status}")
        self._buffer = rest

    # --- frames ---------------------------------------------------------
    def _send(self, opcode: int, payload: bytes) -> None:
        if self._socket is None:
            raise WebSocketError("not connected")
        header = bytearray([0x80 | opcode])
        mask = os.urandom(4)
        size = len(payload)
        if size < 126:
            header.append(0x80 | size)
        elif size < (1 << 16):
            header.append(0x80 | 126)
            header += struct.pack(">H", size)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", size)
        header += mask
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._socket.sendall(bytes(header) + masked)

    def send_json(self, payload: dict) -> None:
        self._send(OP_TEXT, json.dumps(payload).encode("utf-8"))

    def send_bytes(self, payload: bytes) -> None:
        self._send(OP_BINARY, payload)

    def _read(self, count: int) -> bytes:
        while len(self._buffer) < count:
            piece = self._socket.recv(65536)
            if not piece:
                raise WebSocketError("the connection closed")
            self._buffer += piece
        out, self._buffer = self._buffer[:count], self._buffer[count:]
        return out

    def recv(self) -> tuple[int, bytes]:
        """One message, with fragments joined and pings answered."""
        pieces: list[bytes] = []
        first_opcode = 0
        while True:
            head = self._read(2)
            final = bool(head[0] & 0x80)
            opcode = head[0] & 0x0F
            size = head[1] & 0x7F
            if size == 126:
                size = struct.unpack(">H", self._read(2))[0]
            elif size == 127:
                size = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(size) if size else b""

            if opcode == OP_PING:
                self._send(OP_PONG, payload)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                raise WebSocketError("the server closed the connection")
            if opcode != 0:
                first_opcode = opcode
            pieces.append(payload)
            if final:
                return first_opcode, b"".join(pieces)

    def recv_json(self) -> dict:
        opcode, payload = self.recv()
        if opcode != OP_TEXT:
            return {"type": "binary", "bytes": payload}
        return json.loads(payload.decode("utf-8"))

    def settimeout(self, seconds: float | None) -> None:
        if self._socket is not None:
            self._socket.settimeout(seconds)

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._send(OP_CLOSE, b"")
        except OSError:
            pass
        try:
            self._socket.close()
        finally:
            self._socket = None


# ---------------------------------------------------------------------------
# the satellite
# ---------------------------------------------------------------------------


@dataclass
class Server:
    """What the daemon says about itself, so this file holds no settings of its own."""

    rate: int = DEFAULT_RATE
    language: str = "en"
    wake_words: list[str] = field(default_factory=list)
    max_utterance_seconds: float = 20.0
    barge_in: bool = True
    name: str = "gustav"


class Satellite:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.server = Server()
        self.mic = Microphone(settings)
        self.speaker = Speaker(settings)
        self.music_requested = False

    # --- talking to the daemon ------------------------------------------
    def _http(self, path: str) -> dict:
        import urllib.request

        request = urllib.request.Request(self.settings.url.rstrip("/") + path)
        if self.settings.token:
            request.add_header("Authorization", f"Bearer {self.settings.token}")
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    def hello(self) -> None:
        """Ask the server how it wants to be spoken to."""
        found = self._http("/voice/config")
        self.server = Server(
            rate=int(found.get("stt", {}).get("sample_rate") or DEFAULT_RATE),
            language=found.get("language", "en"),
            wake_words=list(found.get("wake_words") or []),
            max_utterance_seconds=float(found.get("max_utterance_seconds") or 20.0),
            barge_in=bool(found.get("barge_in", True)),
            name=found.get("instance", "gustav"),
        )
        self.settings.rate = self.server.rate
        say(f"{self.server.name}: {self.server.language}, {self.server.rate} Hz"
            + (f", wake word {self.server.wake_words}" if self.server.wake_words else ""))

    def socket(self) -> WebSocket:
        url = self.settings.url.rstrip("/")
        url = url.replace("https://", "wss://").replace("http://", "ws://")
        headers = {}
        if self.settings.token:
            headers["Authorization"] = f"Bearer {self.settings.token}"
        connection = WebSocket(f"{url}/voice/stream", headers=headers)
        connection.connect()
        return connection

    # --- listening -------------------------------------------------------
    def wait_for_speech(self, connection: WebSocket) -> bool:
        """Block until someone starts talking, then stream until they stop.

        Returns False if the loop should give up on this utterance (too short to be
        anything, or too long to be a question).
        """
        settings = self.settings
        size = frame_bytes(settings.rate)
        per_frame_ms = FRAME_MS
        preroll_frames = max(settings.preroll_ms // per_frame_ms, 1)
        silence_frames = max(settings.silence_ms // per_frame_ms, 1)
        minimum_frames = max(settings.min_speech_ms // per_frame_ms, 1)
        maximum_frames = max(settings.max_speech_ms // per_frame_ms, 1)

        preroll: list[bytes] = []
        speaking = False
        quiet = 0
        sent = 0

        while True:
            frame = self.mic.read(timeout=1.0)
            if frame is None:
                if speaking:
                    break
                continue
            loud = rms(frame) >= settings.threshold

            if not speaking:
                preroll.append(frame)
                if len(preroll) > preroll_frames:
                    preroll.pop(0)
                if not loud:
                    continue
                # Speech. Send what came just before it, or the first word is clipped.
                speaking = True
                connection.send_json({"type": "start", "sample_rate": settings.rate})
                for kept in preroll:
                    connection.send_bytes(kept)
                sent = len(preroll)
                preroll.clear()
                if settings.verbose:
                    say("listening")
                continue

            connection.send_bytes(frame)
            sent += 1
            quiet = 0 if loud else quiet + 1
            if quiet >= silence_frames:
                break
            if sent >= maximum_frames:
                if settings.verbose:
                    say("that is enough for one question")
                break

        if sent < minimum_frames:
            connection.send_json({"type": "stop"})
            return False
        connection.send_json({"type": "end"})
        return True

    # --- answering -------------------------------------------------------
    def play_answer(self, connection: WebSocket) -> None:
        """Play what comes back without mistaking the speaker for its listener."""
        settings = self.settings
        started = time.monotonic()
        first_audio_at = None
        self.mic.drain()

        while True:
            connection.settimeout(60.0)
            event = connection.recv_json()
            kind = event.get("type")

            if kind in ("audio", "filler"):
                if first_audio_at is None:
                    first_audio_at = time.monotonic()
                    say(f"answering after {int((first_audio_at - started) * 1000)} ms"
                        + (" (filler)" if kind == "filler" else ""))
                self.speaker.play(base64.b64decode(event["pcm"]), int(event["sample_rate"]))
            elif kind == "transcript":
                say(f'heard: "{event.get("text", "")}"')
            elif kind == "error":
                say(event.get("error", "something went wrong"), error=True)
            elif kind == "mode" and event.get("mode") == "music":
                self.music_requested = True
                say("switching to music mode")
                say_something(self.settings,
                              "Pair your phone with me for music. I will listen again when it disconnects.")
            elif kind == "done":
                timings = event.get("timings") or {}
                if event.get("ignored"):
                    say(f"ignored: {event['ignored']}")
                elif settings.verbose:
                    reply = event.get("reply", "")
                    say(f"reply: {reply or '(empty)'}")
                    say("  ".join(f"{name}={value}" for name, value in timings.items()))
                elif timings:
                    say("  ".join(f"{name}={value}" for name, value in timings.items()))
                break

        self.speaker.finish()
        # The microphone has to stay open while ALSA is playing, but its queue is
        # full of Gustav's own voice by then. Keeping those frames would make the next
        # turn start with an answer the satellite just played back to itself.
        self.mic.drain()

    # --- the loop --------------------------------------------------------
    def run(self) -> int:
        for binary in ("arecord", "aplay"):
            if shutil.which(binary) is None:
                say(f"{binary} is not installed: apt install alsa-utils", error=True)
                return 2
        try:
            self.hello()
        except Exception as exc:  # noqa: BLE001
            say(f"cannot reach {self.settings.url}: {why(exc)}", error=True)
            return 3

        self.mic.start()
        if self.mic.failed:
            say(f"the microphone would not open: {self.mic.failed}", error=True)
            for device in capture_devices():
                say(f"  try:  input_device = {device}   (in your config)", error=True)
            return 4
        say("listening. Ctrl-C to stop.")
        backoff = 1.0
        try:
            while True:
                try:
                    connection = self.socket()
                except Exception as exc:  # noqa: BLE001
                    say(f"cannot open the voice socket: {exc}", error=True)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                backoff = 1.0
                try:
                    while True:
                        if not self.wait_for_speech(connection):
                            continue
                        self.play_answer(connection)
                        if self.music_requested:
                            self.music_requested = False
                            BluetoothMusic.wait_for_phone()
                except WebSocketError as exc:
                    say(f"the connection dropped: {exc}", error=True)
                except OSError as exc:
                    say(f"the connection failed: {exc}", error=True)
                finally:
                    connection.close()
        except KeyboardInterrupt:
            say("stopping")
            return 0
        finally:
            self.speaker.stop()
            self.mic.stop()


# ---------------------------------------------------------------------------
# the other things you do with it once
# ---------------------------------------------------------------------------


def capture_devices() -> list[str]:
    """Every capture device `arecord -l` knows, as something you can pass to -D."""
    try:
        listed = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in listed.splitlines():
        match = re.match(r"^card (\d+):.*device (\d+):", line)
        if match:
            found.append(f"plughw:{match.group(1)},{match.group(2)}")
    return found


def playback_devices() -> list[str]:
    """Every playback device `aplay -l` knows, as something you can pass to -D."""
    try:
        listed = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in listed.splitlines():
        match = re.match(r"^card (\d+):.*device (\d+):", line)
        if match:
            found.append(f"plughw:{match.group(1)},{match.group(2)}")
    return found


def save_settings(settings: Settings, path: Path) -> None:
    """Change the satellite's answers while keeping its notes and other settings."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "url": settings.url,
        "token": settings.token,
        "input_device": settings.input_device,
        "output_device": settings.output_device,
        "threshold": f"{settings.threshold:g}",
    }
    try:
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    except OSError as exc:
        raise OSError(f"could not read {path}: {exc}") from exc
    written: set[str] = set()
    output = []
    for line in lines:
        match = re.match(r"^(\s*)(url|token|input_device|output_device|threshold)\s*=", line)
        if match:
            key = match.group(2)
            output.append(f"{match.group(1)}{key} = {values[key]}")
            written.add(key)
        else:
            output.append(line)
    output.extend(f"{key} = {value}" for key, value in values.items() if key not in written)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write("\n".join(output) + "\n")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _choose_device(label: str, current: str, devices: list[str]) -> str:
    if devices:
        say(f"{label} devices:")
        for number, device in enumerate(devices, 1):
            say(f"  {number}. {device}")
    else:
        say(f"no {label.lower()} devices were listed; you can enter an ALSA device name")
    answer = input(f"{label} [Enter keeps {current or 'system default'}]: ").strip()
    if not answer:
        return current
    if answer.lower() in ("default", "system default", "none"):
        return ""
    if answer.isdecimal() and 1 <= int(answer) <= len(devices):
        return devices[int(answer) - 1]
    return answer


def configure(settings: Settings, path: Path) -> int:
    """Ask the person at the satellite for the few things this room knows."""
    import getpass

    say("Satellite setup. Press Enter to keep a saved answer.")
    try:
        while True:
            address = input(f"Gustav server [{settings.url}]: ").strip()
            if not address:
                break
            if not address.startswith(("http://", "https://")):
                address = "http://" + address
            if urlparse(address).netloc:
                settings.url = address.rstrip("/")
                break
            say("enter a server name such as http://box.local:8713")

        token_prompt = "Token [saved; Enter keeps it, - clears it]: " \
            if settings.token else "Token [none; - leaves it empty]: "
        token = getpass.getpass(token_prompt)
        if token == "-":
            settings.token = ""
        elif token:
            settings.token = token

        settings.input_device = _choose_device(
            "Microphone", settings.input_device, capture_devices()
        )
        settings.output_device = _choose_device(
            "Speaker", settings.output_device, playback_devices()
        )
        while True:
            raw = input(f"Speech threshold, 0..1 [{settings.threshold:g}]: ").strip()
            if not raw:
                break
            try:
                threshold = float(raw)
            except ValueError:
                threshold = -1
            if 0 < threshold <= 1:
                settings.threshold = threshold
                break
            say("enter a number greater than 0 and no more than 1")

        if input("Measure the room and calibrate the mic? [y/N]: ").strip().lower() in (
            "y", "yes",
        ):
            if calibrate(settings, save_threshold=True) != 0:
                say("keeping the threshold you entered")

        save_settings(settings, path)
    except (EOFError, KeyboardInterrupt):
        say("setup cancelled; no changes were saved")
        return 1
    except OSError as exc:
        say(str(exc), error=True)
        return 1

    say(f"saved {path} (readable by your account only)")
    systemctl = shutil.which("systemctl")
    if systemctl:
        try:
            active = subprocess.run(
                [systemctl, "--user", "is-active", "--quiet", "gustav-satellite"],
                capture_output=True, timeout=5, check=False,
            ).returncode == 0
            if active:
                restarted = subprocess.run(
                    [systemctl, "--user", "restart", "gustav-satellite"],
                    capture_output=True, text=True, timeout=15, check=False,
                )
                if restarted.returncode == 0:
                    say("restarted the satellite service with the new settings")
                else:
                    say("settings are saved, but the satellite service did not restart; "
                        "run systemctl --user restart gustav-satellite", error=True)
        except (OSError, subprocess.SubprocessError):
            say("settings are saved; restart gustav-satellite if it is running", error=True)
    if input("Play a test through the satellite speaker? [Y/n]: ").strip().lower() \
            not in ("n", "no"):
        say_something(settings, "Gustav is ready to listen.")
    return 0


def calibrate(settings: Settings, seconds: float = 6.0, *, save_threshold: bool = False) -> int:
    """Measure the room. Print the threshold to put in the config."""
    mic = Microphone(settings)
    mic.start()
    say(f"say nothing for {seconds:.0f} seconds…")
    quiet: list[float] = []
    until = time.monotonic() + seconds
    try:
        while time.monotonic() < until:
            frame = mic.read(timeout=1.0)
            if frame:
                quiet.append(rms(frame))
        say("now say something, at the distance you will usually be…")
        loud: list[float] = []
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            frame = mic.read(timeout=1.0)
            if frame:
                loud.append(rms(frame))
    finally:
        mic.stop()
    if not quiet or not loud:
        say("no audio arrived.", error=True)
        if mic.failed:
            say(f"  arecord said: {mic.failed}", error=True)
        if settings.input_device:
            say(f"  it was recording from {settings.input_device}.", error=True)
        else:
            say("  it was recording from the default device, which on a Pi is usually",
                error=True)
            say("  card 0 — HDMI, or nothing at all. The microphone is the other one.",
                error=True)
        for device in capture_devices():
            say(f"  try:  gustav-satellite --calibrate --input {device}", error=True)
        say("  then put it in the config:  input_device = <that>", error=True)
        return 1
    floor = sum(quiet) / len(quiet)
    peak = sorted(loud)[int(len(loud) * 0.9)]
    suggested = max(floor * 3, (floor + peak) / 3)
    say(f"background {floor:.4f}, speech {peak:.4f}")
    say(f"threshold = {suggested:.4f}")
    if peak < floor * 2:
        say("those are too close together: move the microphone or speak up", error=True)
        return 1
    if save_threshold:
        settings.threshold = suggested
    return 0


def say_something(settings: Settings, text: str) -> int:
    """Make the daemon say a sentence through this speaker. The first thing to try."""
    import urllib.request

    request = urllib.request.Request(
        settings.url.rstrip("/") + "/voice/speak",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    if settings.token:
        request.add_header("Authorization", f"Bearer {settings.token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            wav = response.read()
    except Exception as exc:  # noqa: BLE001
        say(f"the daemon would not speak: {why(exc)}", error=True)
        return 1
    argv = ["aplay", "-q"]
    if settings.output_device:
        argv += ["-D", settings.output_device]
    subprocess.run(argv, input=wav, check=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gustav-satellite", description=__doc__.split("\n")[0]
    )
    parser.add_argument("command", nargs="?", choices=("configure",))
    parser.add_argument("--config", default=os.environ.get(
        "GUSTAV_SATELLITE_CONFIG", str(Path.home() / ".config" / "gustav-satellite.conf")))
    parser.add_argument("--url", help="the daemon, e.g. http://box.local:8713")
    parser.add_argument("--token", help="a token from `ai user token <name>`")
    parser.add_argument("--threshold", type=float, help="loudness that counts as speech")
    parser.add_argument("--input", dest="input_device", help="ALSA input, e.g. plughw:1,0")
    parser.add_argument("--output", dest="output_device", help="ALSA output")
    parser.add_argument("--calibrate", action="store_true", help="measure this room and stop")
    parser.add_argument("--say", help="have the daemon say this through the speaker, and stop")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)

    if args.version:
        print(f"gustav-satellite {VERSION}")
        return 0

    config_path = Path(args.config).expanduser()
    settings = Settings.load(config_path)
    for name in ("url", "token", "threshold", "input_device", "output_device"):
        value = getattr(args, name, None)
        if value:
            setattr(settings, name, value)
    settings.verbose = settings.verbose or args.verbose
    if os.environ.get("GUSTAV_URL"):
        settings.url = os.environ["GUSTAV_URL"]
    if os.environ.get("GUSTAV_TOKEN"):
        settings.token = os.environ["GUSTAV_TOKEN"]

    if args.command == "configure":
        return configure(settings, config_path)
    if args.calibrate:
        return calibrate(settings)
    if args.say:
        return say_something(settings, args.say)
    return Satellite(settings).run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
