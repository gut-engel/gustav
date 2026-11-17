"""The assistant API: the endpoints a satellite, a browser or a script talks to.

Five of them, and the shape of each is decided by what is on the other end:

``GET  /voice/config``     everything a satellite needs to configure itself from the
                          server, so the Pi's own config file holds an address and a
                          token and nothing else worth getting wrong.
``POST /voice/transcribe`` audio in, text out. For anything that already has its own
                          idea of what to do with the words.
``POST /voice/speak``      text in, WAV out. The other half, on its own.
``POST /voice/ask``        audio in, the whole turn back as JSON, audio included. One
                          request, no streaming, no protocol — the version you can
                          test with curl.
``WS   /voice/stream``     the fast path. Frames of audio up, events down as they
                          happen: the transcript the moment it exists, a filler if the
                          model is slow, then each sentence of the answer as it is
                          synthesised. This is the one that gets under half a second,
                          because nothing waits for the end of anything.

All five are behind the same identity rules as everything else: an account, a token,
or the Unix socket. A microphone in a hallway is not a reason to open a hole.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

from gustav.voice.audio import Audio, join

if TYPE_CHECKING:  # pragma: no cover
    from gustav.daemon import Daemon

#: A satellite is not a browser: it streams 20 ms frames and expects the first audio
#: back before it has finished thinking about the next one. Anything bigger than this
#: in one message is a client that has misunderstood, or a bug.
MAX_FRAME_BYTES = 1 << 20
#: And a whole utterance has a ceiling too, or a microphone next to a television
#: becomes a denial of service with extra steps.
MAX_UTTERANCE_BYTES = 32 << 20


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str = ""
    voice: str = ""


def encode(audio: Audio) -> dict[str, Any]:
    """Audio as JSON: base64 PCM plus the rate, which is all a player needs.

    Base64 rather than a second binary frame, because the whole point of this path is
    that there is no protocol to get wrong. At a sentence at a time over a LAN the
    encoding costs about a millisecond.
    """
    return {
        "pcm": base64.b64encode(audio.pcm).decode("ascii"),
        "sample_rate": audio.sample_rate,
        "seconds": round(audio.seconds, 3),
    }


def decode(payload: str | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return base64.b64decode(payload)


def voice_routes(daemon: "Daemon", guard: list[Any], account: list[Any],
                 identify) -> APIRouter:
    """``identify`` is the app's own: the socket is held to the same rules as the API."""
    router = APIRouter(prefix="/voice", tags=["voice"])

    def service():
        voice = daemon.voice
        if voice is None or not daemon.config.voice.enabled:
            raise HTTPException(
                status_code=503,
                detail="the voice assistant is switched off (voice.enabled in config.yaml)",
            )
        return voice

    def audio_from(body: bytes, content_type: str, sample_rate: int) -> Audio:
        """WAV if it looks like one, raw 16-bit PCM otherwise.

        A satellite sends PCM because it has it already and a header is pointless;
        curl sends a WAV because that is what is on disk.
        """
        if len(body) > MAX_UTTERANCE_BYTES:
            raise HTTPException(status_code=413, detail="that is more audio than anyone says")
        if not body:
            raise HTTPException(status_code=400, detail="no audio arrived")
        if body[:4] == b"RIFF" or "wav" in content_type:
            try:
                return Audio.from_wav(body)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"that WAV would not open: {exc}")
        return Audio(body, sample_rate)

    @router.get("/config", dependencies=account)
    async def voice_config() -> dict[str, Any]:
        """What a satellite needs to set itself up. The Pi's own file holds no settings."""
        found = service().status()
        found["instance"] = daemon.config.instance.name
        return found

    @router.get("/settings", dependencies=guard)
    async def read_settings() -> dict[str, Any]:
        """What is configured, what has been set over it, and what may be set."""
        from gustav.voice.service import SETTABLE

        voice = daemon.config.voice
        return {
            "settable": {key: kind.__name__ for key, kind in SETTABLE.items()},
            "set_here": daemon.voice_settings(),
            "effective": {
                "language": voice.language,
                "reply_language": voice.reply_language,
                "filler_after_ms": voice.filler_after_ms,
                "stream_sentences": voice.stream_sentences,
                "barge_in": voice.barge_in,
                "max_reply_tokens": voice.max_reply_tokens,
                "wake_words": list(voice.wake_words),
                "fillers": list(voice.fillers),
                "stt.engine": voice.stt.engine,
                "stt.model": voice.stt.model,
                "stt.device": voice.stt.device,
                "tts.engine": voice.tts.engine,
                "tts.voice": voice.tts.voice,
                "tts.speed": voice.tts.speed,
            },
        }

    @router.post("/settings", dependencies=guard)
    async def write_settings(settings: dict[str, Any]) -> dict[str, Any]:
        """Change the voice. config.yaml is never rewritten; this is an overlay."""
        try:
            stored = daemon.set_voice_settings(settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"set_here": stored, "status": service().status()}

    @router.post("/warm", dependencies=guard)
    async def warm() -> dict[str, Any]:
        """Load the models now. Worth doing before the first person speaks, not after."""
        return await asyncio.to_thread(service().warm)

    @router.post("/transcribe", dependencies=account)
    async def transcribe(request: Request, language: str = "", sample_rate: int = 16000):
        voice = service()
        body = await request.body()
        audio = audio_from(body, request.headers.get("content-type", ""), sample_rate)
        return await asyncio.to_thread(voice.transcribe, audio, language=language)

    @router.post("/speak", dependencies=account)
    async def speak(req: SpeakRequest):
        voice = service()
        audio = await asyncio.to_thread(
            voice.speak, req.text, language=req.language, voice=req.voice
        )
        return Response(content=audio.to_wav(), media_type="audio/wav")

    @router.post("/ask", dependencies=account)
    async def ask(request: Request, session: str = "voice", language: str = "",
                  sample_rate: int = 16000) -> dict[str, Any]:
        """One request, one spoken turn. The version you can test with curl."""
        voice = service()
        body = await request.body()
        audio = audio_from(body, request.headers.get("content-type", ""), sample_rate)
        pipeline = voice.pipeline()
        turn = await pipeline.turn(audio, session=session)
        return {
            "transcript": turn.transcript,
            "language": turn.language,
            "reply": turn.reply,
            "ignored": turn.ignored,
            "audio": encode(join(turn.audio)) if turn.audio else None,
            "timings": turn.timings.to_dict(),
        }

    @router.websocket("/stream")
    async def stream(socket: WebSocket) -> None:
        """The fast path.

        The client sends, in order: an optional JSON ``{"type":"start", …}``, then
        binary frames of 16-bit PCM, then ``{"type":"end"}``. The server answers with
        ``transcript``, perhaps ``filler``, then an ``audio`` event per sentence, then
        ``done``. ``{"type":"stop"}`` at any point abandons the answer, which is how
        barge-in works: the satellite hears its person start talking and says stop.
        """
        # A WebSocket carries the same cookies and headers as a request, so it is held
        # to the same rule. Refused before the handshake completes: an unauthenticated
        # microphone should not get as far as sending audio.
        try:
            identity = identify(socket)
        except Exception:  # noqa: BLE001 - HTTPException, or anything identify throws
            await socket.close(code=4401)
            return
        await socket.accept()

        voice = daemon.voice
        if voice is None or not daemon.config.voice.enabled:
            await socket.send_json({"type": "error",
                                    "error": "the voice assistant is switched off"})
            await socket.close()
            return

        sample_rate = daemon.config.voice.stt.sample_rate
        session = f"voice-{identity.name}"
        try:
            while True:
                frames: list[bytes] = []
                total = 0
                opened = time.monotonic()
                # --- listen ---------------------------------------------
                while True:
                    message = await socket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if (payload := message.get("bytes")) is not None:
                        if len(payload) > MAX_FRAME_BYTES:
                            await socket.send_json(
                                {"type": "error", "error": "that frame is far too big"})
                            return
                        total += len(payload)
                        if total > MAX_UTTERANCE_BYTES:
                            await socket.send_json(
                                {"type": "error", "error": "that is more audio than anyone says"})
                            return
                        frames.append(payload)
                        continue
                    text = message.get("text")
                    if not text:
                        continue
                    try:
                        event = json.loads(text)
                    except ValueError:
                        continue
                    kind = event.get("type")
                    if kind == "start":
                        frames, total = [], 0
                        sample_rate = int(event.get("sample_rate") or sample_rate)
                        opened = time.monotonic()
                    elif kind == "audio" and event.get("pcm"):
                        frames.append(decode(event["pcm"]))
                    elif kind in ("end", "ask"):
                        break
                    elif kind == "stop":
                        frames = []
                        break
                    elif kind == "ping":
                        await socket.send_json({"type": "pong"})

                if not frames:
                    continue
                audio = Audio(b"".join(frames), sample_rate)
                await socket.send_json({
                    "type": "heard",
                    "seconds": round(audio.seconds, 2),
                    "waited_ms": int((time.monotonic() - opened) * 1000),
                })

                # --- answer ---------------------------------------------
                pipeline = voice.pipeline()
                async for event in pipeline.converse(audio, session=session):
                    if event.get("type") in ("audio", "filler"):
                        await socket.send_json({
                            "type": event["type"],
                            "text": event.get("text", ""),
                            **encode(event["audio"]),
                        })
                    else:
                        await socket.send_json(event)
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001 - the satellite gets the reason, not a dead socket
            daemon.log.exception("the voice socket failed")
            with_reason = {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
            try:
                await socket.send_json(with_reason)
            except Exception:  # noqa: BLE001
                pass

    return router
