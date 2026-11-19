"""Talking to it.

No microphone, no speaker, no models: the recogniser hears what the test tells it to
hear and the voice produces silence of a believable length, which is enough to test
everything that actually goes wrong — the order things come back in, what happens
while the model is slow, who is allowed to connect, and whether the numbers the
latency claim rests on are being measured at all.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket as socket_module
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from gustav.config.load import load_config
from gustav.daemon import Daemon
from gustav.server.app import create_app
from gustav.voice.audio import Audio, join, rms, sentences
from gustav.voice.engines import NullStt, NullTts, build_stt, build_tts
from gustav.voice.fillers import Fillers, phrases_for
from gustav.voice.pipeline import VoicePipeline, strip_wake_word, wanted
from gustav.voice.service import VoiceService


# --- the audio plumbing ---------------------------------------------------
def test_a_wav_survives_the_round_trip():
    original = Audio(b"\x01\x02" * 1000, 16000)
    back = Audio.from_wav(original.to_wav())
    assert back.pcm == original.pcm
    assert back.sample_rate == 16000
    assert back.samples == 1000


def test_resampling_changes_the_length_and_not_much_else():
    quiet = Audio.silence(1.0, 16000)
    louder = quiet.resampled(22050)
    assert louder.sample_rate == 22050
    assert abs(louder.seconds - 1.0) < 0.01
    assert rms(louder.pcm) == 0.0


def test_sentences_come_out_whole_or_not_at_all():
    done, rest = sentences("The driver is too old. Run ai doctor on the")
    assert done == ["The driver is too old."]
    assert rest == " Run ai doctor on the"


def test_a_decimal_point_is_not_the_end_of_a_sentence():
    done, rest = sentences("Version 2.15 needs a newer driver than this one has")
    assert done == []
    assert "2.15" in rest


def test_a_very_short_sentence_waits_for_the_next_one():
    """"Yes." on its own makes an answer sound stitched together."""
    done, _ = sentences("Yes. ")
    assert done == []


# --- being addressed ------------------------------------------------------
@pytest.mark.parametrize(
    ("said", "heard"),
    [
        ("Hey Gustav, what time is it", True),
        ("hey gustav what time is it", True),
        ("Hey, Gustav: what time is it", True),
        ("what time is it", False),
        ("I was telling Gustav about it yesterday and he said no", False),
    ],
)
def test_the_wake_word_is_matched_loosely_but_only_at_the_start(said, heard):
    assert wanted(said, ["hey gustav"]) is heard


def test_with_no_wake_word_everything_is_for_us():
    assert wanted("what time is it", []) is True


def test_the_wake_word_is_taken_off_before_the_model_sees_it():
    assert strip_wake_word("Hey Gustav, what time is it", ["hey gustav"]) == "what time is it"
    assert strip_wake_word("what time is it", ["hey gustav"]) == "what time is it"


# --- the turn -------------------------------------------------------------
def answerer(pieces: list[str], *, delay: float = 0.0):
    async def ask(question: str, session: str | None = None) -> AsyncIterator[dict]:
        ask.asked.append(question)
        for piece in pieces:
            if delay:
                await asyncio.sleep(delay)
            yield {"type": "chunk", "text": piece}
        yield {"type": "done"}

    ask.asked = []
    return ask


def pipeline(config, *, script=None, pieces=None, delay=0.0, room=None, tts=None):
    tts = tts or NullTts()
    fillers = Fillers(tts, Path(room or "/nonexistent"), language=config.language)
    return VoicePipeline(
        stt=NullStt(script=script or ["hello"]),
        tts=tts,
        config=config,
        ask=answerer(pieces or ["Hello back."], delay=delay),
        fillers=fillers,
    )


async def test_a_spoken_turn_comes_back_in_the_order_a_speaker_plays_it(voice_config, tmp_path):
    pipe = pipeline(voice_config, script=["what is wrong"],
                    pieces=["The driver ", "is too old. ", "Run ai doctor."], room=tmp_path)
    kinds = []
    async for event in pipe.converse(Audio.silence(1.0)):
        kinds.append(event["type"])
    assert kinds[0] == "transcript"
    assert kinds[-1] == "done"
    assert "audio" in kinds


async def test_the_first_sentence_is_spoken_before_the_answer_is_finished(voice_config, tmp_path):
    """The single biggest thing in the latency budget."""
    tts = NullTts()
    pipe = pipeline(voice_config, pieces=["The driver is too old. ", "Run ai doctor on it."],
                    delay=0.05, room=tmp_path, tts=tts)
    first_audio_after = None
    deltas_after_first_audio = 0
    async for event in pipe.converse(Audio.silence(1.0)):
        if event["type"] == "audio" and first_audio_after is None:
            first_audio_after = event["text"]
        elif event["type"] == "reply" and first_audio_after is not None:
            deltas_after_first_audio += 1
    assert first_audio_after == "The driver is too old."
    assert deltas_after_first_audio >= 1, "it waited for the whole answer"
    assert tts.said[0] == "The driver is too old."


async def test_nothing_is_spoken_twice_and_the_tail_is_not_dropped(voice_config, tmp_path):
    tts = NullTts()
    pipe = pipeline(
        voice_config,
        pieces=["The driver is too old. ", "Torch cannot start with it. ",
                "Run ai doctor and it will say which"],
        room=tmp_path, tts=tts,
    )
    turn = await pipe.turn(Audio.silence(0.5))
    spoken = " ".join(tts.said)
    assert spoken.count("Torch cannot start") == 1
    # The last piece has no full stop at all and is still said.
    assert tts.said[-1].endswith("which")
    assert turn.reply.strip().endswith("which")


async def test_short_sentences_are_grouped_rather_than_stuttered(voice_config, tmp_path):
    """Three two-word clips in a row sound like a machine reading a list."""
    tts = NullTts()
    pipe = pipeline(voice_config, pieces=["One. ", "Two. ", "Three."], room=tmp_path, tts=tts)
    await pipe.turn(Audio.silence(0.5))
    assert len(tts.said) == 1
    assert tts.said[0] == "One. Two. Three."


async def test_a_slow_model_gets_a_filler_rather_than_silence(voice_config, tmp_path):
    voice_config.filler_after_ms = 1
    tts = NullTts()
    pipe = pipeline(voice_config, pieces=["Thinking about it. "], delay=0.02,
                    room=tmp_path, tts=tts)
    kinds = [event["type"] async for event in pipe.converse(Audio.silence(1.0))]
    assert "filler" in kinds
    assert kinds.index("filler") < kinds.index("audio")


async def test_a_quick_model_is_not_interrupted_by_a_filler(voice_config, tmp_path):
    voice_config.filler_after_ms = 10_000
    pipe = pipeline(voice_config, pieces=["Right away."], room=tmp_path)
    kinds = [event["type"] async for event in pipe.converse(Audio.silence(1.0))]
    assert "filler" not in kinds


async def test_silence_is_not_a_question(voice_config, tmp_path):
    pipe = pipeline(voice_config, script=[""], room=tmp_path)
    turn = await pipe.turn(Audio.silence(1.0))
    assert turn.ignored == "nothing was said"
    assert turn.audio == []


async def test_something_said_to_somebody_else_is_left_alone(voice_config, tmp_path):
    voice_config.wake_words = ["hey gustav"]
    pipe = pipeline(voice_config, script=["pass me the salt"], room=tmp_path)
    turn = await pipe.turn(Audio.silence(1.0))
    assert turn.ignored == "not addressed to me"
    assert turn.audio == []


async def test_a_spoken_music_mode_request_is_sent_to_the_satellite(voice_config, tmp_path):
    voice_config.wake_words = ["hey gustav"]
    pipe = pipeline(voice_config, script=["Hey Gustav, put yourself in music mode"],
                    room=tmp_path)

    events = [event async for event in pipe.converse(Audio.silence(1.0))]

    assert [event["type"] for event in events] == ["transcript", "mode", "done"]
    assert events[1] == {"type": "mode", "mode": "music"}
    assert pipe.ask.asked == [], "a mode change belongs to the satellite, not the model"


async def test_the_model_is_asked_in_the_language_that_was_configured(voice_config, tmp_path):
    voice_config.reply_language = "German"
    pipe = pipeline(voice_config, script=["what time is it"], room=tmp_path)
    await pipe.turn(Audio.silence(1.0))
    assert "Answer in German" in pipe.ask.asked[0]
    assert "read aloud" in pipe.ask.asked[0], "and told to keep it short"


async def test_every_part_of_the_latency_is_measured(voice_config, tmp_path):
    """A latency claim nobody measures stops being true within a month."""
    pipe = pipeline(voice_config, pieces=["Here it is. "], room=tmp_path)
    turn = await pipe.turn(Audio.silence(1.0))
    numbers = turn.timings.to_dict()
    assert set(numbers) >= {
        "transcribe_ms", "first_token_ms", "first_audio_ms", "first_audio_was_filler",
        "total_ms", "spoken_sentences",
    }
    assert numbers["spoken_sentences"] == 1
    assert numbers["first_audio_ms"] <= numbers["total_ms"]


# --- the fillers ----------------------------------------------------------
def test_the_filler_is_in_the_language_that_is_spoken():
    assert "Einen Moment." in phrases_for("de")
    assert "One moment." in phrases_for("en")
    assert phrases_for("de-DE") == phrases_for("de")
    assert phrases_for("xx") == phrases_for("en"), "an unknown language still says something"
    assert phrases_for("de", ["Sekunde."]) == ["Sekunde."], "yours wins"


def test_fillers_are_rendered_once_and_kept(tmp_path):
    tts = NullTts()
    fillers = Fillers(tts, tmp_path, language="en")
    assert fillers.prepare() == len(phrases_for("en"))
    said = len(tts.said)
    assert fillers.pick() is not None
    assert len(tts.said) == said, "it was already rendered"
    assert list(tmp_path.glob("*.wav"))

    # A fresh one reads them off the disk rather than speaking again.
    again = Fillers(NullTts(), tmp_path, language="en")
    assert again.pick() is not None
    assert again.tts.said == []


def test_a_voice_that_will_not_speak_costs_the_filler_and_nothing_else(tmp_path):
    class Broken(NullTts):
        def speak(self, text, *, language="", voice=""):
            raise RuntimeError("no voice is installed")

    fillers = Fillers(Broken(), tmp_path, language="en")
    assert fillers.prepare() == 0
    assert fillers.pick() is None


# --- choosing engines -----------------------------------------------------
def test_null_engines_are_asked_for_by_name(voice_config):
    voice_config.stt.engine = "null"
    voice_config.tts.engine = "null"
    assert build_stt(voice_config.stt).name == "null"
    assert build_tts(voice_config.tts).name == "null"


def test_auto_falls_back_to_something_that_works(voice_config):
    """A machine with voice on and no models says so; it does not explode."""
    voice_config.stt.engine = "auto"
    voice_config.tts.engine = "auto"
    stt, tts = build_stt(voice_config.stt), build_tts(voice_config.tts)
    assert stt.ready()[1]
    assert tts.ready()[1]


def test_an_engine_nobody_has_heard_of_is_an_error(voice_config):
    voice_config.stt.engine = "cheese"
    with pytest.raises(ValueError, match="cheese"):
        build_stt(voice_config.stt)


# --- through the daemon ---------------------------------------------------
@pytest_asyncio.fixture
async def daemon(project: Path, repo_root: Path) -> AsyncIterator[Daemon]:
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    config, paths = load_config(project / "config.yaml")
    config.voice.enabled = True
    config.voice.stt.engine = "null"
    config.voice.tts.engine = "null"
    instance = Daemon(config, paths)
    await instance.start()
    # The echo responder answers; the null engines hear and speak.
    instance.voice._stt = NullStt(script=["what can you do"])
    instance.voice._tts = NullTts()
    try:
        yield instance
    finally:
        await instance.shutdown("test finished")


@pytest_asyncio.fixture
async def http(daemon: Daemon) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(daemon), client=None)
    async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as client:
        yield client


async def test_the_satellite_is_configured_by_the_server(http):
    """The Pi's own file holds an address and a token; everything else comes from here."""
    found = (await http.get("/voice/config")).json()
    assert found["enabled"] is True
    assert found["stt"]["sample_rate"] == 16000
    assert found["language"] == "en"
    assert found["fillers"], "and what to say while it thinks"
    assert "max_utterance_seconds" in found


async def test_audio_in_text_out(http, daemon):
    audio = Audio.silence(0.6, 16000)
    said = await http.post(
        "/voice/transcribe", content=audio.to_wav(),
        headers={"content-type": "audio/wav"},
    )
    assert said.status_code == 200
    assert said.json()["text"] == "what can you do"


async def test_text_in_audio_out(http):
    spoken = await http.post("/voice/speak", json={"text": "the driver is too old"})
    assert spoken.status_code == 200
    assert spoken.headers["content-type"] == "audio/wav"
    assert Audio.from_wav(spoken.content).seconds > 0


async def test_one_request_is_a_whole_spoken_turn(http):
    audio = Audio.silence(0.6, 16000)
    turn = (await http.post(
        "/voice/ask", content=audio.pcm,
        headers={"content-type": "application/octet-stream"},
    )).json()
    assert turn["transcript"] == "what can you do"
    assert turn["reply"]
    assert turn["audio"]["sample_rate"] > 0
    assert base64.b64decode(turn["audio"]["pcm"])
    assert turn["timings"]["first_audio_ms"] >= 0


async def test_raw_pcm_and_a_wav_are_both_understood(http):
    audio = Audio.silence(0.4, 16000)
    for body, kind in ((audio.pcm, "application/octet-stream"), (audio.to_wav(), "audio/wav")):
        said = await http.post("/voice/transcribe", content=body,
                               headers={"content-type": kind})
        assert said.status_code == 200, kind


async def test_an_empty_request_is_refused_rather_than_transcribed(http):
    said = await http.post("/voice/transcribe", content=b"",
                           headers={"content-type": "application/octet-stream"})
    assert said.status_code == 400


async def test_with_voice_switched_off_the_endpoints_are_not_there(project, repo_root):
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    config, paths = load_config(project / "config.yaml")
    assert config.voice.enabled is False, "off until it is asked for"
    instance = Daemon(config, paths)
    await instance.start()
    try:
        transport = httpx.ASGITransport(app=create_app(instance), client=None)
        async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as client:
            assert (await client.get("/voice/config")).status_code == 404
    finally:
        await instance.shutdown("test finished")


async def test_the_voice_is_not_built_until_it_is_switched_on(project, repo_root):
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths)
    assert instance.voice is None, "a silent daemon holds no speech models"


async def test_warming_reports_what_it_loaded(daemon):
    report = await asyncio.to_thread(daemon.voice.warm)
    assert report["stt"] and report["tts"]
    assert report["fillers"] >= 1
    assert daemon.voice.status()["warmed"] is True


def test_the_service_says_what_is_missing(project):
    """voice.enabled with nothing installed is a sentence, not a stack trace."""
    config, paths = load_config(project / "config.yaml")
    config.voice.enabled = True
    config.voice.stt.engine = "whisper"
    config.voice.tts.engine = "piper"
    service = VoiceService(config, paths, ask=None)
    found = service.status()
    if not found["stt"]["ok"]:
        assert "faster-whisper is not installed" in found["stt"]["detail"]
    if not found["tts"]["ok"]:
        assert "piper is not installed" in found["tts"]["detail"]
    assert found["ready"] == (found["stt"]["ok"] and found["tts"]["ok"])


@pytest.fixture
def daemon_for_socket(project: Path, repo_root: Path):
    """A started daemon for the synchronous WebSocket client.

    TestClient drives the app on its own event loop, so the daemon is started and
    stopped around it rather than by an async fixture that owns a different one.
    """
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    config, paths = load_config(project / "config.yaml")
    config.voice.enabled = True
    config.voice.stt.engine = "null"
    config.voice.tts.engine = "null"
    instance = Daemon(config, paths)
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(instance.start())
    instance.voice._stt = NullStt(script=["what can you do"])
    instance.voice._tts = NullTts()
    try:
        yield instance
    finally:
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            instance.shutdown("test finished")
        )


# --- the fast path --------------------------------------------------------
# The WebSocket is what a satellite uses and what the latency claim rests on, so it is
# driven here the way a satellite drives it: frames up, events down.
def socket_client(daemon: Daemon):
    from starlette.testclient import TestClient

    daemon.config.instance.http_token = "a-token-for-the-satellite"
    return TestClient(create_app(daemon))


def test_the_socket_answers_a_spoken_turn(daemon_for_socket):
    daemon = daemon_for_socket
    client = socket_client(daemon)
    frames = Audio.silence(0.6, 16000).pcm
    with client.websocket_connect(
        "/voice/stream", headers={"Authorization": "Bearer a-token-for-the-satellite"}
    ) as socket:
        socket.send_json({"type": "start", "sample_rate": 16000})
        for at in range(0, len(frames), 640):          # 20 ms frames, as a Pi sends them
            socket.send_bytes(frames[at : at + 640])
        socket.send_json({"type": "end"})

        kinds, events = [], []
        while True:
            event = socket.receive_json()
            kinds.append(event["type"])
            events.append(event)
            if event["type"] == "done":
                break

    assert kinds[0] == "heard", "it says it has the audio before it starts thinking"
    assert "transcript" in kinds
    assert "audio" in kinds
    spoken = next(e for e in events if e["type"] == "audio")
    assert base64.b64decode(spoken["pcm"])
    assert spoken["sample_rate"] > 0
    assert events[-1]["timings"]["first_audio_ms"] >= 0


def test_the_socket_can_be_told_to_stop(daemon_for_socket):
    """Barge-in: the satellite hears its person start talking and abandons the answer."""
    client = socket_client(daemon_for_socket)
    with client.websocket_connect(
        "/voice/stream", headers={"Authorization": "Bearer a-token-for-the-satellite"}
    ) as socket:
        socket.send_json({"type": "start", "sample_rate": 16000})
        socket.send_bytes(Audio.silence(0.2, 16000).pcm)
        socket.send_json({"type": "stop"})
        socket.send_json({"type": "ping"})
        assert socket.receive_json()["type"] == "pong", "the turn was dropped, not the socket"


def test_an_unauthenticated_microphone_does_not_get_to_send_audio(daemon_for_socket):
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect as Refused

    daemon_for_socket.config.instance.http_token = "a-token-for-the-satellite"
    client = TestClient(create_app(daemon_for_socket))
    with pytest.raises(Refused) as closed:
        with client.websocket_connect("/voice/stream") as socket:
            socket.send_bytes(b"\x00" * 320)
            socket.receive_json()
    assert closed.value.code == 4401


# --- the satellite, against a real server ---------------------------------
# The WebSocket client in satellite/gustav_satellite.py is two hundred lines of RFC
# 6455 written to avoid a dependency on a Raspberry Pi. That is a reasonable trade
# only if it is tested against a real server rather than against my reading of the
# specification, so here it is: a daemon on a port, and the satellite's own client
# doing a whole spoken turn through it.
@pytest.mark.slow
def test_the_satellites_own_websocket_client_works(daemon_for_socket, tmp_path):
    import importlib.util
    import threading

    import uvicorn

    if not any(importlib.util.find_spec(name) for name in ("websockets", "wsproto")):
        pytest.skip("uvicorn speaks WebSocket only with websockets/wsproto: pip install .[voice]")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    daemon = daemon_for_socket
    daemon.config.instance.http_token = "a-token-for-the-satellite"

    with socket_module.socket() as finder:
        finder.bind(("127.0.0.1", 0))
        port = finder.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(create_app(daemon), host="127.0.0.1", port=port,
                       log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "the test server did not come up"

    try:
        connection = satellite.WebSocket(
            f"ws://127.0.0.1:{port}/voice/stream",
            headers={"Authorization": "Bearer a-token-for-the-satellite"},
        )
        connection.connect()
        try:
            frames = Audio.silence(0.5, 16000).pcm
            connection.send_json({"type": "start", "sample_rate": 16000})
            for at in range(0, len(frames), 640):
                connection.send_bytes(frames[at : at + 640])
            connection.send_json({"type": "end"})

            kinds = []
            played = 0
            while True:
                event = connection.recv_json()
                kinds.append(event["type"])
                if event["type"] in ("audio", "filler"):
                    played += len(base64.b64decode(event["pcm"]))
                if event["type"] == "done":
                    break
        finally:
            connection.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert "transcript" in kinds and "audio" in kinds and kinds[-1] == "done"
    assert played > 0, "no sound came back"


@pytest.mark.slow
def test_the_satellite_sends_a_frame_the_server_can_read(daemon_for_socket):
    """A masked frame larger than 125 bytes takes the two-byte length path."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    client = socket_client(daemon_for_socket)
    with client.websocket_connect(
        "/voice/stream", headers={"Authorization": "Bearer a-token-for-the-satellite"}
    ) as server_side:
        # The client's own framing of a 640-byte frame, unmasked by hand the way the
        # server does it, is the audio that went in.
        payload = Audio.silence(0.02, 16000).pcm
        assert len(payload) == 640
        server_side.send_bytes(payload)
        server_side.send_json({"type": "end"})
        assert server_side.receive_json()["type"] == "heard"

    # And the client's masking is reversible, which is the half a server checks.
    connection = satellite.WebSocket("ws://127.0.0.1:1/none")
    header = bytearray([0x80 | satellite.OP_BINARY])
    assert header[0] == 0x82


# --- changing it from the interface ---------------------------------------
async def test_the_voice_can_be_changed_without_rewriting_config_yaml(http, daemon, project):
    """config.yaml is hand-written and full of comments; a program must not own it."""
    before = (project / "config.yaml").read_text(encoding="utf-8")

    saved = await http.post("/voice/settings", json={"language": "de", "tts.speed": 1.2})
    assert saved.status_code == 200
    assert saved.json()["set_here"]["language"] == "de"

    assert (project / "config.yaml").read_text(encoding="utf-8") == before, "untouched"
    assert daemon.config.voice.language == "de", "but it took effect"
    assert daemon.config.voice.tts.speed == 1.2

    found = (await http.get("/voice/settings")).json()
    assert found["effective"]["language"] == "de"
    assert found["set_here"] == {"language": "de", "tts.speed": 1.2}
    assert "stt.engine" in found["settable"]


async def test_the_interface_may_not_set_whatever_it_likes(http):
    refused = await http.post("/voice/settings", json={"enabled": False})
    assert refused.status_code == 400
    assert "not something the interface may set" in refused.json()["detail"]


async def test_settings_survive_a_restart(daemon, project, repo_root):
    daemon.set_voice_settings({"language": "fr", "wake_words": ["hé gustav"]})
    await daemon.shutdown("to restart it")

    config, paths = load_config(project / "config.yaml")
    config.voice.enabled = True
    config.voice.stt.engine = "null"
    config.voice.tts.engine = "null"
    again = Daemon(config, paths)
    await again.start()
    try:
        assert again.config.voice.language == "fr"
        assert again.config.voice.wake_words == ["hé gustav"]
    finally:
        await again.shutdown("test finished")


async def test_changing_the_voice_rebuilds_the_engines(daemon):
    first = daemon.voice
    daemon.set_voice_settings({"language": "it"})
    assert daemon.voice is not first, "the next turn uses the new settings"
    assert daemon.voice.status()["language"] == "it"


def test_auto_that_fell_back_to_nothing_does_not_claim_to_be_ready(project):
    """Reported from the panel: "READY: yes" over a daemon that cannot hear a word."""
    config, paths = load_config(project / "config.yaml")
    config.voice.enabled = True
    config.voice.stt.engine = "auto"
    config.voice.tts.engine = "auto"
    service = VoiceService(config, paths, ask=None)
    found = service.status()

    if found["stt"]["engine"] == "null":        # i.e. nothing is installed here
        assert found["ready"] is False
        assert "pip install -e '.[voice]'" in found["stt"]["detail"]
        assert "pip install -e '.[voice]'" in found["tts"]["detail"]


def test_a_null_engine_that_was_asked_for_is_fine(project):
    """The tests ask for it by name, and that is a working configuration."""
    config, paths = load_config(project / "config.yaml")
    config.voice.enabled = True
    config.voice.stt.engine = "null"
    config.voice.tts.engine = "null"
    found = VoiceService(config, paths, ask=None).status()
    assert found["ready"] is True
    assert found["ready_for_a_satellite"] == found["socket"]["ok"]


# --- voices ---------------------------------------------------------------
# Reported from the server: `gustav-satellite --say hello` came back as HTTP 500.
# piper was installed and had nothing to speak with, and the reason was only in the
# daemon's log.
def test_a_voice_name_is_its_path_in_the_collection():
    from gustav.voice import voices as catalogue

    voice = catalogue.parse("de_DE-thorsten-low")
    assert voice.language == "de" and voice.locale == "de_DE"
    assert voice.speaker == "thorsten" and voice.quality == "low"
    assert voice.directory == "de/de_DE/thorsten/low"
    assert voice.model_url.endswith("/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx")
    assert voice.config_url == voice.model_url + ".json"


@pytest.mark.parametrize(
    "name", ["thorsten", "de-thorsten-low", "de_DE-thorsten-enormous", "", "../../etc/passwd"],
)
def test_something_that_is_not_a_voice_name_is_refused(name):
    from gustav.voice import voices as catalogue

    with pytest.raises(catalogue.VoiceError, match="not a piper voice name"):
        catalogue.parse(name)


def test_what_is_installed_is_read_off_the_disk(tmp_path):
    from gustav.voice import voices as catalogue

    (tmp_path / "de_DE-thorsten-low.onnx").write_bytes(b"x" * 2048)
    (tmp_path / "de_DE-thorsten-low.onnx.json").write_text(
        '{"language": {"code": "de_DE"}, "audio": {"sample_rate": 16000}}', encoding="utf-8")
    # A download that was interrupted: a model with no settings beside it.
    (tmp_path / "en_GB-alba-medium.onnx").write_bytes(b"x" * 2048)

    found = {entry["name"]: entry for entry in catalogue.installed(tmp_path)}
    assert found["de_DE-thorsten-low"]["complete"] is True
    assert found["de_DE-thorsten-low"]["sample_rate"] == 16000
    assert found["en_GB-alba-medium"]["complete"] is False, "half a voice is not a voice"
    assert catalogue.for_language(tmp_path, "de").endswith("de_DE-thorsten-low.onnx")
    assert catalogue.for_language(tmp_path, "fr") is None


def test_installing_a_voice_fetches_both_files(tmp_path):
    import io

    from gustav.voice import voices as catalogue

    asked = []

    def fake_open(url, timeout=0):
        asked.append(url)
        return io.BytesIO(b"x" * 4096)

    got = catalogue.install("de_DE-thorsten-low", tmp_path, open_url=fake_open)
    assert got["name"] == "de_DE-thorsten-low" and got["already"] is False
    assert (tmp_path / "de_DE-thorsten-low.onnx").is_file()
    assert (tmp_path / "de_DE-thorsten-low.onnx.json").is_file()
    assert len(asked) == 2

    # Asked for again, it does not fetch it again.
    asked.clear()
    assert catalogue.install("de_DE-thorsten-low", tmp_path, open_url=fake_open)["already"]
    assert asked == []


def test_a_model_without_its_settings_is_not_left_behind(tmp_path):
    import io

    from gustav.voice import voices as catalogue

    def half(url, timeout=0):
        if url.endswith(".json"):
            raise OSError("the network went away")
        return io.BytesIO(b"x" * 4096)

    with pytest.raises(catalogue.VoiceError):
        catalogue.install("de_DE-thorsten-low", tmp_path, open_url=half)
    assert not list(tmp_path.glob("*.onnx")), "it would claim a voice that cannot load"


def test_no_voice_is_a_sentence_rather_than_a_500(tmp_path):
    from gustav.voice.engines import PiperTts, VoiceNotInstalled

    piper = PiperTts(voices_dir=tmp_path)
    with pytest.raises(VoiceNotInstalled) as refused:
        piper.speak("hello", language="de")
    assert "ai voice install de_DE-thorsten-low" in str(refused.value)


def test_a_voice_that_was_named_but_not_installed_says_which(tmp_path):
    from gustav.voice.engines import PiperTts, VoiceNotInstalled

    piper = PiperTts(voice="de_DE-kerstin-low", voices_dir=tmp_path)
    with pytest.raises(VoiceNotInstalled, match="de_DE-kerstin-low"):
        piper.speak("hallo", language="de")


async def test_the_panel_can_list_and_install_voices(http, daemon, monkeypatch, tmp_path):
    import io

    from gustav.voice import voices as catalogue

    monkeypatch.setattr(
        catalogue.urllib.request, "urlopen",
        lambda url, timeout=0: io.BytesIO(b"x" * 4096),
    )
    listed = (await http.get("/voice/voices")).json()
    assert listed["installed"] == []
    assert "en_GB-alba-medium" in listed["suggested"]

    # The suggestions follow the language, because a German speaker being offered
    # three English voices is a panel that has not been paying attention.
    await http.post("/voice/settings", json={"language": "de"})
    assert "de_DE-thorsten-low" in (await http.get("/voice/voices")).json()["suggested"]

    got = (await http.post("/voice/voices", json={"name": "de_DE-thorsten-low"})).json()
    assert got["name"] == "de_DE-thorsten-low"
    assert daemon.config.voice.tts.voice == "de_DE-thorsten-low", "and it is used"

    again = (await http.get("/voice/voices")).json()
    assert [one["name"] for one in again["installed"]] == ["de_DE-thorsten-low"]


async def test_a_voice_name_that_is_nonsense_is_a_400(http):
    refused = await http.post("/voice/voices", json={"name": "the nice one"})
    assert refused.status_code == 400


def test_half_a_voice_is_fetched_again_rather_than_trusted(tmp_path):
    import io

    from gustav.voice import voices as catalogue

    # What an interrupted download leaves behind.
    (tmp_path / "de_DE-thorsten-low.onnx").write_bytes(b"half")
    got = catalogue.install(
        "de_DE-thorsten-low", tmp_path,
        open_url=lambda url, timeout=0: io.BytesIO(b"x" * 4096),
    )
    assert got["already"] is False
    assert (tmp_path / "de_DE-thorsten-low.onnx").stat().st_size == 4096
    assert (tmp_path / "de_DE-thorsten-low.onnx.json").is_file()


# --- what the satellite shows you -----------------------------------------
def test_the_satellite_prints_the_reason_not_just_the_status():
    """Reported: "503 Service Unavailable" and nothing else.

    The daemon's 503 carries a sentence naming the command that fixes it. Printing
    the status line and discarding the body leaves someone with a number.
    """
    import io
    import urllib.error

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    refused = urllib.error.HTTPError(
        "http://box:8713/voice/speak", 503, "Service Unavailable", {},
        io.BytesIO(json.dumps({
            "detail": "piper has no voice installed, so it cannot say anything. "
                      "Install one — `ai voice install de_DE-thorsten-low`."
        }).encode()),
    )
    said = satellite.why(refused)
    assert "503" in said
    assert "ai voice install de_DE-thorsten-low" in said
    assert "no voice installed" in said


def test_a_body_that_is_not_json_is_still_shown():
    import io
    import urllib.error

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    said = satellite.why(urllib.error.HTTPError(
        "http://box:8713/voice/speak", 502, "Bad Gateway", {},
        io.BytesIO(b"<html>nginx</html>"),
    ))
    assert "502" in said and "nginx" in said


def test_an_ordinary_failure_is_left_alone():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    assert satellite.why(OSError("connection refused")) == "connection refused"


def test_the_satellite_discards_its_own_playback_before_listening_again():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    class Mic:
        frames = []

        def drain(self):
            self.frames.clear()

    class Speaker:
        def play(self, pcm, rate):
            pass

        def finish(self):
            pass

    class Connection:
        events = iter((
            {"type": "audio", "pcm": base64.b64encode(b"voice").decode(),
             "sample_rate": 16000},
            {"type": "done"},
        ))

        def settimeout(self, timeout):
            pass

        def recv_json(self):
            event = next(self.events)
            if event["type"] == "done":
                mic.frames.append(b"Gustav echoed by the speaker")
            return event

    mic = Mic()
    client = satellite.Satellite.__new__(satellite.Satellite)
    client.settings = satellite.Settings()
    client.server = satellite.Server()
    client.mic = mic
    client.speaker = Speaker()

    client.play_answer(Connection())

    assert mic.frames == []


def test_music_mode_uses_the_connected_bluetooth_audio_source(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    class Result:
        stdout = "0 alsa_output.platform-sound.analog-stereo.monitor\n"
        returncode = 0

    monkeypatch.setattr(satellite.subprocess, "run", lambda *args, **kwargs: Result())

    assert satellite.BluetoothMusic.source() == ""

    Result.stdout = "4 bluez_input.00_11_22_33_44_55.1 a2dp_source\n"
    assert satellite.BluetoothMusic.source() == "bluez_input.00_11_22_33_44_55.1"


def test_the_satellite_captures_pcm_without_an_audio_card(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arecord = fake_bin / "arecord"
    arecord.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "frame = b'\\x00\\x08' * 320\n"
        "while True:\n"
        "    sys.stdout.buffer.write(frame)\n"
        "    sys.stdout.buffer.flush()\n"
        "    time.sleep(0.02)\n",
        encoding="utf-8",
    )
    arecord.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    mic = satellite.Microphone(satellite.Settings())
    mic.start()
    try:
        frame = mic.read(timeout=2)
        assert frame is not None
        assert len(frame) == satellite.frame_bytes(16000)
        assert satellite.rms(frame) > 0
    finally:
        mic.stop()


def test_the_satellite_sends_pcm_to_the_speaker_without_an_audio_card(tmp_path,
                                                                     monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aplay = fake_bin / "aplay"
    aplay.write_text("#!/bin/sh\ncat > \"$GUSTAV_CAPTURED_PCM\"\n", encoding="utf-8")
    aplay.chmod(0o755)
    capture = tmp_path / "played.pcm"
    monkeypatch.setenv("GUSTAV_CAPTURED_PCM", str(capture))
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    expected = b"\x34\x12" * 640

    speaker = satellite.Speaker(satellite.Settings())
    speaker.play(expected, 16000)
    speaker.finish()

    assert capture.read_bytes() == expected


def test_music_mode_returns_to_gustav_when_the_phone_disconnects(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    import gustav_satellite as satellite

    commands = []

    class Result:
        stdout = "42\n"

    def fake_run(argv, **kwargs):
        commands.append(argv)
        return Result()

    sources = iter(("bluez_input.00_11_22_33_44_55.1", ""))
    monkeypatch.setattr(satellite.shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(satellite.subprocess, "run", fake_run)
    monkeypatch.setattr(satellite.BluetoothMusic, "source", lambda: next(sources))
    monkeypatch.setattr(satellite.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(satellite, "say", lambda *args, **kwargs: None)

    satellite.BluetoothMusic.wait_for_phone()

    assert ["pactl", "load-module", "module-loopback",
            "source=bluez_input.00_11_22_33_44_55.1", "latency_msec=80"] in commands
    assert ["pactl", "unload-module", "42"] in commands
    assert ["bluetoothctl", "discoverable", "off"] in commands
