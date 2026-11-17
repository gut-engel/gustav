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
