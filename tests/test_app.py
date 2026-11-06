"""The whole loop against a fake Core: wake, ask, hear the reply."""

import asyncio
import io
import wave

import numpy as np
import pytest

from reactor.app import Reactor
from reactor.config import Config
from reactor.player import Player
from tests.fakes import FakeCore, ScriptedMic, ScriptedWake, Speakers, quiet, sentence, tone


class Lights:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def show(self, state: str) -> None:
        self.seen.append(state)


def build(core: FakeCore, audio: np.ndarray, wake_at: set[int], token: str = "secret"):
    config = Config(core_url=core.url, transcript_timeout_s=3.0)
    config.audio.chimes = False
    mic = ScriptedMic(audio, wake_at)
    speakers = Speakers()
    lights = Lights()
    reactor = Reactor(config, mic=mic, wake=ScriptedWake(mic),
                      player=Player(speakers.open, lead_in_ms=300, tail_ms=0),
                      lights=lights, token=token)
    return reactor, mic, speakers, lights


async def until(check, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


@pytest.fixture
async def core():
    c = await FakeCore().start()
    yield c
    await c.stop()


async def test_wake_ask_and_hear(core: FakeCore) -> None:
    # 1 s quiet, wake fires at chunk 12, he speaks 1.5 s, then quiet
    audio = np.concatenate([quiet(1.0), tone(1.5), quiet(3.0)])
    reactor, mic, speakers, lights = build(core, audio, {12})
    mic.gate.clear()
    task = asyncio.create_task(reactor.run())
    await until(lambda: reactor.state == "idle")
    mic.gate.set()
    await until(lambda: core.questions)
    with wave.open(io.BytesIO(core.questions[0])) as w:
        assert w.getframerate() == 16000 and 1.5 < w.getnframes() / 16000 < 3.5
    await until(lambda: reactor.state == "idle" and lights.seen[-2:] == ["speaking", "idle"])
    assert lights.seen == ["offline", "idle", "listening", "thinking", "speaking", "idle"]
    assert len(speakers.written()) > 0.5 * 24000 * 2          # both sentences played
    assert core.hellos[0]["payload"]["client_kind"] == "voice.rpi"
    task.cancel()


async def test_sentences_out_of_order_still_play_in_order() -> None:
    a, b = sentence(0.1, value=111), sentence(0.2, value=222)
    core = await FakeCore(reply=[a, b], order=[1, 0]).start()
    try:
        audio = np.concatenate([quiet(0.5), tone(1.0), quiet(3.0)])
        reactor, mic, speakers, lights = build(core, audio, {6})
        task = asyncio.create_task(reactor.run())
        await until(lambda: lights.seen[-2:] == ["speaking", "idle"])
        from reactor.player import read_wav
        played = speakers.written()
        assert 0 < played.find(read_wav(a)[2]) < played.find(read_wav(b)[2])
        task.cancel()
    finally:
        await core.stop()


async def test_wake_word_cuts_jarvis_off() -> None:
    hold = asyncio.Event()
    core = await FakeCore(hold=hold).start()
    try:
        audio = np.concatenate([quiet(0.5), tone(1.0), quiet(1.5), tone(1.0), quiet(3.0)])
        reactor, mic, speakers, lights = build(core, audio, {6, 40})
        task = asyncio.create_task(reactor.run())
        await until(lambda: len(core.questions) == 1)
        await until(lambda: reactor.state == "listening", timeout=8)   # second wake, mid-think
        hold.set()
        await until(lambda: len(core.questions) == 2, timeout=8)
        await until(lambda: lights.seen[-1] == "idle", timeout=8)
        assert lights.seen.count("listening") == 2
        task.cancel()
    finally:
        await core.stop()


async def test_a_wrong_token_never_connects(core: FakeCore) -> None:
    reactor, mic, speakers, lights = build(core, quiet(1.0), set(), token="wrong")
    task = asyncio.create_task(reactor.link.run())
    await until(lambda: core.hellos)
    await asyncio.sleep(0.2)
    assert not reactor.link.connected and reactor.state == "offline"
    task.cancel()


async def test_it_reconnects_after_a_drop(core: FakeCore) -> None:
    reactor, mic, speakers, lights = build(core, quiet(1.0), set())
    task = asyncio.create_task(reactor.link.run())
    await until(lambda: reactor.state == "idle")
    await core.drop_all()
    await until(lambda: reactor.state == "offline")
    await until(lambda: reactor.state == "idle", timeout=6)
    assert len(core.hellos) == 2
    task.cancel()


async def test_no_answer_times_out(core: FakeCore) -> None:
    core._answer = lambda ws: asyncio.sleep(0)          # the Core goes quiet
    audio = np.concatenate([quiet(0.5), tone(1.0), quiet(6.0)])
    reactor, mic, speakers, lights = build(core, audio, {6})
    task = asyncio.create_task(reactor.run())
    await until(lambda: reactor.state == "thinking")
    await until(lambda: reactor.state == "idle", timeout=6)
    task.cancel()


async def test_an_old_core_that_says_done_too_early() -> None:
    """Cores before JARVIS batch 16 could send audio_done before the audio."""
    core = await FakeCore().start()

    async def early(ws) -> None:
        await core._send(ws, "core.transcript", {"text": "hi"})
        await core._send(ws, "core.audio_done", {})
        await core._send(ws, "core.assistant_done", {"turn_id": "t", "job_refs": []})
        await asyncio.sleep(0.2)
        for seq, data in enumerate(core.reply):
            import base64
            await core._send(ws, "core.audio", {"seq": seq, "mime": "audio/wav",
                                                "data_b64": base64.b64encode(data).decode()})
    core._answer = early
    try:
        audio = np.concatenate([quiet(0.5), tone(1.0), quiet(3.0)])
        reactor, mic, speakers, lights = build(core, audio, {6})
        task = asyncio.create_task(reactor.run())
        await until(lambda: lights.seen[-2:] == ["speaking", "idle"])
        assert len(speakers.written()) > 0
        task.cancel()
    finally:
        await core.stop()


async def test_the_talk_button_asks_without_the_wake_word(core: FakeCore) -> None:
    audio = np.concatenate([quiet(1.0), tone(3.0), quiet(4.0)])
    reactor, mic, speakers, lights = build(core, audio, set())
    task = asyncio.create_task(reactor.run())
    await until(lambda: reactor.state == "idle" and mic.index >= 8)
    await reactor.on_button("play")
    assert reactor.state == "listening"
    await until(lambda: mic.index >= 30)                  # he is mid-sentence
    await reactor.on_button("pause")                       # second press: send now
    await until(lambda: core.questions)
    with wave.open(io.BytesIO(core.questions[0])) as w:
        assert w.getnframes() / 16000 < 2.5                # cut, not the full 3 s
    await until(lambda: lights.seen[-2:] == ["speaking", "idle"])
    task.cancel()


async def test_the_toggle_switches_the_wake_word_off(core: FakeCore) -> None:
    audio = np.concatenate([quiet(1.0), tone(1.0), quiet(3.0)])
    reactor, mic, speakers, lights = build(core, audio, {20})
    mic.gate.clear()
    task = asyncio.create_task(reactor.run())
    await until(lambda: reactor.state == "idle")
    await reactor.on_button("next")
    assert reactor.wake_on is False
    mic.gate.set()
    await until(lambda: mic.index >= 40)
    assert reactor.state == "idle" and not core.questions   # the wake was ignored
    await reactor.on_button("next")
    assert reactor.wake_on is True
    task.cancel()


async def test_cancel_drops_the_question(core: FakeCore) -> None:
    reactor, mic, speakers, lights = build(core, np.concatenate([quiet(0.5), tone(4.0)]), set())
    task = asyncio.create_task(reactor.run())
    await until(lambda: reactor.state == "idle")
    await reactor.on_button("playpause")
    await reactor.on_button("previous")
    assert reactor.state == "idle"
    await asyncio.sleep(0.3)
    assert not core.questions
    task.cancel()
