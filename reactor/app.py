"""The reactor itself: wake word -> question -> JARVIS -> spoken reply.

    offline ──connected──> idle ──"hey jarvis"──> listening
       ^                    ^                        │ pause
       └──disconnected──────┤                        v
                            ├──<──played out── speaking <─first sentence── thinking
                            └──<──error / timeout / no audio ──────────────┘

Everything the mic hears goes through the wake word detector, all the
time; nothing leaves the device until the wake word fires. Then the
question is recorded until he pauses, sent as one WAV, and the reply
is played as it arrives.

Saying the wake word while JARVIS is thinking or talking cuts him off
and starts a new question; the Core drops the old reply when the new
message lands.

The earphones' buttons do the same without speaking: the talk button
(play/pause) starts a question - pressed again while recording, it
sends at once; the cancel button (previous) drops a question or stops
JARVIS mid-sentence; the toggle button (next) switches the wake word
on and off. With the wake word off, the detector is not run at all.

This is the only place that decides the state. Mic, player, link and
lights just report and obey.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

import numpy as np

from reactor import sounds
from reactor.config import Config
from reactor.link import Link
from reactor.listen import CHUNK_S, Background, Preroll, Utterance, chunk_frames, level_dbfs
from reactor.player import Player

log = logging.getLogger("reactor")


class Mic(Protocol):
    def chunks(self) -> AsyncIterator[np.ndarray]: ...


class Detector(Protocol):
    def process(self, chunk: np.ndarray) -> float: ...


class Lights(Protocol):
    def show(self, state: str) -> None: ...


class Keys(Protocol):
    def presses(self) -> AsyncIterator[str]: ...


class Reactor:
    def __init__(self, config: Config, *, mic: Mic, wake: Detector, player: Player,
                 lights: Lights, token: str, buttons: Keys | None = None) -> None:
        self._c = config
        self._mic = mic
        self._wake = wake
        self._buttons = buttons
        self.wake_on = config.wake.enabled
        self._player = player
        self._lights = lights
        self.link = Link(config.core_url, token, config.label, self.on_frame,
                         on_up=self._on_up, on_down=self._on_down)
        self.state = "offline"
        self._background = Background()
        self._preroll = Preroll(config.listen.preroll_s)
        self._utterance: Utterance | None = None
        self._clock = 0.0                # seconds of audio heard
        self._last_wake = -1e9
        self._accept_audio = False       # True from the transcript until played out
        self._done_early = False         # audio_done came before any audio (old Core)
        self._reply_text: list[str] = []
        self._watchdog: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        lights.show(self.state)

    # -- running --------------------------------------------------------------

    async def run(self) -> None:
        jobs = [self.link.run(), self._hear()]
        if self._buttons is not None:
            jobs.append(self._press())
        await asyncio.gather(*jobs)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _set(self, state: str) -> None:
        if state != self.state:
            log.debug("%s -> %s", self.state, state)
            self.state = state
            self._lights.show(state)

    def _rest(self) -> None:
        self._set("idle" if self.link.connected else "offline")

    def _chime(self, pcm: bytes) -> None:
        if self._c.audio.chimes:
            self._spawn(self._player.chime(pcm))

    # -- the mic --------------------------------------------------------------

    async def _hear(self) -> None:
        async for chunk in self._mic.chunks():
            score = await asyncio.to_thread(self._wake.process, chunk) if self.wake_on else 0.0
            self._clock += CHUNK_S
            await self.on_audio(chunk, score)

    # -- the buttons ----------------------------------------------------------

    async def _press(self) -> None:
        assert self._buttons is not None
        async for key in self._buttons.presses():
            await self.on_button(key)

    async def on_button(self, key: str) -> None:
        b = self._c.buttons
        if key in b.talk:
            log.info("talk button")
            if self.state == "listening" and self._utterance is not None:
                result = self._utterance.finish()
                self._utterance = None
                if result.status == "done":
                    await self._send(result.audio)
                else:
                    self._chime(sounds.CANCELLED)
                    self._rest()
            else:
                await self._wake_up(None)
        elif b.cancel and key == b.cancel:
            if self.state == "listening":
                log.info("question cancelled")
                self._utterance = None
                self._chime(sounds.CANCELLED)
                self._rest()
            elif self.state == "speaking":
                log.info("reply stopped")
                await self._player.stop()
                self._accept_audio = False
                self._rest()
        elif b.toggle and key == b.toggle:
            self.wake_on = not self.wake_on
            reset = getattr(self._wake, "reset", None)
            if self.wake_on and reset is not None:
                reset()
            log.info("wake word %s", "on" if self.wake_on else "off - talk button only")
            self._chime(sounds.WAKE_ON if self.wake_on else sounds.WAKE_OFF)

    async def on_audio(self, chunk: np.ndarray, score: float) -> None:
        if self.state == "listening" and self._utterance is not None:
            result = self._utterance.feed(chunk)
            if result.status == "done":
                self._utterance = None
                await self._send(result.audio)
            elif result.status == "nothing":
                self._utterance = None
                log.info("heard nothing after the wake word")
                self._chime(sounds.CANCELLED)
                self._rest()
            return

        self._background.update(level_dbfs(chunk))
        self._preroll.add(chunk)
        if score < self._c.wake.threshold:
            return
        if self._clock - self._last_wake < self._c.wake.refractory_s:
            return
        if self.state in ("thinking", "speaking") and not self._c.wake.barge_in:
            return
        self._last_wake = self._clock
        await self._wake_up(score)

    async def _wake_up(self, score: float | None) -> None:
        if score is not None:
            log.info("wake word (%.2f)", score)
        if not self.link.connected:
            log.warning("woken, but not connected to JARVIS")
            self._chime(sounds.ERROR)
            return
        if self.state in ("thinking", "speaking"):
            log.info("cutting off the current reply")
            await self._player.stop()
        self._accept_audio = False
        self._cancel_watchdog()
        self._set("listening")
        self._chime(sounds.LISTENING)
        self._utterance = Utterance(self._c.listen, self._background.dbfs, self._preroll.take())

    async def _send(self, wav: bytes) -> None:
        self._chime(sounds.SENT)
        # Thinking BEFORE the last frame goes: the transcript can come
        # back while the send is still finishing.
        self._reply_text = []
        self._done_early = False
        self._set("thinking")
        sent = True
        for frame in chunk_frames(wav):
            sent = sent and await self.link.send("client.audio_chunk", frame)
        sent = sent and await self.link.send("client.audio_end", {})
        if not sent:
            log.warning("could not send the question - not connected")
            self._chime(sounds.ERROR)
            self._rest()
            return
        log.info("question sent (%.1f s)", (len(wav) - 44) / 32000)
        self._watchdog = asyncio.create_task(self._transcript_watchdog())

    async def _transcript_watchdog(self) -> None:
        await asyncio.sleep(self._c.transcript_timeout_s)
        if self.state == "thinking" and not self._accept_audio:
            log.warning("JARVIS did not answer in %.0f s", self._c.transcript_timeout_s)
            self._chime(sounds.ERROR)
            self._rest()

    def _cancel_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    # -- the Core -------------------------------------------------------------

    def _on_up(self) -> None:
        if self.state == "offline":
            self._set("idle")

    def _on_down(self) -> None:
        if self.state == "thinking":
            self._cancel_watchdog()
            self._chime(sounds.ERROR)
            self._set("offline")
        elif self.state == "idle":
            self._set("offline")

    async def on_frame(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "core.transcript":
            log.info("heard: %s", payload.get("text", ""))
            if self.state == "thinking":
                self._cancel_watchdog()
                self._accept_audio = True

        elif kind == "core.assistant_delta":
            if self._accept_audio:
                self._reply_text.append(str(payload.get("text", "")))

        elif kind == "core.audio":
            if not self._accept_audio or self.state not in ("thinking", "speaking"):
                return
            data = payload.get("data_b64") or ""
            self._player.add(int(payload.get("seq", 0)), base64.b64decode(data) if data else b"")
            if self._done_early:
                self._player.end()
            if self.state == "thinking":
                self._set("speaking")
                self._spawn(self._after_reply())

        elif kind == "core.audio_done":
            if self._accept_audio and self.state == "speaking":
                self._player.end()
            elif self._accept_audio and self.state == "thinking":
                self._done_early = True

        elif kind == "core.assistant_done":
            text = "".join(self._reply_text).strip()
            if text:
                log.info("jarvis: %s", text if len(text) <= 300 else text[:300] + "...")
            if self.state == "thinking" and self._accept_audio:
                # No voice yet. Either speech is off on the Core, or (a
                # Core from before batch 16) the sentences are still on
                # their way. Give them a moment before calling it.
                self._spawn(self._voice_grace())

        elif kind == "core.error":
            log.warning("jarvis: %s", payload.get("message", ""))
            if self.state == "thinking":
                self._cancel_watchdog()
                self._accept_audio = False
                self._chime(sounds.ERROR)
                self._rest()

        elif kind == "core.notification":
            log.info("notification: %s", payload.get("text", ""))
            if self.state == "idle":
                self._chime(sounds.NOTICE)

    async def _voice_grace(self, seconds: float = 8.0) -> None:
        await asyncio.sleep(seconds)
        if self.state == "thinking" and self._accept_audio:
            log.info("the reply came without a voice")
            self._accept_audio = False
            self._chime(sounds.NOTICE)
            self._rest()

    async def _after_reply(self) -> None:
        finished = await self._player.finished()
        if finished and self.state == "speaking":
            self._accept_audio = False
            self._rest()

    # -- one question without the wake word -----------------------------------

    async def ask(self, wav: bytes, timeout_s: float = 120.0) -> bool:
        """Send a recorded question, play the reply, return when done."""
        link = asyncio.create_task(self.link.run())
        try:
            for _ in range(150):
                if self.link.connected:
                    break
                await asyncio.sleep(0.1)
            else:
                log.error("could not connect to JARVIS")
                return False
            await self._send(wav)
            deadline = asyncio.get_running_loop().time() + timeout_s
            while self.state in ("thinking", "speaking"):
                if asyncio.get_running_loop().time() > deadline:
                    return False
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.3)
            return True
        finally:
            link.cancel()
