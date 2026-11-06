"""Playing JARVIS's voice: one stream per reply, sentences in order.

The Core sends a reply as numbered sentences, each a complete WAV
(Kokoro: 24 kHz mono 16-bit). They are synthesised at the same time on
the server, so they can arrive out of order. The player:

  - opens ONE output stream (`pacat`) when the first sentence is ready,
    starting with a little silence so Bluetooth earphones are awake for
    the first word - a new process per sentence was what clipped words
    in the first test;
  - writes sentence 0, 1, 2... into it as each becomes next in line, so
    playback starts while later sentences are still being made;
  - skips a sentence that arrives empty (the Core's "this one failed");
  - waits a few seconds for a missing number, then moves on;
  - ends with a short silence so the last word is not cut, then lets
    the stream drain.

stop() cuts it off at once - that is what the wake word does when he
interrupts.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from collections.abc import Awaitable, Callable
from typing import Protocol

from reactor.sounds import RATE as CHIME_RATE

log = logging.getLogger("reactor.player")

_GAP_WAIT_S = 8.0
_IDLE_LIMIT_S = 90.0


class Stream(Protocol):
    async def write(self, data: bytes) -> None: ...
    async def finish(self) -> None: ...
    def kill(self) -> None: ...


Opener = Callable[[int, int], Awaitable[Stream]]


class PacatStream:
    """A `pacat` process playing raw 16-bit audio from its stdin."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._p = process

    @classmethod
    async def open(cls, device: str | None, rate: int, channels: int) -> "PacatStream":
        args = ["pacat", "--playback", "--raw", "--format=s16le", f"--rate={rate}",
                f"--channels={channels}", "--latency-msec=100",
                "--client-name=reactor", "--stream-name=jarvis"]
        if device:
            args.append(f"--device={device}")
        process = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        return cls(process)

    async def write(self, data: bytes) -> None:
        assert self._p.stdin is not None
        self._p.stdin.write(data)
        await self._p.stdin.drain()

    async def finish(self) -> None:
        assert self._p.stdin is not None
        self._p.stdin.close()
        await self._p.wait()

    def kill(self) -> None:
        if self._p.returncode is None:
            self._p.kill()


def silence(ms: int, rate: int, channels: int) -> bytes:
    return b"\x00\x00" * (rate * ms // 1000) * channels


def read_wav(data: bytes) -> tuple[int, int, bytes]:
    """(rate, channels, 16-bit PCM) from a WAV file."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{wav.getsampwidth() * 8}-bit audio, need 16-bit")
        return wav.getframerate(), wav.getnchannels(), wav.readframes(wav.getnframes())


class Player:
    def __init__(self, opener: Opener, *, lead_in_ms: int, tail_ms: int) -> None:
        self._open = opener
        self._lead_in = lead_in_ms
        self._tail = tail_ms
        self._task: asyncio.Task[bool] | None = None
        self._pending: dict[int, bytes] = {}
        self._arrived = asyncio.Event()
        self._ended = False
        self._stream: Stream | None = None

    @property
    def playing(self) -> bool:
        return self._task is not None and not self._task.done()

    # -- a reply --------------------------------------------------------------

    def add(self, seq: int, wav: bytes) -> None:
        """One sentence of the reply (empty means skip that number)."""
        if self._task is None or self._task.done():
            self._pending, self._ended = {}, False
            self._task = asyncio.create_task(self._feed())
        self._pending[seq] = wav
        self._arrived.set()

    def end(self) -> None:
        """The Core sent audio_done: no new numbers after the last one."""
        self._ended = True
        self._arrived.set()

    async def finished(self) -> bool:
        """Wait for the reply to play out. False if it was stopped."""
        task = self._task
        if task is None:
            return True
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                return False
            raise

    async def stop(self) -> None:
        """Cut the reply off now."""
        if self._stream is not None:
            self._stream.kill()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._pending.clear()

    async def _feed(self) -> bool:
        next_seq = 0
        params: tuple[int, int] | None = None
        self._stream = None
        try:
            while True:
                if next_seq in self._pending:
                    data = self._pending.pop(next_seq)
                    next_seq += 1
                    if not data:
                        continue
                    try:
                        rate, channels, pcm = read_wav(data)
                    except Exception as exc:
                        log.warning("unplayable sentence %d: %s", next_seq - 1, exc)
                        continue
                    if self._stream is not None and params != (rate, channels):
                        await self._stream.finish()
                        self._stream = None
                    if self._stream is None:
                        lead = self._lead_in if params is None else 0
                        params = (rate, channels)
                        self._stream = await self._open(rate, channels)
                        await self._stream.write(silence(lead, rate, channels))
                    await self._stream.write(pcm)
                    continue

                if self._ended and not self._pending:
                    return True
                self._arrived.clear()
                # A gap: a later sentence is here but this one is not.
                waiting = _GAP_WAIT_S if self._pending else _IDLE_LIMIT_S
                try:
                    await asyncio.wait_for(self._arrived.wait(), timeout=waiting)
                except TimeoutError:
                    if not self._pending:
                        log.warning("reply stalled - no audio for %.0f s", waiting)
                        return True
                    log.warning("sentence %d never came - skipping it", next_seq)
                    next_seq += 1
        finally:
            if self._stream is not None and params is not None:
                stream, self._stream = self._stream, None
                try:
                    await asyncio.shield(self._close(stream, params))
                except asyncio.CancelledError:
                    stream.kill()
                    raise

    async def _close(self, stream: Stream, params: tuple[int, int]) -> None:
        try:
            await stream.write(silence(self._tail, *params))
            await stream.finish()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- chimes ---------------------------------------------------------------

    async def chime(self, pcm: bytes) -> None:
        """A short sound, unless a reply is playing."""
        if self.playing:
            return
        try:
            stream = await self._open(CHIME_RATE, 1)
            await stream.write(silence(self._lead_in // 2, CHIME_RATE, 1) + pcm)
            await stream.finish()
        except Exception as exc:
            log.debug("chime failed: %s", exc)
