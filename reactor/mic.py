"""The microphone, as a stream of 80 ms chunks of 16 kHz audio.

PulseMic runs `parec` on the chosen PipeWire source and reads its raw
output. When the router picks a different mic (earphones connected or
gone), or the stream goes silent at the byte level for a few seconds
(device vanished), it restarts parec on the new one. With no usable
mic at all it waits, and says so once.

FileMic plays a WAV file instead, in real time, then silence - for
trying the whole loop without speaking, and for the tests.
"""

from __future__ import annotations

import asyncio
import logging
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from reactor.listen import CHUNK, RATE
from reactor.routes import Router

log = logging.getLogger("reactor.mic")

_BYTES = CHUNK * 2
_STALL_S = 3.0


class PulseMic:
    def __init__(self, router: Router) -> None:
        self._router = router

    async def chunks(self) -> AsyncIterator[np.ndarray]:
        warned = False
        while True:
            source = self._router.route.mic
            if source is None:
                if not warned:
                    log.warning("no microphone - waiting for one (connect the earphones)")
                    warned = True
                self._router.changed.clear()
                try:
                    await asyncio.wait_for(self._router.changed.wait(), timeout=5)
                except TimeoutError:
                    pass
                continue
            warned = False
            log.info("listening on %s", source)
            process = await asyncio.create_subprocess_exec(
                "parec", "--raw", "--format=s16le", f"--rate={RATE}", "--channels=1",
                "--latency-msec=40", f"--device={source}",
                "--client-name=reactor", "--stream-name=wake word",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            assert process.stdout is not None
            try:
                while self._router.route.mic == source:
                    try:
                        data = await asyncio.wait_for(
                            process.stdout.readexactly(_BYTES), timeout=_STALL_S)
                    except (TimeoutError, asyncio.IncompleteReadError):
                        log.warning("microphone %s stopped - restarting", source)
                        break
                    yield np.frombuffer(data, dtype="<i2")
            finally:
                if process.returncode is None:
                    process.kill()
                await process.wait()
            await asyncio.sleep(0.5)


class FileMic:
    """A WAV file (16 kHz mono 16-bit) heard as if spoken, then silence."""

    def __init__(self, path: Path, *, realtime: bool = True, repeat_after_s: float | None = None) -> None:
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
                raise SystemExit(f"{path} must be 16 kHz, mono, 16-bit")
            self._audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        self._realtime = realtime
        self._repeat = repeat_after_s

    async def chunks(self) -> AsyncIterator[np.ndarray]:
        silence = np.zeros(CHUNK, dtype="<i2")
        while True:
            for start in range(0, len(self._audio) - CHUNK + 1, CHUNK):
                yield self._audio[start:start + CHUNK]
                await asyncio.sleep(CHUNK / RATE if self._realtime else 0)
            quiet = 0.0
            while self._repeat is None or quiet < self._repeat:
                yield silence
                await asyncio.sleep(CHUNK / RATE if self._realtime else 0)
                quiet += CHUNK / RATE
