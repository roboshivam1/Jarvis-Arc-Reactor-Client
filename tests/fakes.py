"""Stand-ins for the Pi's audio and for the JARVIS Core."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import numpy as np
import websockets

from reactor.listen import CHUNK, to_wav


class FakeStream:
    def __init__(self, log: list[tuple[str, Any]], rate: int, channels: int) -> None:
        self.log, self.params, self.data = log, (rate, channels), bytearray()
        log.append(("open", (rate, channels)))

    async def write(self, data: bytes) -> None:
        self.data += data
        self.log.append(("write", bytes(data)))
        await asyncio.sleep(0)

    async def finish(self) -> None:
        self.log.append(("finish", None))

    def kill(self) -> None:
        self.log.append(("kill", None))


class Speakers:
    def __init__(self) -> None:
        self.log: list[tuple[str, Any]] = []

    async def open(self, rate: int, channels: int) -> FakeStream:
        return FakeStream(self.log, rate, channels)

    def written(self) -> bytes:
        return b"".join(d for kind, d in self.log if kind == "write")


def tone(seconds: float, level: int = 6000) -> np.ndarray:
    n = int(16000 * seconds) // CHUNK * CHUNK
    t = np.arange(n) / 16000
    return (np.sin(2 * np.pi * 300 * t) * level).astype("<i2")


def quiet(seconds: float) -> np.ndarray:
    n = int(16000 * seconds) // CHUNK * CHUNK
    return (np.random.default_rng(3).normal(0, 20, n)).astype("<i2")


class ScriptedMic:
    """Plays audio in 80 ms chunks; `wake_at` are chunk numbers where the
    fake detector should fire. Pauses whenever `gate` is cleared."""

    def __init__(self, audio: np.ndarray, wake_at: set[int]) -> None:
        self.audio, self.wake_at = audio, wake_at
        self.index = -1
        self.gate = asyncio.Event()
        self.gate.set()

    async def chunks(self):  # type: ignore[no-untyped-def]
        while True:
            for start in range(0, len(self.audio) - CHUNK + 1, CHUNK):
                await self.gate.wait()
                self.index += 1
                yield self.audio[start:start + CHUNK]
                await asyncio.sleep(0.001)
            self.audio = quiet(1.0)


class ScriptedWake:
    def __init__(self, mic: ScriptedMic) -> None:
        self.mic = mic

    def process(self, chunk: np.ndarray) -> float:
        return 0.9 if self.mic.index in self.mic.wake_at else 0.01


def sentence(seconds: float, rate: int = 24000, value: int = 1000) -> bytes:
    pcm = (np.ones(int(rate * seconds)) * value).astype("<i2").tobytes()
    return to_wav(pcm, rate)


class FakeCore:
    """Speaks the client protocol the way the real gateway does."""

    def __init__(self, token: str = "secret", reply: list[bytes] | None = None,
                 order: list[int] | None = None, hold: asyncio.Event | None = None) -> None:
        self.token = token
        self.reply = reply if reply is not None else [sentence(0.2), sentence(0.3)]
        self.order = order or list(range(len(self.reply)))
        self.hold = hold
        self.questions: list[bytes] = []
        self.hellos: list[dict[str, Any]] = []
        self.server: Any = None
        self.connections: list[Any] = []

    @property
    def url(self) -> str:
        port = self.server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}/ws/client"

    async def start(self) -> "FakeCore":
        self.server = await websockets.serve(self._client, "127.0.0.1", 0, max_size=None)
        return self

    async def stop(self) -> None:
        self.server.close()
        await self.server.wait_closed()

    async def drop_all(self) -> None:
        for ws in list(self.connections):
            await ws.close()

    async def _send(self, ws: Any, kind: str, payload: dict[str, Any]) -> None:
        await ws.send(json.dumps({"v": 1, "kind": kind, "payload": payload}))

    async def _client(self, ws: Any) -> None:
        hello = json.loads(await ws.recv())
        self.hellos.append(hello)
        if hello["payload"]["token"] != self.token:
            await self._send(ws, "core.error", {"message": "invalid token"})
            await ws.close(code=1008)
            return
        self.connections.append(ws)
        await self._send(ws, "core.ready", {"session_id": "s", "owner_timezone": "Asia/Kolkata",
                                            "protocol_v": 1})
        audio: dict[int, bytes] = {}
        try:
            async for raw in ws:
                frame = json.loads(raw)
                if frame["kind"] == "client.audio_chunk":
                    audio[frame["payload"]["seq"]] = base64.b64decode(frame["payload"]["data_b64"])
                elif frame["kind"] == "client.audio_end":
                    self.questions.append(b"".join(audio[k] for k in sorted(audio)))
                    audio.clear()
                    asyncio.ensure_future(self._answer(ws))
        finally:
            self.connections.remove(ws)

    async def _answer(self, ws: Any) -> None:
        await self._send(ws, "core.transcript", {"text": "what is for dinner"})
        if self.hold is not None:
            await self.hold.wait()
        await self._send(ws, "core.assistant_delta", {"text": "Rajma, sir."})
        for seq in self.order:
            data = self.reply[seq]
            await self._send(ws, "core.audio", {"seq": seq, "mime": "audio/wav",
                                                "data_b64": base64.b64encode(data).decode()})
        await self._send(ws, "core.audio_done", {})
        await self._send(ws, "core.assistant_done", {"turn_id": "t", "job_refs": []})
