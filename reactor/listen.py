"""Hearing a question: where it starts, where it ends, and packing it up.

The reactor does not stream sound to JARVIS. After the wake word it
records until he stops talking, then sends ONE WAV - the Core joins
the chunks and hands the file to Whisper.

"Stopped talking" is decided by loudness against the background. The
background level is learned all the time while the reactor is idle,
so a noisy mess and a quiet room both work. The earphones' mic already
suppresses noise, which keeps this simple.
"""

from __future__ import annotations

import base64
import io
import math
import wave
from collections import deque
from dataclasses import dataclass

import numpy as np

from reactor.config import Listen

RATE = 16000
CHUNK = 1280                      # 80 ms - the same step as the wake word
CHUNK_S = CHUNK / RATE
SEND_BYTES = 48_000               # per client.audio_chunk frame


def level_dbfs(chunk: np.ndarray) -> float:
    """Loudness of a chunk, in dB below full scale."""
    rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return 20 * math.log10(max(rms, 1.0) / 32768)


class Background:
    """The room's background loudness: the quiet moments of the last few
    seconds (their 20th percentile). Talk has gaps between words, so
    chatting with people nearby does not teach it that speech is the
    background, and a noisy mess raises it within seconds."""

    def __init__(self, seconds: float = 5.0) -> None:
        self._levels: deque[float] = deque(maxlen=round(seconds / CHUNK_S))
        self.dbfs = -60.0

    def update(self, dbfs: float) -> None:
        self._levels.append(dbfs)
        self.dbfs = float(np.percentile(self._levels, 20))


class Preroll:
    """The last moment of sound, kept so a question can include it."""

    def __init__(self, seconds: float) -> None:
        self._chunks: deque[bytes] = deque(maxlen=max(1, round(seconds / CHUNK_S)))

    def add(self, chunk: np.ndarray) -> None:
        self._chunks.append(chunk.astype("<i2").tobytes())

    def take(self) -> list[bytes]:
        taken = list(self._chunks)
        self._chunks.clear()
        return taken


@dataclass
class Result:
    status: str          # "recording", "done" or "nothing"
    audio: bytes = b""


class Utterance:
    """One question being recorded, from the wake word to the pause."""

    def __init__(self, settings: Listen, background_dbfs: float, preroll: list[bytes]) -> None:
        self._s = settings
        self._threshold = max(background_dbfs + settings.speech_margin_db,
                              settings.speech_floor_dbfs)
        self._audio = list(preroll)
        self._elapsed = 0.0
        self._loud_run = 0
        self._started = False
        self._quiet = 0.0

    @property
    def started(self) -> bool:
        return self._started

    def feed(self, chunk: np.ndarray) -> Result:
        self._audio.append(chunk.astype("<i2").tobytes())
        self._elapsed += CHUNK_S
        loud = level_dbfs(chunk) >= self._threshold

        if not self._started:
            if self._elapsed > self._s.ignore_s and loud:
                self._loud_run += 1
                # Two loud chunks in a row (160 ms): a click is not a word.
                if self._loud_run >= 2:
                    self._started = True
            else:
                self._loud_run = 0
            if not self._started and self._elapsed >= self._s.start_timeout_s:
                return Result("nothing")
            return Result("recording")

        self._quiet = 0.0 if loud else self._quiet + CHUNK_S
        if self._quiet >= self._s.end_silence_s or self._elapsed >= self._s.max_s:
            return Result("done", to_wav(b"".join(self._audio)))
        return Result("recording")


    def finish(self) -> Result:
        """Stop now (the talk button pressed again): send what there is,
        unless he never started talking."""
        if not self._started:
            return Result("nothing")
        return Result("done", to_wav(b"".join(self._audio)))


def to_wav(pcm: bytes, rate: int = RATE) -> bytes:
    """16-bit mono PCM -> one WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buffer.getvalue()


def chunk_frames(wav: bytes) -> list[dict[str, object]]:
    """A WAV as client.audio_chunk payloads. Only the first carries the
    header - the Core joins them back into one file."""
    return [
        {"seq": seq, "data_b64": base64.b64encode(wav[start:start + SEND_BYTES]).decode("ascii"),
         "mime": "audio/wav"}
        for seq, start in enumerate(range(0, len(wav), SEND_BYTES))
    ]
