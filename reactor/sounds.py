"""The chimes, made from sine waves - no sound files to ship.

There are no LEDs yet, so these are how the reactor says what it is
doing: a rising pair when it starts listening, a falling pair when the
question is sent, a low double when something failed, a soft single
note when a notification arrives, a rising three-note run when the wake
word is switched on and a falling pair of notes when it is switched off.
"""

from __future__ import annotations

import numpy as np

RATE = 24000


def _tone(freq: float, ms: int, volume: float = 0.25) -> np.ndarray:
    n = int(RATE * ms / 1000)
    t = np.arange(n) / RATE
    wave = np.sin(2 * np.pi * freq * t)
    fade = min(n // 4, int(RATE * 0.008))
    envelope = np.ones(n)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return wave * envelope * volume


def _gap(ms: int) -> np.ndarray:
    return np.zeros(int(RATE * ms / 1000))


def _pcm(*parts: np.ndarray) -> bytes:
    audio = np.concatenate(parts)
    return bytes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


LISTENING = _pcm(_tone(660, 70), _gap(20), _tone(880, 90))
SENT = _pcm(_tone(880, 60), _gap(20), _tone(660, 80))
ERROR = _pcm(_tone(220, 120, 0.3), _gap(60), _tone(196, 160, 0.3))
CANCELLED = _pcm(_tone(440, 90, 0.15))
NOTICE = _pcm(_tone(523, 140, 0.18))
WAKE_ON = _pcm(_tone(523, 60), _gap(15), _tone(659, 60), _gap(15), _tone(784, 90))
WAKE_OFF = _pcm(_tone(784, 60), _gap(15), _tone(523, 110))
