"""Settings: built-in defaults, overridden by ~/.config/reactor/config.toml.

Every setting has a working default, so the file only needs the lines
that differ. The token is never in this file - it stays in its own
file (token_file), readable only by this user.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("~/.config/reactor/config.toml").expanduser()


@dataclass
class Wake:
    models_dir: str = "~/reactor/models"
    model: str = "hey_jarvis_v0.1.onnx"
    # Score from 0 to 1 that counts as the wake word. Higher means fewer
    # false wakes and more missed ones; `python -m reactor scores`
    # shows live scores for tuning.
    threshold: float = 0.5
    # After a wake, ignore the wake word this long (it is still being said).
    refractory_s: float = 2.0
    # Listen for the wake word while JARVIS is talking, to cut him off.
    barge_in: bool = True
    # Start with the wake word on. The earphones' toggle button (see
    # [buttons]) switches it on and off while running; with it off, the
    # talk button is the only way in, and the detector stops using CPU.
    enabled: bool = True


@dataclass
class Buttons:
    # The earphones' buttons, through BlueZ's "(AVRCP)" input device.
    enabled: bool = True
    # Part of the input device's name to look for.
    device: str = "AVRCP"
    # Keys that mean "talk": starts a question (or cuts JARVIS off); a
    # second press while recording sends it straight away.
    talk: list[str] = field(default_factory=lambda: ["playpause", "play", "pause"])
    # The key that switches the wake word on and off. "" for none.
    toggle: str = "next"
    # The key that cancels: stops a question being recorded, or JARVIS
    # mid-sentence. "" for none.
    cancel: str = "previous"


@dataclass
class Listen:
    # After the wake word: how long to wait for him to start talking.
    start_timeout_s: float = 5.0
    # How long a pause ends the question.
    end_silence_s: float = 0.9
    # The longest question, after which it is sent anyway.
    max_s: float = 20.0
    # Audio kept from just before the wake fired, so a question said
    # straight after "hey jarvis" does not lose its first word.
    preroll_s: float = 0.3
    # Ignore sound this long after the wake for deciding he has started
    # talking - it covers the listening chime.
    ignore_s: float = 0.3
    # Loudness, in dB above the background, that counts as speech.
    speech_margin_db: float = 10.0
    # Never treat anything quieter than this as speech (dBFS).
    speech_floor_dbfs: float = -50.0


@dataclass
class Audio:
    # "auto" picks by priority (bluetooth > usb > onboard); or give a
    # PipeWire node name from `python -m reactor check`.
    mic: str = "auto"
    speaker: str = "auto"
    # Use the Bluetooth earphones' own mic. This switches them to the
    # headset (call) profile: the mic works, but sound drops to call
    # quality. Set false once another mic is fitted, and they go back
    # to high-quality playback.
    bluetooth_mic: bool = True
    # Silence played before a reply, so the earphones are awake for the
    # first word.
    lead_in_ms: int = 300
    # Silence after the last sentence, so the tail is not cut off.
    tail_ms: int = 250
    # Play the short chimes (listening, sent, error, notification).
    chimes: bool = True


@dataclass
class Config:
    core_url: str = "wss://jarvis-core.tail831914.ts.net/ws/client"
    token_file: str = "~/.jarvis_token"
    label: str = "arc reactor"
    log_level: str = "INFO"
    # Seconds to wait for JARVIS to hear the question before giving up.
    transcript_timeout_s: float = 25.0
    wake: Wake = field(default_factory=Wake)
    listen: Listen = field(default_factory=Listen)
    audio: Audio = field(default_factory=Audio)
    buttons: Buttons = field(default_factory=Buttons)

    @property
    def models_path(self) -> Path:
        return Path(self.wake.models_dir).expanduser()

    def token(self) -> str:
        path = Path(self.token_file).expanduser()
        try:
            token = path.read_text().strip()
        except FileNotFoundError:
            raise SystemExit(f"No token file at {path}. Put the gateway token there "
                             f"and run: chmod 600 {path}") from None
        if not token:
            raise SystemExit(f"The token file {path} is empty.")
        return token


def _apply(target: Any, values: dict[str, Any], where: str) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise SystemExit(f"Unknown setting {where}{key} in {CONFIG_PATH}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise SystemExit(f"[{where}{key}] must be a section in {CONFIG_PATH}")
            _apply(current, value, f"{where}{key}.")
        else:
            if isinstance(current, float) and isinstance(value, int):
                value = float(value)
            if type(value) is not type(current):
                raise SystemExit(f"{where}{key} should be {type(current).__name__} "
                                 f"in {CONFIG_PATH}")
            setattr(target, key, value)


def load(path: Path = CONFIG_PATH) -> Config:
    config = Config()
    if path.is_file():
        with path.open("rb") as handle:
            _apply(config, tomllib.load(handle), "")
    return config
