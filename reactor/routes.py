"""Which mic and which speaker, and keeping the earphones in the right mode.

PipeWire knows every audio device; this asks it with `pactl` (from
pulseaudio-utils, talking to pipewire-pulse) every few seconds and
picks:

    mic       a real input, never a ".monitor" (a monitor is the sound
              going OUT - listening to it would mean hearing JARVIS)
    speaker   a real output, never the dummy "auto_null"

by priority: bluetooth > usb > anything else (the onboard jack, and
later the I2S amp). A name in the config (audio.mic / audio.speaker)
beats the priority.

Bluetooth earphones have two modes. A2DP is high-quality sound with no
mic; the headset profile (HFP) has the mic but call-quality sound.
With audio.bluetooth_mic on, the earphones are kept in headset mode,
because the wake word needs the mic open all the time; with it off
they are kept in A2DP. Earphones that reconnect come back in A2DP, so
this is checked again on every look.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

log = logging.getLogger("reactor.routes")


@dataclass(frozen=True)
class Route:
    mic: str | None
    speaker: str | None


async def pactl(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "pactl", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"pactl {' '.join(args)}: {err.decode().strip()}")
    return out.decode()


def short_names(listing: str) -> list[str]:
    """Node names from `pactl list short sinks|sources`."""
    return [line.split("\t")[1] for line in listing.splitlines() if line.count("\t") >= 1]


def _rank(name: str) -> int:
    if name.startswith("bluez_"):
        return 0
    if ".usb-" in name or name.startswith(("alsa_input.usb", "alsa_output.usb")):
        return 1
    return 2


def pick(names: list[str], wanted: str) -> str | None:
    """The device to use from a list of node names."""
    usable = [n for n in names if not n.endswith(".monitor") and n != "auto_null"]
    if wanted != "auto":
        return wanted if wanted in usable else None
    return min(usable, key=lambda n: (_rank(n), n), default=None)


@dataclass(frozen=True)
class Card:
    name: str
    active: str
    profiles: dict[str, dict[str, object]]


def bluetooth_cards(listing: str) -> list[Card]:
    """Bluetooth cards from `pactl -f json list cards`."""
    cards = []
    for card in json.loads(listing or "[]"):
        if not str(card.get("name", "")).startswith("bluez_card."):
            continue
        profiles = card.get("profiles") or {}
        if isinstance(profiles, list):
            profiles = {p.get("name", ""): p for p in profiles}
        cards.append(Card(card["name"], str(card.get("active_profile", "")), profiles))
    return cards


def best_profile(card: Card, *, headset: bool) -> str | None:
    """The profile to use: the best headset (mic) profile, or the best
    A2DP one. 'Best' is PipeWire's own priority - mSBC beats CVSD."""
    def good(name: str, info: dict[str, object]) -> bool:
        if name == "off" or info.get("available") in (False, "no"):
            return False
        if headset:
            return name.startswith("headset-head-unit") or int(str(info.get("sources", 0))) > 0
        return name.startswith("a2dp")

    candidates = [(int(str(info.get("priority", 0))), name)
                  for name, info in card.profiles.items() if good(name, info)]
    return max(candidates)[1] if candidates else None


class Router:
    """Looks at the audio devices every few seconds and keeps the
    current choice in .route. Waiters are woken when it changes."""

    def __init__(self, mic: str, speaker: str, bluetooth_mic: bool, every_s: float = 4.0) -> None:
        self._mic, self._speaker, self._bt_mic = mic, speaker, bluetooth_mic
        self._every = every_s
        self.route = Route(None, None)
        self.changed = asyncio.Event()
        self._warned_json = False

    async def look(self) -> Route:
        await self._keep_profiles()
        sources = short_names(await pactl("list", "short", "sources"))
        sinks = short_names(await pactl("list", "short", "sinks"))
        route = Route(pick(sources, self._mic), pick(sinks, self._speaker))
        if route != self.route:
            log.info("audio: mic %s, speaker %s", route.mic or "NONE", route.speaker or "NONE")
            self.route = route
            self.changed.set()
        return route

    async def _keep_profiles(self) -> None:
        try:
            cards = bluetooth_cards(await pactl("-f", "json", "list", "cards"))
        except (RuntimeError, json.JSONDecodeError, KeyError) as exc:
            if not self._warned_json:
                log.warning("cannot read bluetooth profiles (%s) - set the earphones' "
                            "profile by hand", exc)
                self._warned_json = True
            return
        for card in cards:
            if not card.active:
                continue        # cannot tell what it is on; leave it alone
            want = best_profile(card, headset=self._bt_mic)
            if want and want != card.active:
                log.info("earphones: switching %s from %s to %s", card.name, card.active, want)
                try:
                    await pactl("set-card-profile", card.name, want)
                except RuntimeError as exc:
                    log.warning("could not switch profile: %s", exc)
            elif want is None and self._bt_mic:
                log.warning("earphones %s offer no headset (mic) profile", card.name)

    async def run(self) -> None:
        while True:
            try:
                await self.look()
            except Exception as exc:
                log.warning("audio check failed: %s", exc)
            await asyncio.sleep(self._every)
