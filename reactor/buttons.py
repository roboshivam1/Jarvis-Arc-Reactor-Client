"""The earphones' own buttons, as a talk button and a mode switch.

When Bluetooth earphones are connected, their play/pause/next presses
arrive over AVRCP, and BlueZ turns them into key presses on a virtual
keyboard called "<earphones> (AVRCP)" under /dev/input. This reads
that device directly (24-byte input_event records; no extra library).

Two things must be true on the Pi for it to work, both set up once:
  - the uinput kernel module is loaded, so BlueZ can make the device;
  - the user is in the `input` group, so it can be read.
`python -m reactor buttons` shows what arrives, for checking.

Whether presses come through while the earphones are in headset (mic)
mode depends on the earphones: some keep AVRCP, some treat a press as
"answer / end call", which never reaches here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import struct
from collections.abc import AsyncIterator
from pathlib import Path

log = logging.getLogger("reactor.buttons")

_EVENT = struct.Struct("llHHi")          # timeval, type, code, value
_EV_KEY = 1
KEYS = {
    164: "playpause", 200: "play", 201: "pause", 166: "stop",
    163: "next", 165: "previous", 115: "volumeup", 114: "volumedown",
}
_REPEAT_S = 0.35                          # a press counted twice is one press


def find_devices(listing: str, match: str) -> list[tuple[str, str]]:
    """(name, /dev/input/eventN) of input devices whose name contains
    `match`, from /proc/bus/input/devices."""
    found = []
    for block in listing.split("\n\n"):
        name = re.search(r'^N: Name="(.*)"$', block, re.M)
        handlers = re.search(r"^H: Handlers=(.*)$", block, re.M)
        if not name or not handlers:
            continue
        event = re.search(r"\b(event\d+)\b", handlers.group(1))
        if event and match.lower() in name.group(1).lower():
            found.append((name.group(1), f"/dev/input/{event.group(1)}"))
    return found


def decode(data: bytes) -> list[tuple[str, int]]:
    """(key name, value) for each key event: 1 press, 0 release, 2 repeat."""
    out = []
    for offset in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
        _, _, kind, code, value = _EVENT.unpack_from(data, offset)
        if kind == _EV_KEY and code in KEYS:
            out.append((KEYS[code], value))
    return out


class Buttons:
    def __init__(self, match: str = "AVRCP",
                 listing: Path = Path("/proc/bus/input/devices")) -> None:
        self._match = match
        self._listing = listing

    async def presses(self) -> AsyncIterator[str]:
        """Key names as they are pressed, across disconnects, forever."""
        warned = ""
        while True:
            try:
                devices = find_devices(self._listing.read_text(), self._match)
            except OSError:
                devices = []
            if not devices:
                if warned != "none":
                    log.info("no earphone buttons found yet")
                    warned = "none"
                await asyncio.sleep(3)
                continue
            name, path = devices[-1]
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except PermissionError:
                if warned != "perm":
                    log.warning("cannot read %s (%s) - add this user to the 'input' "
                                "group and reboot", path, name)
                    warned = "perm"
                await asyncio.sleep(10)
                continue
            except OSError:
                await asyncio.sleep(3)
                continue
            log.info("earphone buttons: %s", name)
            warned = ""
            try:
                async for key in self._read(fd):
                    yield key
            finally:
                os.close(fd)
            log.info("earphone buttons gone")
            await asyncio.sleep(1)

    async def _read(self, fd: int) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        def readable() -> None:
            try:
                data = os.read(fd, _EVENT.size * 32)
            except BlockingIOError:
                return
            except OSError:
                data = b""
            queue.put_nowait(data or None)

        loop.add_reader(fd, readable)
        last: dict[str, float] = {}
        try:
            while True:
                data = await queue.get()
                if data is None:            # the earphones disconnected
                    return
                for key, value in decode(data):
                    now = loop.time()
                    if value == 1 and now - last.get(key, -9.0) >= _REPEAT_S:
                        last[key] = now
                        yield key
        finally:
            loop.remove_reader(fd)
