import asyncio
import os
import struct
from pathlib import Path

from reactor.buttons import Buttons, decode, find_devices

DEVICES = """I: Bus=0019 Vendor=0000 Product=0001 Version=0000
N: Name="gpio-keys"
H: Handlers=kbd event0

I: Bus=0005 Vendor=0000 Product=0000 Version=0000
N: Name="Boult Z40 (AVRCP)"
P: Phys=b8:27:eb:00:00:01
H: Handlers=kbd event3
B: EV=100007
"""


def event(code: int, value: int, kind: int = 1) -> bytes:
    return struct.pack("llHHi", 0, 0, kind, code, value)


def test_the_earphones_device_is_found() -> None:
    assert find_devices(DEVICES, "AVRCP") == [("Boult Z40 (AVRCP)", "/dev/input/event3")]
    assert find_devices(DEVICES, "nothing") == []


def test_key_events_are_decoded() -> None:
    data = event(164, 1) + event(0, 0, kind=0) + event(164, 0) + event(163, 1) + event(999, 1)
    assert decode(data) == [("playpause", 1), ("playpause", 0), ("next", 1)]


async def test_presses_are_read_and_double_reports_merged(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    listing = tmp_path / "devices"
    listing.write_text(DEVICES.replace("event3", "event99"))
    buttons = Buttons(listing=listing)

    async def fake_read(fd: int):  # type: ignore[no-untyped-def]
        async for key in Buttons._read(buttons, read_fd):
            yield key

    seen: list[str] = []

    async def collect() -> None:
        async for key in fake_read(read_fd):
            seen.append(key)

    task = asyncio.create_task(collect())
    os.write(write_fd, event(200, 1) + event(200, 0) + event(200, 1) + event(163, 1))
    await asyncio.sleep(0.1)
    os.close(write_fd)            # the earphones go away
    await asyncio.wait_for(task, 2)
    os.close(read_fd)
    assert seen == ["play", "next"]   # the second "play" came too fast: one press
