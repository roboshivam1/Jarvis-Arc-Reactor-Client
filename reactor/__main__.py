"""Command line.

    python -m reactor                  run (what the service does)
    python -m reactor --mic-file F     run, hearing a WAV file instead of the mic
    python -m reactor check            config, token, models, audio, connection
    python -m reactor scores           live wake word scores, for tuning
    python -m reactor buttons          show the earphones' button presses
    python -m reactor ask F.wav        send one recorded question, play the reply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from reactor import __version__
from reactor.config import CONFIG_PATH, Config, load
from reactor.lights import NoLights
from reactor.listen import level_dbfs
from reactor.player import PacatStream, Player
from reactor.routes import Router, best_profile, bluetooth_cards, pactl

log = logging.getLogger("reactor")


def _player(config: Config, router: Router) -> Player:
    async def opener(rate: int, channels: int) -> PacatStream:
        return await PacatStream.open(router.route.speaker, rate, channels)
    return Player(opener, lead_in_ms=config.audio.lead_in_ms, tail_ms=config.audio.tail_ms)


def _router(config: Config) -> Router:
    return Router(config.audio.mic, config.audio.speaker, config.audio.bluetooth_mic)


async def _first_look(router: Router) -> None:
    """Find the devices once before starting; PipeWire may not be up yet
    at boot, which is fine - the router keeps looking."""
    try:
        await router.look()
    except (RuntimeError, FileNotFoundError) as exc:
        log.warning("cannot see the audio devices yet: %s", exc)


async def run(config: Config, mic_file: Path | None) -> None:
    from reactor.app import Reactor
    from reactor.mic import FileMic, PulseMic
    from reactor.wake import WakeWord

    router = _router(config)
    await _first_look(router)
    mic = FileMic(mic_file, repeat_after_s=10.0) if mic_file else PulseMic(router)
    wake = WakeWord(config.models_path, config.wake.model)
    from reactor.buttons import Buttons
    buttons = Buttons(config.buttons.device) if config.buttons.enabled else None
    reactor = Reactor(config, mic=mic, wake=wake, player=_player(config, router),
                      lights=NoLights(), token=config.token(), buttons=buttons)
    log.info("arc reactor %s - %s", __version__,
             "say \"hey jarvis\"" if config.wake.enabled else "wake word off - use the talk button")
    await asyncio.gather(router.run(), reactor.run())


async def ask(config: Config, wav_path: Path) -> int:
    from reactor.app import Reactor

    router = _router(config)
    await _first_look(router)

    class NoMic:
        async def chunks(self):  # type: ignore[no-untyped-def]
            return
            yield

    class NoWake:
        def process(self, chunk):  # type: ignore[no-untyped-def]
            return 0.0

    reactor = Reactor(config, mic=NoMic(), wake=NoWake(), player=_player(config, router),
                      lights=NoLights(), token=config.token())
    return 0 if await reactor.ask(wav_path.read_bytes()) else 1


async def scores(config: Config) -> None:
    from reactor.mic import PulseMic
    from reactor.wake import WakeWord

    router = _router(config)
    await _first_look(router)
    watcher = asyncio.create_task(router.run())
    wake = WakeWord(config.models_path, config.wake.model)
    print(f"Say \"hey jarvis\". Threshold is {config.wake.threshold}. Ctrl-C to stop.")
    try:
        async for chunk in PulseMic(router).chunks():
            score = await asyncio.to_thread(wake.process, chunk)
            level = level_dbfs(chunk)
            if score >= 0.05:
                bar = "#" * int(score * 40)
                mark = "  <- WAKE" if score >= config.wake.threshold else ""
                print(f"score {score:4.2f} {bar:<40} level {level:5.1f} dBFS{mark}")
    finally:
        watcher.cancel()


async def buttons(config: Config) -> None:
    from reactor.buttons import Buttons
    b = config.buttons
    print(f"Press the earphones' buttons. talk = {', '.join(b.talk)}; "
          f"cancel = {b.cancel or 'none'}; toggle = {b.toggle or 'none'}. Ctrl-C to stop.")
    async for key in Buttons(b.device).presses():
        role = ("talk" if key in b.talk else "cancel" if key == b.cancel
                else "toggle wake word" if key == b.toggle else "not used")
        print(f"  {key:<10} -> {role}")


async def check(config: Config) -> int:
    problems = 0

    def report(ok: bool, text: str) -> None:
        nonlocal problems
        problems += not ok
        print(("  ok    " if ok else "  FIX   ") + text)

    print(f"arc reactor {__version__}")
    report(True, f"config: {CONFIG_PATH}" + ("" if CONFIG_PATH.is_file() else " (not there - defaults)"))
    try:
        token = config.token()
        report(True, f"token file {config.token_file} ({len(token)} characters)")
    except SystemExit as exc:
        token = ""
        report(False, str(exc))
    for name in ("melspectrogram.onnx", "embedding_model.onnx", config.wake.model):
        path = config.models_path / name
        report(path.is_file(), f"model {path}")

    if config.buttons.enabled:
        import os
        from reactor.buttons import find_devices
        found = find_devices(Path("/proc/bus/input/devices").read_text(), config.buttons.device) \
            if Path("/proc/bus/input/devices").exists() else []
        if not found:
            report(False, "earphone buttons: no '(AVRCP)' input device - is uinput loaded "
                          "and are the earphones connected?")
        for device_name, device in found:
            readable = os.access(device, os.R_OK)
            report(readable, f"earphone buttons: {device_name} at {device}"
                   + ("" if readable else " - not readable: join the 'input' group"))

    print("audio:")
    try:
        router = _router(config)
        for card in bluetooth_cards(await pactl("-f", "json", "list", "cards")):
            want = best_profile(card, headset=config.audio.bluetooth_mic)
            report(want is not None,
                   f"earphones {card.name}: profile {card.active or '?'}"
                   + (f" (want {want})" if want and want != card.active else "")
                   + ("" if want else " - no usable profile"))
        route = await router.look()
        report(route.mic is not None, f"mic: {route.mic or 'none found'}")
        report(route.speaker is not None, f"speaker: {route.speaker or 'none found'}")
        if route.mic and not route.mic.startswith("bluez") and config.audio.bluetooth_mic:
            print("        (the earphones' mic is not active - are they connected?)")
    except (RuntimeError, FileNotFoundError, json.JSONDecodeError) as exc:
        report(False, f"cannot ask PipeWire: {exc}")

    print("JARVIS:")
    if token:
        import websockets
        try:
            async with websockets.connect(config.core_url, open_timeout=15, max_size=None) as ws:
                await ws.send(json.dumps({"v": 1, "kind": "client.hello", "payload": {
                    "token": token, "client_kind": "voice.rpi", "label": config.label}}))
                first = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                ok = first.get("kind") == "core.ready"
                report(ok, f"{config.core_url}: " + ("connected" if ok else
                       str(first.get("payload", {}).get("message", first.get("kind")))))
        except Exception as exc:
            report(False, f"{config.core_url}: {exc or type(exc).__name__}")
    print("all good" if not problems else f"{problems} thing(s) to fix")
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="reactor", description="JARVIS arc reactor client")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "check", "scores", "buttons", "ask"])
    parser.add_argument("file", nargs="?", type=Path, help="the WAV for 'ask'")
    parser.add_argument("--mic-file", type=Path, help="hear this WAV instead of the mic")
    parser.add_argument("--debug", action="store_true", help="log everything")
    args = parser.parse_args()

    config = load()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S",
    )
    # websockets logs every ping at debug level; that is noise here.
    logging.getLogger("websockets").setLevel(logging.INFO)
    try:
        if args.command == "check":
            sys.exit(asyncio.run(check(config)))
        if args.command == "scores":
            asyncio.run(scores(config))
        elif args.command == "buttons":
            asyncio.run(buttons(config))
        elif args.command == "ask":
            if not args.file:
                parser.error("ask needs a WAV file")
            sys.exit(asyncio.run(ask(config, args.file)))
        else:
            asyncio.run(run(config, args.mic_file))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
