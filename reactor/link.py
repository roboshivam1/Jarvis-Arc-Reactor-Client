"""The connection to the JARVIS Core: one WebSocket, kept up forever.

Every frame is a JSON envelope {"v", "kind", "payload"} (the Core fills
in id and ts). The first frame is client.hello with the token; the Core
answers core.ready or closes. After that, frames from the Core go to
the app's handler one by one, in order.

When the connection drops - Wi-Fi, a Core restart, the VPS rebooting -
it reconnects with a growing wait (1 s up to 30 s), says hello again
and carries on. Nothing is lost: the conversation lives on the Core.
A wrong token is not retried quickly; it waits five minutes, because
retrying a wrong token will not fix it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger("reactor.link")

Handler = Callable[[str, dict[str, Any]], Awaitable[None]]


class BadToken(Exception):
    pass


class Link:
    def __init__(self, url: str, token: str, label: str, handler: Handler,
                 on_up: Callable[[], None], on_down: Callable[[], None]) -> None:
        self._url = url
        self._token = token
        self._label = label
        self._handler = handler
        self._on_up = on_up
        self._on_down = on_down
        self._ws: Any = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def send(self, kind: str, payload: dict[str, Any]) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(json.dumps({"v": 1, "kind": kind, "payload": payload}))
            return True
        except ConnectionClosed:
            return False

    async def run(self) -> None:
        wait = 1.0
        while True:
            try:
                await self._session()
                wait = 1.0
            except BadToken:
                log.error("JARVIS refused the token - check %s. Retrying in 5 minutes.",
                          "the token file")
                await asyncio.sleep(300)
                continue
            except (OSError, ConnectionClosed, TimeoutError,
                    websockets.InvalidHandshake, websockets.InvalidURI) as exc:
                log.warning("not connected to JARVIS: %s", exc or type(exc).__name__)
            await asyncio.sleep(wait + random.uniform(0, wait / 4))
            wait = min(wait * 2, 30.0)

    async def _session(self) -> None:
        # max_size=None: one spoken sentence in base64 can be over the
        # library's 1 MB default.
        async with websockets.connect(self._url, max_size=None, open_timeout=15,
                                      ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps({"v": 1, "kind": "client.hello", "payload": {
                "token": self._token, "client_kind": "voice.rpi", "label": self._label}}))
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if first.get("kind") != "core.ready":
                message = first.get("payload", {}).get("message", "")
                if "token" in message:
                    raise BadToken()
                raise ConnectionError(f"unexpected first frame: {first.get('kind')} {message}")
            log.info("connected to JARVIS")
            self._ws = ws
            self._on_up()
            try:
                async for raw in ws:
                    try:
                        frame = json.loads(raw)
                        kind, payload = frame["kind"], frame.get("payload") or {}
                    except (ValueError, KeyError, TypeError):
                        log.warning("unreadable frame from the Core")
                        continue
                    try:
                        await self._handler(kind, payload)
                    except Exception:
                        log.exception("handling %s failed", kind)
            finally:
                self._ws = None
                self._on_down()
                log.info("disconnected from JARVIS")
