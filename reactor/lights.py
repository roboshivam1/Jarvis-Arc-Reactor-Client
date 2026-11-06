"""The LED ring, behind one small interface.

The state machine calls show() on every change. Until the new rings
arrive this only logs; the SPI ring driver will be another class with
the same show() method.

States and their planned colours:
    offline     amber      - no connection to JARVIS
    idle        blue       - listening for the wake word
    listening   red        - recording the question
    thinking    pulsing white
    speaking    green
"""

from __future__ import annotations

import logging

log = logging.getLogger("reactor.lights")


class NoLights:
    def show(self, state: str) -> None:
        log.debug("lights: %s", state)
