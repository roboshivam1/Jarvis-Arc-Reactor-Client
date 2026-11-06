import asyncio

import pytest

from reactor import player as player_module
from reactor.player import Player, read_wav, silence
from tests.fakes import Speakers, sentence


async def test_sentences_play_in_order_in_one_stream() -> None:
    speakers = Speakers()
    p = Player(speakers.open, lead_in_ms=300, tail_ms=250)
    one, two, three = sentence(0.1), sentence(0.2), sentence(0.3)
    p.add(2, three)
    p.add(0, one)
    p.add(1, two)
    p.end()
    assert await p.finished()
    opens = [d for k, d in speakers.log if k == "open"]
    assert opens == [(24000, 1)]                                  # one stream
    expected = (silence(300, 24000, 1) + read_wav(one)[2] + read_wav(two)[2]
                + read_wav(three)[2] + silence(250, 24000, 1))
    assert speakers.written() == expected
    assert speakers.log[-1] == ("finish", None)


async def test_playback_starts_before_the_rest_arrives() -> None:
    speakers = Speakers()
    p = Player(speakers.open, lead_in_ms=0, tail_ms=0)
    p.add(0, sentence(0.1))
    await asyncio.sleep(0.05)
    assert len(speakers.written()) > 0                            # already playing
    p.add(1, sentence(0.1))
    p.end()
    assert await p.finished()


async def test_empty_and_missing_sentences_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(player_module, "_GAP_WAIT_S", 0.1)
    speakers = Speakers()
    p = Player(speakers.open, lead_in_ms=0, tail_ms=0)
    p.add(0, b"")                  # the Core's "this one failed"
    p.add(2, sentence(0.1))        # 1 never comes
    p.end()
    assert await p.finished()
    assert speakers.written() == read_wav(sentence(0.1))[2]


async def test_a_new_sample_rate_reopens_the_stream() -> None:
    speakers = Speakers()
    p = Player(speakers.open, lead_in_ms=100, tail_ms=0)
    p.add(0, sentence(0.1, 24000))
    p.add(1, sentence(0.1, 22050))
    p.end()
    assert await p.finished()
    assert [d for k, d in speakers.log if k == "open"] == [(24000, 1), (22050, 1)]


async def test_stop_cuts_it_off() -> None:
    speakers = Speakers()
    p = Player(speakers.open, lead_in_ms=0, tail_ms=0)
    p.add(0, sentence(0.1))
    await asyncio.sleep(0.02)
    await p.stop()
    assert ("kill", None) in speakers.log
    assert not p.playing
    assert await p.finished()      # nothing left to wait for


async def test_chimes_wait_for_quiet() -> None:
    speakers = Speakers()
    p = Player(speakers.open, lead_in_ms=0, tail_ms=0)
    await p.chime(b"\x01\x00" * 10)
    assert speakers.written() == b"\x01\x00" * 10
