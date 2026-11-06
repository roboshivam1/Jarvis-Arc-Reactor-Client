import io
import wave

import numpy as np

from reactor.config import Listen
from reactor.listen import CHUNK, Background, Preroll, Utterance, chunk_frames, level_dbfs, to_wav
from tests.fakes import quiet, tone


def feed(u: Utterance, audio: np.ndarray):
    result = None
    for start in range(0, len(audio), CHUNK):
        result = u.feed(audio[start:start + CHUNK])
        if result.status != "recording":
            return result
    return result


def test_a_question_ends_at_the_pause() -> None:
    u = Utterance(Listen(), -60.0, [b"\x00\x00" * CHUNK])
    result = feed(u, np.concatenate([quiet(0.4), tone(1.5), quiet(2.0)]))
    assert result.status == "done"
    with wave.open(io.BytesIO(result.audio)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        seconds = w.getnframes() / 16000
    # preroll + lead + speech + about 0.9 s of pause, not the whole 2 s
    assert 2.6 < seconds < 3.1


def test_silence_after_the_wake_word_is_nothing() -> None:
    u = Utterance(Listen(start_timeout_s=2.0), -60.0, [])
    assert feed(u, quiet(3.0)).status == "nothing"


def test_the_chime_does_not_count_as_speech() -> None:
    u = Utterance(Listen(start_timeout_s=1.0), -60.0, [])
    # loud for the first 0.24 s (the chime), then quiet
    assert feed(u, np.concatenate([tone(0.24), quiet(2.0)])).status == "nothing"


def test_long_questions_are_cut_at_the_limit() -> None:
    u = Utterance(Listen(max_s=3.0), -60.0, [])
    assert feed(u, tone(5.0)).status == "done"


def test_loud_rooms_raise_the_bar() -> None:
    noisy = (np.random.default_rng(1).normal(0, 3000, 16000 * 2)).astype("<i2")
    background = Background()
    for start in range(0, len(noisy), CHUNK):
        background.update(level_dbfs(noisy[start:start + CHUNK]))
    assert background.dbfs > -25
    u = Utterance(Listen(start_timeout_s=2.0), background.dbfs, [])
    assert feed(u, noisy).status == "nothing"       # the room is not a question


def test_frames_rejoin_into_the_same_wav() -> None:
    wav = to_wav(tone(4.0).tobytes())
    frames = chunk_frames(wav)
    assert len(frames) > 1 and all(f["mime"] == "audio/wav" for f in frames)
    import base64
    assert b"".join(base64.b64decode(f["data_b64"]) for f in frames) == wav
    assert [f["seq"] for f in frames] == list(range(len(frames)))


def test_preroll_keeps_only_the_last_moment() -> None:
    p = Preroll(0.3)
    for i in range(10):
        p.add(np.full(CHUNK, i, dtype="<i2"))
    taken = p.take()
    assert len(taken) == 4 and np.frombuffer(taken[-1], "<i2")[0] == 9
    assert p.take() == []
