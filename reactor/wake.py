"""Wake word detection: openWakeWord's "hey jarvis", run directly.

openWakeWord is three small ONNX models in a row:

    16 kHz audio --melspectrogram--> 32-band mel frames (8 per 80 ms)
                 --embedding-------> one 96-number feature per 80 ms,
                                     from the last 76 mel frames
                 --wake model------> a score from 0 to 1,
                                     from the last 16 features (1.28 s)

This file runs those three models with onnxruntime and nothing else.
The openwakeword package itself is not used: on Linux it insists on
tflite-runtime, which has no build for Python 3.13 on the Pi. The
pipeline below follows the package's own streaming code step for step,
and a test checks its scores against the package's.

Audio goes in 1280 samples (80 ms) at a time; each call returns the
score for the latest 1.28 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

CHUNK = 1280                 # 80 ms at 16 kHz - the models' step
_CONTEXT = 160 * 3           # extra samples the mel model needs before a chunk
_MEL_WINDOW = 76             # mel frames per embedding
_MEL_KEEP = 970              # about 10 s of mel frames
_FEATURES = 16               # embeddings per wake decision
_FEATURE_KEEP = 120          # about 10 s of embeddings
_WARMUP_CHUNKS = 5           # scores before the buffers fill are noise


class WakeWord:
    """Streaming detector for one wake word."""

    def __init__(self, models_dir: Path, model_file: str, *, threads: int = 1) -> None:
        options = ort.SessionOptions()
        # One thread: the Pi has four cores and the rest of the client,
        # PipeWire and Bluetooth need them more than this needs speed.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1

        def load(name: str) -> ort.InferenceSession:
            path = models_dir / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"wake word model missing: {path} - run scripts/fetch_models.sh"
                )
            return ort.InferenceSession(str(path), options,
                                        providers=["CPUExecutionProvider"])

        self._mel = load("melspectrogram.onnx")
        self._embed = load("embedding_model.onnx")
        self._wake = load(model_file)
        self._mel_in = self._mel.get_inputs()[0].name
        self._embed_in = self._embed.get_inputs()[0].name
        self._wake_in = self._wake.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        """Forget everything heard so far."""
        self._raw = np.zeros(_CONTEXT, dtype=np.int16)
        self._mel_frames = np.ones((_MEL_WINDOW, 32), dtype=np.float32)
        # The package seeds its history with the features of 4 s of
        # quiet noise, so the first decisions are not made on garbage.
        noise = np.random.default_rng(0).integers(-1000, 1000, 16000 * 4).astype(np.int16)
        self._features = self._embeddings_of(noise)
        self._chunks = 0

    def process(self, chunk: np.ndarray) -> float:
        """Feed 1280 samples of 16-bit audio; get the wake score (0-1)."""
        if chunk.dtype != np.int16 or chunk.shape != (CHUNK,):
            raise ValueError("wake word needs 1280 int16 samples at a time")
        audio = np.concatenate((self._raw[-_CONTEXT:], chunk))
        self._raw = audio
        mel = self._mel_of(audio)
        self._mel_frames = np.vstack((self._mel_frames, mel))[-_MEL_KEEP:]
        window = self._mel_frames[-_MEL_WINDOW:][None, :, :, None]
        embedding = self._embed.run(None, {self._embed_in: window})[0].reshape(1, -1)
        self._features = np.vstack((self._features, embedding))[-_FEATURE_KEEP:]
        self._chunks += 1
        if self._chunks <= _WARMUP_CHUNKS:
            return 0.0
        recent = self._features[-_FEATURES:][None].astype(np.float32)
        return float(np.asarray(self._wake.run(None, {self._wake_in: recent})[0]).squeeze())

    def _mel_of(self, audio: np.ndarray) -> np.ndarray:
        out = self._mel.run(None, {self._mel_in: audio[None].astype(np.float32)})[0]
        # The package's adjustment to match Google's original features.
        return np.asarray(np.squeeze(out) / 10 + 2, dtype=np.float32)

    def _embeddings_of(self, audio: np.ndarray) -> np.ndarray:
        mel = self._mel_of(audio)
        windows = [mel[i:i + _MEL_WINDOW] for i in range(0, mel.shape[0], 8)
                   if mel[i:i + _MEL_WINDOW].shape[0] == _MEL_WINDOW]
        batch = np.array(windows, dtype=np.float32)[..., None]
        return np.asarray(self._embed.run(None, {self._embed_in: batch})[0]).reshape(len(windows), -1)
