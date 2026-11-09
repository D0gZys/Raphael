"""Micro et haut-parleur (via sounddevice / PortAudio)."""

from __future__ import annotations

import asyncio
import threading

import numpy as np

SAMPLE_RATE = 24_000  # les modèles Kyutai travaillent en 24 kHz mono
FRAME_SIZE = 1_920  # 80 ms, la cadence des modèles Kyutai


class Microphone:
    """Envoie le son du micro, par blocs de 80 ms, dans une file asyncio."""

    def __init__(self, loop: asyncio.AbstractEventLoop, device: int | None = None) -> None:
        import sounddevice as sd

        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        def callback(indata, frames, time, status):  # appelé dans un thread audio
            chunk = indata[:, 0].astype(np.float32).copy()
            loop.call_soon_threadsafe(self.queue.put_nowait, chunk)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            device=device,
            callback=callback,
        )

    def __enter__(self) -> "Microphone":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()


class Speaker:
    """Joue des échantillons au fil de l'eau, sans se soucier de la taille des blocs reçus."""

    def __init__(self, device: int | None = None) -> None:
        import sounddevice as sd

        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

        def callback(outdata, frames, time, status):
            with self._lock:
                n = min(frames, len(self._buffer))
                outdata[:n, 0] = self._buffer[:n]
                outdata[n:, 0] = 0
                self._buffer = self._buffer[n:]

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            device=device,
            callback=callback,
        )

    def __enter__(self) -> "Speaker":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()

    def play(self, pcm: np.ndarray) -> None:
        with self._lock:
            self._buffer = np.concatenate([self._buffer, pcm.astype(np.float32)])

    def pending_seconds(self) -> float:
        with self._lock:
            return len(self._buffer) / SAMPLE_RATE

    async def wait_until_done(self) -> None:
        while self.pending_seconds() > 0:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.15)  # laisse le dernier bloc sortir du haut-parleur
