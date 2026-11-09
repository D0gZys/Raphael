"""Reconnaissance vocale en continu avec le serveur Kyutai STT (moshi-server).

Le serveur renvoie les mots au fil de l'eau ("Word") et, à chaque pas de 80 ms,
des probabilités de pause ("Step"). Quand une pause est détectée après des mots,
on envoie un "Marker" : le serveur le renvoie une fois tout l'audio précédent
transcrit, ce qui garantit qu'on ne perd pas les derniers mots de la phrase.
"""

from __future__ import annotations

import asyncio
import logging

import msgpack
import numpy as np

from .audio import FRAME_SIZE

log = logging.getLogger(__name__)

# Le détecteur de pause a 4 têtes : pauses de 0,5 / 1 / 2 / 3 secondes.
# Unmute utilise la tête 2 ; la tête 1 rend Raphaël plus réactif mais il risque de couper la parole.
PAUSE_HEAD = 2
PAUSE_THRESHOLD = 0.5
# Les premières prédictions de pause sont peu fiables : on les ignore.
WARMUP_STEPS = 12


def _pack(message: dict) -> bytes:
    return msgpack.packb(message, use_bin_type=True, use_single_float=True)


class UtteranceDetector:
    """Logique pure (testable sans serveur) : assemble les mots en phrases."""

    def __init__(self) -> None:
        self.words: list[str] = []
        self.steps_seen = 0
        self.pending_marker: int | None = None
        self._next_marker = 0

    def reset(self) -> None:
        self.words.clear()
        self.pending_marker = None

    def on_message(self, data: dict) -> tuple[str, object] | None:
        """Traite un message du serveur.

        Renvoie ("marker", id) quand il faut envoyer un marqueur au serveur,
        ("utterance", texte) quand une phrase complète est prête, sinon None.
        """
        kind = data.get("type")
        if kind == "Word":
            self.words.append(data.get("text", ""))
        elif kind == "Step":
            self.steps_seen += 1
            prs = data.get("prs") or []
            if (
                self.steps_seen > WARMUP_STEPS
                and self.words
                and self.pending_marker is None
                and len(prs) > PAUSE_HEAD
                and prs[PAUSE_HEAD] > PAUSE_THRESHOLD
            ):
                self.pending_marker = self._next_marker
                self._next_marker += 1
                return ("marker", self.pending_marker)
        elif kind == "Marker" and data.get("id") == self.pending_marker:
            text = " ".join(w for w in self.words if w).strip()
            self.reset()
            if text:
                return ("utterance", text)
        elif kind == "Error":
            log.error("Erreur du serveur STT : %s", data.get("message", data))
        return None


class SpeechToText:
    """Garde une connexion ouverte et publie chaque phrase dite dans `utterances`."""

    def __init__(self, url: str, api_key: str) -> None:
        self._url = url.rstrip("/") + "/api/asr-streaming"
        self._api_key = api_key
        self.utterances: asyncio.Queue[str] = asyncio.Queue()
        self.muted = False  # pendant que Raphaël parle, on envoie du silence
        self._detector = UtteranceDetector()

    def mute(self) -> None:
        self.muted = True

    def unmute(self) -> None:
        self._detector.reset()  # oublie ce qui a pu être entendu pendant la réponse
        self.muted = False

    async def run(self, mic_queue: asyncio.Queue[np.ndarray]) -> None:
        from websockets.asyncio.client import connect

        headers = {"kyutai-api-key": self._api_key}
        async with connect(self._url, additional_headers=headers) as ws:
            log.info("Connecté au serveur STT %s", self._url)
            silence = np.zeros(FRAME_SIZE, dtype=np.float32)
            outgoing: asyncio.Queue[bytes] = asyncio.Queue()

            async def send_audio() -> None:
                while True:
                    chunk = await mic_queue.get()
                    if self.muted:
                        chunk = silence[: len(chunk)]
                    await ws.send(_pack({"type": "Audio", "pcm": chunk.tolist()}))
                    while not outgoing.empty():
                        await ws.send(outgoing.get_nowait())

            async def receive() -> None:
                async for raw in ws:
                    event = self._detector.on_message(msgpack.unpackb(raw, raw=False))
                    if event is None:
                        continue
                    kind, value = event
                    if kind == "marker":
                        outgoing.put_nowait(_pack({"type": "Marker", "id": value}))
                    elif kind == "utterance" and not self.muted:
                        await self.utterances.put(value)

            await asyncio.gather(send_audio(), receive())
