"""Synthèse vocale avec le serveur Kyutai TTS (moshi-server)."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlencode

import msgpack
import numpy as np

from .audio import Speaker

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+")


def prepare_text(text: str) -> str:
    """Nettoie le texte pour qu'il se lise bien à voix haute."""
    text = _URL_RE.sub("", text)
    text = re.sub(r"[*_#`>|]", "", text)  # restes de Markdown
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return " ".join(text.split())


class TextToSpeech:
    def __init__(self, url: str, api_key: str, voice: str) -> None:
        params = urlencode({"voice": voice, "format": "PcmMessagePack"})
        self._url = f"{url.rstrip('/')}/api/tts_streaming?{params}"
        self._api_key = api_key

    async def speak(self, text: str, speaker: Speaker) -> None:
        from websockets.asyncio.client import connect

        text = prepare_text(text)
        if not text:
            return
        headers = {"kyutai-api-key": self._api_key}
        async with connect(self._url, additional_headers=headers) as ws:
            for word in text.split():
                await ws.send(msgpack.packb({"type": "Text", "text": word}))
            await ws.send(msgpack.packb({"type": "Eos"}))
            async for raw in ws:  # le serveur ferme la connexion à la fin
                message = msgpack.unpackb(raw, raw=False)
                if message.get("type") == "Audio":
                    speaker.play(np.asarray(message["pcm"], dtype=np.float32))
                elif message.get("type") == "Error":
                    log.error("Erreur du serveur TTS : %s", message.get("message", message))
        await speaker.wait_until_done()
