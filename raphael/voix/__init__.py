"""Mode vocal : micro -> Kyutai STT -> cerveau -> Kyutai TTS -> haut-parleur."""

from __future__ import annotations

import asyncio
import logging

from ..brain import Brain
from ..config import Settings

log = logging.getLogger(__name__)


async def run_voice(settings: Settings, brain: Brain, device: int | None = None) -> None:
    from .audio import Microphone, Speaker
    from .stt import SpeechToText
    from .tts import TextToSpeech

    stt = SpeechToText(settings.stt_url, settings.kyutai_api_key)
    tts = TextToSpeech(settings.tts_url, settings.kyutai_api_key, settings.tts_voice)
    loop = asyncio.get_running_loop()

    with Microphone(loop, device) as mic, Speaker(device) as speaker:
        stt_task = asyncio.create_task(stt.run(mic.queue))
        print("Raphaël t'écoute. Parle normalement (Ctrl+C pour quitter).")
        print("Conseil : utilise un casque, sinon Raphaël risque de s'entendre parler.\n")
        try:
            while True:
                get_utterance = asyncio.create_task(stt.utterances.get())
                done, _ = await asyncio.wait(
                    {get_utterance, stt_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if stt_task in done:
                    get_utterance.cancel()
                    stt_task.result()  # remonte l'erreur de connexion éventuelle
                    raise RuntimeError("La connexion au serveur STT s'est fermée.")
                question = get_utterance.result()
                print(f"Toi     : {question}")

                stt.mute()
                try:
                    answer = await asyncio.to_thread(brain.ask, question)
                    print(f"Raphaël : {answer}\n")
                    await tts.speak(answer, speaker)
                except Exception:
                    log.exception("Erreur pendant la réponse")
                finally:
                    stt.unmute()
        finally:
            stt_task.cancel()
