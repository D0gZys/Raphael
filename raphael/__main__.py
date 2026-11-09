"""Point d'entrée : `python -m raphael` (voix) ou `python -m raphael --texte`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .brain import Brain
from .config import ConfigError, Settings
from .tools import Toolbox
from .tools import web_search


def build_brain(settings: Settings) -> Brain:
    from openai import OpenAI

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    toolbox = Toolbox()
    toolbox.register(web_search.SPEC, web_search.WebSearch.from_api_key(settings.tavily_api_key))
    return Brain(client, settings.llm_model, toolbox, settings.llm_reasoning_effort)


def run_text(brain: Brain) -> None:
    print("Mode texte. Écris ta question (Entrée vide ou Ctrl+C pour quitter).\n")
    while True:
        try:
            question = input("Toi     : ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        try:
            print(f"Raphaël : {brain.ask(question)}\n")
        except Exception as exc:  # erreur réseau, clé invalide, quota...
            print(f"[Erreur] {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="raphael", description="Raphaël, ton assistant vocal.")
    parser.add_argument("--texte", action="store_true", help="discuter au clavier (sans micro)")
    parser.add_argument("--peripherique", type=int, help="numéro du périphérique audio")
    parser.add_argument("--liste-audio", action="store_true", help="lister les périphériques audio")
    parser.add_argument("-v", "--verbeux", action="store_true", help="afficher les détails")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbeux else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.liste_audio:
        import sounddevice as sd

        print(sd.query_devices())
        return

    try:
        settings = Settings.load()
    except ConfigError as exc:
        sys.exit(str(exc))

    brain = build_brain(settings)
    if args.texte:
        run_text(brain)
        return

    from .voix import run_voice

    try:
        asyncio.run(run_voice(settings, brain, args.peripherique))
    except KeyboardInterrupt:
        print("\nÀ bientôt !")


if __name__ == "__main__":
    main()
