#!/bin/bash
# Lance le serveur de synthèse vocale (Kyutai TTS, ~5,3 Go de VRAM) sur le port 8089.
# À lancer dans WSL (Ubuntu), depuis n'importe quel dossier.
source "$(dirname "$0")/commun.sh"
uv run --locked --project ./tts-python moshi-server worker --config configs/tts.toml --port 8089
