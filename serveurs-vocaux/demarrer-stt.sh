#!/bin/bash
# Lance le serveur de reconnaissance vocale (Kyutai STT, ~2,5 Go de VRAM) sur le port 8090.
# À lancer dans WSL (Ubuntu), depuis n'importe quel dossier.
source "$(dirname "$0")/commun.sh"
moshi-server worker --config configs/stt.toml --port 8090
