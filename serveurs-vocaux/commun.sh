#!/bin/bash
# Préparation commune aux serveurs STT et TTS (sourcé par demarrer-*.sh).
# Adapté de github.com/kyutai-labs/unmute/dockerless (licence MIT).
set -ex
cd "$(dirname "${BASH_SOURCE[0]}")"

# moshi-server embarque une partie Python (PyTorch) pour la TTS et a besoin de
# libpython même pour la STT : on prépare l'environnement Python officiel.
# Commit de kyutai-labs/moshi (version 0.6.4) dont l'environnement Python utilise PyTorch 2.9,
# compatible avec les RTX 50xx. L'ancien commit utilisé par Unmute (PyTorch 2.6) ne l'est pas.
MOSHI_COMMIT=e6a55d2722a65870ef52a6c9f6ecfc0e90f38362
mkdir -p tts-python
[ -f tts-python/pyproject.toml ] || wget -q -P tts-python https://raw.githubusercontent.com/kyutai-labs/moshi/$MOSHI_COMMIT/rust/moshi-server/pyproject.toml
[ -f tts-python/uv.lock ] || wget -q -P tts-python https://raw.githubusercontent.com/kyutai-labs/moshi/$MOSHI_COMMIT/rust/moshi-server/uv.lock
(cd tts-python && uv venv --allow-existing)
source tts-python/.venv/bin/activate

# Doit être défini AVANT `cargo install`, sinon erreur "no module named huggingface_hub".
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH}

# Correctifs pour Ubuntu 25+/26 + RTX 50xx (voir le guide de test Moshi).
export CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-120}
export NVCC_CCBIN=${NVCC_CCBIN:-g++-14}
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CFLAGS="-std=gnu17"
export CXXFLAGS="-include cstdint"

command -v moshi-server >/dev/null || cargo install --features cuda moshi-server@0.6.4
