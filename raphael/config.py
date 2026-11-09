"""Lecture de la configuration depuis le fichier `.env` (ou les variables d'environnement)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Une variable obligatoire manque dans le `.env`."""


def _get(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip()
    return value or None


def _require(name: str) -> str:
    value = _get(name)
    if not value or value.endswith("..."):
        raise ConfigError(
            f"La variable {name} est manquante. Copie `.env.example` en `.env` et remplis-la."
        )
    return value


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_reasoning_effort: str | None
    tavily_api_key: str
    stt_url: str
    tts_url: str
    kyutai_api_key: str
    tts_voice: str

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            llm_base_url=_get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            llm_api_key=_require("LLM_API_KEY"),
            llm_model=_get("LLM_MODEL", "openai/gpt-oss-120b"),
            llm_reasoning_effort=_get("LLM_REASONING_EFFORT"),
            tavily_api_key=_require("TAVILY_API_KEY"),
            stt_url=_get("STT_URL", "ws://127.0.0.1:8090"),
            tts_url=_get("TTS_URL", "ws://127.0.0.1:8089"),
            kyutai_api_key=_get("KYUTAI_API_KEY", "public_token"),
            tts_voice=_get("TTS_VOICE", "unmute-prod-website/developpeuse-3.wav"),
        )
