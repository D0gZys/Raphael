"""Le cerveau de Raphaël : un LLM (interchangeable) qui peut appeler des outils."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .tools import Toolbox

log = logging.getLogger(__name__)

_MOIS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]
_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

SYSTEM_PROMPT = """Tu es Raphaël, un assistant vocal personnel, dans l'esprit de Jarvis.
Tu parles français, avec un ton naturel, chaleureux et un brin d'humour discret.

Tes réponses sont lues à voix haute par une synthèse vocale, donc :
- Réponds en une à trois phrases courtes, comme dans une vraie conversation.
- N'utilise jamais de Markdown, de listes, d'émojis, de tableaux ni d'adresses web.
- Écris les nombres et les unités comme on les dit si c'est plus naturel.
- Si la question est vague, pose une seule question courte pour préciser.

Pour tout ce qui est récent ou que tu ne sais pas avec certitude (actualités,
météo, sport, prix, horaires...), utilise l'outil recherche_web, puis résume
l'essentiel en une ou deux phrases. Ne cite pas les sources sauf si on te le demande.
N'invente jamais un fait : si tu ne trouves pas, dis-le simplement.

Nous sommes le {date}, il est {heure}."""


def _date_fr(now: datetime) -> tuple[str, str]:
    date = f"{_JOURS[now.weekday()]} {now.day} {_MOIS[now.month - 1]} {now.year}"
    return date, now.strftime("%H:%M")


class Brain:
    """Garde l'historique de la conversation et gère la boucle d'appels d'outils."""

    def __init__(
        self,
        client: Any,
        model: str,
        toolbox: Toolbox,
        reasoning_effort: str | None = None,
        max_history_turns: int = 10,
        max_tool_rounds: int = 4,
    ) -> None:
        self._client = client  # un client `openai.OpenAI` (ou compatible)
        self._model = model
        self._toolbox = toolbox
        self._reasoning_effort = reasoning_effort
        self._max_history_turns = max_history_turns
        self._max_tool_rounds = max_tool_rounds
        # Seuls les échanges finaux (question / réponse) sont gardés en mémoire.
        self._history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._history.clear()

    def _system_message(self) -> dict[str, str]:
        date, heure = _date_fr(datetime.now())
        return {"role": "system", "content": SYSTEM_PROMPT.format(date=date, heure=heure)}

    def _complete(self, messages: list[dict[str, Any]], allow_tools: bool) -> Any:
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if allow_tools and self._toolbox.specs:
            kwargs["tools"] = self._toolbox.specs
            kwargs["tool_choice"] = "auto"
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        return self._client.chat.completions.create(**kwargs)

    def ask(self, user_text: str) -> str:
        """Envoie la phrase de l'utilisateur et renvoie la réponse finale de Raphaël."""
        messages: list[dict[str, Any]] = [self._system_message(), *self._history]
        messages.append({"role": "user", "content": user_text})

        answer = ""
        for round_index in range(self._max_tool_rounds + 1):
            # Au dernier tour, on interdit les outils pour forcer une réponse.
            allow_tools = round_index < self._max_tool_rounds
            message = self._complete(messages, allow_tools).choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                answer = (message.content or "").strip()
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                log.info("Outil %s(%s)", call.function.name, call.function.arguments)
                result = self._toolbox.call(call.function.name, call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        if not answer:
            answer = "Désolé, je n'ai pas réussi à formuler de réponse."

        self._history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
        self._history = self._history[-2 * self._max_history_turns :]
        return answer
