"""Outil de recherche internet via Tavily."""

from __future__ import annotations

from typing import Any, Protocol

SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "recherche_web",
        "description": (
            "Cherche sur internet des informations récentes ou que tu ne connais pas : "
            "actualités, météo, résultats sportifs, prix, horaires, faits précis. "
            "Formule une requête courte et précise."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "requete": {
                    "type": "string",
                    "description": "La requête de recherche, par exemple 'météo Lyon demain'.",
                }
            },
            "required": ["requete"],
            "additionalProperties": False,
        },
    },
}

# Longueur max de chaque extrait renvoyé au LLM (garde le contexte court = réponses rapides).
_MAX_SNIPPET = 500


class _SearchClient(Protocol):
    def search(self, query: str, **kwargs: Any) -> dict: ...


class WebSearch:
    """Encapsule le client Tavily et formate les résultats pour le LLM."""

    def __init__(self, client: _SearchClient, max_results: int = 4) -> None:
        self._client = client
        self._max_results = max_results

    @classmethod
    def from_api_key(cls, api_key: str) -> "WebSearch":
        from tavily import TavilyClient

        return cls(TavilyClient(api_key=api_key))

    def __call__(self, requete: str) -> str:
        try:
            data = self._client.search(
                requete,
                search_depth="fast",  # bon compromis vitesse / pertinence
                max_results=self._max_results,
                include_answer="basic",  # Tavily fournit aussi un résumé direct
                language="fr",  # favorise les résultats en français (sans les imposer)
                timeout=10,
            )
        except Exception as exc:  # le LLM doit pouvoir dire que la recherche a échoué
            return f"La recherche a échoué : {exc}"
        return format_results(data)


def format_results(data: dict) -> str:
    lines: list[str] = []
    answer = (data.get("answer") or "").strip()
    if answer:
        lines.append(f"Résumé : {answer}")
    for i, result in enumerate(data.get("results") or [], start=1):
        title = (result.get("title") or "").strip()
        url = (result.get("url") or "").strip()
        content = " ".join((result.get("content") or "").split())
        if len(content) > _MAX_SNIPPET:
            content = content[:_MAX_SNIPPET].rsplit(" ", 1)[0] + "…"
        lines.append(f"[{i}] {title} ({url})\n{content}")
    return "\n\n".join(lines) if lines else "Aucun résultat trouvé."
