"""Les outils que le cerveau de Raphaël peut appeler.

Pour ajouter un outil (domotique, agenda...), crée un module avec une `SPEC`
au format "function calling" d'OpenAI et une fonction, puis enregistre-le
dans `Toolbox` (voir `raphael/__main__.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Toolbox:
    specs: list[dict[str, Any]] = field(default_factory=list)
    functions: dict[str, Callable[..., str]] = field(default_factory=dict)

    def register(self, spec: dict[str, Any], function: Callable[..., str]) -> None:
        self.specs.append(spec)
        self.functions[spec["function"]["name"]] = function

    def call(self, name: str, arguments_json: str) -> str:
        function = self.functions.get(name)
        if function is None:
            return f"Erreur : l'outil {name!r} n'existe pas."
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return f"Erreur : arguments invalides pour {name!r} : {arguments_json!r}"
        if not isinstance(arguments, dict):
            return f"Erreur : arguments invalides pour {name!r}."
        try:
            return str(function(**arguments))
        except TypeError as exc:
            return f"Erreur : mauvais arguments pour {name!r} : {exc}"
