import json
from types import SimpleNamespace as NS

from raphael.brain import Brain
from raphael.tools import Toolbox
from raphael.tools.web_search import SPEC, WebSearch, format_results


def _reply(content=None, tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


def _tool_call(call_id, name, args):
    return NS(id=call_id, function=NS(name=name, arguments=json.dumps(args)))


class FakeLLM:
    """Imite `client.chat.completions.create` en rejouant des réponses prévues."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


class FakeTavily:
    def __init__(self):
        self.queries = []

    def search(self, query, **kwargs):
        self.queries.append((query, kwargs))
        return {
            "answer": "Il fera 12 degrés à Lyon demain.",
            "results": [{"title": "Météo Lyon", "url": "https://meteo.example", "content": "Nuageux, 12°C."}],
        }


def _toolbox(tavily):
    box = Toolbox()
    box.register(SPEC, WebSearch(tavily))
    return box


def test_simple_answer_without_tools():
    llm = FakeLLM([_reply("Bonjour Thomas !")])
    brain = Brain(llm, "modele", _toolbox(FakeTavily()))

    assert brain.ask("Salut") == "Bonjour Thomas !"
    sent = llm.calls[0]
    assert sent["model"] == "modele"
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][-1] == {"role": "user", "content": "Salut"}
    assert sent["tools"][0]["function"]["name"] == "recherche_web"
    assert "reasoning_effort" not in sent


def test_web_search_tool_loop():
    tavily = FakeTavily()
    llm = FakeLLM(
        [
            _reply(tool_calls=[_tool_call("c1", "recherche_web", {"requete": "météo Lyon demain"})]),
            _reply("Demain à Lyon, ciel nuageux et douze degrés."),
        ]
    )
    brain = Brain(llm, "modele", _toolbox(tavily), reasoning_effort="low")

    answer = brain.ask("Quel temps demain à Lyon ?")

    assert answer == "Demain à Lyon, ciel nuageux et douze degrés."
    assert tavily.queries[0][0] == "météo Lyon demain"
    assert tavily.queries[0][1]["language"] == "fr"
    second = llm.calls[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1]["role"] == "tool" and "12 degrés" in second[-1]["content"]
    assert llm.calls[1]["reasoning_effort"] == "low"


def test_history_keeps_only_final_exchanges():
    llm = FakeLLM(
        [
            _reply(tool_calls=[_tool_call("c1", "recherche_web", {"requete": "x"})]),
            _reply("Réponse 1"),
            _reply("Réponse 2"),
        ]
    )
    brain = Brain(llm, "modele", _toolbox(FakeTavily()))
    brain.ask("Question 1")
    brain.ask("Question 2")

    roles = [m["role"] for m in llm.calls[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_history_is_trimmed():
    llm = FakeLLM([_reply(f"R{i}") for i in range(5)])
    brain = Brain(llm, "modele", _toolbox(FakeTavily()), max_history_turns=2)
    for i in range(5):
        brain.ask(f"Q{i}")
    contents = [m["content"] for m in llm.calls[-1]["messages"][1:]]
    assert contents == ["Q2", "R2", "Q3", "R3", "Q4"]


def test_tools_disabled_after_max_rounds():
    looping = _reply(tool_calls=[_tool_call("c", "recherche_web", {"requete": "x"})])
    llm = FakeLLM([looping, looping, _reply("Voilà.")])
    brain = Brain(llm, "modele", _toolbox(FakeTavily()), max_tool_rounds=2)

    assert brain.ask("?") == "Voilà."
    assert "tools" in llm.calls[1] and "tools" not in llm.calls[2]


def test_unknown_tool_and_bad_arguments_are_reported_to_llm():
    box = _toolbox(FakeTavily())
    assert "n'existe pas" in box.call("inconnu", "{}")
    assert "invalides" in box.call("recherche_web", "pas du json")
    assert "mauvais arguments" in box.call("recherche_web", '{"autre": 1}')


def test_search_failure_is_returned_as_text():
    class Broken:
        def search(self, query, **kwargs):
            raise TimeoutError("trop lent")

    assert "échoué" in WebSearch(Broken())("x")


def test_format_results_truncates_and_handles_empty():
    assert format_results({}) == "Aucun résultat trouvé."
    long = format_results({"results": [{"title": "T", "url": "u", "content": "mot " * 500}]})
    assert len(long) < 600 and long.endswith("…")
