import pytest

pytest.importorskip("msgpack")

from raphael.voix.stt import WARMUP_STEPS, UtteranceDetector  # noqa: E402
from raphael.voix.tts import prepare_text  # noqa: E402


def _step(pause):
    return {"type": "Step", "step_idx": 0, "prs": [pause, pause, pause, pause]}


def _warm_up(detector):
    for _ in range(WARMUP_STEPS):
        assert detector.on_message(_step(0.9)) is None


def test_utterance_flushed_with_marker_after_pause():
    d = UtteranceDetector()
    _warm_up(d)
    d.on_message({"type": "Word", "text": "Quel"})
    assert d.on_message(_step(0.1)) is None
    d.on_message({"type": "Word", "text": "temps"})

    assert d.on_message(_step(0.8)) == ("marker", 0)
    assert d.on_message(_step(0.9)) is None  # un seul marqueur à la fois
    d.on_message({"type": "Word", "text": "fait-il ?"})  # mot arrivé en retard

    assert d.on_message({"type": "Marker", "id": 0}) == ("utterance", "Quel temps fait-il ?")
    assert d.words == []


def test_no_marker_without_words_or_during_warmup():
    d = UtteranceDetector()
    d.on_message({"type": "Word", "text": "bonjour"})
    assert d.on_message(_step(0.9)) is None  # encore en échauffement
    d2 = UtteranceDetector()
    _warm_up(d2)
    assert d2.on_message(_step(0.9)) is None  # pas de mots


def test_reset_forgets_pending_marker():
    d = UtteranceDetector()
    _warm_up(d)
    d.on_message({"type": "Word", "text": "écho"})
    assert d.on_message(_step(0.9)) == ("marker", 0)
    d.reset()
    assert d.on_message({"type": "Marker", "id": 0}) is None
    d.on_message({"type": "Word", "text": "salut"})
    assert d.on_message(_step(0.9)) == ("marker", 1)


def test_prepare_text_for_speech():
    text = "**Demain** il fera 12°C, voir https://meteo.example/lyon. C’est `top` !"
    assert prepare_text(text) == "Demain il fera 12°C, voir C'est top !"
