"""Provider-Auswahl, Logging und Prompt-Schärfung."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

from core import llm
from core.question_generator import (
    APP_DEV_HINT,
    SYSTEM_PROMPT,
    _prompt,
    generate_question,
    needs_app_dev_hint,
)
from core.scoring import score_programs
from tests.test_scoring import FLOOR, CEIL, make_snapshot, unit


@pytest.fixture
def provider(monkeypatch):
    """Setzt Provider-Einstellungen und leert den Auswahl-Cache; propagiert hsd.* für caplog."""
    def configure(**overrides):
        monkeypatch.setattr(llm, "settings", dataclasses.replace(llm.settings, **overrides))
        llm._select.cache_clear()
    monkeypatch.setattr(logging.getLogger("hsd"), "propagate", True)
    yield configure
    llm._select.cache_clear()


# ----------------------------------------------------------- Provider-Transparenz
def test_none_provider_is_rule_based_and_says_so(provider, caplog):
    provider(llm_provider="none")
    with caplog.at_level(logging.INFO, logger="hsd.llm"):
        label, model = llm.get_chat_model()
    assert model is None and label == llm.FALLBACK_LABEL
    assert "LLM deaktiviert" in caplog.text
    assert any("bewusst deaktiviert" in n for n in llm.provider_notes())


def test_auto_without_any_provider_lists_every_reason(provider, caplog, monkeypatch):
    provider(llm_provider="auto", anthropic_api_key="", openai_api_key="")
    monkeypatch.setattr(llm, "_ollama_running", lambda: False)
    with caplog.at_level(logging.WARNING, logger="hsd.llm"):
        label, model = llm.get_chat_model()
    assert model is None and label == llm.FALLBACK_LABEL
    notes = " ".join(llm.provider_notes())
    assert "ANTHROPIC_API_KEY nicht gesetzt" in notes and "OPENAI_API_KEY nicht gesetzt" in notes and "Ollama nicht erreichbar" in notes
    assert "Kein LLM verfügbar" in caplog.text


def test_auto_prefers_ollama_when_it_is_the_only_one_running(provider, monkeypatch, caplog):
    provider(llm_provider="auto", anthropic_api_key="", openai_api_key="")
    monkeypatch.setattr(llm, "_ollama_running", lambda: True)
    with caplog.at_level(logging.INFO, logger="hsd.llm"):
        label, model = llm.get_chat_model()
    assert model is not None and label.startswith("Ollama")
    assert "LLM-Provider aktiv: Ollama" in caplog.text


def test_explicit_provider_without_key_falls_back(provider):
    provider(llm_provider="anthropic", anthropic_api_key="")
    label, model = llm.get_chat_model()
    assert model is None and label == llm.FALLBACK_LABEL
    assert llm.provider_notes() == ["anthropic: ANTHROPIC_API_KEY nicht gesetzt"]


class _Echo:
    label = "Echo"

    def complete(self, system, user, json_mode=False, max_tokens=None):
        return "OK"


class _Boom:
    label = "Boom"

    def complete(self, system, user, json_mode=False, max_tokens=None):
        raise TimeoutError("zu langsam")


def test_logged_chat_logs_success_and_failure(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("hsd"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="hsd.llm"):
        assert llm.LoggedChat(_Echo()).complete("s", "u") == "OK"
        with pytest.raises(TimeoutError):
            llm.LoggedChat(_Boom()).complete("s", "u")
    assert "Echo: Antwort in" in caplog.text
    assert "Boom: Aufruf fehlgeschlagen" in caplog.text and "TimeoutError" in caplog.text


def test_failed_llm_is_visible_in_question_source():
    snapshot = make_snapshot({"A": unit(1, 0.2, 0, 0), "B": unit(1, -0.2, 0, 0), "C": unit(0, 0, 1, 0)})
    scores = score_programs(snapshot.prior, snapshot, FLOOR, CEIL)
    q = generate_question(snapshot, snapshot.prior, scores, [], [], chat=_Boom())
    assert q.source.startswith(llm.FALLBACK_LABEL) and "Boom" in q.source and "TimeoutError" in q.source


# ----------------------------------------------------------------- Prompt-Schärfung
def test_app_dev_hint_triggers_on_web_mobile_ux_topics():
    assert needs_app_dev_hint(["Modul Webprogrammierung: HTML, CSS"], n_candidates=2)
    assert needs_app_dev_hint(["Mobile Apps für Smartphones"], n_candidates=3)
    assert needs_app_dev_hint(["Usability und User Experience (UX)"], n_candidates=2)
    assert not needs_app_dev_hint(["Mikrofone, Raumakustik und Tonstudio"], n_candidates=2)
    assert not needs_app_dev_hint(["Webprogrammierung"], n_candidates=1)  # kein Vergleich nötig


def _snapshot_with_texts(text: str):
    snapshot = make_snapshot({"A": unit(1, 0.2, 0, 0), "B": unit(1, -0.2, 0, 0)})
    snapshot.chunk_texts = [f"Modul – {text}" for _ in snapshot.chunk_texts]
    return snapshot


def test_prompt_includes_hint_only_for_app_development_content():
    for text, expected in [("Webanwendungen und Usability", True), ("Akustik und Tonstudio", False)]:
        snapshot = _snapshot_with_texts(text)
        scores = score_programs(snapshot.prior, snapshot, FLOOR, CEIL)
        prompt = _prompt(snapshot, snapshot.prior, scores, [0, 1], [0, 4], [])
        assert (APP_DEV_HINT in prompt) is expected


def test_system_prompt_is_compact_but_keeps_the_essentials():
    assert len(SYSTEM_PROMPT) < 700                       # kurz, damit Ollama wenig Kontext verarbeiten muss
    for essential in ("Ja/Nein", "JSON", "feature", "Modulabschnitte", "8 bis 15 Wörter"):
        assert essential in SYSTEM_PROMPT
    assert "Anwendungsentwicklung" in APP_DEV_HINT and "Medien & Interaktion" in APP_DEV_HINT
    assert "Daten & Intelligenz" in APP_DEV_HINT and len(APP_DEV_HINT) < 600
