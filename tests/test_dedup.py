"""Duplikat-Prüfung der Fragen, Ollama-Parameter und Wiederverwendung der Embeddings in der Session."""

from __future__ import annotations

import dataclasses
import json

import httpx
import numpy as np
import pytest

from core import akinator, llm
from core.question_generator import (
    Question,
    generate_question,
    history_similarity,
    separating_chunks,
)
from core.scoring import score_programs
from tests.test_scoring import CEIL, FLOOR, make_snapshot, unit

DIRS = {"A": unit(1, 0.2, 0, 0), "B": unit(1, -0.2, 0, 0), "C": unit(0, 0, 1, 0)}


@pytest.fixture
def snapshot():
    return make_snapshot(DIRS, chunks_per_program=4)


def question(text: str, probe: str = "Merkmal") -> Question:
    return Question(text, "", "", probe, "Test")


def embed_by_chunk(snapshot):
    """Fake-Embedding: bekannte Merkmalstexte → Chunk-Vektor, alles andere → orthogonaler Vektor."""
    table = {snapshot.chunk_texts[i].split(" – ", 1)[-1]: snapshot.chunk_embeddings[i] for i in range(len(snapshot.chunk_ids))}

    def embed(text: str) -> np.ndarray:
        return table.get(text, unit(0, 0, 0, 1))

    return embed


# -------------------------------------------------------------- history_similarity
def test_identical_and_reworded_questions_are_similar_by_words():
    asked = [question("Magst du Webanwendungen entwickeln und testen?", "Webanwendungen entwickeln und testen")]
    same = question("Magst du Webanwendungen entwickeln und testen?", "Webanwendungen entwickeln und testen")
    other = question("Interessiert dich Tonstudiotechnik?", "Mikrofone und Tonstudiotechnik")
    assert history_similarity(same, None, asked, []) == pytest.approx(1.0)
    assert history_similarity(other, None, asked, []) < 0.3


def test_template_words_do_not_make_different_topics_look_alike():
    a = question("Würde dich das Thema „Akustik“ reizen?", "Schallausbreitung")
    b = question("Würde dich das Thema „Statistik“ reizen?", "Wahrscheinlichkeitsrechnung")
    assert history_similarity(b, None, [a], []) == 0.0


def test_embedding_similarity_catches_paraphrases():
    probe = unit(1, 0, 0, 0)
    near = unit(1, 0.3, 0, 0)       # cos ≈ 0.96
    far = unit(0, 1, 0, 0)
    q = question("Neue Formulierung?", "ganz andere Wörter")
    assert history_similarity(q, near, [], [probe]) > 0.9
    assert history_similarity(q, far, [], [probe]) < 0.1


# ----------------------------------------------------------------- generate_question
def test_rule_based_skips_chunks_already_asked(snapshot):
    scores = score_programs(snapshot.prior, snapshot, FLOOR, CEIL)
    embed = embed_by_chunk(snapshot)
    first = generate_question(snapshot, snapshot.prior, scores, [], [], chat=None, embed=embed)
    asked, probes = [first], [first.probe_vec]
    second = generate_question(snapshot, snapshot.prior, scores, asked, probes, chat=None, embed=embed)
    assert second.chunk_id != first.chunk_id and second.text != first.text
    # auch ohne Abwertung durch die Chunk-Auswahl: die Embedding-Prüfung allein verwirft den Wiederholer
    candidates = [0, 1, 2]
    picks = separating_chunks(snapshot, candidates, [], n=8)
    assert first.chunk_id in {snapshot.chunk_ids[i] for i in picks}
    assert second.probe_vec is not None and float(second.probe_vec @ first.probe_vec) <= 0.70


def test_modules_of_earlier_questions_are_never_proposed_again(snapshot):
    scores = score_programs(snapshot.prior, snapshot, FLOOR, CEIL)
    first = generate_question(snapshot, snapshot.prior, scores, [], [], chat=None)
    module = snapshot.chunk_meta[snapshot.chunk_ids.index(first.chunk_id)]["module"]
    later = generate_question(snapshot, snapshot.prior, scores, [first], [], chat=None)
    assert snapshot.chunk_meta[snapshot.chunk_ids.index(later.chunk_id)]["module"] != module


class _ScriptedChat:
    label = "Skript-LLM"

    def __init__(self, *replies: dict):
        self.replies, self.calls = list(replies), []

    def complete(self, system, user, json_mode=False, max_tokens=None):
        self.calls.append({"user": user, "max_tokens": max_tokens})
        return json.dumps(self.replies[min(len(self.calls), len(self.replies)) - 1])


def test_duplicate_llm_question_is_retried_with_next_chunk(snapshot):
    scores = score_programs(snapshot.prior, snapshot, FLOOR, CEIL)
    old = question("Magst du Webanwendungen entwickeln?", "Webanwendungen entwickeln und testen")
    chat = _ScriptedChat(
        {"question": "Magst du Webanwendungen entwickeln?", "feature": "Webanwendungen entwickeln und testen"},
        {"question": "Interessiert dich Raumakustik?", "feature": "Schallmessung in Räumen durchführen"},
    )
    q = generate_question(snapshot, snapshot.prior, scores, [old], [], chat=chat)
    assert len(chat.calls) == 2
    assert q.source == "Skript-LLM" and "Raumakustik" in q.text
    assert "Verworfen" in chat.calls[1]["user"] and "Webanwendungen" in chat.calls[1]["user"]
    assert all(c["max_tokens"] == 100 for c in chat.calls)


def test_two_duplicate_llm_questions_fall_back_to_next_rag_chunk(snapshot):
    scores = score_programs(snapshot.prior, snapshot, FLOOR, CEIL)
    old = question("Magst du Webanwendungen entwickeln?", "Webanwendungen entwickeln und testen")
    dup = {"question": "Magst du Webanwendungen entwickeln?", "feature": "Webanwendungen entwickeln und testen"}
    chat = _ScriptedChat(dup, dup)
    q = generate_question(snapshot, snapshot.prior, scores, [old], [], chat=chat)
    assert len(chat.calls) == 2                       # kein dritter, teurer LLM-Aufruf
    assert q.source.startswith("Regelbasierter Fallback") and "zu ähnlich" in q.source
    assert q.chunk_id is not None


def test_everything_asked_returns_least_similar_instead_of_crashing(snapshot):
    scores = score_programs(snapshot.prior, snapshot, FLOOR, CEIL)
    embed = embed_by_chunk(snapshot)
    asked, probes = [], []
    for _ in range(20):                               # mehr Fragen, als der Index Module hat
        q = generate_question(snapshot, snapshot.prior, scores, asked, probes, chat=None, embed=embed)
        asked.append(q)
        probes.append(q.probe_vec)
    assert len(asked) == 20 and all(q.text for q in asked)


# ---------------------------------------------------------------------- Ollama
def test_ollama_request_is_short_cold_start_free_and_deterministic(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, payload=json)
        return httpx.Response(200, json={"message": {"content": "{}"}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    llm.OllamaChat().complete("sys", "user", json_mode=True, max_tokens=100)
    options = sent["payload"]["options"]
    assert options["num_predict"] <= 100
    assert options["temperature"] == pytest.approx(0.3)
    assert sent["payload"]["format"] == "json" and sent["payload"]["keep_alive"]


# ------------------------------------------------------------------ Session
def test_session_reuses_probe_embedding_and_profile_vector(snapshot, monkeypatch):
    calls = []
    table = embed_by_chunk(snapshot)

    def counting_embed(text):
        calls.append(text)
        return table(text)

    monkeypatch.setattr(akinator, "embed_text", counting_embed)
    monkeypatch.setattr(akinator.llm, "get_chat_model", lambda: (llm.FALLBACK_LABEL, None))
    monkeypatch.setattr(akinator, "settings", dataclasses.replace(akinator.settings, max_questions=3, min_questions=9))
    session = akinator.AkinatorSession.create(snapshot)
    session.start()
    first_calls = len(calls)
    assert session.question is not None and first_calls >= 1
    assert calls.count(session.question.probe_text) == 1          # Merkmal wurde nur einmal eingebettet
    vector_before = session._u
    session.answer("Ja")
    assert session._u is not vector_before and np.isclose(np.linalg.norm(session._u), 1)
    assert session.question.text != session.turns[0].question.text
    session.answer("Nein")
    session.answer("Eher ja")
    assert session.question is None and len(session.turns) == 3   # max_questions erreicht
