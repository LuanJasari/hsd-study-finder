"""Ergebnisdarstellung: Rangliste ohne hartes „0 %“ und Begründungen für die Top-3."""

from __future__ import annotations

import numpy as np
import pytest

from core.explain import build_explanations, answer_effects
from core.ranking import LOW_MATCH_LABEL, MIN_BAR_PERCENT, bar_percent, low_match_note, percent_label
from core.scoring import ProgramScore, score_programs
from tests.test_scoring import CEIL, FLOOR, make_snapshot, unit

DIRS = {"A": unit(1, 0.2, 0, 0), "B": unit(1, -0.2, 0, 0), "C": unit(0, 0, 1, 0), "D": unit(0, 0, 0, 1)}


@pytest.fixture
def snapshot():
    return make_snapshot(DIRS, chunks_per_program=4)


# ------------------------------------------------------------------- Rangliste
def test_below_floor_is_flagged_and_not_shown_as_hard_zero():
    low = ProgramScore("X", "Studiengang X", cosine=0.30, percent=0.0, below_floor=True)
    assert percent_label(low) == "< 1 %"
    assert bar_percent(low) == MIN_BAR_PERCENT
    note = low_match_note(low)
    assert note and LOW_MATCH_LABEL in note and "0.30" in note


def test_normal_scores_keep_their_value_and_have_no_warning():
    ok = ProgramScore("Y", "Studiengang Y", cosine=0.7, percent=62.4, below_floor=False)
    assert percent_label(ok) == "62 %" and bar_percent(ok) == pytest.approx(62.4)
    assert low_match_note(ok) is None


def test_score_programs_marks_below_floor(snapshot):
    scores = {s.program_id: s for s in score_programs(unit(1, 0, 0, 0), snapshot, FLOOR, CEIL)}
    assert scores["C"].below_floor and scores["C"].percent == 0
    assert not scores["A"].below_floor and scores["A"].percent > 90
    # alle vier Studiengänge bleiben in der Liste, auch die mit Mindestwert
    assert len(scores) == 4 and percent_label(scores["C"]) == "< 1 %"


# ---------------------------------------------------------------- Begründungen
def run(snapshot, chat=None, u=None, answers=None, probes=None, top_n=3):
    u = unit(1, 0, 0, 0) if u is None else u
    scores = score_programs(u, snapshot, FLOOR, CEIL)
    return scores, build_explanations(snapshot, u, scores, answers or [], probes or [], chat, top_n=top_n)


def test_top3_in_ranking_order_with_rule_based_text(snapshot):
    scores, explanations = run(snapshot)
    assert [e.program_id for e in explanations] == [s.program_id for s in scores[:3]]
    for e in explanations:
        assert e.text and e.name in e.text
        assert 1 <= len(e.modules) <= 3 and len({m for m, _ in e.modules}) == len(e.modules)


def test_low_match_program_is_called_out(snapshot):
    scores, explanations = run(snapshot, top_n=4)
    last = explanations[-1]
    assert last.low_match and "geringe Übereinstimmung" in last.text and last.percent_label == "< 1 %"


def test_answer_effects_are_relative_to_other_programs(snapshot):
    probe = snapshot.centroids[2]                       # Thema von Studiengang C
    answers = [("Mögen Sie Thema C?", "Ja", 1.0), ("Egal?", "Weiß nicht", 0.0)]
    effect_c = answer_effects(snapshot, answers, [probe, probe], 2)
    effect_a = answer_effects(snapshot, answers, [probe, probe], 0)
    assert len(effect_c) == 1                           # neutrale Antwort (Gewicht 0) zählt nicht
    assert effect_c[0][0] > 0 > effect_a[0][0]
    assert "Ja" in effect_c[0][1]


def test_pro_and_contra_lines_are_listed(snapshot):
    probe = snapshot.centroids[2]
    answers = [("Thema C?", "Ja", 1.0)]
    _, explanations = run(snapshot, u=unit(0, 0, 1, 0), answers=answers, probes=[probe])
    assert explanations[0].program_id == "C"
    assert any("Thema C?" in line for line in explanations[0].pro) and not explanations[0].contra
    for other in explanations[1:]:
        assert any("Thema C?" in line for line in other.contra) and not other.pro


class _Chat:
    label = "Test"

    def __init__(self, reply):
        self.reply, self.seen = reply, None

    def complete(self, system, user, json_mode=False, max_tokens=None):
        self.seen = {"user": user, "max_tokens": max_tokens}
        return self.reply


def test_one_llm_call_covers_all_programs_and_gets_a_scaled_budget(snapshot):
    scores = score_programs(unit(1, 0, 0, 0), snapshot, FLOOR, CEIL)
    chat = _Chat("\n".join(f"## {s.name}\nText für {s.program_id}." for s in scores[:3]))
    _, explanations = run(snapshot, chat=chat)
    assert [e.text for e in explanations] == [f"Text für {s.program_id}." for s in scores[:3]]
    assert chat.seen["max_tokens"] == 300 * 3
    assert "Belege je Studiengang" in chat.seen["user"]


def test_missing_llm_section_falls_back_per_program(snapshot):
    scores = score_programs(unit(1, 0, 0, 0), snapshot, FLOOR, CEIL)
    first, second = scores[0], scores[1]
    chat = _Chat(f"## {first.name}\nNur dieser Studiengang wurde begründet.")
    _, explanations = run(snapshot, chat=chat)
    by_id = {e.program_id: e for e in explanations}
    assert by_id[first.program_id].text == "Nur dieser Studiengang wurde begründet."
    assert "liegt bei" in by_id[second.program_id].text      # Regeltext für den ausgelassenen


def test_llm_failure_gives_rule_based_texts_for_all(snapshot):
    class Boom(_Chat):
        def complete(self, *a, **k):
            raise TimeoutError("zu langsam")

    _, explanations = run(snapshot, chat=Boom(""))
    assert len(explanations) == 3 and all(e.text for e in explanations)
