"""Tests der Kernmathematik – vor allem: Die Scores sind unabhängig (kein Softmax)."""

from __future__ import annotations

import numpy as np
import pytest

from core.question_generator import (
    Question,
    _escape,
    _parse_json,
    generate_question,
    separating_chunks,
)
from core.scoring import UserProfile, cosine_scores, score_programs
from core.similarity import cosine, cosine_to_percent, normalize
from embeddings.loader import OVERVIEW_MODULE, _is_informative, detect_module, is_toc_page
from embeddings.vector_store import IndexSnapshot

FLOOR, CEIL = 0.10, 0.60


def unit(*values: float) -> np.ndarray:
    return normalize(np.array(values, dtype=np.float32))


def make_snapshot(directions: dict[str, np.ndarray], chunks_per_program: int = 4) -> IndexSnapshot:
    """Synthetischer Index: je Studiengang ein paar Chunks leicht um seine Richtung gestreut."""
    rng = np.random.default_rng(0)
    ids, names, centroids, embs, texts, metas, owner = [], [], [], [], [], [], []
    for p, (pid, direction) in enumerate(directions.items()):
        ids.append(pid)
        names.append(f"Studiengang {pid}")
        centroids.append(direction)
        for i in range(chunks_per_program):
            embs.append(normalize(direction + rng.normal(0, 0.03, direction.shape)))
            texts.append(f"Modul {pid}{i}: {pid}{i} – Inhalte des Moduls {pid}{i} mit genug Text für einen Auszug.")
            metas.append({"program_id": pid, "program_name": f"Studiengang {pid}", "module": f"M{pid}{i}: Modul {pid}{i}", "page": i})
            owner.append(p)
    return IndexSnapshot(
        ids, names, np.stack(centroids), [f"{m['program_id']}-{i}" for i, m in enumerate(metas)],
        texts, metas, np.stack(embs).astype(np.float32), np.array(owner),
    )


@pytest.fixture
def snapshot() -> IndexSnapshot:
    return make_snapshot({
        "A": unit(1, 0.2, 0, 0),
        "B": unit(1, -0.2, 0, 0),
        "C": unit(0, 0, 1, 0),
    })


def by_id(scores):
    return {s.program_id: s for s in scores}


# ------------------------------------------------------------ Kosinus → Prozent
def test_percent_is_linear_between_floor_and_ceil():
    assert cosine_to_percent(0.10, FLOOR, CEIL) == pytest.approx(0)
    assert cosine_to_percent(0.35, FLOOR, CEIL) == pytest.approx(50)
    assert cosine_to_percent(0.60, FLOOR, CEIL) == pytest.approx(100)


def test_percent_is_clipped():
    assert cosine_to_percent(-0.4, FLOOR, CEIL) == 0
    assert cosine_to_percent(0.99, FLOOR, CEIL) == 100


def test_percent_rejects_inverted_range():
    with pytest.raises(ValueError):
        cosine_to_percent(0.3, 0.6, 0.1)


def test_cosine_basics():
    assert cosine(unit(1, 0), unit(1, 0)) == pytest.approx(1)
    assert cosine(unit(1, 0), unit(0, 1)) == pytest.approx(0)
    assert cosine(unit(1, 0), unit(-1, 0)) == pytest.approx(-1)
    assert cosine(np.zeros(2), unit(1, 0)) == 0  # kein NaN beim Nullvektor


# ------------------------------------------------------------------ Kein Softmax
def test_two_programs_can_both_exceed_90_percent(snapshot):
    scores = by_id(score_programs(unit(1, 0, 0, 0), snapshot, FLOOR, CEIL))
    assert scores["A"].percent > 90
    assert scores["B"].percent > 90
    assert scores["C"].percent < 10


def test_scores_do_not_sum_to_100(snapshot):
    scores = score_programs(unit(1, 0, 0, 0), snapshot, FLOOR, CEIL)
    assert sum(s.percent for s in scores) > 180  # bei Softmax wäre die Summe exakt 100


def test_all_scores_can_be_low_when_nothing_fits(snapshot):
    scores = score_programs(unit(0, 0, 0, 1), snapshot, FLOOR, CEIL)  # Richtung, die kein Studiengang hat
    assert all(s.percent < 20 for s in scores)


def test_score_is_independent_of_other_programs(snapshot):
    u = unit(1, 0.1, 0.3, 0)
    with_all = by_id(score_programs(u, snapshot, FLOOR, CEIL))["A"]
    only_ab = make_snapshot({"A": unit(1, 0.2, 0, 0), "B": unit(1, -0.2, 0, 0)})
    alone = make_snapshot({"A": unit(1, 0.2, 0, 0)})
    assert by_id(score_programs(u, only_ab, FLOOR, CEIL))["A"].percent == pytest.approx(with_all.percent)
    assert by_id(score_programs(u, alone, FLOOR, CEIL))["A"].percent == pytest.approx(with_all.percent)


def test_scores_are_sorted_descending(snapshot):
    scores = score_programs(unit(0.2, 0, 1, 0), snapshot, FLOOR, CEIL)
    assert [s.program_id for s in scores][0] == "C"
    assert [s.cosine for s in scores] == sorted((s.cosine for s in scores), reverse=True)


def test_alpha_blends_centroid_and_chunk_cosine(snapshot):
    u = unit(1, 0, 0, 0)
    pure_centroid = cosine_scores(u, snapshot, alpha=1.0)
    assert pure_centroid[0] == pytest.approx(float(snapshot.centroids[0] @ u))
    pure_chunks = cosine_scores(u, snapshot, alpha=0.0, top_k=2)
    own = np.sort(snapshot.chunk_embeddings[snapshot.chunks_of(0)] @ u)[-2:]
    assert pure_chunks[0] == pytest.approx(own.mean())


# -------------------------------------------------------------------- Nutzerprofil
def test_profile_starts_at_prior(snapshot):
    assert np.allclose(UserProfile(prior=snapshot.prior).vector(), snapshot.prior)


def test_yes_moves_towards_and_no_moves_away(snapshot):
    probe = unit(0, 0, 1, 0)
    prior = snapshot.prior
    yes, no = UserProfile(prior=prior), UserProfile(prior=prior)
    yes.add_feedback(1.0, probe)
    no.add_feedback(-1.0, probe)
    assert cosine(yes.vector(), probe) > cosine(prior, probe) > cosine(no.vector(), probe)


def test_dont_know_leaves_profile_unchanged(snapshot):
    profile = UserProfile(prior=snapshot.prior)
    profile.add_feedback(0.0, unit(0, 0, 1, 0))
    assert np.allclose(profile.vector(), snapshot.prior)


def test_profile_vector_is_unit_length_and_survives_cancellation(snapshot):
    profile = UserProfile(prior=snapshot.prior, prior_weight=1.0)
    profile.add_feedback(1.0, -snapshot.prior)  # hebt den Prior exakt auf
    v = profile.vector()
    assert np.linalg.norm(v) == pytest.approx(1)
    assert not np.isnan(v).any()


def test_answers_shift_scores_towards_chosen_program(snapshot):
    profile = UserProfile(prior=snapshot.prior)
    before = by_id(score_programs(profile.vector(), snapshot, FLOOR, CEIL))
    profile.add_feedback(1.0, snapshot.centroids[2])
    after = by_id(score_programs(profile.vector(), snapshot, FLOOR, CEIL))
    assert after["C"].percent > before["C"].percent
    assert after["A"].percent < before["A"].percent


# ------------------------------------------------------------- Fragenauswahl
def test_separating_chunks_prefer_discriminative_and_skip_overview(snapshot):
    snapshot.chunk_meta[0]["module"] = OVERVIEW_MODULE
    picks = separating_chunks(snapshot, [0, 1, 2], [])
    assert picks and 0 not in picks
    modules = [(snapshot.chunk_meta[i]["program_id"], snapshot.chunk_meta[i]["module"]) for i in picks]
    assert len(set(modules)) == len(modules)


def test_asked_topics_are_penalised(snapshot):
    first = separating_chunks(snapshot, [0, 1, 2], [])[0]
    again = separating_chunks(snapshot, [0, 1, 2], [snapshot.chunk_embeddings[first]])
    assert again[0] != first


def test_rule_based_question_without_llm(snapshot):
    u = snapshot.prior
    scores = score_programs(u, snapshot, FLOOR, CEIL)
    q = generate_question(snapshot, u, scores, [], [], chat=None)
    assert q.text.endswith("?") and q.probe_text and q.chunk_id in snapshot.chunk_ids


class _BrokenChat:
    label = "kaputt"

    def complete(self, system, user, json_mode=False, max_tokens=None):
        raise RuntimeError("Rate-Limit")


class _JsonChat:
    label = "Test-LLM"

    def complete(self, system, user, json_mode=False, max_tokens=None):
        return 'Gern!\n{"question": "Magst du <b>Mathe</b>?", "detail": "Z. B. Statistik.", "feature": "Mathematik und Statistik", "rationale": "trennt A von C"}'


def test_llm_failure_falls_back_to_rules(snapshot):
    scores = score_programs(snapshot.prior, snapshot, FLOOR, CEIL)
    q = generate_question(snapshot, snapshot.prior, scores, [], [], chat=_BrokenChat())
    assert q.source != "kaputt" and q.text


def test_llm_question_is_parsed_and_html_escaped(snapshot):
    scores = score_programs(snapshot.prior, snapshot, FLOOR, CEIL)
    q = generate_question(snapshot, snapshot.prior, scores, [], [], chat=_JsonChat())
    assert q.source == "Test-LLM" and q.probe_text == "Mathematik und Statistik"
    assert "<b>" not in q.text and "&lt;b&gt;" in q.text


def test_parse_json_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_json("keine Ahnung")
    with pytest.raises(ValueError):
        _parse_json('{"question": "", "feature": "x"}')
    assert _escape("a  <i>b</i>") == "a &lt;i&gt;b&lt;/i&gt;"


# ------------------------------------------------------------------ Loader
@pytest.mark.parametrize("line, expected", [
    ("BMI 05: Mathematik 1", "BMI 05: Mathematik 1"),
    ("BMT 11 - Grundlagen Elektrotechnik 1 - Physik", "BMT 11: Grundlagen Elektrotechnik 1 - Physik"),
    ("D3.2: Information Security and Privacy", "D3.2: Information Security and Privacy"),
    ("PF 5.3 Business Analytics: Innovationsmanagement", "PF 5.3: Business Analytics: Innovationsmanagement"),
])
def test_detect_module_headings(line, expected):
    assert detect_module(["Seitenkopf", line, "Credits Workload"]) == expected


def test_detect_module_ignores_toc_lines_and_plain_text():
    assert detect_module(["BMI 05: Mathematik 1 ........ 17", "Die Studierenden kennen …"]) is None


def test_toc_pages_are_recognised():
    toc = [f"BMI {i:02d}: Modul {i} .............. {i + 10}" for i in range(8)]
    assert is_toc_page(toc)
    assert not is_toc_page(["Lehrinhalte:", "Mengenlehre und Logik", "Literatur: Papula"])


def test_question_dataclass_defaults():
    assert Question("a?", "", "", "p", "s").chunk_id is None


def test_loader_drops_layout_boilerplate_and_literature_but_keeps_content():
    header = ("Verwendung in anderen Studiengängen: Modulbeauftragte/r: Prof. Dr. Dozent/in: Lehrende "
              "Hochschule / Studiengang: HSD Credits Workload Kontaktzeit Selbststudium Dauer 1 Sem")
    literature = "M. Beispiel: Webentwicklung für Einsteiger, 4. Auflage, Springer Verlag 2019, ISBN 123"
    content = ("Die Studierenden können interaktive Webanwendungen entwerfen, mit Nutzern testen und "
               "anhand der Ergebnisse verbessern. Sie kennen Grundlagen der Usability.")
    assert not _is_informative(header)
    assert not _is_informative(literature)
    assert _is_informative(content)
    assert not _is_informative("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30")
