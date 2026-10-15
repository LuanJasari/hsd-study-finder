"""Kalibrierung: Kurzsatz-Proben, Wahl des konservativen COS_CEIL, .env-Schreiben."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from core import calibrate
from core.calibrate import PROBE_STYLES, calibrate as run_calibration, short_probe, write_env
from tests.test_scoring import make_snapshot, unit


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministische Pseudo-Embeddings (Einheitsvektoren) ohne Modell."""
    rows = [np.random.default_rng(zlib.crc32(t.encode())).normal(size=4) for t in texts]
    return np.stack([r / np.linalg.norm(r) for r in rows]).astype(np.float32)


@pytest.fixture
def snapshot():
    return make_snapshot({"A": unit(1, 0.2, 0, 0), "B": unit(1, -0.2, 0, 0), "C": unit(0, 0, 1, 0)}, chunks_per_program=6)


def test_short_probe_drops_module_prefix_and_bullets_and_truncates():
    text = "BMI 05: Mathematik 1 – . • Mengen, Relationen und Funktionen sowie Beweismethoden und Logik der Aussagen und Prädikate in Anwendungen"
    probe = short_probe(text, words=6)
    assert probe == "Mengen, Relationen und Funktionen sowie Beweismethoden"
    assert short_probe("nur ein kurzer Text") == "nur ein kurzer Text"


def test_unknown_probe_style_is_rejected(snapshot):
    with pytest.raises(ValueError):
        run_calibration(snapshot, style="lang", embed=fake_embed)
    assert set(PROBE_STYLES) == {"chunk", "feature", "both"}


def test_both_uses_the_lower_ceil(snapshot, monkeypatch):
    monkeypatch.setattr(calibrate, "measure_floor", lambda *a, **k: np.array([0.40, 0.44]))
    measured = {"chunk": np.array([0.88, 0.92]), "feature": np.array([0.70, 0.78])}
    monkeypatch.setattr(calibrate, "measure_ceil", lambda snap, rng, style, embed: measured[style])
    both = run_calibration(snapshot, "both", embed=fake_embed)
    assert both.floor == pytest.approx(0.42)
    assert both.ceil == pytest.approx(0.74)                  # Kurzsätze, nicht die ganzen Chunks (0.90)
    assert run_calibration(snapshot, "chunk", embed=fake_embed).ceil == pytest.approx(0.90)
    assert run_calibration(snapshot, "feature", embed=fake_embed).ceil == pytest.approx(0.74)


def test_feature_style_embeds_short_probes_only_once_per_chunk(snapshot):
    seen: list[list[str]] = []

    def spy(texts):
        seen.append(list(texts))
        return fake_embed(texts)

    calibrate.measure_ceil(snapshot, np.random.default_rng(0), "feature", embed=spy)
    assert len(seen) == 1                                   # ein Batch, nicht je Versuch
    assert all(len(t.split()) <= calibrate.SHORT_WORDS for t in seen[0])


def test_scale_never_collapses(snapshot, monkeypatch):
    monkeypatch.setattr(calibrate, "measure_floor", lambda *a, **k: np.array([0.9]))
    monkeypatch.setattr(calibrate, "measure_ceil", lambda *a, **k: np.array([0.5]))
    result = run_calibration(snapshot, "both", embed=fake_embed)
    assert result.ceil - result.floor >= calibrate.MIN_SPAN - 1e-9


def test_write_env_updates_existing_and_appends_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text("LLM_PROVIDER=auto\nCOS_FLOOR=0.10\nCOS_CEIL=0.60\n")
    write_env(0.436, 0.758)
    assert (tmp_path / ".env").read_text() == "LLM_PROVIDER=auto\nCOS_FLOOR=0.436\nCOS_CEIL=0.758\n"
    (tmp_path / ".env").write_text("LLM_PROVIDER=none")
    write_env(0.4, 0.7)
    assert (tmp_path / ".env").read_text() == "LLM_PROVIDER=none\nCOS_FLOOR=0.4\nCOS_CEIL=0.7\n"
