"""Nutzerprofil (Rocchio-artiges Relevance Feedback) und unabhängige Studiengangs-Scores.

    u       = normalize(w_p·c̄ + w_0·e(intro) + Σ aᵢ·e(probeᵢ))
    cos_p   = α·cos(u, c_p) + (1−α)·mean(top-k cos(u, chunk ∈ p))
    Passung = clip((cos_p − floor) / (ceil − floor), 0, 1) · 100 %

Jeder Studiengang wird für sich bewertet; die Scores werden nie gegeneinander normiert.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import settings
from core.similarity import cosine_to_percent, normalize
from embeddings.vector_store import IndexSnapshot


@dataclass
class ProgramScore:
    program_id: str
    name: str
    cosine: float
    percent: float


@dataclass
class UserProfile:
    """Der wandernde Suchvektor: Startpunkt + optionaler Intro-Text + gewichtete Antworten."""

    prior: np.ndarray
    prior_weight: float = field(default_factory=lambda: settings.prior_weight)
    intro_weight: float = field(default_factory=lambda: settings.intro_weight)
    intro: np.ndarray | None = None
    feedback: list[tuple[float, np.ndarray]] = field(default_factory=list)  # (aᵢ, e(probeᵢ))

    def add_feedback(self, weight: float, probe: np.ndarray) -> None:
        self.feedback.append((weight, probe))

    def vector(self) -> np.ndarray:
        total = self.prior_weight * self.prior
        if self.intro is not None:
            total = total + self.intro_weight * self.intro
        for weight, probe in self.feedback:
            total = total + weight * probe
        norm = float(np.linalg.norm(total))
        # Heben sich alle Anteile exakt auf, bleibt nur der neutrale Startpunkt
        return total / norm if norm > 1e-9 else self.prior


def cosine_scores(
    u: np.ndarray,
    snapshot: IndexSnapshot,
    alpha: float | None = None,
    top_k: int | None = None,
) -> np.ndarray:
    """Gemischter Kosinus je Studiengang, in Reihenfolge von ``snapshot.program_ids``."""
    alpha = settings.score_alpha if alpha is None else alpha
    top_k = settings.score_top_k_chunks if top_k is None else top_k
    u = normalize(u)
    centroid_cos = snapshot.centroids @ u
    chunk_cos = snapshot.chunk_embeddings @ u
    result = np.empty(len(snapshot.program_ids))
    for p in range(len(result)):
        own = chunk_cos[snapshot.chunks_of(p)]
        best = np.sort(own)[-top_k:] if own.size else np.array([centroid_cos[p]])
        result[p] = alpha * centroid_cos[p] + (1 - alpha) * best.mean()
    return result


def score_programs(
    u: np.ndarray,
    snapshot: IndexSnapshot,
    floor: float | None = None,
    ceil: float | None = None,
    alpha: float | None = None,
    top_k: int | None = None,
) -> list[ProgramScore]:
    """Unabhängige Scores aller Studiengänge, absteigend nach Passung sortiert."""
    floor = settings.cos_floor if floor is None else floor
    ceil = settings.cos_ceil if ceil is None else ceil
    cosines = cosine_scores(u, snapshot, alpha, top_k)
    percents = cosine_to_percent(cosines, floor, ceil)
    scores = [
        ProgramScore(pid, name, float(c), float(pct))
        for pid, name, c, pct in zip(snapshot.program_ids, snapshot.program_names, cosines, percents)
    ]
    return sorted(scores, key=lambda s: s.cosine, reverse=True)
