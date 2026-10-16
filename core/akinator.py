"""Session-Zustand und Frage-Antwort-Loop des Akinators."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from config import settings
from core import llm
from core.explain import ProgramExplanation, build_explanations
from core.question_generator import Question, generate_question
from core.scoring import ProgramScore, UserProfile, score_programs
from embeddings.embedder import embed_text
from embeddings.vector_store import IndexSnapshot

log = logging.getLogger("hsd.akinator")

EXTRA_QUESTIONS = 3  # so viele Fragen kommen nach „Weiter fragen“ mindestens dazu


@dataclass
class Turn:
    question: Question
    answer: str


@dataclass
class AkinatorSession:
    snapshot: IndexSnapshot
    profile: UserProfile
    scores: list[ProgramScore]
    question: Question | None = None
    turns: list[Turn] = field(default_factory=list)
    _u: np.ndarray | None = field(default=None, repr=False)  # aktueller Profilvektor, einmal je Antwort berechnet
    _probe: np.ndarray | None = field(default=None, repr=False)
    _asked_probes: list[np.ndarray] = field(default_factory=list, repr=False)
    _ask_until: int = field(default=0, repr=False)

    @classmethod
    def create(cls, snapshot: IndexSnapshot) -> "AkinatorSession":
        session = cls(snapshot, UserProfile(prior=snapshot.prior), scores=[])
        session._rescore()
        return session

    # -------------------------------------------------------------- Ablauf
    def start(self, intro: str = "") -> None:
        if intro.strip():
            self.profile.intro = embed_text(intro.strip())
            self._rescore()
        self._next_question()

    def answer(self, label: str) -> None:
        if self.question is None or self._probe is None:
            raise RuntimeError("Es gibt keine offene Frage.")
        self.profile.add_feedback(settings.answer_weights[label], self._probe)
        self._asked_probes.append(self._probe)
        self.turns.append(Turn(self.question, label))
        self._rescore()
        self._next_question()

    def continue_asking(self) -> None:
        self._ask_until = len(self.turns) + EXTRA_QUESTIONS
        self._next_question()

    def explanations(self) -> list[ProgramExplanation]:
        """Begründungen für die Top-3 der Rangliste."""
        _, chat = llm.get_chat_model()
        answers = [(t.question.text, t.answer, settings.answer_weights[t.answer]) for t in self.turns]
        return build_explanations(self.snapshot, self._u, self.scores, answers, self._asked_probes, chat)

    # ------------------------------------------------------------- intern
    def _rescore(self) -> None:
        self._u = self.profile.vector()
        self.scores = score_programs(self._u, self.snapshot)

    def _should_stop(self) -> bool:
        n = len(self.turns)
        if n < self._ask_until:
            return False
        if n >= settings.max_questions:
            return True
        if n < settings.min_questions or len(self.scores) < 2:
            return False
        return (self.scores[0].percent >= settings.stop_percent
                and self.scores[0].percent - self.scores[1].percent >= settings.stop_margin)

    def _next_question(self) -> None:
        if self._should_stop():
            self.question, self._probe = None, None
            return
        _, chat = llm.get_chat_model()
        start = time.perf_counter()
        self.question = generate_question(
            self.snapshot, self._u, self.scores,
            [t.question for t in self.turns], self._asked_probes, chat, embed=embed_text,
        )
        # Das Embedding stammt aus der Duplikat-Prüfung und wird nicht ein zweites Mal berechnet
        self._probe = self.question.probe_vec if self.question.probe_vec is not None else embed_text(self.question.probe_text)
        log.debug("Frage %d in %.0f ms erzeugt", len(self.turns) + 1, (time.perf_counter() - start) * 1000)
