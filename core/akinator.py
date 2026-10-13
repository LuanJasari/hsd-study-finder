"""Session-Zustand und Frage-Antwort-Loop des Akinators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import settings
from core import llm
from core.question_generator import Question, build_explanation, generate_question
from core.scoring import ProgramScore, UserProfile, score_programs
from embeddings.embedder import embed_text
from embeddings.vector_store import IndexSnapshot

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
    _probe: np.ndarray | None = field(default=None, repr=False)
    _asked_probes: list[np.ndarray] = field(default_factory=list, repr=False)
    _ask_until: int = field(default=0, repr=False)

    @classmethod
    def create(cls, snapshot: IndexSnapshot) -> "AkinatorSession":
        profile = UserProfile(prior=snapshot.prior)
        return cls(snapshot, profile, score_programs(profile.vector(), snapshot))

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

    def explanation(self) -> str:
        _, chat = llm.get_chat_model()
        answers = [(t.question.text, t.answer) for t in self.turns]
        return build_explanation(self.snapshot, self.profile.vector(), self.scores, answers, chat)

    # ------------------------------------------------------------- intern
    def _rescore(self) -> None:
        self.scores = score_programs(self.profile.vector(), self.snapshot)

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
        self.question = generate_question(
            self.snapshot, self.profile.vector(), self.scores,
            [t.question for t in self.turns], self._asked_probes, chat,
        )
        self._probe = embed_text(self.question.probe_text)
