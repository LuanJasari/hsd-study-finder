"""Begründung der Empfehlung für die Top-Studiengänge – nachvollziehbar statt nur „passt am besten“.

Je Studiengang gibt es immer die nachprüfbaren Belege (nächste Module, Antworten, die ihn hoben oder
senkten). Das LLM formuliert daraus in einem einzigen Aufruf einen kurzen Fließtext je Studiengang;
fehlt das LLM oder liefert es für einen Studiengang nichts Brauchbares, baut eine Regel den Text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from config import settings
from core.question_generator import _context, _excerpt, _module_title
from core.ranking import LOW_MATCH_LABEL, percent_label
from core.scoring import ProgramScore
from embeddings.vector_store import IndexSnapshot

log = logging.getLogger("hsd.explain")

TOP_N = 3
MODULES_PER_PROGRAM = 3
MIN_EFFECT = 0.005   # Antworten mit kleinerem Einfluss gelten als neutral

EXPLAIN_SYSTEM = (
    "Du bist der HSD Study-Finder. Begründe für JEDEN genannten Studiengang in 2 bis 3 Sätzen auf Deutsch, "
    "warum er zu den Antworten der Person passt oder nur schwach passt. Nenne 1 bis 2 konkrete Module aus den "
    "Belegen (kursiv) und erfinde nichts dazu. Ausgabe: je Studiengang eine Zeile `## <Name>`, darunter der Text."
)


@dataclass
class ProgramExplanation:
    program_id: str
    name: str
    percent_label: str
    cosine: float
    low_match: bool
    modules: list[tuple[str, str]]                # (Modultitel, Auszug)
    pro: list[str] = field(default_factory=list)  # Antworten, die diesen Studiengang gehoben haben
    contra: list[str] = field(default_factory=list)
    text: str = ""


def _evidence(snapshot: IndexSnapshot, u: np.ndarray, p: int) -> list[tuple[str, str, int]]:
    """Die dem Profil nächsten Abschnitte des Studiengangs, je Modul einer."""
    seen: set[str] = set()
    result = []
    for idx in snapshot.retrieve(u, k=3 * MODULES_PER_PROGRAM, program_idx=p):
        module = snapshot.chunk_meta[idx]["module"]
        if module in seen:
            continue
        seen.add(module)
        result.append((_module_title(module), _excerpt(snapshot.chunk_texts[idx], 180), idx))
        if len(result) == MODULES_PER_PROGRAM:
            break
    return result


def answer_effects(
    snapshot: IndexSnapshot,
    answers: list[tuple[str, str, float]],
    probes: list[np.ndarray],
    program_idx: int,
) -> list[tuple[float, str]]:
    """Einfluss jeder Antwort auf einen Studiengang *relativ zu den anderen*.

    effekt = Gewicht · (cos(Probe, Zentroid) − Mittel über alle Studiengänge). Positiv heißt: Diese Antwort hat
    den Studiengang gegenüber den anderen nach vorn geschoben (z. B. „Ja“ zu einem Thema, das er abdeckt).
    """
    effects = []
    for (question, label, weight), probe in zip(answers, probes):
        if weight == 0:
            continue
        cos = snapshot.centroids @ probe
        effects.append((float(weight * (cos[program_idx] - cos.mean())), f"{question} → {label}"))
    return sorted(effects, reverse=True)


def _rule_text(ex: ProgramExplanation) -> str:
    modules = ", ".join(f"*{title}*" for title, _ in ex.modules) or "keine passenden Module gefunden"
    if ex.low_match:
        return (f"Zu {ex.name} gibt es nur eine geringe Übereinstimmung ({ex.percent_label}). "
                f"Am nächsten liegen noch: {modules}.")
    return f"{ex.name} liegt bei {ex.percent_label}. Deinem Profil am nächsten liegen: {modules}."


def _llm_texts(chat, explanations: list[ProgramExplanation], evidence: dict[str, list[str]],
               answers: list[tuple[str, str, float]]) -> dict[str, str]:
    given = [f"- {q} → {label}" for q, label, _ in answers] or ["- (keine)"]
    blocks = []
    for ex in explanations:
        blocks.append(f"### {ex.name} ({ex.percent_label}{', ' + LOW_MATCH_LABEL if ex.low_match else ''})")
        blocks += [f"- {line}" for line in evidence[ex.program_id]]
        blocks += [f"- Dafür sprach: {line}" for line in ex.pro] + [f"- Dagegen sprach: {line}" for line in ex.contra]
    user = "\n".join(["Antworten der Person:", *given, "", "Belege je Studiengang:", *blocks])
    reply = chat.complete(EXPLAIN_SYSTEM, user, max_tokens=settings.explain_max_tokens * len(explanations))
    parts = re.split(r"(?m)^#{2,3}\s*(.+?)\s*$", reply)  # [vorher, Name, Text, Name, Text, …]
    texts: dict[str, str] = {}
    for heading, body in zip(parts[1::2], parts[2::2]):
        for ex in explanations:
            if ex.program_id not in texts and (ex.name.lower() in heading.lower() or ex.program_id.lower() in heading.lower().split()):
                texts[ex.program_id] = body.strip()
    return {k: v for k, v in texts.items() if v}


def build_explanations(
    snapshot: IndexSnapshot,
    u: np.ndarray,
    scores: list[ProgramScore],
    answers: list[tuple[str, str, float]],
    probes: list[np.ndarray],
    chat,
    top_n: int = TOP_N,
) -> list[ProgramExplanation]:
    """Begründungen für die ``top_n`` bestplatzierten Studiengänge, in Reihenfolge der Rangliste."""
    explanations: list[ProgramExplanation] = []
    evidence: dict[str, list[str]] = {}
    for score in scores[:top_n]:
        p = snapshot.program_ids.index(score.program_id)
        found = _evidence(snapshot, u, p)
        effects = answer_effects(snapshot, answers, probes, p)
        explanations.append(ProgramExplanation(
            program_id=score.program_id,
            name=score.name,
            percent_label=percent_label(score),
            cosine=score.cosine,
            low_match=score.below_floor,
            modules=[(title, excerpt) for title, excerpt, _ in found],
            pro=[text for effect, text in effects[:2] if effect > MIN_EFFECT],
            contra=[text for effect, text in effects[-1:] if effect < -MIN_EFFECT],
        ))
        evidence[score.program_id] = [_context(snapshot, idx) for _, _, idx in found]

    texts: dict[str, str] = {}
    if chat is not None and explanations:
        try:
            texts = _llm_texts(chat, explanations, evidence, answers)
        except Exception as exc:
            log.warning("LLM-Begründung fehlgeschlagen (%s: %s) – nutze regelbasierte Begründung.", type(exc).__name__, exc)
    for ex in explanations:
        ex.text = texts.get(ex.program_id) or _rule_text(ex)
    missing = [ex.program_id for ex in explanations if ex.program_id not in texts]
    if chat is not None and missing:
        log.info("Regelbasierte Begründung für: %s", ", ".join(missing))
    return explanations
