"""Nächste Akinator-Frage: trennscharfe Modulabschnitte finden, per RAG anreichern, vom LLM formulieren lassen.

1. Für jeden Chunk der Top-Kandidaten wird die Standardabweichung seiner Kosinus-Ähnlichkeit zu den
   Kandidaten-Zentroiden berechnet. Hohe Streuung ⇒ ein „Ja“ hebt manche Kandidaten und senkt andere.
2. Bereits gefragte Themen werden über ihre Ähnlichkeit zu früheren Fragen abgewertet.
3. Je Kandidat liefert das Retrieval den Abschnitt, der dem aktuellen Profil am nächsten liegt.
4. Das LLM formuliert Frage und Merkmalssatz (JSON). Das Merkmal wird später eingebettet und ist
   der „Probe“-Vektor, den die Antwort verschiebt. Ohne LLM baut eine Regel die Frage aus dem besten Chunk.
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass

import numpy as np

from config import settings
from core.llm import FALLBACK_LABEL, ChatModel
from core.scoring import ProgramScore
from embeddings.loader import OVERVIEW_MODULE
from embeddings.vector_store import IndexSnapshot

log = logging.getLogger("hsd.questions")

SEPARATING_CHUNKS = 4
CONTEXT_CHARS = 420

SYSTEM_PROMPT = (
    "Du bist der HSD Study-Finder, ein Akinator für die Bachelor-Studiengänge des Fachbereichs Medien "
    "der Hochschule Düsseldorf. Du stellst einer Studieninteressierten oder einem Studieninteressierten "
    "EINE kurze, gut verständliche Entscheidungsfrage, die zwischen den aktuellen Kandidaten trennt. "
    "Stütze dich ausschließlich auf die gelieferten Modulabschnitte. Keine Fachbegriffe ohne Erklärung, "
    "kein Siezen, keine Suggestivfragen, und wiederhole keine bereits gestellte Frage.\n"
    "Das Feld \"feature\" wird später eingebettet und verschiebt das Profil der Person. Es muss deshalb "
    "konkret sein: ein Satz mit 8 bis 20 Wörtern, der eine Tätigkeit oder einen Anwendungsfall nennt und "
    "ein bis zwei Fachbegriffe aus den Abschnitten enthält. Keine Floskeln (\"Die Studierenden kennen …\"), "
    "keine Studiengangsnamen, keine allgemeinen Sätze wie \"Interesse an Technik\".\n"
    "Gute Merkmale: \"Interaktive Webanwendungen mit Frontend-Frameworks entwickeln und in Nutzertests prüfen\" · "
    "\"Statistische Modelle und maschinelles Lernen auf Daten anwenden, um Vorhersagen zu treffen\" · "
    "\"Audiosignale aufnehmen, mischen und im Studio produzieren\".\n"
    'Antworte NUR mit JSON: {"question": "Frage mit höchstens 20 Wörtern, auf die man mit Ja/Nein antworten kann", '
    '"detail": "ein erklärender Satz mit Beispiel", '
    '"feature": "das konkrete Merkmal wie oben beschrieben, positiv formuliert", '
    '"rationale": "ein Satz: warum diese Frage zwischen den Kandidaten trennt"}'
)

# Anwendungsentwicklung (Web, Mobile, UX) ist der Bereich, in dem sich Medieninformatik und Data-Science-/KI-Studiengänge
# überschneiden. Eine Frage wie „Programmierst du gern?“ trennt dort nicht, ein konkreter Anwendungsfall schon.
APP_DEV_PATTERN = re.compile(
    r"\bweb\w*|\bapps?\b|mobile?\b|smartphone|front-?end|\bui\b|\bux\b|usab\w+|benutzer(?:oberfläche|freundlich)\w*|"
    r"interaktiv\w*|software-?(?:entwicklung|engineering)|anwendungsentwicklung|programmier\w+",
    re.I,
)
APP_DEV_HINT = (
    "\nHinweis Anwendungsentwicklung: Mehrere Kandidaten behandeln Programmierung, Web, Apps oder UX. "
    "\"Programmierst du gern?\" oder \"Magst du Apps?\" trennt hier nicht. Frage stattdessen nach EINEM konkreten "
    "Merkmal und wähle die Dimension, die sich in den Abschnitten am deutlichsten zwischen den Kandidaten unterscheidet:\n"
    "- Medien & Interaktion: Web-Frontends und Responsive Design, Mobile Apps, Usability-/UX-Tests, Grafik, Animation, 3D/VR, Medienproduktion\n"
    "- Daten & Intelligenz: Datenanalyse, Statistik, maschinelles Lernen, Empfehlungs- und Assistenzsysteme, Informationssicherheit, KI-Infrastruktur\n"
    "- Software-Handwerk: Objektorientierung, Architektur, Datenbanken, Teamprojekte\n"
    "Das \"feature\" benennt die gewählte Dimension mit einem konkreten Anwendungsfall (z. B. \"eine Web-App entwerfen "
    "und in Nutzertests verbessern\" statt \"Software entwickeln\")."
)


@dataclass
class Question:
    text: str
    detail: str
    rationale: str
    probe_text: str   # wird eingebettet und verschiebt das Nutzerprofil
    source: str       # Anzeigename des Erzeugers (LLM oder Fallback)
    chunk_id: str | None = None


def _module_title(module: str) -> str:
    return module.split(": ", 1)[-1].strip()


def _excerpt(chunk_text: str, limit: int) -> str:
    body = chunk_text.split(" – ", 1)[-1].strip()
    if len(body) <= limit:
        return body
    return body[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " …"


def separating_chunks(
    snapshot: IndexSnapshot,
    candidates: list[int],
    asked_probes: list[np.ndarray],
    n: int = SEPARATING_CHUNKS,
) -> list[int]:
    """Indizes der trennschärfsten Chunks der Kandidaten, je Modul höchstens einer."""
    pool = np.concatenate([snapshot.chunks_of(p) for p in candidates])
    to_candidates = snapshot.chunk_vs_centroid[pool][:, candidates]
    # Streuung × Typizität: Ausreißer (Literaturlisten, Kopfzeilen) streuen zwar stark, sind aber für
    # keinen Kandidaten charakteristisch
    spread = to_candidates.std(axis=1) * np.clip(to_candidates.max(axis=1), 0.0, 1.0)
    if asked_probes:
        overlap = (snapshot.chunk_embeddings[pool] @ np.stack(asked_probes).T).max(axis=1)
        spread = spread * (1.0 - np.clip(overlap, 0.0, 1.0))
    picked: list[int] = []
    seen_modules: set[tuple[str, str]] = set()
    for i in np.argsort(-spread):
        idx = int(pool[i])
        meta = snapshot.chunk_meta[idx]
        key = (meta["program_id"], meta["module"])
        if meta["module"] == OVERVIEW_MODULE or key in seen_modules:
            continue
        seen_modules.add(key)
        picked.append(idx)
        if len(picked) == n:
            break
    return picked


def _context(snapshot: IndexSnapshot, idx: int) -> str:
    meta = snapshot.chunk_meta[idx]
    return f"[{meta['program_name']} · {_module_title(meta['module'])}] {_excerpt(snapshot.chunk_texts[idx], CONTEXT_CHARS)}"


def needs_app_dev_hint(texts: list[str], n_candidates: int) -> bool:
    """Der Hinweis greift nur, wenn mehrere Kandidaten im Rennen sind und die Abschnitte/Fragen Anwendungsentwicklung berühren."""
    return n_candidates >= 2 and any(APP_DEV_PATTERN.search(text) for text in texts)


def _prompt(
    snapshot: IndexSnapshot,
    u: np.ndarray,
    scores: list[ProgramScore],
    candidates: list[int],
    picks: list[int],
    asked: list[Question],
) -> str:
    separating = [_context(snapshot, idx) for idx in picks]
    closest = [_context(snapshot, best[0]) for p in candidates if (best := snapshot.retrieve(u, k=1, program_idx=p))]
    lines = ["Aktuelle Scores (unabhängig, Summe ≠ 100 %):"]
    lines += [f"- {s.name}: {s.percent:.0f} %" for s in scores]
    lines.append("\nTrennscharfe Modulabschnitte der Kandidaten:")
    lines += [f"{i}. {text}" for i, text in enumerate(separating, 1)]
    lines.append("\nWas dem bisherigen Profil je Kandidat am nächsten liegt:")
    lines += [f"- {text}" for text in closest]
    lines.append("\nBereits gestellte Fragen (nicht wiederholen):")
    lines += [f"- {q.text}" for q in asked] or ["- (noch keine)"]
    if needs_app_dev_hint(separating + closest + [q.text for q in asked], len(candidates)):
        log.debug("Hinweis Anwendungsentwicklung im Prompt")
        lines.append(APP_DEV_HINT)
    lines.append("\nFormuliere jetzt die nächste Frage als JSON.")
    return "\n".join(lines)


def _parse_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("keine JSON-Antwort")
    data = json.loads(raw[start:end + 1])
    for key in ("question", "feature"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"Feld '{key}' fehlt")
    return data


def _escape(text: str) -> str:
    """Fragen landen in HTML-Karten der App, daher werden Sonderzeichen maskiert."""
    return html.escape(re.sub(r"\s+", " ", text).strip(), quote=False)


def _rule_based(snapshot: IndexSnapshot, scores: list[ProgramScore], candidates: list[int], picks: list[int]) -> Question:
    idx = picks[0]
    meta = snapshot.chunk_meta[idx]
    others = [s.name for s in scores[: len(candidates)] if s.program_id != meta["program_id"]]
    rationale = (f"Der Abschnitt gehört zu {meta['program_name']} und passt deutlich weniger zu "
                 f"{' und '.join(others) if others else 'den übrigen Kandidaten'}.")
    return Question(
        text=_escape(f"Würde dich das Thema „{_module_title(meta['module'])}“ reizen?"),
        detail=_escape(_excerpt(snapshot.chunk_texts[idx], 220)),
        rationale=rationale,
        probe_text=_excerpt(snapshot.chunk_texts[idx], 300),
        source=FALLBACK_LABEL,
        chunk_id=snapshot.chunk_ids[idx],
    )


def generate_question(
    snapshot: IndexSnapshot,
    u: np.ndarray,
    scores: list[ProgramScore],
    asked: list[Question],
    asked_probes: list[np.ndarray],
    chat: ChatModel | None,
) -> Question:
    candidates = [snapshot.program_ids.index(s.program_id) for s in scores[: settings.candidate_pool]]
    picks = separating_chunks(snapshot, candidates, asked_probes)
    if not picks:  # sehr kleiner Index: ohne Abwertung neu versuchen
        picks = separating_chunks(snapshot, candidates, [])
    if chat is not None:
        try:
            data = _parse_json(chat.complete(SYSTEM_PROMPT, _prompt(snapshot, u, scores, candidates, picks, asked), json_mode=True))
            question = Question(
                text=_escape(data["question"]),
                detail=_escape(str(data.get("detail", ""))),
                rationale=str(data.get("rationale", "")).strip(),
                probe_text=data["feature"].strip(),
                source=chat.label,
                chunk_id=snapshot.chunk_ids[picks[0]],
            )
            log.info("Frage %d von %s: %s | Merkmal: %s", len(asked) + 1, chat.label, question.text, question.probe_text)
            return question
        except Exception as exc:  # Netzwerk, Rate-Limit, kaputtes JSON … die Demo läuft regelbasiert weiter
            log.warning("LLM-Frage fehlgeschlagen (%s: %s) – nutze regelbasierte Frage.", type(exc).__name__, exc)
            question = _rule_based(snapshot, scores, candidates, picks)
            question.source = f"{FALLBACK_LABEL} · {chat.label} nicht erreichbar oder unbrauchbar ({type(exc).__name__})"
            return question
    question = _rule_based(snapshot, scores, candidates, picks)
    log.info("Frage %d regelbasiert: %s", len(asked) + 1, question.text)
    return question


EXPLAIN_SYSTEM = (
    "Du bist der HSD Study-Finder. Begründe in 3 bis 4 Sätzen auf Deutsch, warum der genannte Studiengang "
    "zu den Antworten der Person passt. Nenne zwei bis drei konkrete Module aus den gelieferten Abschnitten "
    "(kursiv) und erfinde nichts dazu. Erwähne in einem Satz, welcher andere Studiengang ebenfalls in Frage kommt."
)


def build_explanation(
    snapshot: IndexSnapshot,
    u: np.ndarray,
    scores: list[ProgramScore],
    answers: list[tuple[str, str]],
    chat: ChatModel | None,
) -> str:
    top = scores[0]
    p = snapshot.program_ids.index(top.program_id)
    best = snapshot.retrieve(u, k=3, program_idx=p)
    others = ", ".join(f"{s.name} ({s.percent:.0f} %)" for s in scores[1:3])
    if chat is not None:
        try:
            given = [f"- {q} → {a}" for q, a in answers] or ["- (keine)"]
            user = "\n".join([
                f"Bester Studiengang: {top.name} ({top.percent:.0f} %). Weitere: {others or '–'}.",
                "Antworten der Person:", *given,
                "Passende Modulabschnitte:", *[f"- {_context(snapshot, i)}" for i in best],
            ])
            return chat.complete(EXPLAIN_SYSTEM, user).strip()
        except Exception as exc:
            log.warning("LLM-Begründung fehlgeschlagen (%s: %s) – nutze regelbasierte Begründung.", type(exc).__name__, exc)
    bullets = "\n".join(
        f"- *{_module_title(snapshot.chunk_meta[i]['module'])}*: {_excerpt(snapshot.chunk_texts[i], 200)}" for i in best
    )
    tail = f"\n\nAuch interessant: {others}." if others else ""
    return f"Diese Modulabschnitte von **{top.name}** liegen deinem Profil am nächsten:\n\n{bullets}{tail}"
