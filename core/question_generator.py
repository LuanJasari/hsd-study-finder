"""Nächste Akinator-Frage: trennscharfe Modulabschnitte finden, per RAG anreichern, vom LLM formulieren lassen.

1. Für jeden Chunk der Top-Kandidaten wird die Standardabweichung seiner Kosinus-Ähnlichkeit zu den
   Kandidaten-Zentroiden berechnet. Hohe Streuung ⇒ ein „Ja“ hebt manche Kandidaten und senkt andere.
2. Bereits gefragte Themen werden über ihre Ähnlichkeit zu früheren Fragen abgewertet.
3. Je Kandidat liefert das Retrieval den Abschnitt, der dem aktuellen Profil am nächsten liegt.
4. Das LLM formuliert Frage und Merkmalssatz (JSON). Das Merkmal wird später eingebettet und ist
   der „Probe“-Vektor, den die Antwort verschiebt. Ohne LLM baut eine Regel die Frage aus dem besten Chunk.
5. Jede Frage wird gegen die Historie geprüft (Merkmal-Kosinus und Wort-Jaccard > ``DUPLICATE_SIMILARITY``).
   Duplikate werden verworfen; dann kommt der nächstbeste Chunk an die Reihe.
"""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from config import settings
from core.llm import FALLBACK_LABEL, ChatModel
from core.scoring import ProgramScore
from embeddings.loader import OVERVIEW_MODULE
from embeddings.vector_store import IndexSnapshot

log = logging.getLogger("hsd.questions")

SEPARATING_CHUNKS = 4   # so viele Abschnitte landen im Prompt
CANDIDATE_CHUNKS = 8    # so viele stehen als Reserve bereit, falls eine Frage ein Duplikat ist
MAX_LLM_ATTEMPTS = 2    # LLM-Aufrufe je Frage (der zweite mit dem nächstbesten Chunk als Ausgangspunkt)
CONTEXT_CHARS = 300

# Bewusst kurz: Der Prompt wird bei lokalen Modellen (Ollama) bei jedem Aufruf neu verarbeitet, und die
# Antwort muss in ~100 Tokens passen (daher nur drei kurze Felder; die Begründung liefert die Regel).
SYSTEM_PROMPT = (
    "Du bist der HSD Study-Finder, ein Akinator für Medien-Studiengänge der HS Düsseldorf. "
    "Stelle EINE Ja/Nein-Frage (du-Form, höchstens 15 Wörter), die die Kandidaten trennt. "
    "Nutze nur die gelieferten Modulabschnitte und wiederhole keine bisherige Frage.\n"
    'Antworte NUR mit JSON: {"question": "...", "detail": "höchstens 10 Wörter", '
    '"feature": "8 bis 15 Wörter: konkrete Tätigkeit mit 1-2 Fachbegriffen, '
    'z. B. Interaktive Webanwendungen entwickeln und in Nutzertests prüfen"}'
)

# Anwendungsentwicklung (Web, Mobile, UX) ist der Bereich, in dem sich Medieninformatik und Data-Science-/KI-Studiengänge
# überschneiden. Eine Frage wie „Programmierst du gern?“ trennt dort nicht, ein konkreter Anwendungsfall schon.
APP_DEV_PATTERN = re.compile(
    r"\bweb\w*|\bapps?\b|mobile?\b|smartphone|front-?end|\bui\b|\bux\b|usab\w+|benutzer(?:oberfläche|freundlich)\w*|"
    r"interaktiv\w*|software-?(?:entwicklung|engineering)|anwendungsentwicklung|programmier\w+",
    re.I,
)
APP_DEV_HINT = (
    "\nHinweis Anwendungsentwicklung: „Programmierst du gern?“ trennt hier nicht. Frage nach EINEM konkreten "
    "Anwendungsfall aus einer Dimension: Medien & Interaktion (Web-Frontends, Mobile Apps, Usability/UX, Grafik, "
    "Animation, 3D/VR) · Daten & Intelligenz (Datenanalyse, maschinelles Lernen, Empfehlungssysteme, "
    "KI-Infrastruktur) · Software-Handwerk (Architektur, Datenbanken, Teamprojekte)."
)


@dataclass
class Question:
    text: str
    detail: str
    rationale: str
    probe_text: str   # wird eingebettet und verschiebt das Nutzerprofil
    source: str       # Anzeigename des Erzeugers (LLM oder Fallback)
    chunk_id: str | None = None
    probe_vec: np.ndarray | None = field(default=None, repr=False, compare=False)  # schon berechnetes Embedding


def _module_title(module: str) -> str:
    return module.split(": ", 1)[-1].strip()


def _excerpt(chunk_text: str, limit: int) -> str:
    body = chunk_text.split(" – ", 1)[-1].strip()
    if len(body) <= limit:
        return body
    return body[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " …"


# ------------------------------------------------------------ Chunk-Auswahl
def separating_chunks(
    snapshot: IndexSnapshot,
    candidates: list[int],
    asked_probes: list[np.ndarray],
    n: int = SEPARATING_CHUNKS,
    asked_modules: set[tuple[str, str]] | frozenset = frozenset(),
) -> list[int]:
    """Indizes der trennschärfsten Chunks der Kandidaten, je Modul höchstens einer.

    Module aus ``asked_modules`` (Studiengang, Modul) kamen schon in einer früheren Frage vor und entfallen.
    """
    pool = np.concatenate([snapshot.chunks_of(p) for p in candidates])
    to_candidates = snapshot.chunk_vs_centroid[pool][:, candidates]
    # Streuung × Typizität: Ausreißer (Literaturlisten, Kopfzeilen) streuen zwar stark, sind aber für
    # keinen Kandidaten charakteristisch
    spread = to_candidates.std(axis=1) * np.clip(to_candidates.max(axis=1), 0.0, 1.0)
    stale = np.zeros(len(pool), dtype=bool)
    if asked_probes:
        overlap = (snapshot.chunk_embeddings[pool] @ np.stack(asked_probes).T).max(axis=1)
        spread = spread * (1.0 - np.clip(overlap, 0.0, 1.0))
        stale = overlap > settings.duplicate_similarity  # Thema ist praktisch schon gefragt → gar nicht erst vorschlagen
    picked: list[int] = []
    seen_modules: set[tuple[str, str]] = set(asked_modules)
    for i in np.argsort(-spread):
        if stale[i]:
            continue
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


# ------------------------------------------------------ Duplikat-Erkennung
_TEMPLATE_WORDS = {
    "würde", "dich", "thema", "reizen", "magst", "gern", "gerne", "interessiert", "interessierst", "dass",
    "oder", "eine", "einen", "einer", "sich", "wenn", "auch", "haben", "habe", "dir", "mehr", "über",
    "wie", "wäre", "wärst", "möchtest", "willst", "lernen", "arbeiten",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zäöüß0-9]{4,}", text.lower()) if t not in _TEMPLATE_WORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def history_similarity(
    question: Question,
    probe_vec: np.ndarray | None,
    asked: list[Question],
    asked_probes: list[np.ndarray],
) -> float:
    """Höchste Ähnlichkeit der Frage zur Historie (0–1).

    * Kosinus der Merkmal-Embeddings: erkennt Umformulierungen derselben Idee. Die Fragetexte selbst
      eignen sich dafür nicht, die Schablone „Würde dich das Thema …“ macht alle ähnlich.
    * Jaccard der Inhaltswörter von Frage und Merkmal: erkennt wörtliche Wiederholungen ohne Modell.
    """
    best = 0.0
    if probe_vec is not None and asked_probes:
        best = float(np.max(np.stack(asked_probes) @ probe_vec))
    text, probe = _content_tokens(question.text), _content_tokens(question.probe_text)
    for q in asked:
        best = max(best, _jaccard(text, _content_tokens(q.text)), _jaccard(probe, _content_tokens(q.probe_text)))
    return best


# ------------------------------------------------------------------ Prompt
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
    discarded: list[Question] | None = None,
) -> str:
    separating = [_context(snapshot, idx) for idx in picks]
    closest = [_context(snapshot, best[0]) for p in candidates if (best := snapshot.retrieve(u, k=1, program_idx=p))]
    lines = ["Scores: " + "; ".join(f"{s.name} {s.percent:.0f} %" for s in scores)]
    lines.append("Trennscharfe Abschnitte:")
    lines += [f"{i}. {text}" for i, text in enumerate(separating, 1)]
    lines.append("Nächste Abschnitte zum Profil:")
    lines += [f"- {text}" for text in closest]
    lines.append("Schon gefragt (nicht wiederholen):")
    lines += [f"- {q.text}" for q in asked] or ["- (noch nichts)"]
    if discarded:
        lines.append("Verworfen, zu ähnlich zu früheren Fragen (anderes Thema wählen):")
        lines += [f"- {q.text}" for q in discarded]
    if needs_app_dev_hint(separating + closest + [q.text for q in asked], len(candidates)):
        log.debug("Hinweis Anwendungsentwicklung im Prompt")
        lines.append(APP_DEV_HINT)
    lines.append("Nächste Frage als JSON:")
    return "\n".join(lines)


def _parse_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("keine vollständige JSON-Antwort (evtl. durch das Token-Limit abgeschnitten)")
    data = json.loads(raw[start:end + 1])
    for key in ("question", "feature"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"Feld '{key}' fehlt")
    return data


def _escape(text: str) -> str:
    """Fragen landen in HTML-Karten der App, daher werden Sonderzeichen maskiert."""
    return html.escape(re.sub(r"\s+", " ", text).strip(), quote=False)


# ------------------------------------------------------------ Fragen bauen
def _rationale(snapshot: IndexSnapshot, scores: list[ProgramScore], candidates: list[int], idx: int) -> str:
    meta = snapshot.chunk_meta[idx]
    others = [s.name for s in scores[: len(candidates)] if s.program_id != meta["program_id"]]
    return (f"Der Abschnitt „{_module_title(meta['module'])}“ gehört zu {meta['program_name']} und passt deutlich "
            f"weniger zu {' und '.join(others) if others else 'den übrigen Kandidaten'}.")


def _rule_based(snapshot: IndexSnapshot, scores: list[ProgramScore], candidates: list[int], idx: int) -> Question:
    meta = snapshot.chunk_meta[idx]
    return Question(
        text=_escape(f"Würde dich das Thema „{_module_title(meta['module'])}“ reizen?"),
        detail=_escape(_excerpt(snapshot.chunk_texts[idx], 220)),
        rationale=_rationale(snapshot, scores, candidates, idx),
        probe_text=_excerpt(snapshot.chunk_texts[idx], 300),
        source=FALLBACK_LABEL,
        chunk_id=snapshot.chunk_ids[idx],
    )


def _llm_question(
    chat: ChatModel,
    snapshot: IndexSnapshot,
    u: np.ndarray,
    scores: list[ProgramScore],
    candidates: list[int],
    picks: list[int],
    asked: list[Question],
    discarded: list[Question],
) -> Question:
    prompt = _prompt(snapshot, u, scores, candidates, picks, asked, discarded)
    data = _parse_json(chat.complete(SYSTEM_PROMPT, prompt, json_mode=True, max_tokens=settings.question_max_tokens))
    return Question(
        text=_escape(data["question"]),
        detail=_escape(str(data.get("detail", ""))),
        rationale=str(data.get("rationale", "")).strip() or _rationale(snapshot, scores, candidates, picks[0]),
        probe_text=data["feature"].strip(),
        source=chat.label,
        chunk_id=snapshot.chunk_ids[picks[0]],
    )


def generate_question(
    snapshot: IndexSnapshot,
    u: np.ndarray,
    scores: list[ProgramScore],
    asked: list[Question],
    asked_probes: list[np.ndarray],
    chat: ChatModel | None,
    embed: Callable[[str], np.ndarray] | None = None,
) -> Question:
    """Liefert die nächste Frage, die sich ausreichend von allen bisherigen unterscheidet.

    ``embed`` bettet den Merkmalssatz ein (für die Duplikat-Prüfung); das Ergebnis liegt danach in
    ``Question.probe_vec`` und muss vom Aufrufer nicht erneut berechnet werden. Ohne ``embed`` prüft
    nur der Wort-Jaccard.
    """
    candidates = [snapshot.program_ids.index(s.program_id) for s in scores[: settings.candidate_pool]]
    asked_modules = {
        (snapshot.chunk_meta[i]["program_id"], snapshot.chunk_meta[i]["module"])
        for q in asked if q.chunk_id in snapshot.chunk_ids for i in [snapshot.chunk_ids.index(q.chunk_id)]
    }
    picks = separating_chunks(snapshot, candidates, asked_probes, n=CANDIDATE_CHUNKS, asked_modules=asked_modules)
    if not picks:  # Pool der Kandidaten ist erschöpft: ohne Ausschlüsse neu versuchen, Duplikate fängt vet() ab
        picks = separating_chunks(snapshot, candidates, [], n=CANDIDATE_CHUNKS)
    if not picks:
        picks = snapshot.retrieve(u, k=1)
    number = len(asked) + 1
    discarded: list[Question] = []
    least_similar: tuple[float, Question] | None = None

    def vet(question: Question) -> bool:
        """True, wenn die Frage neu genug ist; sonst wird sie vermerkt und verworfen."""
        nonlocal least_similar
        question.probe_vec = embed(question.probe_text) if embed is not None else None
        similarity = history_similarity(question, question.probe_vec, asked, asked_probes)
        if similarity <= settings.duplicate_similarity:
            return True
        log.info("Frage %d verworfen (Ähnlichkeit %.3f > %.2f): %s", number, similarity,
                 settings.duplicate_similarity, question.text)
        discarded.append(question)
        if least_similar is None or similarity < least_similar[0]:
            least_similar = (similarity, question)
        return False

    note = ""
    if chat is not None:
        for lead in range(min(MAX_LLM_ATTEMPTS, len(picks))):
            try:  # der nächstbeste Chunk steht beim zweiten Versuch an erster Stelle
                question = _llm_question(chat, snapshot, u, scores, candidates,
                                         picks[lead:lead + SEPARATING_CHUNKS], asked, discarded)
            except Exception as exc:  # Netzwerk, Rate-Limit, kaputtes JSON … die Demo läuft regelbasiert weiter
                log.warning("LLM-Frage fehlgeschlagen (%s: %s) – nutze regelbasierte Frage.", type(exc).__name__, exc)
                note = f"{chat.label} nicht erreichbar oder unbrauchbar ({type(exc).__name__})"
                break
            if vet(question):
                log.info("Frage %d von %s: %s | Merkmal: %s", number, chat.label, question.text, question.probe_text)
                return question
        else:
            note = f"{chat.label}: Fragen zu ähnlich zu früheren"

    for idx in picks:  # nächstbester Chunk, bis eine neue Frage gefunden ist
        question = _rule_based(snapshot, scores, candidates, idx)
        if note:
            question.source = f"{FALLBACK_LABEL} · {note}"
        if vet(question):
            log.info("Frage %d regelbasiert: %s", number, question.text)
            return question

    # Alles wurde schon so ähnlich gefragt (kleiner Index, lange Session): die unähnlichste nehmen
    assert least_similar is not None
    log.warning("Frage %d: keine neue Frage gefunden, nehme die unähnlichste (%.2f).", number, least_similar[0])
    return least_similar[1]
