"""Modulhandbücher einlesen (PDF / Markdown) und in Chunks zerlegen.

PDFs werden seitenweise mit ``PyPDFLoader`` gelesen, von Kopf-/Fußzeilen und Inhaltsverzeichnissen
befreit und anhand der Modul-Überschriften (``BMI 05: …``, ``D3.2: …``, ``PF 5.3 …``) einem Modul
zugeordnet. Markdown-Dateien dürfen YAML-artigen Front-Matter (``id``, ``name``) und ``#``-Überschriften
enthalten. Jeder Chunk trägt Studiengang, Modul und Seite als Metadaten.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PROGRAM_NAMES, settings

OVERVIEW_MODULE = "Studiengangsübersicht"
MIN_CHUNK_CHARS = 80
MIN_ALPHA_RATIO = 0.55

# Modul-Überschriften der vorhandenen Handbücher; der Titel steht in der Gruppe "title"
_MODULE_HEADINGS = [
    # BMI 05: Mathematik 1 · BMT 11 - Grundlagen Elektrotechnik · D3.2: Information Security
    re.compile(r"^(?P<id>(?:BMI|BMT|BTB|PF|D)\s?[A-Z]?\d+(?:\.\d+)?)\s*[:–-]\s*(?P<title>[^\W\d_].{2,120})$"),
    # PF 5.3 Business Analytics: Innovationsmanagement (hier ohne Doppelpunkt nach der Kennung)
    re.compile(r"^(?P<id>PF\s?\d+(?:\.\d+)?)\s+(?P<title>[^\W\d_].{2,120})$"),
    # Studiengang Ton und Bild Modul 3.1: Instrument / Gesang III · … – Schwerpunkt: Musikproduktion 2
    re.compile(r"^Studiengang\s+Ton und Bild\s*(?:[–-]\s*)?(?:Modul\s+)?(?P<title>\S.{3,140})$"),
]
# Endet ein Titel auf eines dieser Wörter, ist er in der nächsten Zeile umgebrochen
_DANGLING = {"der", "des", "die", "das", "und", "für", "von", "in", "im", "zu", "zur", "to", "of", "and", "for", "&", "-"}
_DOT_LEADER = re.compile(r"\.{4,}")
_HEADER_LINES, _FOOTER_LINES = 6, 3
_HEADING_SCAN_LINES = 10
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass(frozen=True)
class Chunk:
    id: str
    program_id: str
    program_name: str
    module: str
    page: int
    text: str


def program_id_from_path(path: Path) -> str:
    """``BMT(Medientechnik).pdf`` → ``BMT``."""
    return re.split(r"[\s(_.\-]", path.stem, maxsplit=1)[0].upper()


def _digits_masked(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip()))


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def strip_boilerplate(pages: list[str]) -> list[list[str]]:
    """Entfernt Kopf-/Fußzeilen, die (Ziffern maskiert) auf mindestens der Hälfte der Seiten stehen."""
    page_lines = [_lines(t) for t in pages]
    seen: Counter[str] = Counter()
    for lines in page_lines:
        edges = lines[:_HEADER_LINES] + lines[-_FOOTER_LINES:]
        seen.update({_digits_masked(line) for line in edges})
    limit = max(3, len(pages) * 0.5)
    repeated = {line for line, n in seen.items() if n >= limit}
    return [
        [line for i, line in enumerate(lines)
         if not ((i < _HEADER_LINES or i >= len(lines) - _FOOTER_LINES) and _digits_masked(line) in repeated)]
        for lines in page_lines
    ]


def is_toc_page(lines: list[str]) -> bool:
    """Inhaltsverzeichnis-Seiten erkennt man an den Punktlinien vor den Seitenzahlen."""
    return bool(lines) and sum(bool(_DOT_LEADER.search(line)) for line in lines) / len(lines) > 0.3


def detect_module(lines: list[str]) -> str | None:
    """Sucht im Seitenkopf eine Modul-Überschrift und liefert ``"<Kennung> <Titel>"``."""
    for i, line in enumerate(lines[:_HEADING_SCAN_LINES]):
        if _DOT_LEADER.search(line):
            continue
        for pattern in _MODULE_HEADINGS:
            match = pattern.match(line)
            if match:
                ident = match.groupdict().get("id")
                title = match["title"].strip().rstrip(":")
                if title.split()[-1].lower() in _DANGLING and i + 1 < len(lines):
                    title = f"{title} {lines[i + 1]}"
                return f"{ident.replace('  ', ' ')}: {title}" if ident else title
    return None


def _flatten(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(?=[a-zäöüß])", r"\1", text)  # Silbentrennung am Zeilenende
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _is_informative(text: str) -> bool:
    if len(text) < MIN_CHUNK_CHARS:
        return False
    return sum(c.isalpha() for c in text) / len(text) >= MIN_ALPHA_RATIO


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", ". ", "; ", " "],
    )


def _make_chunks(program_id: str, program_name: str, segments: list[tuple[str, int, str]]) -> list[Chunk]:
    """``segments`` = (Modul, Seite, Text). Das Modul wird dem Chunk-Text als Kontext vorangestellt."""
    splitter = _splitter()
    chunks: list[Chunk] = []
    for module, page, text in segments:
        for part in splitter.split_text(text):
            if _is_informative(part):
                chunks.append(Chunk(f"{program_id}-{len(chunks):05d}", program_id, program_name, module, page, f"{module} – {part}"))
    return chunks


def load_pdf(path: Path, program_id: str, program_name: str) -> list[Chunk]:
    documents = PyPDFLoader(str(path)).load()
    cleaned = strip_boilerplate([doc.page_content for doc in documents])
    segments: list[tuple[str, int, str]] = []
    module = OVERVIEW_MODULE
    for doc, lines in zip(documents, cleaned):
        if is_toc_page(lines):
            continue
        module = detect_module(lines) or module
        text = _flatten(lines)
        if text:
            segments.append((module, int(doc.metadata.get("page", 0)) + 1, text))
    return _make_chunks(program_id, program_name, segments)


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if value:
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, raw[match.end():]


def load_markdown(path: Path) -> list[Chunk]:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    program_id = meta.get("id", program_id_from_path(path)).upper()
    program_name = meta.get("name", PROGRAM_NAMES.get(program_id, path.stem))
    segments: list[tuple[str, int, str]] = []
    module = OVERVIEW_MODULE
    for block in re.split(r"(?m)^#{1,3}\s+", body):
        title, _, text = block.partition("\n")
        if text.strip():
            module = title.strip() or module
            segments.append((module, 0, _flatten(_lines(text))))
    return _make_chunks(program_id, program_name, segments)


def load_all_chunks(data_dir: Path | None = None) -> list[Chunk]:
    """Liest alle ``*.pdf`` in ``data/`` sowie ``*.md`` in ``data/`` und ``data/programs/``."""
    data_dir = data_dir or settings.data_dir
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.pdf")):
        program_id = program_id_from_path(path)
        chunks += load_pdf(path, program_id, PROGRAM_NAMES.get(program_id, path.stem))
    for path in sorted([*data_dir.glob("*.md"), *data_dir.glob("programs/*.md")]):
        if path.name.lower() != "readme.md":
            chunks += load_markdown(path)
    return chunks
