"""Zentrale Einstellungen des HSD Study-Finders.

Alle Werte lassen sich per Umgebungsvariable bzw. ``.env`` überschreiben (siehe ``.env.example``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Reihenfolge = Reihenfolge der Antwort-Buttons in der App
DEFAULT_ANSWER_WEIGHTS = {"Ja": 1.0, "Eher ja": 0.5, "Weiß nicht": 0.0, "Eher nein": -0.5, "Nein": -1.0}

# Anzeigenamen je Studiengangskürzel (Kürzel = Dateiname bis zur ersten Klammer)
PROGRAM_NAMES = {
    "BMI": "Medieninformatik (B.Sc.)",
    "BMT": "Medientechnik (B.Eng.)",
    "BTB": "Ton und Bild (B.Eng.)",
    "DAISY": "Data Science, AI und Intelligente Systeme (B.Sc.)",
}


def _get(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _path(name: str, default: str) -> Path:
    path = Path(_get(name, default))
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    # Embeddings & Index
    embedding_model: str = field(default_factory=lambda: _get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", "data"))
    chroma_dir: Path = field(default_factory=lambda: _path("CHROMA_DIR", ".chroma"))
    chunks_collection: str = "hsd_chunks"
    programs_collection: str = "hsd_programs"
    chunk_size: int = field(default_factory=lambda: int(_get("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(_get("CHUNK_OVERLAP", "80")))

    # Scoring: cos → Prozent (linear, ohne Softmax)
    cos_floor: float = field(default_factory=lambda: float(_get("COS_FLOOR", "0.436")))
    cos_ceil: float = field(default_factory=lambda: float(_get("COS_CEIL", "0.758")))
    score_alpha: float = field(default_factory=lambda: float(_get("SCORE_ALPHA", "0.5")))
    score_top_k_chunks: int = field(default_factory=lambda: int(_get("SCORE_TOP_K_CHUNKS", "5")))
    prior_weight: float = field(default_factory=lambda: float(_get("PRIOR_WEIGHT", "1.0")))
    intro_weight: float = field(default_factory=lambda: float(_get("INTRO_WEIGHT", "1.0")))
    answer_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ANSWER_WEIGHTS))

    # Akinator-Loop
    max_questions: int = field(default_factory=lambda: int(_get("MAX_QUESTIONS", "12")))
    min_questions: int = field(default_factory=lambda: int(_get("MIN_QUESTIONS", "5")))
    duplicate_similarity: float = field(default_factory=lambda: float(_get("DUPLICATE_SIMILARITY", "0.70")))
    candidate_pool: int = field(default_factory=lambda: int(_get("CANDIDATE_POOL", "3")))
    stop_percent: float = field(default_factory=lambda: float(_get("STOP_PERCENT", "75")))
    stop_margin: float = field(default_factory=lambda: float(_get("STOP_MARGIN", "15")))

    # LLM
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "auto").lower())
    anthropic_api_key: str = field(default_factory=lambda: _get("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _get("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _get("OPENAI_MODEL", "gpt-4o-mini"))
    ollama_base_url: str = field(default_factory=lambda: _get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _get("OLLAMA_MODEL", "llama3.1:8b"))
    llm_timeout: float = 30.0
    question_max_tokens: int = field(default_factory=lambda: int(_get("QUESTION_MAX_TOKENS", "100")))
    explain_max_tokens: int = field(default_factory=lambda: int(_get("EXPLAIN_MAX_TOKENS", "300")))
    ollama_temperature: float = field(default_factory=lambda: float(_get("OLLAMA_TEMPERATURE", "0.3")))
    ollama_keep_alive: str = field(default_factory=lambda: _get("OLLAMA_KEEP_ALIVE", "30m"))

    # Logging (Logger-Familie "hsd.*": LLM-Auswahl, LLM-Aufrufe, Fallbacks)
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO").upper())


settings = Settings()


def _configure_logging() -> None:
    """Eigener Handler für ``hsd.*``, damit die Meldungen im Terminal auch unter Streamlit erscheinen."""
    logger = logging.getLogger("hsd")
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    logger.propagate = False


_configure_logging()
