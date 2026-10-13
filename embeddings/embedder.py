"""Sentence-Transformers-Embeddings, L2-normalisiert (Skalarprodukt = Kosinus-Ähnlichkeit)."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from config import settings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Liefert eine ``(n, d)``-Matrix aus Einheitsvektoren (float32)."""
    if not texts:
        return np.zeros((0, get_model().get_sentence_embedding_dimension()), dtype=np.float32)
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
