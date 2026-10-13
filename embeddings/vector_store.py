"""ChromaDB-Anbindung (Kosinus-Distanz) und der im Speicher gehaltene Index-Snapshot.

``hsd_chunks``   – ein Eintrag je Chunk (Embedding, Text, Metadaten)
``hsd_programs`` – ein Eintrag je Studiengang (Zentroid = normalisiertes Mittel seiner Chunks)

Zur Laufzeit wird der Index einmal in numpy-Arrays kopiert (:func:`load_snapshot`). Der Akinator-Loop
rechnet nur noch auf diesem Snapshot; ChromaDB und torch laufen so nie gleichzeitig im Hot Path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import settings
from embeddings.loader import Chunk

_BATCH = 500
_COSINE = {"hnsw:space": "cosine"}


def _client(path: Path | None = None):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(
        path=str(path or settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def program_centroids(chunks: list[Chunk], embeddings: np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    """Zentroid je Studiengang in Reihenfolge des ersten Auftretens."""
    ids = list(dict.fromkeys(c.program_id for c in chunks))
    names = [next(c.program_name for c in chunks if c.program_id == pid) for pid in ids]
    rows = np.array([[c.program_id == pid for c in chunks] for pid in ids])
    return ids, names, _normalize_rows(np.stack([embeddings[mask].mean(axis=0) for mask in rows]))


def rebuild_index(chunks: list[Chunk], embeddings: np.ndarray, path: Path | None = None) -> None:
    """Ersetzt beide Collections vollständig durch den übergebenen Stand."""
    client = _client(path)
    for name in (settings.chunks_collection, settings.programs_collection):
        try:
            client.delete_collection(name)
        except Exception:  # Collection existiert beim ersten Lauf noch nicht
            pass
    chunk_col = client.create_collection(settings.chunks_collection, metadata=_COSINE)
    for start in range(0, len(chunks), _BATCH):
        batch = chunks[start:start + _BATCH]
        chunk_col.add(
            ids=[c.id for c in batch],
            embeddings=embeddings[start:start + _BATCH].tolist(),
            documents=[c.text for c in batch],
            metadatas=[{"program_id": c.program_id, "program_name": c.program_name,
                        "module": c.module, "page": c.page} for c in batch],
        )
    ids, names, centroids = program_centroids(chunks, embeddings)
    client.create_collection(settings.programs_collection, metadata=_COSINE).add(
        ids=ids,
        embeddings=centroids.tolist(),
        metadatas=[{"program_id": i, "program_name": n} for i, n in zip(ids, names)],
    )


def query_chunks(query: np.ndarray, k: int = 3, program_id: str | None = None, path: Path | None = None) -> list[dict]:
    """RAG-Retrieval direkt gegen ChromaDB (Kosinus-Distanz → Ähnlichkeit = 1 − Distanz)."""
    collection = _client(path).get_collection(settings.chunks_collection)
    result = collection.query(
        query_embeddings=[query.tolist()],
        n_results=k,
        where={"program_id": program_id} if program_id else None,
    )
    return [
        {"id": i, "text": t, "similarity": 1.0 - d, **m}
        for i, t, d, m in zip(result["ids"][0], result["documents"][0], result["distances"][0], result["metadatas"][0])
    ]


@dataclass
class IndexSnapshot:
    program_ids: list[str]
    program_names: list[str]
    centroids: np.ndarray                 # (P, d), Einheitsvektoren
    chunk_ids: list[str]
    chunk_texts: list[str]
    chunk_meta: list[dict]
    chunk_embeddings: np.ndarray          # (N, d), Einheitsvektoren
    chunk_program: np.ndarray             # (N,), Index in program_ids
    _by_program: list[np.ndarray] = field(init=False, repr=False)
    _chunk_vs_centroid: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_program = [np.flatnonzero(self.chunk_program == p) for p in range(len(self.program_ids))]

    @property
    def is_empty(self) -> bool:
        return len(self.chunk_ids) == 0 or len(self.program_ids) == 0

    @property
    def prior(self) -> np.ndarray:
        """Neutraler Startpunkt: normalisiertes Mittel aller Studiengangs-Zentroide."""
        mean = self.centroids.mean(axis=0)
        return mean / max(float(np.linalg.norm(mean)), 1e-12)

    def chunks_of(self, program_idx: int) -> np.ndarray:
        return self._by_program[program_idx]

    @property
    def chunk_vs_centroid(self) -> np.ndarray:
        """``(N, P)``-Matrix cos(Chunk, Studiengangs-Zentroid), einmal berechnet und gecacht."""
        if self._chunk_vs_centroid is None:
            self._chunk_vs_centroid = self.chunk_embeddings @ self.centroids.T
        return self._chunk_vs_centroid

    def retrieve(self, query: np.ndarray, k: int = 3, program_idx: int | None = None) -> list[int]:
        """Indizes der ``k`` ähnlichsten Chunks (optional nur eines Studiengangs)."""
        sims = self.chunk_embeddings @ query
        pool = np.arange(len(sims)) if program_idx is None else self.chunks_of(program_idx)
        return pool[np.argsort(-sims[pool])[:k]].tolist()

    def subset(self, chunk_idx: np.ndarray) -> "IndexSnapshot":
        """Teil-Snapshot mit neu berechneten Zentroiden (für die Kalibrierung)."""
        chunk_idx = np.asarray(chunk_idx)
        emb = self.chunk_embeddings[chunk_idx]
        prog = self.chunk_program[chunk_idx]
        centroids = _normalize_rows(np.stack([emb[prog == p].mean(axis=0) for p in range(len(self.program_ids))]))
        return IndexSnapshot(
            self.program_ids, self.program_names, centroids,
            [self.chunk_ids[i] for i in chunk_idx], [self.chunk_texts[i] for i in chunk_idx],
            [self.chunk_meta[i] for i in chunk_idx], emb, prog,
        )


def empty_snapshot() -> IndexSnapshot:
    return IndexSnapshot([], [], np.zeros((0, 0), np.float32), [], [], [], np.zeros((0, 0), np.float32), np.zeros(0, int))


def load_snapshot(path: Path | None = None) -> IndexSnapshot:
    """Liest den persistierten Index komplett in den Speicher; fehlt er, ist der Snapshot leer."""
    client = _client(path)
    try:
        chunk_col = client.get_collection(settings.chunks_collection)
        program_col = client.get_collection(settings.programs_collection)
    except Exception:
        return empty_snapshot()

    programs = program_col.get(include=["embeddings", "metadatas"])
    order = sorted(range(len(programs["ids"])), key=lambda i: programs["ids"][i])
    program_ids = [programs["ids"][i] for i in order]
    program_names = [programs["metadatas"][i]["program_name"] for i in order]
    centroids = np.asarray(programs["embeddings"], dtype=np.float32)[order]

    chunks = chunk_col.get(include=["embeddings", "documents", "metadatas"])
    order = sorted(range(len(chunks["ids"])), key=lambda i: chunks["ids"][i])
    index = {pid: i for i, pid in enumerate(program_ids)}
    metas = [chunks["metadatas"][i] for i in order]
    return IndexSnapshot(
        program_ids, program_names, centroids,
        [chunks["ids"][i] for i in order],
        [chunks["documents"][i] for i in order],
        metas,
        np.asarray(chunks["embeddings"], dtype=np.float32)[order],
        np.array([index[m["program_id"]] for m in metas], dtype=int),
    )
